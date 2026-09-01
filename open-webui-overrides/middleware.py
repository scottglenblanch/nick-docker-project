import logging
import sys

from open_webui.env import ENABLE_API_OUTLET_FILTERS, GLOBAL_LOG_LEVEL
from open_webui.events import EVENTS, publish_event
from open_webui.models.chats import Chats
from open_webui.models.config import Config
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.misc import get_reasoning_details
from open_webui.utils.response import normalize_usage
from starlette.responses import StreamingResponse

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)



from open_webui.utils.middleware_helpers import output_id, process_tool_result as _process_tool_result_impl, publish_chat_finished_event


from open_webui.utils.middleware_runtime import (
    background_tasks_handler,
    build_chat_response_context as _build_chat_response_context_impl,
    build_response_object,
    drain_approved_tool_calls as _drain_approved_tool_calls_impl,
    get_response_data,
    get_system_oauth_token as _get_system_oauth_token_impl,
    merge_events_into_response,
    outlet_filter_handler,
)


from open_webui.utils.middleware_pipeline import process_chat_payload as _process_chat_payload_impl


async def process_tool_result(request, tool_function_name, tool_result, tool_type, direct_tool=False, metadata=None, user=None):
    return await _process_tool_result_impl(
        request,
        tool_function_name,
        tool_result,
        tool_type,
        direct_tool=direct_tool,
        metadata=metadata,
        user=user,
    )


async def build_chat_response_context(request, form_data, user, model, metadata, tasks, events):
    return await _build_chat_response_context_impl(request, form_data, user, model, metadata, tasks, events)


async def drain_approved_tool_calls(request, form_data, user, model, metadata):
    return await _drain_approved_tool_calls_impl(request, form_data, user, model, metadata)


async def get_system_oauth_token(request, user):
    return await _get_system_oauth_token_impl(request, user)


async def process_chat_payload(request, form_data, user, metadata, model):
    return await _process_chat_payload_impl(request, form_data, user, metadata, model)

async def non_streaming_chat_response_handler(response, ctx):
    request = ctx['request']

    user = ctx['user']
    metadata = ctx['metadata']
    events = ctx['events']

    event_emitter = ctx['event_emitter']

    response, response_data = get_response_data(response)
    if response_data is None:
        return response

    chat_id = metadata.get('chat_id') or ''
    save_to_chat = is_saved_chat_id(chat_id)

    if event_emitter:
        try:
            if 'error' in response_data:
                error = response_data.get('error')

                if isinstance(error, dict):
                    error = error.get('detail', error)
                else:
                    error = str(error)

                log.error('Provider returned error (non-streaming): %s', error)

                if save_to_chat:
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata['chat_id'],
                        metadata['message_id'],
                        {
                            'error': {'content': error},
                        },
                    )
                if isinstance(error, str) or isinstance(error, dict):
                    await event_emitter(
                        {
                            'type': 'chat:message:error',
                            'data': {'error': {'content': error}},
                        }
                    )

            if 'selected_model_id' in response_data and save_to_chat:
                await Chats.upsert_message_to_chat_by_id_and_message_id(
                    metadata['chat_id'],
                    metadata['message_id'],
                    {
                        'selectedModelId': response_data['selected_model_id'],
                    },
                    touch=False,
                )

            choices = response_data.get('choices', [])
            response_output = response_data.get('output')
            content = choices[0].get('message', {}).get('content') if choices else ''

            if choices and (content or response_output):
                if content or response_output:
                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': response_data,
                        }
                    )

                    title = await Chats.get_chat_title_by_id(metadata['chat_id']) if save_to_chat else ''

                    # Use output from backend if provided (OR-compliant backends),
                    # otherwise generate from response content
                    if not response_output:
                        choice_message = choices[0].get('message', {})
                        reasoning_content = choice_message.get('reasoning_content') or choice_message.get('reasoning')
                        reasoning_details = get_reasoning_details(choice_message)
                        response_output = []
                        if reasoning_content or reasoning_details:
                            reasoning_item = {
                                'type': 'reasoning',
                                'id': output_id('r'),
                                'status': 'completed',
                                'start_tag': '<think>',
                                'end_tag': '</think>',
                                'attributes': {'type': 'reasoning_content'},
                                'content': (
                                    [{'type': 'output_text', 'text': reasoning_content}] if reasoning_content else []
                                ),
                                'summary': None,
                            }
                            if reasoning_details:
                                reasoning_item['reasoning_details'] = (
                                    reasoning_details if isinstance(reasoning_details, list) else [reasoning_details]
                                )
                            response_output.append(reasoning_item)
                        response_output.append(
                            {
                                'type': 'message',
                                'id': output_id('msg'),
                                'status': 'completed',
                                'role': 'assistant',
                                'content': [{'type': 'output_text', 'text': content}],
                            }
                        )

                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': {
                                'done': True,
                                'output': response_output,
                                'title': title,
                            },
                        }
                    )

                    # Save message in the database
                    usage = normalize_usage(response_data.get('usage', {}) or {})

                    if save_to_chat:
                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                            metadata['chat_id'],
                            metadata['message_id'],
                            {
                                'done': True,
                                'role': 'assistant',
                                'output': response_output,
                                **({'usage': usage} if usage else {}),
                            },
                        )

                    await publish_chat_finished_event(request, user, metadata, title, content, response_output)

                    ctx['assistant_message'] = {
                        'content': content,
                        'output': response_output,
                        **({'usage': usage} if usage else {}),
                    }
                    await outlet_filter_handler(ctx)
                    await background_tasks_handler(ctx)

            response = build_response_object(response, merge_events_into_response(response_data, events))
        except Exception as e:
            log.debug('Error occurred while processing request: %s', e)
            chat_id = metadata.get('chat_id')
            if getattr(request.state, 'internal', False) is not True and chat_id and is_saved_chat_id(chat_id):
                webui_url = await Config.get('webui.url')
                await publish_event(
                    request,
                    EVENTS.CHAT_FAILED,
                    actor=user,
                    subject_id=chat_id,
                    subject_type='chat',
                    data={
                        'user_id': user.id,
                        'chat_id': chat_id,
                        'message_id': metadata.get('message_id'),
                        'model_id': metadata.get('model_id'),
                        'url': f'{webui_url}/c/{chat_id}' if webui_url else f'/c/{chat_id}',
                        'message': str(e),
                    },
                    message='Chat failed',
                )
            pass

        return response

    choices = response_data.get('choices', [])
    output = response_data.get('output')
    content = choices[0].get('message', {}).get('content') if choices else ''
    if ENABLE_API_OUTLET_FILTERS and (content or output):
        usage = normalize_usage(response_data.get('usage', {}) or {})
        ctx['assistant_message'] = {
            **({'content': content} if content else {}),
            **({'output': output} if output else {}),
            **({'usage': usage} if usage else {}),
        }
        await outlet_filter_handler(ctx)

    if isinstance(response, dict):
        response = merge_events_into_response(response_data, events)

    return response



from open_webui.utils.middleware_streaming import (
    streaming_chat_response_handler as _streaming_chat_response_handler_impl,
)


async def streaming_chat_response_handler(response, ctx):
    return await _streaming_chat_response_handler_impl(response, ctx)

async def process_chat_response(response, ctx):
    # Non-streaming response
    if not isinstance(response, StreamingResponse):
        return await non_streaming_chat_response_handler(response, ctx)

    # Non standard response
    if not any(
        content_type in response.headers['Content-Type']
        for content_type in ['text/event-stream', 'application/x-ndjson']
    ):
        return response

    # Streaming response
    return await streaming_chat_response_handler(response, ctx)
