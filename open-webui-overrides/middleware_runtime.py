import ast
import copy
import logging
import re
import sys
import time
from uuid import uuid4

from fastapi.responses import JSONResponse
from open_webui.constants import TASKS
from open_webui.env import ENABLE_PLUGINS, GLOBAL_LOG_LEVEL
from open_webui.models.chats import Chats
from open_webui.models.oauth_sessions import OAuthSessions
from open_webui.models.users import UserModel
from open_webui.routers.pipelines import get_sorted_filters, process_pipeline_outlet_filter
from open_webui.routers.tasks import generate_chat_tags, generate_follow_ups, generate_title
from open_webui.socket.main import (
    get_event_call,
    get_event_emitter,
)
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.filter import get_filter_context, get_filter_functions, process_filter_functions
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.memory import review_memory_after_turn
from open_webui.utils.misc import get_last_user_message, get_last_user_message_item, get_message_list, get_output_text
from open_webui.utils.response import merge_usage
from open_webui.utils.tools import get_updated_tool_function

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)



from open_webui.utils.middleware_helpers import MESSAGE_REPLAY_KEYS, _is_tool_result_error, build_terminal_file_tool_result, get_reasoning_format, load_messages_from_db, normalize_messages_for_model, output_id, process_messages_with_output, process_tool_result, sanitize_tool_pairs, strip_reasoning_details, terminal_event_handler, tool_result_content, handle_responses_streaming_event

async def get_event_emitter_and_caller(metadata):
    event_emitter = None
    event_caller = None

    # event_emitter only needs user_id + chat_id + message_id.
    # It broadcasts to user:{user_id} room AND persists to DB,
    # so it works for backend-initiated calls (automations, API).
    if metadata.get('chat_id') and metadata.get('message_id'):
        event_emitter = await get_event_emitter(metadata)

    # event_caller needs session_id — it calls back to a specific
    # websocket session (used by direct tools, pyodide code interpreter).
    if metadata.get('session_id') and metadata.get('chat_id') and metadata.get('message_id'):
        event_caller = await get_event_call(metadata)

    return event_emitter, event_caller


async def build_chat_response_context(request, form_data, user, model, metadata, tasks, events):
    event_emitter, event_caller = await get_event_emitter_and_caller(metadata)
    return {
        'request': request,
        'form_data': form_data,
        'user': user,
        'model': model,
        'metadata': metadata,
        'tasks': tasks,
        'events': events,
        'event_emitter': event_emitter,
        'event_caller': event_caller,
    }


async def execute_tool_call_for_output(request, form_data, user, metadata, event_caller, event_emitter, tool_call):
    tools = metadata.get('tools', {})
    name = tool_call.get('function', {}).get('name', '')
    tool_args = tool_call.get('function', {}).get('arguments', '{}')
    params = {}
    if tool_args and tool_args.strip():
        try:
            params = JSONCodec.loads(tool_args)
        except Exception:
            try:
                params = ast.literal_eval(tool_args)
            except Exception as e:
                log.debug(e)
                return {
                    'tool_call_id': tool_call.get('id', ''),
                    'content': (
                        'Error: Tool call arguments could not be parsed. '
                        'The model generated malformed or incomplete JSON.'
                    ),
                }
    tool_call.setdefault('function', {})['arguments'] = JSONCodec.dumps(params)

    tool = tools.get(name)
    if not tool:
        return {'tool_call_id': tool_call.get('id', ''), 'content': f'Error: Tool "{name}" not found.'}

    spec = tool.get('spec', {})
    tool_type = tool.get('type', '')
    direct_tool = tool.get('direct', False)
    allowed_params = spec.get('parameters', {}).get('properties', {}).keys()
    params = {key: value for key, value in params.items() if key in allowed_params}

    try:
        if direct_tool:
            if not event_caller:
                result = 'Error: Browser session is not connected for this direct tool.'
            else:
                result = await event_caller(
                    {
                        'type': 'execute:tool',
                        'data': {
                            'id': str(uuid4()),
                            'name': name,
                            'params': params,
                            'server': tool.get('server', {}),
                            'session_id': metadata.get('session_id'),
                        },
                    }
                )
        else:
            function = await get_updated_tool_function(
                function=tool['callable'],
                extra_params={
                    '__messages__': form_data.get('messages', []),
                    '__files__': metadata.get('files', []),
                },
            )
            result = await function(**params)
    except Exception as e:
        result = {'error': str(e)}

    terminal_file_result = build_terminal_file_tool_result(name, params, result, tool, metadata)
    if terminal_file_result:
        result = terminal_file_result

    result, files, embeds = await process_tool_result(
        request,
        name,
        result,
        tool_type,
        direct_tool,
        metadata,
        user,
    )

    await terminal_event_handler(name, params, result, event_emitter)

    return {
        'tool_call_id': tool_call.get('id', ''),
        'content': tool_result_content(result),
        **({'files': files} if files else {}),
        **({'embeds': embeds} if embeds else {}),
    }


async def drain_approved_tool_calls(request, form_data, user, model, metadata) -> bool:
    chat_id = metadata.get('chat_id')
    assistant_message_id = metadata.get('assistant_message_id')
    # Only a resume/continue payload re-enters an existing message; other paths mint a fresh id with nothing to drain.
    if not is_saved_chat_id(chat_id) or not assistant_message_id:
        return False

    message_id = metadata.get('message_id') or assistant_message_id
    message = await Chats.get_message_by_id_and_message_id(chat_id, message_id)
    output = message.get('output') if message else None
    if not isinstance(output, list):
        return False

    result_call_ids = {
        item.get('call_id') for item in output if item.get('type') == 'function_call_output' and item.get('call_id')
    }
    approved_calls = [
        item
        for item in output
        if item.get('type') == 'function_call'
        and item.get('call_id')
        and item.get('status') == 'queued'
        and item.get('approved') is True
        and item.get('call_id') not in result_call_ids
    ]
    if not approved_calls:
        if metadata.get('params', {}).get('tool_approval_mode', 'full') == 'ask' and any(
            item.get('type') == 'function_call'
            and item.get('name') != 'ask_user'
            and (item.get('call_id') or item.get('id'))
            and item.get('status') == 'queued'
            and item.get('approved') is not True
            and (item.get('call_id') or item.get('id')) not in result_call_ids
            for item in output
        ):
            event_emitter, _ = await get_event_emitter_and_caller(metadata)
            await pause_for_tool_approval(chat_id, message_id, output, form_data, metadata)
            if event_emitter:
                await event_emitter({'type': 'chat:completion', 'data': {'done': False, 'output': output}})
            return True
        return False

    event_emitter, event_caller = await get_event_emitter_and_caller(metadata)
    changed = False
    for item in approved_calls:
        if item.get('name') == 'ask_user':
            item['status'] = 'pending'
            item.pop('approved', None)
            changed = True
            continue

        tool_call = {
            'id': item.get('call_id', ''),
            'type': 'function',
            'function': {
                'name': item.get('name', ''),
                'arguments': item.get('arguments', '{}'),
            },
        }
        result = await execute_tool_call_for_output(
            request,
            form_data,
            user,
            metadata,
            event_caller,
            event_emitter,
            tool_call,
        )
        item['arguments'] = tool_call.get('function', {}).get('arguments', '{}')
        output_parts = [{'type': 'input_text', 'text': result.get('content', '')}]
        item['status'] = 'failed' if _is_tool_result_error(result.get('content', '')) else 'completed'
        display_files = []
        for file_item in result.get('files', []):
            if file_item.get('type') == 'image' and file_item.get('url', '').startswith('data:'):
                output_parts.append({'type': 'input_image', 'image_url': file_item['url']})
            else:
                display_files.append(file_item)

        output.append(
            {
                'type': 'function_call_output',
                'id': output_id('fco'),
                'call_id': result.get('tool_call_id', ''),
                'output': output_parts,
                'status': item['status'],
                **({'files': display_files} if display_files else {}),
                **({'embeds': result.get('embeds')} if result.get('embeds') else {}),
            }
        )
        changed = True

    if changed:
        result_call_ids = {
            item.get('call_id') for item in output if item.get('type') == 'function_call_output' and item.get('call_id')
        }
        if metadata.get('params', {}).get('tool_approval_mode', 'full') == 'ask' and any(
            item.get('type') == 'function_call'
            and item.get('name') != 'ask_user'
            and (item.get('call_id') or item.get('id'))
            and item.get('status') == 'queued'
            and item.get('approved') is not True
            and (item.get('call_id') or item.get('id')) not in result_call_ids
            for item in output
        ):
            await pause_for_tool_approval(chat_id, message_id, output, form_data, metadata)
            result_call_ids = {
                item.get('call_id')
                for item in output
                if item.get('type') == 'function_call_output' and item.get('call_id')
            }
        paused = any(
            item.get('type') == 'function_call'
            and item.get('call_id')
            and item.get('status') in {'pending', 'queued', 'requires_approval'}
            and item.get('call_id') not in result_call_ids
            for item in output
        )
        if not paused:
            output.append(
                {
                    'type': 'message',
                    'id': output_id('msg'),
                    'status': 'in_progress',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': ''}],
                }
            )

        await Chats.upsert_message_to_chat_by_id_and_message_id(
            chat_id,
            message_id,
            {'done': False, 'output': output},
            touch=False,
        )
        if event_emitter:
            await event_emitter(
                {
                    'type': 'chat:completion',
                    'data': {
                        'done': False,
                        'output': output,
                    },
                }
            )

        db_messages = await load_messages_from_db(chat_id, metadata.get('user_message_id'))
        if db_messages:
            assistant_message = await Chats.get_message_by_id_and_message_id(chat_id, message_id)
            if assistant_message:
                db_messages.append({k: v for k, v in assistant_message.items() if k in MESSAGE_REPLAY_KEYS})
            for message in db_messages:
                output = message.get('output')
                # reasoning_details can be model/provider-bound, so only replay them
                # for output produced by the same model.
                if (
                    message.get('role') == 'assistant'
                    and message.get('model') != model['id']
                    and isinstance(output, list)
                ):
                    message['output'] = strip_reasoning_details(output)

            form_data['messages'] = process_messages_with_output(
                db_messages,
                reasoning_format=get_reasoning_format(model),
            )
            form_data['messages'] = sanitize_tool_pairs(form_data['messages'])

        if not paused and ENABLE_PLUGINS:
            filter_functions = await get_filter_functions(request, model, metadata.get('filter_ids', []))
            if filter_functions:
                filtered_form_data, _ = await process_filter_functions(
                    request=request,
                    filter_context=get_filter_context(request),
                    filter_functions=filter_functions,
                    filter_type='request',
                    form_data=form_data,
                    extra_params={
                        '__event_emitter__': event_emitter,
                        '__event_call__': event_caller,
                        '__user__': user.model_dump() if isinstance(user, UserModel) else {},
                        '__metadata__': metadata,
                        '__oauth_token__': await get_system_oauth_token(request, user),
                        '__request__': request,
                        '__model__': model,
                        '__chat_id__': metadata.get('chat_id'),
                        '__message_id__': metadata.get('message_id'),
                    },
                )
                if filtered_form_data is not form_data:
                    form_data.clear()
                    form_data.update(filtered_form_data)

        if not paused:
            normalize_messages_for_model(form_data)

        return paused

    return False


async def pause_for_tool_approval(chat_id: str, message_id: str, output: list[dict], form_data: dict, metadata: dict):
    result_call_ids = {
        item.get('call_id') for item in output if item.get('type') == 'function_call_output' and item.get('call_id')
    }
    has_pending_approval = False
    for item in output:
        if item.get('type') == 'function_call' and not item.get('call_id') and item.get('id'):
            item['call_id'] = item['id']

        if (
            item.get('type') == 'function_call'
            and item.get('call_id')
            and item.get('call_id') not in result_call_ids
            and item.get('status') != 'rejected'
        ):
            if not has_pending_approval:
                item['status'] = 'pending'
                has_pending_approval = True
            elif item.get('status') == 'in_progress':
                item['status'] = 'queued'

    await Chats.upsert_message_to_chat_by_id_and_message_id(
        chat_id,
        message_id,
        {
            'done': False,
            'output': output,
            'meta': {
                **(metadata.get('tool_approval') or {}),
                'session_id': metadata.get('session_id'),
                'tool_ids': metadata.get('tool_ids') or [],
                'skill_ids': metadata.get('skill_ids') or [],
                'terminal_id': metadata.get('terminal_id'),
                'tool_servers': metadata.get('tool_servers'),
                'filter_ids': metadata.get('filter_ids') or [],
                'features': metadata.get('features') or {},
                'variables': metadata.get('variables') or {},
                'files': metadata.get('files') or [],
                'params': metadata.get('params') or {},
            },
        },
        touch=False,
    )


def get_response_data(response):
    if isinstance(response, list) and len(response) == 1:
        # If the response is a single-item list, unwrap it #17213
        response = response[0]

    if isinstance(response, JSONResponse):
        if isinstance(response.body, bytes):
            try:
                response_data = JSONCodec.loads(response.body.decode('utf-8', 'replace'))
            except JSONCodec.JSONDecodeError:
                response_data = {'error': {'detail': 'Invalid JSON response'}}
        else:
            response_data = response
    elif isinstance(response, dict):
        response_data = response
    else:
        response_data = None

    return response, response_data


def merge_events_into_response(response_data, events):
    if events and isinstance(events, list):
        extra_response = {}
        for event in events:
            if isinstance(event, dict):
                extra_response.update(event)
            else:
                extra_response[event] = True

        return {
            **extra_response,
            **response_data,
        }
    return response_data


def build_response_object(response, response_data):
    if isinstance(response, dict):
        return response_data
    if isinstance(response, JSONResponse):
        return JSONResponse(
            content=response_data,
            headers=response.headers,
            status_code=response.status_code,
        )
    return response


def update_assistant_message_from_stream(assistant_message, raw):
    line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
    if not isinstance(line, str):
        return

    def append_output_text(item, text):
        parts = item.setdefault('content', [])
        if parts and parts[-1].get('type') == 'output_text':
            parts[-1]['text'] += text
        else:
            parts.append({'type': 'output_text', 'text': text})

    for raw_part in line.splitlines():
        part = raw_part.removeprefix('data:').strip()
        if not part or part == '[DONE]':
            continue

        try:
            data = JSONCodec.loads(part)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if data.get('type', '').startswith('response.'):
            output, meta = handle_responses_streaming_event(data, assistant_message.get('output', []))
            if output:
                assistant_message['output'] = output
            if meta and meta.get('usage'):
                assistant_message['usage'] = merge_usage(assistant_message.get('usage'), meta['usage'])
            continue

        raw_usage = data.get('usage', {}) or {}
        raw_usage.update(data.get('timings', {}))
        if raw_usage:
            assistant_message['usage'] = merge_usage(assistant_message.get('usage'), raw_usage)

        for choice in data.get('choices', []):
            delta = choice.get('delta', {}) or {}
            content = delta.get('content')
            reasoning_content = delta.get('reasoning_content') or delta.get('reasoning') or delta.get('thinking')

            if reasoning_content:
                output = assistant_message.setdefault('output', [])
                if not output or output[-1].get('type') != 'reasoning':
                    output.append(
                        {
                            'type': 'reasoning',
                            'id': output_id('r'),
                            'status': 'in_progress',
                            'start_tag': '<think>',
                            'end_tag': '</think>',
                            'attributes': {'type': 'reasoning_content'},
                            'content': [],
                            'summary': None,
                            'started_at': time.time(),
                        }
                    )

                append_output_text(output[-1], reasoning_content)

            if content:
                output = assistant_message.get('output')
                if output:
                    if output[-1].get('type') == 'reasoning':
                        output[-1]['status'] = 'completed'
                        output[-1]['ended_at'] = time.time()
                        output[-1]['duration'] = int(output[-1]['ended_at'] - output[-1]['started_at'])

                    if not output or output[-1].get('type') != 'message':
                        output.append(
                            {
                                'type': 'message',
                                'id': output_id('msg'),
                                'status': 'in_progress',
                                'role': 'assistant',
                                'content': [],
                            }
                        )

                    append_output_text(output[-1], content)

                assistant_message['content'] = assistant_message.get('content', '') + content


async def get_system_oauth_token(request, user):
    """Get the system OAuth token for a user.

    Primary path: use the oauth_session_id cookie (browser requests).
    Fallback: look up the user's most recent OAuth session from the DB
    (covers automations, API calls, and other cookie-less contexts).
    """
    oauth_token = None
    try:
        oauth_session_id = request.cookies.get('oauth_session_id', None)
        if oauth_session_id:
            oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                user.id,
                oauth_session_id,
            )

        # Fallback: no cookie (automation, API key, etc.) — use most recent session
        if oauth_token is None:
            from open_webui.models.oauth_sessions import OAuthSessions

            sessions = await OAuthSessions.get_sessions_by_user_id(user.id)
            # Filter out MCP-provider sessions — their token refresh is handled
            # separately by oauth_client_manager.  Passing them to the SSO
            # oauth_manager causes a failed refresh and session deletion (#24618).
            sessions = [s for s in sessions if not (s.provider or '').startswith('mcp:')]
            if sessions:
                best = max(sessions, key=lambda s: s.updated_at)
                oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                    user.id,
                    best.id,
                )
    except Exception as e:
        log.error(f'Error getting OAuth token: {e}')
    return oauth_token


async def background_tasks_handler(ctx):
    request = ctx['request']
    form_data = ctx['form_data']
    user = ctx['user']
    metadata = ctx['metadata']
    tasks = ctx['tasks']
    event_emitter = ctx['event_emitter']

    message = None
    messages = []

    if is_saved_chat_id(metadata.get('chat_id')):
        messages_map = await Chats.get_messages_map_by_chat_id(metadata['chat_id'])
        if not messages_map:
            # Chat was deleted while the response was streaming — skip background tasks
            return
        message = messages_map.get(metadata['message_id'])

        message_list = get_message_list(messages_map, metadata['message_id'])

        # Remove details tags and files from the messages.
        # as get_message_list creates a new list, it does not affect
        # the original messages outside of this handler

        messages = []
        for message in message_list:
            content = message.get('content', '')
            if isinstance(content, list):
                for item in content:
                    if item.get('type') == 'text':
                        content = item['text']
                        break

            if isinstance(content, str):
                content = re.sub(
                    r'<details\b[^>]*>.*?<\/details>|!\[.*?\]\(.*?\)',
                    '',
                    content,
                    flags=re.S | re.I,
                ).strip()

            messages.append(
                {
                    **message,
                    'role': message.get('role', 'assistant'),  # Safe fallback for missing role
                    'content': content,
                }
            )
    else:
        # Local temp chat, get the model and message from the form_data
        message = get_last_user_message_item(form_data.get('messages', []))
        messages = form_data.get('messages', [])
        if message:
            message['model'] = form_data.get('model')

    if message and 'model' in message:
        if tasks and messages:
            if TASKS.FOLLOW_UP_GENERATION in tasks and tasks[TASKS.FOLLOW_UP_GENERATION]:
                res = await generate_follow_ups(
                    request,
                    {
                        'model': message['model'],
                        'messages': messages,
                        'message_id': metadata['message_id'],
                        'chat_id': metadata['chat_id'],
                    },
                    user,
                )

                if res and isinstance(res, dict):
                    if len(res.get('choices', [])) == 1:
                        response_message = res.get('choices', [])[0].get('message', {})

                        follow_ups_string = response_message.get('content') or response_message.get(
                            'reasoning_content', ''
                        )
                    else:
                        follow_ups_string = ''

                    follow_ups_string = follow_ups_string[
                        follow_ups_string.find('{') : follow_ups_string.rfind('}') + 1
                    ]

                    try:
                        follow_ups = JSONCodec.loads(follow_ups_string).get('follow_ups', [])
                        await event_emitter(
                            {
                                'type': 'chat:message:follow_ups',
                                'data': {
                                    'follow_ups': follow_ups,
                                },
                            }
                        )

                        if is_saved_chat_id(metadata.get('chat_id')):
                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                metadata['chat_id'],
                                metadata['message_id'],
                                {
                                    'followUps': follow_ups,
                                },
                                touch=False,
                            )

                    except Exception as e:
                        pass

            if is_saved_chat_id(metadata.get('chat_id')):  # Only update titles and tags for saved chats
                if TASKS.TITLE_GENERATION in tasks:
                    user_message = get_last_user_message(messages)
                    if user_message and len(user_message) > 100:
                        user_message = user_message[:100] + '...'

                    title = None
                    if tasks[TASKS.TITLE_GENERATION]:
                        res = await generate_title(
                            request,
                            {
                                'model': message['model'],
                                'messages': messages,
                                'chat_id': metadata['chat_id'],
                            },
                            user,
                        )

                        if res and isinstance(res, dict):
                            if len(res.get('choices', [])) == 1:
                                response_message = res.get('choices', [])[0].get('message', {})

                                title_string = (
                                    response_message.get('content')
                                    or response_message.get(
                                        'reasoning_content',
                                    )
                                    or message.get('content', user_message)
                                )
                            else:
                                title_string = ''

                            title_string = title_string[title_string.find('{') : title_string.rfind('}') + 1]

                            try:
                                title = JSONCodec.loads(title_string).get('title', user_message)
                            except Exception as e:
                                title = ''

                            if not title:
                                title = messages[0].get('content', user_message)

                            await Chats.update_chat_title_by_id(metadata['chat_id'], title)

                            await event_emitter(
                                {
                                    'type': 'chat:title',
                                    'data': title,
                                }
                            )

                    if title == None and len(messages) == 2 and (not messages_map or len(messages_map) <= 2):
                        title = messages[0].get('content', user_message)

                        await Chats.update_chat_title_by_id(metadata['chat_id'], title)

                        await event_emitter(
                            {
                                'type': 'chat:title',
                                'data': message.get('content', user_message),
                            }
                        )

                if TASKS.TAGS_GENERATION in tasks and tasks[TASKS.TAGS_GENERATION]:
                    res = await generate_chat_tags(
                        request,
                        {
                            'model': message['model'],
                            'messages': messages,
                            'chat_id': metadata['chat_id'],
                        },
                        user,
                    )

                    if res and isinstance(res, dict):
                        if len(res.get('choices', [])) == 1:
                            response_message = res.get('choices', [])[0].get('message', {})

                            tags_string = response_message.get('content') or response_message.get(
                                'reasoning_content', ''
                            )
                        else:
                            tags_string = ''

                        tags_string = tags_string[tags_string.find('{') : tags_string.rfind('}') + 1]

                        try:
                            tags = JSONCodec.loads(tags_string).get('tags', [])
                            await Chats.update_chat_tags_by_id(metadata['chat_id'], tags, user)

                            await event_emitter(
                                {
                                    'type': 'chat:tags',
                                    'data': tags,
                                }
                            )
                        except Exception as e:
                            pass

        if messages:
            await review_memory_after_turn(
                request=request,
                user=user,
                model=ctx['model'],
                metadata=metadata,
                form_data=form_data,
                assistant_message=ctx.get('assistant_message') or {},
                messages=messages,
            )


async def outlet_filter_handler(ctx):
    """Run outlet filters inline after chat completion.

    Replaces the separate POST /api/chat/completed round-trip.
    Persists outlet-modified content to DB and emits a chat:outlet event
    so the frontend can sync its in-memory state. Returns immediately when
    the model has no filters.

    For temp/API chats, messages are built from form_data plus ctx['assistant_message'].
    """
    request = ctx['request']
    user = ctx['user']
    model = ctx['model']
    metadata = ctx['metadata']
    event_emitter = ctx.get('event_emitter')
    event_caller = ctx.get('event_caller')

    chat_id = metadata.get('chat_id', '')
    message_id = metadata.get('message_id')

    if not chat_id and not ctx.get('assistant_message'):
        return

    if not message_id:
        message_id = output_id('msg')

    is_unsaved_chat = not is_saved_chat_id(chat_id)
    try:
        filter_functions = (
            await get_filter_functions(request, model, metadata.get('filter_ids', [])) if ENABLE_PLUGINS else []
        )
        model_id = model.get('id') if isinstance(model, dict) else model
        models = request.app.state.MODELS
        has_pipeline_outlet_filters = bool(
            (isinstance(model, dict) and 'pipeline' in model) or get_sorted_filters(model_id, models)
        )
        if not filter_functions and not has_pipeline_outlet_filters:
            return

        messages_map = None

        if is_unsaved_chat:
            form_messages = ctx.get('form_data', {}).get('messages', [])
            assistant_message = ctx.get('assistant_message', {})

            message_list = [
                {
                    'role': m.get('role'),
                    'content': m.get('content') or get_output_text(m.get('output')),
                }
                for m in form_messages
            ]

            if assistant_message:
                message_list.append(
                    {
                        'id': message_id,
                        'role': 'assistant',
                        **assistant_message,
                    }
                )

            if not message_list:
                return
        else:
            messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
            if not messages_map:
                return

            message_list = get_message_list(messages_map, message_id)
            if not message_list:
                return

        outlet_data = {
            'model': model_id,
            'messages': [
                {
                    'id': m.get('id'),
                    'role': m.get('role'),
                    'content': m.get('content') or get_output_text(m.get('output')),
                    'info': m.get('info'),
                    'timestamp': m.get('timestamp'),
                    # Deepcopy so in-place filter mutations do not alias messages_map's baseline
                    **({'output': copy.deepcopy(m['output'])} if m.get('output') else {}),
                    **({'usage': m['usage']} if m.get('usage') else {}),
                    **({'sources': m['sources']} if m.get('sources') else {}),
                }
                for m in message_list
            ],
            'filter_ids': metadata.get('filter_ids', []),
            'chat_id': chat_id,
            'session_id': metadata.get('session_id'),
            'id': message_id,
        }

        # Pipeline outlet filters
        try:
            outlet_data = await process_pipeline_outlet_filter(request, outlet_data, user, models)
        except Exception as e:
            log.debug('Pipeline outlet filter error: %s', e)

        # Function outlet filters
        extra_params = {
            '__event_emitter__': event_emitter,
            '__event_call__': event_caller,
            '__user__': user.model_dump() if isinstance(user, UserModel) else {},
            '__metadata__': metadata,
            '__request__': request,
            '__model__': model,
        }

        if filter_functions:
            outlet_result, _ = await process_filter_functions(
                request=request,
                filter_context=None,
                filter_functions=filter_functions,
                filter_type='outlet',
                form_data=outlet_data,
                extra_params=extra_params,
            )
        else:
            outlet_result = outlet_data

        if outlet_result and outlet_result.get('messages'):
            if not is_unsaved_chat and messages_map:
                for message in outlet_result['messages']:
                    outlet_message_id = message.get('id')
                    if outlet_message_id and outlet_message_id in messages_map:
                        original_message = messages_map[outlet_message_id]
                        original_content = original_message.get('content') or get_output_text(
                            original_message.get('output')
                        )
                        message_content = message.get('content') or get_output_text(message.get('output'))
                        content_changed = original_content != message_content
                        output_changed = message.get('output') and message.get('output') != original_message.get(
                            'output'
                        )
                        if content_changed or output_changed:
                            message_update = {
                                'originalContent': original_content,
                                **({'output': message['output']} if output_changed else {}),
                            }
                            if content_changed:
                                message_update['content'] = message_content or ''
                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                chat_id,
                                outlet_message_id,
                                message_update,
                            )

            if event_emitter:
                await event_emitter(
                    {
                        'type': 'chat:outlet',
                        'data': {'messages': outlet_result['messages']},
                    }
                )
    except Exception as e:
        log.debug('Error running outlet filters: %s', e)


