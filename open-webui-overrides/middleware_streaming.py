import ast
import asyncio
import logging
import re
import sys
import textwrap
import time
from uuid import uuid4

from fastapi import HTTPException
from open_webui.config import CODE_INTERPRETER_BLOCKED_MODULES
from open_webui.env import CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS, CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE, ENABLE_API_OUTLET_FILTERS, ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION, ENABLE_PLUGINS, ENABLE_RESPONSES_API_STATEFUL, GLOBAL_LOG_LEVEL, RAG_SYSTEM_CONTEXT
from open_webui.models.chats import Chats
from open_webui.models.config import Config
from open_webui.models.users import UserModel
from open_webui.routers.pipelines import get_sorted_filters
from open_webui.tasks import clear_response_stream, save_response_stream
from open_webui.utils.access_control import has_permission
from open_webui.utils.ask_user import stage_ask_user_tool_calls
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.code_interpreter import execute_code_jupyter
from open_webui.utils.files import convert_markdown_base64_images, get_image_url_from_base64
from open_webui.utils.filter import FilterContext, get_filter_functions, process_filter_functions
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.misc import add_or_update_system_message, add_or_update_user_message, convert_output_to_messages, get_content_from_message, get_last_assistant_message, get_last_user_message, get_output_text, get_response_error_detail, get_reasoning_details, get_system_message, replace_system_message_content, set_last_user_message_content
from open_webui.utils.response import merge_usage
from open_webui.utils.sanitize import sanitize_code
from open_webui.utils.task import rag_template
from open_webui.utils.tools import get_updated_tool_function
from starlette.responses import StreamingResponse

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)



from open_webui.utils.middleware_helpers import DEFAULT_CODE_INTERPRETER_TAGS, DEFAULT_REASONING_TAGS, DEFAULT_SOLUTION_TAGS, _is_tool_result_error, _split_tool_calls, _start_tag_pattern, build_terminal_file_tool_result, get_citation_source_from_tool_result, get_image_urls, get_reasoning_format, get_response_completion_event_data, get_source_context, merge_streamed_reasoning_details, normalize_messages_for_model, output_id, process_tool_result, publish_chat_finished_event, terminal_event_handler, tool_result_content, handle_responses_streaming_event
from open_webui.utils.middleware_runtime import background_tasks_handler, get_system_oauth_token, outlet_filter_handler

async def streaming_chat_response_handler(response, ctx):
    request = ctx['request']

    form_data = ctx['form_data']

    user = ctx['user']
    model = ctx['model']

    metadata = ctx['metadata']
    events = ctx['events']

    event_emitter = ctx['event_emitter']
    event_caller = ctx['event_caller']
    chat_id = metadata.get('chat_id') or ''
    save_to_chat = is_saved_chat_id(chat_id)

    extra_params = {
        '__event_emitter__': event_emitter,
        '__event_call__': event_caller,
        '__user__': user.model_dump() if isinstance(user, UserModel) else {},
        '__metadata__': metadata,
        '__oauth_token__': await get_system_oauth_token(request, user),
        '__request__': request,
        '__model__': model,
        '__chat_id__': metadata.get('chat_id'),
        '__message_id__': metadata.get('message_id'),
    }

    filter_functions = (
        await get_filter_functions(request, model, metadata.get('filter_ids', [])) if ENABLE_PLUGINS else []
    )

    # Standard streaming response handler
    # event_caller is optional — only needed for direct (client-side) tools
    # and pyodide code interpreter. Server-side tools work without it.
    if event_emitter:
        task_id = str(uuid4())  # Create a unique task ID.
        model_id = form_data.get('model', '')

        # Handle as a background task
        async def response_handler(response, events):
            filter_context = FilterContext()
            tag_scan_positions = {}
            tag_boundary_positions = {}
            response_stream_task_id = metadata.get('task_id') or metadata.get('message_id')

            def tag_output_handler(content_type, tags, output):
                """
                Detect special tags (reasoning, solution, code_interpreter) in streaming
                content and create corresponding OR-aligned output items directly.
                Operates on output items instead of content_blocks.

                Uses the text from the output items themselves for tag detection,
                eliminating state divergence between accumulated content and items.
                """
                end_flag = False

                def extract_attributes(tag_content):
                    """Extract attributes from a tag if they exist."""
                    attributes = {}
                    if not tag_content:
                        return attributes
                    matches = re.findall(r'(\w+)\s*=\s*"([^"]+)"', tag_content)
                    for key, value in matches:
                        attributes[key] = value
                    return attributes

                def get_last_text(out):
                    """Get text from last message item, or empty string."""
                    if out and out[-1].get('type') == 'message':
                        parts = out[-1].get('content', [])
                        if parts and parts[-1].get('type') == 'output_text':
                            return parts[-1].get('text', '')
                    return ''

                def set_last_text(out, text):
                    """Set text on last message item's output_text."""
                    if out and out[-1].get('type') == 'message':
                        parts = out[-1].get('content', [])
                        if parts and parts[-1].get('type') == 'output_text':
                            parts[-1]['text'] = text

                def get_scanned_length(item, text):
                    item_id = item.get('id')
                    if not item_id:
                        return 0

                    scanned_length = tag_scan_positions.get((item_id, content_type), 0)
                    return scanned_length if scanned_length <= len(text) else 0

                def save_scanned_length(item, text):
                    item_id = item.get('id')
                    if item_id:
                        tag_scan_positions[(item_id, content_type)] = len(text)

                def clear_scanned_length(item):
                    item_id = item.get('id')
                    if item_id:
                        tag_scan_positions.pop((item_id, content_type), None)
                        tag_boundary_positions.pop((item_id, content_type), None)

                def get_tag_boundaries(item, text, scanned_length):
                    """Index of the last '<', and of the last '>' or newline, before scanned_length."""
                    key = (item.get('id'), content_type)
                    scanned, last_open, last_boundary = tag_boundary_positions.get(key, (0, -1, -1))
                    if scanned > scanned_length:  # the item was rewritten, so the cached positions are stale
                        scanned, last_open, last_boundary = 0, -1, -1

                    if scanned < scanned_length:
                        # only text added since the last call can move either position
                        open_tag = text.rfind('<', scanned, scanned_length)
                        if open_tag != -1:
                            last_open = open_tag
                        boundary = max(
                            text.rfind('>', scanned, scanned_length),
                            text.rfind('\n', scanned, scanned_length),
                        )
                        if boundary != -1:
                            last_boundary = boundary
                        tag_boundary_positions[key] = (scanned_length, last_open, last_boundary)

                    return last_open, last_boundary

                # Map content_type to output item type
                output_type_map = {
                    'reasoning': 'reasoning',
                    'solution': 'message',  # solution tags just produce text
                    'code_interpreter': 'open_webui:code_interpreter',
                }
                output_item_type = output_type_map.get(content_type, content_type)

                last_type = output[-1].get('type', '') if output else ''

                if last_type == 'message':
                    # Use the output item's own text for tag detection
                    item = output[-1]
                    item_text = get_last_text(output)
                    scanned_length = get_scanned_length(item, item_text)
                    max_start_tag_length = max((len(start_tag) for start_tag, _ in tags), default=1)
                    search_start = max(0, scanned_length - max_start_tag_length + 1)

                    if scanned_length and any(
                        start_tag.startswith('<') and start_tag.endswith('>') for start_tag, _ in tags
                    ):
                        open_tag_start, last_tag_boundary = get_tag_boundaries(item, item_text, scanned_length)
                        if open_tag_start > last_tag_boundary:
                            search_start = min(search_start, open_tag_start)

                    for start_tag, end_tag in tags:
                        match = re.compile(_start_tag_pattern(start_tag)).search(item_text, search_start)
                        if match:
                            clear_scanned_length(item)
                            try:
                                attr_content = match.group(1) if match.group(1) else ''
                            except Exception:
                                attr_content = ''

                            attributes = extract_attributes(attr_content)

                            before_tag = item_text[: match.start()]
                            after_tag = item_text[match.end() :]

                            # Keep only text before the tag in the message
                            set_last_text(output, before_tag)

                            if not before_tag.strip():
                                # Remove empty message item
                                if output and output[-1].get('type') == 'message':
                                    output.pop()

                            # Append the new output item
                            if output_item_type == 'reasoning':
                                output.append(
                                    {
                                        'type': 'reasoning',
                                        'id': output_id('r'),
                                        'status': 'in_progress',
                                        'start_tag': start_tag,
                                        'end_tag': end_tag,
                                        'attributes': attributes,
                                        'content': [],
                                        'summary': None,
                                        'started_at': time.time(),
                                    }
                                )
                            elif output_item_type == 'open_webui:code_interpreter':
                                output.append(
                                    {
                                        'type': 'open_webui:code_interpreter',
                                        'id': output_id('ci'),
                                        'status': 'in_progress',
                                        'start_tag': start_tag,
                                        'end_tag': end_tag,
                                        'attributes': attributes,
                                        'lang': attributes.get('lang', 'python'),
                                        'code': '',
                                        'output': None,
                                        'started_at': time.time(),
                                    }
                                )
                            else:
                                # solution or other text-producing tag
                                output.append(
                                    {
                                        'type': 'message',
                                        'id': output_id('msg'),
                                        'status': 'in_progress',
                                        'role': 'assistant',
                                        'content': [{'type': 'output_text', 'text': ''}],
                                        '_tag_type': content_type,
                                        'start_tag': start_tag,
                                        'end_tag': end_tag,
                                        'attributes': attributes,
                                        'started_at': time.time(),
                                    }
                                )

                            if after_tag:
                                # Set the after_tag content on the new item
                                if output_item_type == 'reasoning':
                                    output[-1]['content'] = [{'type': 'output_text', 'text': after_tag}]
                                elif output_item_type == 'open_webui:code_interpreter':
                                    output[-1]['code'] = after_tag
                                else:
                                    set_last_text(output, after_tag)

                                _, recursive_end = tag_output_handler(content_type, tags, output)
                                if recursive_end:
                                    end_flag = True

                            break
                    else:
                        save_scanned_length(item, item_text)

                elif (
                    (last_type == 'reasoning' and content_type == 'reasoning')
                    or (last_type == 'open_webui:code_interpreter' and content_type == 'code_interpreter')
                    or (last_type == 'message' and output[-1].get('_tag_type') == content_type)
                ):
                    item = output[-1]
                    start_tag = item.get('start_tag', '')
                    end_tag = item.get('end_tag', '')

                    # Get the block content from the item itself
                    if last_type == 'reasoning':
                        parts = item.get('content', [])
                        block_content = ''
                        if parts and parts[-1].get('type') == 'output_text':
                            block_content = parts[-1].get('text', '')
                    elif last_type == 'open_webui:code_interpreter':
                        block_content = item.get('code', '')
                    else:
                        block_content = get_last_text(output)

                    scanned_length = get_scanned_length(item, block_content)
                    end_tag_search_start = max(0, scanned_length - max(len(end_tag), 1) + 1)

                    if block_content.find(end_tag, end_tag_search_start) != -1:
                        clear_scanned_length(item)
                        end_flag = True

                        # Strip start and end tags from content
                        start_tag_pattern = _start_tag_pattern(start_tag)
                        block_content = re.sub(start_tag_pattern, '', block_content).strip()

                        end_tag_pattern = rf'{re.escape(end_tag)}'
                        end_tag_regex = re.compile(end_tag_pattern, re.DOTALL)
                        split_content = end_tag_regex.split(block_content, maxsplit=1)

                        block_content = split_content[0].strip() if split_content else ''
                        leftover_content = split_content[1].strip() if len(split_content) > 1 else ''

                        if block_content:
                            # Update the item with final content
                            if last_type == 'reasoning':
                                item['content'] = [{'type': 'output_text', 'text': block_content}]
                                item['ended_at'] = time.time()
                                item['duration'] = int(item['ended_at'] - item['started_at'])
                                item['status'] = 'completed'
                            elif last_type == 'open_webui:code_interpreter':
                                item['code'] = block_content
                                item['ended_at'] = time.time()
                                item['duration'] = int(item['ended_at'] - item['started_at'])
                            else:
                                set_last_text(output, block_content)
                                item['ended_at'] = time.time()

                            # Reset by appending a new message item for leftover
                            output.append(
                                {
                                    'type': 'message',
                                    'id': output_id('msg'),
                                    'status': 'in_progress',
                                    'role': 'assistant',
                                    'content': [
                                        {
                                            'type': 'output_text',
                                            'text': leftover_content,
                                        }
                                    ],
                                }
                            )
                        else:
                            # Remove the block if content is empty
                            output.pop()
                            output.append(
                                {
                                    'type': 'message',
                                    'id': output_id('msg'),
                                    'status': 'in_progress',
                                    'role': 'assistant',
                                    'content': [
                                        {
                                            'type': 'output_text',
                                            'text': leftover_content,
                                        }
                                    ],
                                }
                            )
                    else:
                        save_scanned_length(item, block_content)

                return output, end_flag

            message = (
                await Chats.get_message_by_id_and_message_id(metadata['chat_id'], metadata['message_id'])
                if save_to_chat
                else None
            )

            tool_calls = []

            last_assistant_message = None
            try:
                if form_data['messages'][-1]['role'] == 'assistant':
                    last_assistant_message = get_last_assistant_message(form_data['messages'])
            except Exception as e:
                pass

            initial_content = (
                message.get('content', '') if message else last_assistant_message if last_assistant_message else ''
            )
            content_parts = [initial_content] if initial_content else []

            # Initialize output: use existing from message if continuing, else create new
            existing_output = message.get('output') if message else None
            prior_output = []
            if existing_output and metadata.get('assistant_message_id'):
                prior_output = list(existing_output)
                if (
                    prior_output
                    and prior_output[-1].get('type') == 'message'
                    and prior_output[-1].get('status') == 'in_progress'
                ):
                    msg_parts = prior_output[-1].get('content', [])
                    if not msg_parts or (len(msg_parts) == 1 and not msg_parts[0].get('text', '').strip()):
                        prior_output.pop()
                output = []
                content_parts = []
            elif existing_output:
                output = existing_output
            else:
                # Only create an initial message item if there is content to initialize with
                if initial_content:
                    output = [
                        {
                            'type': 'message',
                            'id': output_id('msg'),
                            'status': 'in_progress',
                            'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': initial_content}],
                        }
                    ]
                else:
                    output = []

            usage = None
            last_response_id = None

            def full_output():
                return prior_output + output if prior_output else output

            def get_message_error_content(error):
                if isinstance(error, HTTPException):
                    error = error.detail
                elif isinstance(error, dict):
                    error = error.get('detail', error)
                else:
                    error = str(error)

                return error if isinstance(error, (str, dict)) else str(error)

            async def emit_message_error(error_content):
                if save_to_chat:
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata['chat_id'],
                        metadata['message_id'],
                        {'error': {'content': error_content}},
                    )
                await event_emitter(
                    {
                        'type': 'chat:message:error',
                        'data': {'error': {'content': error_content}},
                    }
                )

            reasoning_tags_param = metadata.get('params', {}).get('reasoning_tags')
            DETECT_REASONING_TAGS = reasoning_tags_param is not False

            # Legacy tool-calling only: native FC gets execute_code as a builtin tool.
            # Same five authz gates as utils/tools.py get_builtin_tools.
            features = metadata.get('features', {}) or {}
            model_capabilities = model.get('info', {}).get('meta', {}).get('capabilities') or {}
            builtin_tools_meta = model.get('info', {}).get('meta', {}).get('builtinTools', {})
            DETECT_CODE_INTERPRETER = (
                metadata.get('params', {}).get('function_calling') == 'legacy'
                and bool(features.get('code_interpreter'))
                and builtin_tools_meta.get('code_interpreter', True)
                and await Config.get('code_interpreter.enable')
                and model_capabilities.get('code_interpreter', True)
                and (
                    getattr(user, 'role', None) == 'admin'
                    or await has_permission(
                        getattr(user, 'id', ''),
                        'features.code_interpreter',
                        await Config.get('user.permissions'),
                    )
                )
            )

            reasoning_tags = []
            if DETECT_REASONING_TAGS:
                if isinstance(reasoning_tags_param, list) and len(reasoning_tags_param) == 2:
                    reasoning_tags = [(reasoning_tags_param[0], reasoning_tags_param[1])]
                else:
                    reasoning_tags = DEFAULT_REASONING_TAGS

            try:
                for event in events:
                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': event,
                        }
                    )

                    # Save message in the database
                    if save_to_chat:
                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                            metadata['chat_id'],
                            metadata['message_id'],
                            {
                                **event,
                            },
                        )

                async def stream_body_handler(response, form_data):
                    nonlocal usage
                    nonlocal output
                    nonlocal prior_output
                    nonlocal last_response_id

                    response_tool_calls = []

                    delta_count = 0
                    delta_chunk_size = max(
                        CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE,
                        int(metadata.get('params', {}).get('stream_delta_chunk_size') or 1),
                    )
                    last_delta_data = None
                    last_delta_type = None
                    last_delta_key = None

                    joined_content = ''
                    joined_part_count = 0

                    async def save_current_response_stream(stream_output: list | None = None):
                        nonlocal joined_content
                        nonlocal joined_part_count

                        if not chat_id or not metadata.get('message_id'):
                            return

                        # content_parts is append-only, so its length tells us when the join is stale
                        if joined_part_count != len(content_parts):
                            joined_content = ''.join(content_parts)
                            joined_part_count = len(content_parts)

                        current_stream_output = stream_output if stream_output is not None else full_output()
                        await save_response_stream(
                            request.app.state.redis,
                            response_stream_task_id,
                            chat_id,
                            metadata.get('message_id'),
                            joined_content or get_output_text(current_stream_output),
                            current_stream_output,
                        )

                    def get_response_delta_key(delta_data: dict):
                        event_type = delta_data.get('type', '')
                        if not event_type.startswith('response.') or not event_type.endswith('.delta'):
                            return None
                        return (
                            event_type,
                            delta_data.get('item_id'),
                            delta_data.get('output_index'),
                            delta_data.get('content_index'),
                            delta_data.get('summary_index'),
                        )

                    def get_response_data_with_full_output_index(response_data: dict):
                        if prior_output and isinstance(response_data.get('output_index'), int):
                            return {
                                **response_data,
                                'output_index': response_data['output_index'] + len(prior_output),
                            }
                        return response_data

                    async def flush_pending_delta_data(threshold: int = 0):
                        nonlocal delta_count
                        nonlocal last_delta_data
                        nonlocal last_delta_type
                        nonlocal last_delta_key

                        if delta_count >= threshold and last_delta_data:
                            await event_emitter(
                                {
                                    'type': 'response:completion',
                                    'data': last_delta_data,
                                }
                            )
                            await save_current_response_stream()
                            delta_count = 0
                            last_delta_data = None
                            last_delta_type = None
                            last_delta_key = None

                    async def queue_pending_delta_data(delta_data: dict, delta_type: str):
                        nonlocal delta_count
                        nonlocal last_delta_data
                        nonlocal last_delta_type
                        nonlocal last_delta_key

                        delta_data = get_response_data_with_full_output_index(delta_data)
                        delta_key = get_response_delta_key(delta_data)
                        if (
                            last_delta_data
                            and last_delta_key == delta_key
                            and isinstance(last_delta_data.get('delta'), str)
                            and isinstance(delta_data.get('delta'), str)
                        ):
                            last_delta_data['delta'] += delta_data['delta']
                            delta_count += 1
                        else:
                            if last_delta_data and (last_delta_type != delta_type or last_delta_key != delta_key):
                                await flush_pending_delta_data()

                            delta_count += 1
                            last_delta_data = delta_data
                            last_delta_type = delta_type
                            last_delta_key = delta_key

                        if delta_count >= delta_chunk_size:
                            await flush_pending_delta_data(delta_chunk_size)

                    async def emit_response_completion_event(response_data: dict, stream_output: list | None = None):
                        if response_data.get('type', '').endswith('.delta'):
                            await queue_pending_delta_data(
                                response_data,
                                response_data.get('type', 'response.delta'),
                            )
                            return

                        response_data = get_response_data_with_full_output_index(response_data)
                        await flush_pending_delta_data()
                        await event_emitter(
                            {
                                'type': 'response:completion',
                                'data': get_response_completion_event_data(response_data),
                            }
                        )
                        await save_current_response_stream(stream_output)

                    filter_extra_params = {'__body__': form_data, **extra_params} if filter_functions else None

                    async for line in response.body_iterator:
                        line = line.decode('utf-8', 'replace') if isinstance(line, bytes) else line
                        data = line

                        # Skip empty lines
                        if not data or data.isspace():
                            continue

                        # "data:" is the prefix for each event
                        if not data.startswith('data:'):
                            # Some upstreams return plain JSON error lines in a streaming response
                            # (without SSE `data:` prefix). Try to normalize these into standard
                            # error events so frontend and DB paths still receive them.
                            try:
                                raw_obj = JSONCodec.loads(data)
                                raw_error = raw_obj.get('error') if isinstance(raw_obj, dict) else None
                                if raw_error:
                                    if save_to_chat:
                                        try:
                                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                                metadata['chat_id'],
                                                metadata['message_id'],
                                                {
                                                    'error': {'content': raw_error},
                                                },
                                            )
                                        except Exception:
                                            pass
                                    await event_emitter({'type': 'chat:completion', 'data': {'error': raw_error}})
                            except Exception:
                                pass
                            continue

                        # Remove the "data:" prefix
                        data = data[5:].strip()

                        try:
                            data = JSONCodec.loads(data)

                            if filter_functions:
                                data, _ = await process_filter_functions(
                                    request=request,
                                    filter_context=filter_context,
                                    filter_functions=filter_functions,
                                    filter_type='stream',
                                    form_data=data,
                                    extra_params=filter_extra_params,
                                )

                            if data:
                                if 'event' in data and not getattr(request.state, 'direct', False):
                                    await event_emitter(data.get('event', {}))

                                if 'selected_model_id' in data:
                                    model_id = data['selected_model_id']
                                    if save_to_chat:
                                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                                            metadata['chat_id'],
                                            metadata['message_id'],
                                            {
                                                'selectedModelId': model_id,
                                            },
                                            touch=False,
                                        )
                                    await event_emitter(
                                        {
                                            'type': 'chat:completion',
                                            'data': data,
                                        }
                                    )
                                # Check for Responses API events (type field starts with "response.")
                                elif data.get('type', '').startswith('response.'):
                                    response_data_type = data.get('type', '')
                                    response_data_is_delta = response_data_type.endswith('.delta')
                                    output, response_metadata = handle_responses_streaming_event(data, output)

                                    if not response_data_is_delta:
                                        await flush_pending_delta_data()

                                    # Emit citation sources from finalized output items
                                    # (mirrors Chat Completions annotation handling at delta level)
                                    if response_data_type == 'response.output_item.done':
                                        item = data.get('item', {})
                                        if item.get('type') == 'message':
                                            for part in item.get('content', []):
                                                for annotation in part.get('annotations', []):
                                                    if annotation.get('type') == 'url_citation':
                                                        # Handle both flat (Responses API) and nested (Chat Completions) formats
                                                        url_citation = annotation.get('url_citation', annotation)

                                                        url = url_citation.get('url', '')
                                                        title = url_citation.get('title', url)

                                                        if url:
                                                            await event_emitter(
                                                                {
                                                                    'type': 'source',
                                                                    'data': {
                                                                        'source': {
                                                                            'name': title,
                                                                            'url': url,
                                                                        },
                                                                        'document': [title],
                                                                        'metadata': [
                                                                            {
                                                                                'source': url,
                                                                                'name': title,
                                                                            }
                                                                        ],
                                                                    },
                                                                }
                                                            )

                                    # Merge any metadata (usage, etc.)
                                    # Strip 'done' — response.completed emits
                                    # it but we may still need to execute tool
                                    # calls. The outer middleware manages the
                                    # actual completion signal.
                                    if response_metadata:
                                        if ENABLE_RESPONSES_API_STATEFUL:
                                            response_id = response_metadata.pop('response_id', None)
                                            if response_id:
                                                last_response_id = response_id

                                        # Normalize and capture usage for DB persistence
                                        if response_metadata.get('usage'):
                                            usage = merge_usage(usage, response_metadata['usage'])
                                            response_metadata['usage'] = usage

                                        if response_metadata.get('error'):
                                            await event_emitter(
                                                {
                                                    'type': 'chat:completion',
                                                    'data': {'error': response_metadata['error']},
                                                }
                                            )

                                    await emit_response_completion_event(data)

                                    if response_metadata and response_metadata.get('usage'):
                                        await event_emitter(
                                            {
                                                'type': 'chat:completion',
                                                'data': {'usage': usage},
                                            }
                                        )
                                    continue
                                else:
                                    choices = data.get('choices', [])

                                    # Normalize usage data to standard format
                                    raw_usage = data.get('usage', {}) or {}
                                    raw_usage.update(data.get('timings', {}))  # llama.cpp
                                    if raw_usage:
                                        usage = merge_usage(usage, raw_usage)
                                        await event_emitter(
                                            {
                                                'type': 'chat:completion',
                                                'data': {
                                                    'usage': usage,
                                                },
                                            }
                                        )

                                    if not choices:
                                        error = data.get('error', {})
                                        if error:
                                            log.error('Provider returned error (streaming): %s', error)
                                            if save_to_chat:
                                                try:
                                                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                                                        metadata['chat_id'],
                                                        metadata['message_id'],
                                                        {
                                                            'error': {'content': error},
                                                        },
                                                    )
                                                except Exception:
                                                    pass
                                            await event_emitter(
                                                {
                                                    'type': 'chat:completion',
                                                    'data': {
                                                        'error': error,
                                                    },
                                                }
                                            )
                                        continue

                                    delta = choices[0].get('delta', {})
                                    delta_type = 'content'

                                    # Handle delta annotations
                                    annotations = delta.get('annotations')
                                    if annotations:
                                        for annotation in annotations:
                                            if (
                                                annotation.get('type') == 'url_citation'
                                                and 'url_citation' in annotation
                                            ):
                                                url_citation = annotation['url_citation']

                                                url = url_citation.get('url', '')
                                                title = url_citation.get('title', url)

                                                await event_emitter(
                                                    {
                                                        'type': 'source',
                                                        'data': {
                                                            'source': {
                                                                'name': title,
                                                                'url': url,
                                                            },
                                                            'document': [title],
                                                            'metadata': [
                                                                {
                                                                    'source': url,
                                                                    'name': title,
                                                                }
                                                            ],
                                                        },
                                                    }
                                                )

                                    delta_tool_calls = delta.get('tool_calls', None)
                                    if delta_tool_calls:
                                        for delta_tool_call in delta_tool_calls:
                                            tool_call_index = delta_tool_call.get('index')

                                            if tool_call_index is not None:
                                                # Check if the tool call already exists
                                                current_response_tool_call = None
                                                for response_tool_call in response_tool_calls:
                                                    if response_tool_call.get('index') == tool_call_index:
                                                        current_response_tool_call = response_tool_call
                                                        break

                                                if current_response_tool_call is None:
                                                    # Add the new tool call
                                                    delta_tool_call.setdefault('function', {})
                                                    delta_tool_call['function'].setdefault('name', '')
                                                    delta_tool_call['id'] = delta_tool_call.get('id') or output_id('fc')
                                                    delta_arguments = delta_tool_call['function'].get('arguments')
                                                    if not isinstance(delta_arguments, str):
                                                        delta_tool_call['function']['arguments'] = (
                                                            ''
                                                            if delta_arguments is None
                                                            else JSONCodec.dumps(delta_arguments)
                                                        )
                                                    response_tool_calls.append(delta_tool_call)
                                                else:
                                                    # Update the existing tool call
                                                    delta_name = delta_tool_call.get('function', {}).get('name')
                                                    delta_arguments = delta_tool_call.get('function', {}).get(
                                                        'arguments'
                                                    )

                                                    if delta_name:
                                                        current_response_tool_call['function']['name'] = delta_name

                                                    if delta_arguments is not None:
                                                        if not isinstance(delta_arguments, str):
                                                            delta_arguments = JSONCodec.dumps(delta_arguments)
                                                        current_response_tool_call.setdefault('function', {})
                                                        if not isinstance(
                                                            current_response_tool_call['function'].get('arguments'),
                                                            str,
                                                        ):
                                                            current_response_tool_call['function']['arguments'] = ''
                                                        current_response_tool_call['function']['arguments'] += (
                                                            delta_arguments
                                                        )

                                        # Emit pending tool calls in real-time as Responses events.
                                        if response_tool_calls:
                                            output_by_call_id = {
                                                item.get('call_id'): (idx, item)
                                                for idx, item in enumerate(output)
                                                if item.get('type') == 'function_call'
                                            }

                                            for tc in response_tool_calls:
                                                call_id = tc.get('id') or output_id('fc')
                                                tc['id'] = call_id
                                                func = tc.get('function', {})
                                                if call_id in output_by_call_id:
                                                    output_index, item = output_by_call_id[call_id]
                                                    item['name'] = func.get('name', item.get('name', ''))
                                                    item['arguments'] = func.get('arguments', item.get('arguments', ''))
                                                    item['status'] = 'in_progress'
                                                else:
                                                    output_index = len(output)
                                                    item = {
                                                        'type': 'function_call',
                                                        'id': call_id,
                                                        'call_id': call_id,
                                                        'name': func.get('name', ''),
                                                        'arguments': '',
                                                        'status': 'in_progress',
                                                    }
                                                    output.append(item)
                                                    output_by_call_id[call_id] = (output_index, item)
                                                    await emit_response_completion_event(
                                                        {
                                                            'type': 'response.output_item.added',
                                                            'output_index': output_index,
                                                            'item': item.copy(),
                                                        }
                                                    )
                                                    item['arguments'] = func.get('arguments', '')

                                            for delta_tool_call in delta_tool_calls:
                                                tool_call_index = delta_tool_call.get('index')
                                                current_response_tool_call = next(
                                                    (
                                                        tc
                                                        for tc in response_tool_calls
                                                        if tc.get('index') == tool_call_index
                                                    ),
                                                    None,
                                                )
                                                if not current_response_tool_call:
                                                    continue
                                                call_id = current_response_tool_call.get('id')
                                                output_index, _ = output_by_call_id.get(call_id, (len(output) - 1, {}))
                                                delta_arguments = delta_tool_call.get('function', {}).get('arguments')
                                                if delta_arguments is not None:
                                                    if not isinstance(delta_arguments, str):
                                                        delta_arguments = JSONCodec.dumps(delta_arguments)
                                                    await emit_response_completion_event(
                                                        {
                                                            'type': 'response.function_call_arguments.delta',
                                                            'item_id': call_id,
                                                            'output_index': output_index,
                                                            'delta': delta_arguments,
                                                        }
                                                    )

                                            await save_current_response_stream()
                                            data = None
                                            delta_type = 'tool_call'

                                    delta_images = delta.get('images')
                                    image_urls = (
                                        await get_image_urls(delta_images, request, metadata, user)
                                        if delta_images
                                        else []
                                    )
                                    if image_urls:
                                        image_file_list = [{'type': 'image', 'url': url} for url in image_urls]
                                        message_files = image_file_list
                                        if save_to_chat:
                                            message_files = await Chats.add_message_files_by_id_and_message_id(
                                                metadata['chat_id'],
                                                metadata['message_id'],
                                                image_file_list,
                                            )
                                            if message_files is None:
                                                message_files = image_file_list

                                        await event_emitter(
                                            {
                                                'type': 'files',
                                                'data': {'files': message_files},
                                            }
                                        )

                                    # content and reasoning deltas are raw JSON: a stream filter can make them any type
                                    value = delta.get('content')
                                    if value and not isinstance(value, str):
                                        value = f'{value}'

                                    reasoning_content = (
                                        delta.get('reasoning_content')
                                        or delta.get('reasoning')
                                        or delta.get('thinking')
                                    )
                                    if reasoning_content and not isinstance(reasoning_content, str):
                                        reasoning_content = f'{reasoning_content}'
                                    reasoning_details = get_reasoning_details(delta)
                                    reasoning_detail_items = (
                                        [item for item in reasoning_details if isinstance(item, dict)]
                                        if isinstance(reasoning_details, list)
                                        else [reasoning_details]
                                        if isinstance(reasoning_details, dict)
                                        else []
                                    )
                                    existing_reasoning_item = next(
                                        (item for item in reversed(output) if item.get('type') == 'reasoning'),
                                        None,
                                    )
                                    message_index = next(
                                        (i for i, item in enumerate(output) if item.get('type') == 'message'),
                                        None,
                                    )
                                    if reasoning_content or (
                                        reasoning_detail_items
                                        and (
                                            existing_reasoning_item
                                            or any(
                                                item.get('text') or item.get('summary') or item.get('data')
                                                for item in reasoning_detail_items
                                            )
                                        )
                                    ):
                                        reasoning_item = (
                                            existing_reasoning_item
                                            if (reasoning_detail_items and not reasoning_content)
                                            or message_index is not None
                                            else None
                                        )

                                        if reasoning_item is None:
                                            if not output or output[-1].get('type') != 'reasoning':
                                                reasoning_item = {
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
                                                if message_index is not None:
                                                    reasoning_item['ended_at'] = time.time()
                                                    reasoning_item['duration'] = 0
                                                    reasoning_item['status'] = 'completed'
                                                    output.insert(message_index, reasoning_item)
                                                else:
                                                    output.append(reasoning_item)
                                            else:
                                                reasoning_item = output[-1]

                                        if reasoning_content:
                                            # Append to reasoning content
                                            parts = reasoning_item.get('content', [])
                                            if parts and parts[-1].get('type') == 'output_text':
                                                parts[-1]['text'] += reasoning_content
                                            else:
                                                reasoning_item['content'] = [
                                                    {
                                                        'type': 'output_text',
                                                        'text': reasoning_content,
                                                    }
                                                ]

                                            reasoning_index = output.index(reasoning_item)
                                            data = {
                                                'type': 'response.reasoning_text.delta',
                                                'item_id': reasoning_item.get('id'),
                                                'output_index': reasoning_index,
                                                'content_index': max(
                                                    len(reasoning_item.get('content', [])) - 1,
                                                    0,
                                                ),
                                                'delta': reasoning_content,
                                            }
                                            delta_type = 'response.reasoning_text.delta'

                                        if reasoning_detail_items:
                                            merge_streamed_reasoning_details(
                                                reasoning_item.setdefault('reasoning_details', []),
                                                reasoning_detail_items,
                                            )
                                            await save_current_response_stream()
                                            # Providers such as OpenRouter send reasoning_details
                                            # alongside the reasoning text: only drop the event when
                                            # the details were all there was to report, otherwise the
                                            # reasoning delta never reaches the client.
                                            if not reasoning_content:
                                                data = None

                                    if value:
                                        if (
                                            output
                                            and output[-1].get('type') == 'reasoning'
                                            and output[-1].get('attributes', {}).get('type') == 'reasoning_content'
                                        ):
                                            reasoning_item = output[-1]
                                            reasoning_item['ended_at'] = time.time()
                                            reasoning_item['duration'] = int(
                                                reasoning_item['ended_at'] - reasoning_item['started_at']
                                            )
                                            reasoning_item['status'] = 'completed'

                                            output.append(
                                                {
                                                    'type': 'message',
                                                    'id': output_id('msg'),
                                                    'status': 'in_progress',
                                                    'role': 'assistant',
                                                    'content': [
                                                        {
                                                            'type': 'output_text',
                                                            'text': '',
                                                        }
                                                    ],
                                                }
                                            )

                                        if ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION:
                                            value = await convert_markdown_base64_images(
                                                request,
                                                value,
                                                {
                                                    'chat_id': metadata.get('chat_id', None),
                                                    'message_id': metadata.get('message_id', None),
                                                },
                                                user,
                                            )

                                        # closure-cell str += recopies per chunk; append + join once at read is O(n)
                                        content_parts.append(value)

                                        # Check if we're inside a tag-based block
                                        # (reasoning, code_interpreter, or solution).
                                        # If so, append to the existing in-progress
                                        # item instead of creating a new message —
                                        # otherwise tag_output_handler re-detects the
                                        # start tag on every chunk and fragments the
                                        # output.
                                        last_item = output[-1] if output else None
                                        last_item_type = last_item.get('type', '') if last_item else ''
                                        inside_tag_block = (
                                            last_item is not None
                                            and last_item.get('status') == 'in_progress'
                                            and last_item.get('attributes', {}).get('type') != 'reasoning_content'
                                            and (
                                                last_item_type == 'reasoning'
                                                or last_item_type == 'open_webui:code_interpreter'
                                                or (
                                                    last_item_type == 'message'
                                                    and last_item.get('_tag_type') is not None
                                                )
                                            )
                                        )

                                        if inside_tag_block:
                                            # Append to the existing tag-based item
                                            if last_item_type == 'open_webui:code_interpreter':
                                                last_item['code'] = last_item.get('code', '') + value
                                            elif last_item_type == 'reasoning':
                                                parts = last_item.get('content', [])
                                                if parts and parts[-1].get('type') == 'output_text':
                                                    parts[-1]['text'] += value
                                                else:
                                                    last_item['content'] = [
                                                        {
                                                            'type': 'output_text',
                                                            'text': value,
                                                        }
                                                    ]
                                            else:
                                                # solution or other _tag_type message
                                                msg_parts = last_item.get('content', [])
                                                if msg_parts and msg_parts[-1].get('type') == 'output_text':
                                                    msg_parts[-1]['text'] += value
                                                else:
                                                    last_item['content'] = [
                                                        {
                                                            'type': 'output_text',
                                                            'text': value,
                                                        }
                                                    ]
                                        else:
                                            if not output or output[-1].get('type') != 'message':
                                                output.append(
                                                    {
                                                        'type': 'message',
                                                        'id': output_id('msg'),
                                                        'status': 'in_progress',
                                                        'role': 'assistant',
                                                        'content': [
                                                            {
                                                                'type': 'output_text',
                                                                'text': '',
                                                            }
                                                        ],
                                                    }
                                                )

                                            # Append value to last message item's text
                                            msg_parts = output[-1].get('content', [])
                                            if msg_parts and msg_parts[-1].get('type') == 'output_text':
                                                msg_parts[-1]['text'] += value
                                            else:
                                                output[-1]['content'] = [
                                                    {
                                                        'type': 'output_text',
                                                        'text': value,
                                                    }
                                                ]

                                        if DETECT_REASONING_TAGS:
                                            output, _ = tag_output_handler(
                                                'reasoning',
                                                reasoning_tags,
                                                output,
                                            )

                                            output, _ = tag_output_handler(
                                                'solution',
                                                DEFAULT_SOLUTION_TAGS,
                                                output,
                                            )

                                        if DETECT_CODE_INTERPRETER:
                                            output, end = tag_output_handler(
                                                'code_interpreter',
                                                DEFAULT_CODE_INTERPRETER_TAGS,
                                                output,
                                            )

                                            if end:
                                                break

                                        target_index = len(output) - 1
                                        target_item = output[target_index] if target_index >= 0 else {}
                                        target_content = target_item.get('content', [])
                                        content_index = max(len(target_content) - 1, 0)
                                        delta_event_type = (
                                            'response.reasoning_text.delta'
                                            if target_item.get('type') == 'reasoning'
                                            else 'response.output_text.delta'
                                        )
                                        data = {
                                            'type': delta_event_type,
                                            'item_id': target_item.get('id'),
                                            'output_index': target_index,
                                            'content_index': content_index,
                                            'delta': value,
                                        }
                                        delta_type = delta_event_type

                                if delta and data:
                                    await queue_pending_delta_data(data, delta_type)
                                elif data:
                                    await event_emitter(
                                        {
                                            'type': 'chat:completion',
                                            'data': data,
                                        }
                                    )
                        except (asyncio.CancelledError, KeyboardInterrupt):
                            raise
                        except Exception as e:
                            done = 'data: [DONE]' in line
                            if done:
                                pass
                            else:
                                log.debug('Error: %s', e)
                                continue
                    await flush_pending_delta_data()

                    if output:
                        # Clean up the last message item
                        if output[-1].get('type') == 'message':
                            parts = output[-1].get('content', [])
                            if parts and parts[-1].get('type') == 'output_text':
                                parts[-1]['text'] = parts[-1]['text'].strip()

                                if not parts[-1]['text']:
                                    output.pop()

                                    if not output:
                                        output.append(
                                            {
                                                'type': 'message',
                                                'id': output_id('msg'),
                                                'status': 'in_progress',
                                                'role': 'assistant',
                                                'content': [{'type': 'output_text', 'text': ''}],
                                            }
                                        )

                        if output[-1].get('type') == 'reasoning':
                            reasoning_item = output[-1]
                            if reasoning_item.get('ended_at') is None:
                                reasoning_item['ended_at'] = time.time()
                                if reasoning_item.get('started_at') is not None:
                                    reasoning_item['duration'] = int(
                                        reasoning_item['ended_at'] - reasoning_item['started_at']
                                    )
                                reasoning_item['status'] = 'completed'

                    if response_tool_calls:
                        for tc in response_tool_calls:
                            call_id = tc.get('id', '')
                            arguments = tc.get('function', {}).get('arguments', '{}')
                            for output_index, item in enumerate(output):
                                if item.get('type') == 'function_call' and item.get('call_id') == call_id:
                                    item['arguments'] = arguments
                                    item['status'] = 'completed'
                                    await emit_response_completion_event(
                                        {
                                            'type': 'response.function_call_arguments.done',
                                            'item_id': item.get('id'),
                                            'output_index': output_index,
                                            'arguments': arguments,
                                        }
                                    )
                                    await emit_response_completion_event(
                                        {
                                            'type': 'response.output_item.done',
                                            'output_index': output_index,
                                            'item': item.copy(),
                                        }
                                    )
                                    break
                        tool_calls.append(_split_tool_calls(response_tool_calls))

                    # Responses API path: extract function_call items from output
                    if not response_tool_calls and output:
                        # Collect call_ids that already have results,
                        # including those from prior_output so we don't
                        # re-process tool calls from a previous turn.
                        handled_call_ids = {
                            item.get('call_id')
                            for item in (prior_output + output)
                            if item.get('type') == 'function_call_output'
                        }
                        responses_api_tool_calls = []
                        for item in output:
                            call_id = item.get('call_id') or item.get('id') or output_id('fc')
                            if item.get('type') == 'function_call' and call_id not in handled_call_ids:
                                arguments = item.get('arguments', '{}')
                                responses_api_tool_calls.append(
                                    {
                                        'id': call_id,
                                        'index': len(responses_api_tool_calls),
                                        'function': {
                                            'name': item.get('name', ''),
                                            'arguments': (
                                                arguments if isinstance(arguments, str) else JSONCodec.dumps(arguments)
                                            ),
                                        },
                                    }
                                )
                        if responses_api_tool_calls:
                            tool_calls.append(_split_tool_calls(responses_api_tool_calls))

                try:
                    await stream_body_handler(response, form_data)
                finally:
                    if response.background:
                        await response.background()

                tool_call_iterations = 0
                max_tool_call_iterations = getattr(
                    request.state,
                    'max_tool_call_iterations',
                    CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS,
                )
                tool_call_sources = []  # Track citation sources from tool results
                all_tool_call_sources = []  # Accumulated sources across all iterations
                user_message = get_last_user_message(form_data['messages'])

                # Check if citations are enabled for this model
                citations_enabled = (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get(
                    'citations', True
                )

                # Use the pre-RAG system content captured before the
                # initial file-source injection in process_chat_payload.
                # This ensures restore truly undoes the RAG template.
                original_system_content = metadata.get('system_prompt')
                if original_system_content is None:
                    original_system_message = get_system_message(form_data['messages'])
                    original_system_content = (
                        get_content_from_message(original_system_message) if original_system_message else None
                    )

                while tool_calls and (
                    max_tool_call_iterations is None or tool_call_iterations < max_tool_call_iterations
                ):
                    tool_call_iterations += 1

                    response_tool_calls = tool_calls.pop(0)
                    ask_user_staged, ask_user_error = stage_ask_user_tool_calls(response_tool_calls, output, output_id)
                    if ask_user_error:
                        response_tool_calls = [
                            tool_call
                            for tool_call in response_tool_calls
                            if tool_call.get('function', {}).get('name') != 'ask_user'
                        ]
                    elif ask_user_staged:
                        if is_saved_chat_id(metadata.get('chat_id')) and metadata.get('message_id'):
                            await pause_for_tool_approval(
                                metadata['chat_id'],
                                metadata['message_id'],
                                full_output(),
                                form_data,
                                metadata,
                            )
                        await event_emitter({'type': 'chat:completion', 'data': {'output': full_output()}})
                        return

                    # Append function_call items for each tool call
                    # (Responses API already has them from streaming, so skip duplicates)
                    existing_call_ids = {item.get('call_id') for item in output if item.get('type') == 'function_call'}
                    for tc in response_tool_calls:
                        call_id = tc.get('id', '')
                        if call_id not in existing_call_ids:
                            func = tc.get('function', {})
                            output.append(
                                {
                                    'type': 'function_call',
                                    'id': call_id or output_id('fc'),
                                    'call_id': call_id,
                                    'name': func.get('name', ''),
                                    'arguments': func.get('arguments', '{}'),
                                    'status': 'in_progress',
                                }
                            )

                    tool_approval_mode = metadata.get('params', {}).get('tool_approval_mode', 'full')
                    if (
                        response_tool_calls
                        and tool_approval_mode == 'ask'
                        and is_saved_chat_id(metadata.get('chat_id'))
                        and metadata.get('message_id')
                    ):
                        await pause_for_tool_approval(
                            metadata['chat_id'],
                            metadata['message_id'],
                            full_output(),
                            form_data,
                            metadata,
                        )
                        await event_emitter(
                            {
                                'type': 'chat:completion',
                                'data': {
                                    'output': full_output(),
                                },
                            }
                        )
                        return

                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': {
                                'output': full_output(),
                            },
                        }
                    )

                    tools = metadata.get('tools', {})

                    results = []

                    def parse_tool_params(tool_call):
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
                                    return None
                        tool_call.setdefault('function', {})['arguments'] = JSONCodec.dumps(params)
                        return params

                    async def execute_tool_call(tool_call):
                        name = tool_call.get('function', {}).get('name', '')
                        params = parse_tool_params(tool_call)
                        if params is None:
                            return {}, None, None, None, False
                        tool = tools.get(name)
                        if not tool:
                            return params, f'Error: Tool "{name}" not found.', None, None, False
                        spec = tool.get('spec', {})
                        tool_type = tool.get('type', '')
                        direct_tool = tool.get('direct', False)
                        allowed_params = spec.get('parameters', {}).get('properties', {}).keys()
                        params = {key: value for key, value in params.items() if key in allowed_params}
                        try:
                            if direct_tool:
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
                        return params, result, tool, tool_type, direct_tool

                    delegate_calls = [
                        tool_call
                        for tool_call in response_tool_calls
                        if tool_call.get('function', {}).get('name') == 'delegate_task'
                    ]
                    tool_results = {}
                    for tool_call in response_tool_calls:
                        if tool_call.get('function', {}).get('name') != 'delegate_task':
                            tool_results[id(tool_call)] = await execute_tool_call(tool_call)
                    tool_results.update(
                        zip(
                            [id(tool_call) for tool_call in delegate_calls],
                            await asyncio.gather(*(execute_tool_call(tool_call) for tool_call in delegate_calls)),
                        )
                    )

                    for tool_call in response_tool_calls:
                        tool_call_id = tool_call.get('id', '')
                        tool_function_name = tool_call.get('function', {}).get('name', '')
                        tool_function_params, tool_result, tool, tool_type, direct_tool = tool_results[id(tool_call)]
                        if tool_result is None:
                            results.append(
                                {
                                    'tool_call_id': tool_call_id,
                                    'content': (
                                        'Error: Tool call arguments could not be parsed. The model generated '
                                        f'malformed or incomplete JSON for `{tool_function_name}`. Please try again.'
                                    ),
                                }
                            )
                            continue

                        terminal_file_result = build_terminal_file_tool_result(
                            tool_function_name,
                            tool_function_params,
                            tool_result,
                            tool,
                            metadata,
                        )
                        if terminal_file_result:
                            tool_result = terminal_file_result

                        tool_result, tool_result_files, tool_result_embeds = await process_tool_result(
                            request,
                            tool_function_name,
                            tool_result,
                            tool_type,
                            direct_tool,
                            metadata,
                            user,
                        )

                        await terminal_event_handler(
                            tool_function_name,
                            tool_function_params,
                            tool_result,
                            event_emitter,
                        )

                        # Extract citation sources from tool results
                        if (
                            citations_enabled
                            and tool_function_name
                            in [
                                'search_web',
                                'fetch_url',
                                'view_file',
                                'view_knowledge_file',
                                'query_knowledge_files',
                                'query_chat_files',
                            ]
                            and tool_result
                        ):
                            try:
                                citation_sources = get_citation_source_from_tool_result(
                                    tool_name=tool_function_name,
                                    tool_params=tool_function_params,
                                    tool_result=tool_result,
                                    tool_id=tool.get('tool_id', '') if tool else '',
                                )
                                tool_call_sources.extend(citation_sources)
                            except Exception as e:
                                log.exception(f'Error extracting citation source: {e}')

                        results.append(
                            {
                                'tool_call_id': tool_call_id,
                                'content': tool_result_content(tool_result),
                                **({'files': tool_result_files} if tool_result_files else {}),
                                **({'embeds': tool_result_embeds} if tool_result_embeds else {}),
                            }
                        )

                    result_status_by_call_id = {}
                    for result in results:
                        output_parts = [{'type': 'input_text', 'text': result.get('content', '')}]
                        local_output_status = (
                            'failed' if _is_tool_result_error(result.get('content', '')) else 'completed'
                        )
                        result_status_by_call_id[result.get('tool_call_id', '')] = local_output_status

                        # Separate image data URIs (for LLM via input_image) from
                        # other files (for frontend display via files attribute).
                        display_files = []
                        for file_item in result.get('files', []):
                            if file_item.get('type') == 'image' and file_item.get('url', '').startswith('data:'):
                                # LLM-only: add as input_image part, not frontend display output.
                                output_parts.append({'type': 'input_image', 'image_url': file_item['url']})
                            else:
                                # Frontend display (MCP images, audio, etc.)
                                display_files.append(file_item)

                        output.append(
                            {
                                'type': 'function_call_output',
                                'id': output_id('fco'),
                                'call_id': result.get('tool_call_id', ''),
                                'output': output_parts,
                                'status': local_output_status,
                                **({'files': display_files} if display_files else {}),
                                **({'embeds': result.get('embeds')} if result.get('embeds') else {}),
                            }
                        )

                    # Update function_call statuses and parsed/sanitized arguments.
                    for tc in response_tool_calls:
                        call_id = tc.get('id', '')
                        for item in output:
                            if item.get('type') == 'function_call' and item.get('call_id') == call_id:
                                item['status'] = result_status_by_call_id.get(call_id, 'completed')
                                item['arguments'] = tc.get('function', {}).get('arguments', '{}')
                                break

                    # Emit citation sources to the frontend for display
                    if citations_enabled:
                        for source in tool_call_sources:
                            await event_emitter({'type': 'source', 'data': source})

                        # Apply tool source context to messages for the model.
                        # Restoring to pre-RAG original prevents duplicating
                        # the RAG template across file and tool sources.
                        all_tool_call_sources.extend(tool_call_sources)
                        if all_tool_call_sources and user_message:
                            # Restore pre-RAG message state before re-applying
                            # to prevent RAG template duplication.
                            original_user_message = metadata.get('user_prompt') or user_message
                            set_last_user_message_content(
                                original_user_message,
                                form_data['messages'],
                            )
                            if original_system_content is not None:
                                if get_system_message(form_data['messages']):
                                    replace_system_message_content(
                                        original_system_content,
                                        form_data['messages'],
                                    )
                                else:
                                    form_data['messages'] = add_or_update_system_message(
                                        original_system_content,
                                        form_data['messages'],
                                    )
                            else:
                                replace_system_message_content('', form_data['messages'])

                            # Build context: file sources with content,
                            # tool sources as citation markers only.
                            source_ids = {}
                            source_context = get_source_context(
                                metadata.get('sources', []), source_ids
                            ) + get_source_context(
                                all_tool_call_sources,
                                source_ids,
                                include_content=False,
                            )
                            source_context = source_context.strip()
                            if source_context:
                                rag_content = await rag_template(
                                    await Config.get('rag.template'),
                                    source_context,
                                    user_message,
                                )
                                if RAG_SYSTEM_CONTEXT:
                                    form_data['messages'] = add_or_update_system_message(
                                        rag_content,
                                        form_data['messages'],
                                        append=True,
                                    )
                                else:
                                    form_data['messages'] = add_or_update_user_message(
                                        rag_content,
                                        form_data['messages'],
                                        append=False,
                                    )
                        tool_call_sources.clear()

                    # Strip input_image parts (large base64 data URIs) from the
                    # output sent to the frontend — they're only for LLM consumption
                    # via convert_output_to_messages.
                    frontend_output = []
                    for item in full_output():
                        if item.get('type') == 'function_call_output':
                            parts = item.get('output', [])
                            if any(p.get('type') == 'input_image' for p in parts):
                                item = {**item, 'output': [p for p in parts if p.get('type') != 'input_image']}
                        frontend_output.append(item)

                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': {
                                'output': frontend_output,
                            },
                        }
                    )

                    try:
                        new_form_data = {
                            **form_data,
                            'model': model_id,
                            'stream': True,
                            'metadata': metadata,
                        }

                        if ENABLE_RESPONSES_API_STATEFUL and last_response_id:
                            system_message = get_system_message(form_data['messages'])
                            new_form_data['messages'] = (
                                [system_message] if system_message else []
                            ) + convert_output_to_messages(
                                output, raw=True, reasoning_format=get_reasoning_format(model)
                            )
                            new_form_data['previous_response_id'] = last_response_id
                        else:
                            tool_messages = convert_output_to_messages(
                                output,
                                raw=True,
                                reasoning_format=get_reasoning_format(model),
                                flatten_tool_images=True,
                            )

                            # Chat Completions providers don't support multimodal
                            # tool messages.  Extract images into a user message.
                            image_urls = []
                            for message in tool_messages:
                                if message.get('role') == 'tool' and isinstance(message.get('content'), list):
                                    text_parts = []
                                    for part in message['content']:
                                        if part.get('type') == 'input_text':
                                            text_parts.append(part.get('text', ''))
                                        elif part.get('type') == 'input_image':
                                            image_urls.append(part.get('image_url', ''))
                                    message['content'] = ''.join(text_parts)

                            new_form_data['messages'] = [
                                *form_data['messages'],
                                *tool_messages,
                            ]

                            if image_urls:
                                new_form_data['messages'].append(
                                    {
                                        'role': 'user',
                                        'content': [
                                            {
                                                'type': 'text',
                                                'text': 'Here are the images from the tool results above. Please analyze them.',
                                            },
                                            *[{'type': 'image_url', 'image_url': {'url': url}} for url in image_urls],
                                        ],
                                    }
                                )

                        if filter_functions:
                            new_form_data, _ = await process_filter_functions(
                                request=request,
                                filter_context=filter_context,
                                filter_functions=filter_functions,
                                filter_type='request',
                                form_data=new_form_data,
                                extra_params=extra_params,
                            )

                        new_form_data = normalize_messages_for_model(new_form_data)

                        res = await generate_chat_completion(
                            request,
                            new_form_data,
                            user,
                            bypass_system_prompt=True,
                        )

                        if isinstance(res, StreamingResponse):
                            # Save accumulated output and start fresh.
                            # Responses API output_index values are relative
                            # to the current response — a clean output list
                            # keeps indices aligned. The display prefix
                            # ensures the UI shows tool history during
                            # streaming.
                            prior_output = list(full_output())
                            # Trim the trailing empty placeholder message
                            # so it doesn't persist as a ghost item once
                            # the new stream produces real content.
                            if (
                                prior_output
                                and prior_output[-1].get('type') == 'message'
                                and prior_output[-1].get('status') == 'in_progress'
                            ):
                                msg_parts = prior_output[-1].get('content', [])
                                if not msg_parts or (len(msg_parts) == 1 and not msg_parts[0].get('text', '').strip()):
                                    prior_output.pop()
                            output = []
                            await stream_body_handler(res, new_form_data)
                            output[:0] = prior_output
                            prior_output = []
                        elif getattr(res, 'status_code', 200) >= 400:
                            await emit_message_error(get_message_error_content(get_response_error_detail(res)))
                            break
                        else:
                            break
                    except Exception as e:
                        error_content = get_message_error_content(e)
                        log.exception('Tool-call continuation failed: %s', error_content)
                        await emit_message_error(error_content)
                        break

                if (
                    max_tool_call_iterations is not None
                    and tool_calls
                    and tool_call_iterations >= max_tool_call_iterations
                ):
                    log.warning('Tool-call iteration limit reached (%s)', max_tool_call_iterations)
                    error_content = f'Tool-call limit reached ({max_tool_call_iterations} iterations).'
                    await emit_message_error(error_content)

                if DETECT_CODE_INTERPRETER:
                    MAX_RETRIES = 5
                    retries = 0

                    while output and output[-1].get('type') == 'open_webui:code_interpreter' and retries < MAX_RETRIES:
                        await event_emitter(
                            {
                                'type': 'chat:completion',
                                'data': {
                                    'output': full_output(),
                                },
                            }
                        )

                        retries += 1
                        log.debug('Attempt count: %s', retries)

                        ci_item = output[-1]
                        ci_output = ''
                        try:
                            if ci_item.get('attributes', {}).get('type') == 'code':
                                code = ci_item.get('code', '')
                                # Sanitize code (strips ANSI codes and markdown fences)
                                code = sanitize_code(code)

                                if CODE_INTERPRETER_BLOCKED_MODULES:
                                    blocking_code = textwrap.dedent(f"""
                                        import builtins
    
                                        BLOCKED_MODULES = {CODE_INTERPRETER_BLOCKED_MODULES}
    
                                        _real_import = builtins.__import__
                                        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
                                            if name.split('.')[0] in BLOCKED_MODULES:
                                                importer_name = globals.get('__name__') if globals else None
                                                if importer_name == '__main__':
                                                    raise ImportError(
                                                        f"Direct import of module {{name}} is restricted."
                                                    )
                                            return _real_import(name, globals, locals, fromlist, level)
    
                                        builtins.__import__ = restricted_import
                                    """)
                                    code = blocking_code + '\n' + code

                                ci_engine = await Config.get('code_interpreter.engine')
                                if ci_engine == 'pyodide':
                                    ci_output = await event_caller(
                                        {
                                            'type': 'execute:python',
                                            'data': {
                                                'id': str(uuid4()),
                                                'code': code,
                                                'session_id': metadata.get('session_id', None),
                                                'files': metadata.get('files', []),
                                            },
                                        }
                                    )
                                elif ci_engine == 'jupyter':
                                    ci_output = await execute_code_jupyter(
                                        await Config.get('code_interpreter.jupyter.url'),
                                        code,
                                        (
                                            await Config.get('code_interpreter.jupyter.auth_token')
                                            if await Config.get('code_interpreter.jupyter.auth') == 'token'
                                            else None
                                        ),
                                        (
                                            await Config.get('code_interpreter.jupyter.auth_password')
                                            if await Config.get('code_interpreter.jupyter.auth') == 'password'
                                            else None
                                        ),
                                        await Config.get('code_interpreter.jupyter.timeout'),
                                    )
                                else:
                                    ci_output = {'stdout': 'Code interpreter engine not configured.'}

                                log.debug('Code interpreter output: %s', ci_output)

                                # Handle error responses from event_caller
                                # (e.g. session disconnected, timeout)
                                if isinstance(ci_output, dict) and ci_output.get('error'):
                                    ci_output = {'stderr': ci_output['error']}

                                if isinstance(ci_output, dict):
                                    stdout = ci_output.get('stdout', '')

                                    if isinstance(stdout, str):
                                        stdoutLines = stdout.split('\n')
                                        for idx, line in enumerate(stdoutLines):
                                            if re.match(r'data:image/\w+;base64', line):
                                                image_url = await get_image_url_from_base64(
                                                    request,
                                                    line,
                                                    metadata,
                                                    user,
                                                )
                                                if image_url:
                                                    stdoutLines[idx] = f'![Output Image]({image_url})'

                                        ci_output['stdout'] = '\n'.join(stdoutLines)

                                    result = ci_output.get('result', '')

                                    if isinstance(result, str):
                                        resultLines = result.split('\n')
                                        for idx, line in enumerate(resultLines):
                                            if re.match(r'data:image/\w+;base64', line):
                                                image_url = await get_image_url_from_base64(
                                                    request,
                                                    line,
                                                    metadata,
                                                    user,
                                                )
                                                resultLines[idx] = f'![Output Image]({image_url})'
                                        ci_output['result'] = '\n'.join(resultLines)
                        except Exception as e:
                            ci_output = str(e)

                        ci_item['output'] = ci_output
                        ci_item['status'] = 'completed'

                        output.append(
                            {
                                'type': 'message',
                                'id': output_id('msg'),
                                'status': 'in_progress',
                                'role': 'assistant',
                                'content': [{'type': 'output_text', 'text': ''}],
                            }
                        )

                        await event_emitter(
                            {
                                'type': 'chat:completion',
                                'data': {
                                    'output': full_output(),
                                },
                            }
                        )

                        try:
                            new_form_data = {
                                **form_data,
                                'model': model_id,
                                'stream': True,
                                'metadata': metadata,
                                'messages': [
                                    *form_data['messages'],
                                    *convert_output_to_messages(
                                        output,
                                        raw=True,
                                        reasoning_format=get_reasoning_format(model),
                                        flatten_tool_images=True,
                                    ),
                                ],
                            }

                            if filter_functions:
                                new_form_data, _ = await process_filter_functions(
                                    request=request,
                                    filter_context=filter_context,
                                    filter_functions=filter_functions,
                                    filter_type='request',
                                    form_data=new_form_data,
                                    extra_params=extra_params,
                                )

                            new_form_data = normalize_messages_for_model(new_form_data)

                            res = await generate_chat_completion(
                                request,
                                new_form_data,
                                user,
                                bypass_system_prompt=True,
                            )

                            if isinstance(res, StreamingResponse):
                                await stream_body_handler(res, new_form_data)
                            elif getattr(res, 'status_code', 200) >= 400:
                                await emit_message_error(get_message_error_content(get_response_error_detail(res)))
                                break
                            else:
                                break
                        except Exception as e:
                            error_content = get_message_error_content(e)
                            log.exception('Code interpreter continuation failed: %s', error_content)
                            await emit_message_error(error_content)
                            break

                # Mark all in-progress items as completed
                for item in output:
                    if item.get('status') == 'in_progress':
                        item['status'] = 'completed'

                current_output = full_output()
                title = await Chats.get_chat_title_by_id(metadata['chat_id']) if save_to_chat else ''
                data = {
                    'done': True,
                    'output': current_output,
                    'title': title,
                    **({'usage': usage} if usage else {}),
                }

                if save_to_chat:
                    # Save final output once. The delta path keeps in-progress
                    # state in response_streams instead of writing tokens to DB.
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata['chat_id'],
                        metadata['message_id'],
                        {
                            'done': True,
                            'output': current_output,
                            **({'usage': usage} if usage else {}),
                        },
                    )

                await clear_response_stream(request.app.state.redis, response_stream_task_id)
                await publish_chat_finished_event(
                    request, user, metadata, title, ''.join(content_parts), current_output
                )

                await event_emitter(
                    {
                        'type': 'chat:completion',
                        'data': data,
                    }
                )

                ctx['assistant_message'] = {
                    'content': ''.join(content_parts) or get_output_text(current_output),
                    'output': current_output,
                    **({'usage': usage} if usage else {}),
                }
                await outlet_filter_handler(ctx)
                await background_tasks_handler(ctx)
            except asyncio.CancelledError:
                log.warning('Task was cancelled!')

                # Close the response body iterator to trigger cleanup
                # in stream_wrapper's finally block and release the
                # upstream connection.  Without this, the async
                # generator is orphaned and may spin in anyio internals.
                if hasattr(response, 'body_iterator') and hasattr(response.body_iterator, 'aclose'):
                    try:
                        await asyncio.shield(response.body_iterator.aclose())
                    except (asyncio.CancelledError, Exception):
                        pass

                async def save_cancelled_state():
                    await event_emitter({'type': 'chat:tasks:cancel'})
                    if save_to_chat:
                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                            metadata['chat_id'],
                            metadata['message_id'],
                            {
                                'done': True,
                                'output': full_output(),
                            },
                        )
                    await clear_response_stream(request.app.state.redis, response_stream_task_id)

                try:
                    await asyncio.shield(save_cancelled_state())
                except (asyncio.CancelledError, Exception):
                    pass
                raise  # re-raise CancelledError for proper propagation

            if response.background is not None:
                await response.background()

        return await response_handler(response, events)

    else:
        # Fallback to the original response
        async def stream_wrapper(original_generator, events):
            def wrap_item(item):
                return f'data: {item}\n\n'

            assistant_message = {}
            filter_context = FilterContext()
            has_api_outlet_filters = ENABLE_API_OUTLET_FILTERS and bool(filter_functions)
            if ENABLE_API_OUTLET_FILTERS and not has_api_outlet_filters:
                try:
                    model_id = model.get('id') if isinstance(model, dict) else model
                    has_api_outlet_filters = bool(
                        (isinstance(model, dict) and 'pipeline' in model)
                        or get_sorted_filters(model_id, request.app.state.MODELS)
                    )
                except Exception:
                    has_api_outlet_filters = True

            for event in events:
                event, _ = await process_filter_functions(
                    request=request,
                    filter_context=filter_context,
                    filter_functions=filter_functions,
                    filter_type='stream',
                    form_data=event,
                    extra_params=extra_params,
                )

                if event:
                    yield wrap_item(JSONCodec.dumps(event))

            async for data in original_generator:
                if filter_functions:
                    line = data.decode('utf-8', 'replace') if isinstance(data, bytes) else data
                    if isinstance(line, str) and line.startswith('data:'):
                        payload = line.removeprefix('data:').strip()
                        if payload and payload != '[DONE]':
                            try:
                                event = JSONCodec.loads(payload)
                            except JSONCodec.JSONDecodeError:
                                event = None

                            if isinstance(event, dict):
                                event, _ = await process_filter_functions(
                                    request=request,
                                    filter_context=filter_context,
                                    filter_functions=filter_functions,
                                    filter_type='stream',
                                    form_data=event,
                                    extra_params=extra_params,
                                )
                                data = wrap_item(JSONCodec.dumps(event)) if event else None

                if data:
                    if has_api_outlet_filters:
                        update_assistant_message_from_stream(assistant_message, data)
                    yield data

            if has_api_outlet_filters and assistant_message:
                ctx['assistant_message'] = assistant_message
                await outlet_filter_handler(ctx)

        return StreamingResponse(
            stream_wrapper(response.body_iterator, events),
            headers=dict(response.headers),
            background=response.background,
        )


