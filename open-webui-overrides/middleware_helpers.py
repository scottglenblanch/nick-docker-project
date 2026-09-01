import copy
import json
import logging
import mimetypes
import os
import re
import sys
from typing import Any, Optional
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from open_webui.config import DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE
from open_webui.constants import TASKS
from open_webui.env import ENABLE_QUERIES_CACHE, GLOBAL_LOG_LEVEL, RAG_SYSTEM_CONTEXT
from open_webui.events import EVENTS, publish_event
from open_webui.models.chats import Chats
from open_webui.models.config import Config
from open_webui.models.users import UserModel
from open_webui.retrieval.utils import get_sources_from_items
from open_webui.routers.images import (
    CreateImageForm,
    EditImageForm,
    image_edits,
    image_generations,
)
from open_webui.routers.retrieval import (
    SearchForm,
    process_web_search,
)
from open_webui.routers.tasks import generate_image_prompt, generate_queries
from open_webui.socket.main import get_event_emitter
from open_webui.utils.access_control import has_connection_access
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.files import get_file_url_from_base64, get_image_base64_from_url, get_image_url_from_base64
from open_webui.utils.json_codec import JSONCodec
from open_webui.utils.mcp.client import MCPClient
from open_webui.utils.misc import add_or_update_system_message, add_or_update_user_message, convert_output_to_messages, get_content_from_message, get_last_user_message, get_message_list, get_output_text, is_string_allowed, merge_system_messages, strip_empty_content_blocks
from open_webui.utils.task import (
    get_task_model_id,
    rag_template,
    tools_function_calling_generation_template,
)
from open_webui.utils.tools import build_tool_server_headers
from starlette.responses import JSONResponse

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)


def _is_tool_result_error(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if (
            text.startswith('error:')
            or text.startswith('exception:')
            or text.startswith('traceback')
            or text.startswith('http error!')
        ):
            return True

    parsed = value
    while isinstance(parsed, str):
        try:
            parsed = JSONCodec.loads(parsed)
        except (JSONCodec.JSONDecodeError, TypeError, ValueError):
            break

    if not isinstance(parsed, dict):
        return False

    error = parsed.get('error')
    if isinstance(error, str):
        has_error = bool(error.strip())
    else:
        has_error = isinstance(error, (dict, list)) and bool(error)
    if has_error:
        return True

    status = parsed.get('status')
    if isinstance(status, str) and status.strip().lower() in {'error', 'failed'}:
        return True

    if parsed.get('success') is False or parsed.get('ok') is False:
        message = parsed.get('message')
        return has_error or (
            bool(message.strip()) if isinstance(message, str) else isinstance(message, (dict, list)) and bool(message)
        )

    return False


def normalize_messages_for_model(form_data: dict) -> dict:
    form_data['messages'] = strip_empty_content_blocks(form_data.get('messages', []))
    form_data['messages'] = merge_system_messages(form_data.get('messages', []))
    return form_data


async def publish_chat_finished_event(
    request: Request, user: UserModel, metadata: dict, title: str, content: str, output: list | None = None
):
    chat_id = metadata.get('chat_id')
    if getattr(request.state, 'internal', False) is True or not is_saved_chat_id(chat_id):
        return

    content = content or get_output_text(output)
    webui_url = await Config.get('webui.url')
    await publish_event(
        request,
        EVENTS.CHAT_FINISHED,
        actor=user,
        subject_id=chat_id,
        subject_type='chat',
        data={
            'user_id': user.id,
            'chat_id': chat_id,
            'message_id': metadata.get('message_id'),
            'model_id': metadata.get('model_id'),
            'title': title,
            'url': f'{webui_url}/c/{chat_id}' if webui_url else f'/c/{chat_id}',
            'message': content,
        },
        message=title or 'Chat finished',
    )
    event_emitter = await get_event_emitter(metadata, update_db=False)
    if event_emitter:
        folder_id = metadata.get('folder_id') or await Chats.get_chat_folder_id(chat_id, metadata.get('user_id'))
        await event_emitter({'type': 'chat:list', 'data': {'chat_id': chat_id, 'folder_id': folder_id}})


# We believe in one maker of all models, seen and unseen,
# and in the reasoning which proceeds from the architect.
# We look for the resurrection of dead processes and the
# inference of the world to come.
DEFAULT_REASONING_TAGS = [
    ('<think>', '</think>'),
    ('<thinking>', '</thinking>'),
    ('<reason>', '</reason>'),
    ('<reasoning>', '</reasoning>'),
    ('<thought>', '</thought>'),
    ('<Thought>', '</Thought>'),
    ('<|begin_of_thought|>', '<|end_of_thought|>'),
    ('◁think▷', '◁/think▷'),
]

DEFAULT_SOLUTION_TAGS = [('<|begin_of_solution|>', '<|end_of_solution|>')]
DEFAULT_CODE_INTERPRETER_TAGS = [('<code_interpreter>', '</code_interpreter>')]


def _start_tag_pattern(start_tag: str) -> str:
    if start_tag.startswith('<') and start_tag.endswith('>'):
        return rf'<{re.escape(start_tag[1:-1])}(\s.*?)?>'
    return re.escape(start_tag)


def output_id(prefix: str) -> str:
    """Generate OR-style ID: prefix + 24-char hex UUID."""
    return f'{prefix}_{uuid4().hex[:24]}'


def build_terminal_file_tool_result(
    tool_function_name: str,
    tool_function_params: dict,
    tool_result: Any,
    tool: dict | None,
    metadata: dict | None,
) -> dict | None:
    if isinstance(tool_result, (list, tuple)) and tool_result and isinstance(tool_result[0], dict):
        tool_result = tool_result[0]

    if tool_function_name != 'display_file' or not isinstance(tool_result, dict) or tool_result.get('exists') is False:
        return None

    tool_id = (tool or {}).get('tool_id', '')
    terminal_id = metadata.get('terminal_id') if metadata else None
    if isinstance(tool_id, str) and tool_id.startswith('terminal:'):
        terminal_id = tool_id.split(':', 1)[1]

    server_url = ((tool or {}).get('server') or {}).get('url')
    terminal_selector = terminal_id or server_url
    path = tool_result.get('path') or tool_function_params.get('path')
    if not terminal_selector or not path:
        return None
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type or 'application/octet-stream'
    page = tool_result.get('page') or tool_function_params.get('page')

    return {
        **tool_result,
        'type': 'file',
        'source': 'open_terminal',
        **({'displayed': True} if tool_function_params.get('inline') is True else {}),
        'terminal_selector': terminal_selector,
        **({'terminal_id': terminal_id} if terminal_id else {}),
        **({'terminal_url': server_url} if server_url and not terminal_id else {}),
        'session_id': metadata.get('chat_id') if metadata else None,
        'path': path,
        'full_path': tool_result.get('full_path') or path,
        'name': tool_result.get('name') or os.path.basename(path),
        'mime_type': tool_result.get('mime_type') or tool_result.get('content_type') or mime_type,
        'content_type': tool_result.get('content_type') or tool_result.get('mime_type') or mime_type,
        **({'page': page} if page else {}),
    }


def tool_result_content(tool_result: Any) -> str:
    if not tool_result:
        return ''
    if isinstance(tool_result, (dict, list)):
        return JSONCodec.dumps(tool_result, ensure_ascii=False)
    return str(tool_result)


def merge_streamed_reasoning_details(target: list, details) -> None:
    items = details if isinstance(details, list) else [details]
    for item in items:
        if not isinstance(item, dict):
            continue

        index = item.get('index')
        existing = (
            next((detail for detail in target if detail.get('index') == index), None)
            if isinstance(index, int)
            else None
        )
        if existing is None:
            target.append(dict(item))
            continue

        for key, value in item.items():
            if key in ('text', 'summary') and isinstance(value, str) and isinstance(existing.get(key), str):
                existing[key] += value
            else:
                existing[key] = value


def _split_tool_calls(
    tool_calls: list[dict],
) -> list[dict]:
    """Expand tool calls whose arguments contain multiple back-to-back JSON objects.

    Some models (e.g. GPT-5.4) send multiple complete JSON argument objects
    under the same tool call index, producing concatenated invalid JSON like:
        '{"query":"A","count":5}{"query":"B","count":5}'

    Each such tool call is split into separate entries so each gets executed
    independently. Single-object arguments pass through unchanged.
    """

    def split_json_objects(raw: str) -> list[str]:
        if not isinstance(raw, str):
            raw = '' if raw is None else JSONCodec.dumps(raw)

        decoder = json.JSONDecoder()
        results = []
        position = 0

        while position < len(raw):
            while position < len(raw) and raw[position].isspace():
                position += 1
            if position >= len(raw):
                break
            try:
                _, end = decoder.raw_decode(raw, position)
                results.append(raw[position:end].strip())
                position = end
            except JSONCodec.JSONDecodeError:
                return [raw]

        return results or [raw]

    expanded = []
    for tool_call in tool_calls:
        function = tool_call.setdefault('function', {})
        arguments = function.get('arguments')
        if not isinstance(arguments, str):
            arguments = '' if arguments is None else JSONCodec.dumps(arguments)
            function['arguments'] = arguments
        split_arguments = split_json_objects(arguments)

        if len(split_arguments) <= 1:
            expanded.append(tool_call)
        else:
            for argument in split_arguments:
                cloned = copy.deepcopy(tool_call)
                cloned['id'] = f'call_{uuid4().hex[:24]}'
                cloned['function']['arguments'] = argument
                expanded.append(cloned)

    return expanded


def get_citation_source_from_tool_result(
    tool_name: str, tool_params: dict, tool_result: str, tool_id: str = ''
) -> list[dict]:
    """
    Parse a tool's result and convert it to source dicts for citation display.

    Follows the source format conventions from get_sources_from_items:
    - source: file/item info object with id, name, type
    - document: list of document contents
    - metadata: list of metadata objects with source, file_id, name fields

    Returns a list of sources (usually one, but query_knowledge_files/query_chat_files may return multiple).
    """
    _EXPECTS_LIST = {'search_web', 'query_knowledge_files', 'query_chat_files'}
    _EXPECTS_DICT = {'view_knowledge_file', 'view_file'}

    try:
        try:
            tool_result = JSONCodec.loads(tool_result)
        except (JSONCodec.JSONDecodeError, TypeError):
            pass  # keep tool_result as-is (e.g. fetch_url returns plain text)
        if isinstance(tool_result, dict) and 'error' in tool_result:
            return []

        # Validate tool_result type based on what the branch expects
        if tool_name in _EXPECTS_LIST and not isinstance(tool_result, list):
            return []
        elif tool_name in _EXPECTS_DICT and not isinstance(tool_result, dict):
            return []

        if tool_name == 'search_web':
            # Parse JSON array: [{"title": "...", "link": "...", "snippet": "..."}]
            results = tool_result
            documents = []
            metadata = []

            for result in results:
                title = result.get('title', '')
                link = result.get('link', '')
                snippet = result.get('snippet', '')

                documents.append(f'{title}\n{snippet}')
                metadata.append(
                    {
                        'source': link,
                        'name': title,
                        'url': link,
                    }
                )

            return [
                {
                    'source': {'name': 'search_web', 'id': 'search_web'},
                    'document': documents,
                    'metadata': metadata,
                }
            ]

        elif tool_name in ('view_knowledge_file', 'view_file'):
            file_data = tool_result
            filename = file_data.get('filename', 'Unknown File')
            file_id = file_data.get('id', '')
            knowledge_name = file_data.get('knowledge_name', '')

            return [
                {
                    'source': {
                        'id': file_id,
                        'name': filename,
                        'type': 'file',
                    },
                    'document': [file_data.get('content', '')],
                    'metadata': [
                        {
                            'file_id': file_id,
                            'name': filename,
                            'source': filename,
                            **({'knowledge_name': knowledge_name} if knowledge_name else {}),
                        }
                    ],
                }
            ]

        elif tool_name == 'fetch_url':
            url = tool_params.get('url', '')
            content = tool_result if isinstance(tool_result, str) else str(tool_result)
            snippet = content[:500] + ('...' if len(content) > 500 else '')

            return [
                {
                    'source': {'name': url or 'fetch_url', 'id': url or 'fetch_url'},
                    'document': [snippet],
                    'metadata': [
                        {
                            'source': url,
                            'name': url,
                            'url': url,
                        }
                    ],
                }
            ]

        elif tool_name in ('query_knowledge_files', 'query_chat_files'):
            chunks = tool_result

            # Group chunks by source for better citation display
            # Each unique source becomes a separate source entry
            sources_by_file = {}

            for chunk in chunks:
                source_name = chunk.get('source', 'Unknown')
                file_id = chunk.get('file_id', '')
                note_id = chunk.get('note_id', '')
                chunk_type = chunk.get('type', 'file')
                content = chunk.get('content', '')

                # Use file_id or note_id as the key
                key = file_id or note_id or source_name

                if key not in sources_by_file:
                    sources_by_file[key] = {
                        'source': {
                            'id': file_id or note_id,
                            'name': source_name,
                            'type': chunk_type,
                        },
                        'document': [],
                        'metadata': [],
                    }

                sources_by_file[key]['document'].append(content)
                sources_by_file[key]['metadata'].append(
                    {
                        'file_id': file_id,
                        'name': source_name,
                        'source': source_name,
                        **({'note_id': note_id} if note_id else {}),
                    }
                )

            # Return all grouped sources as a list
            if sources_by_file:
                return list(sources_by_file.values())

            # Empty result fallback
            return []

        else:
            # Fallback for other tools
            return [
                {
                    'source': {
                        'name': tool_name,
                        'type': 'tool',
                        'id': tool_id or tool_name,
                    },
                    'document': [str(tool_result)],
                    'metadata': [{'source': tool_name, 'name': tool_name}],
                }
            ]
    except Exception as e:
        log.exception(f'Error parsing tool result for {tool_name}: {e}')
        return [
            {
                'source': {'name': tool_name, 'type': 'tool'},
                'document': [str(tool_result)],
                'metadata': [{'source': tool_name}],
            }
        ]


def deep_merge(target, source):
    """
    Merge source into target recursively (returning new structure).
    - Dicts: Recursive merge.
    - Strings: Concatenation.
    - Others: Overwrite.
    """
    if isinstance(target, dict) and isinstance(source, dict):
        new_target = target.copy()
        for k, v in source.items():
            if k in new_target:
                new_target[k] = deep_merge(new_target[k], v)
            else:
                new_target[k] = v
        return new_target
    elif isinstance(target, str) and isinstance(source, str):
        return target + source
    else:
        return source


RESPONSE_COMPLETION_RESPONSE_FIELDS = ('error', 'id', 'output', 'usage')


def get_response_completion_event_data(event: dict) -> dict:
    """Build the data payload for response:completion events."""
    response = event.get('response')
    if not isinstance(response, dict):
        return event

    response_data = {key: response[key] for key in RESPONSE_COMPLETION_RESPONSE_FIELDS if key in response}

    return {
        **event,
        'response': response_data,
    }


def handle_responses_streaming_event(
    data: dict,
    current_output: list,
) -> tuple[list, dict | None]:
    """
    Handle Responses API streaming events in a pure functional way.

    Args:
        data: The event data
        current_output: List of output items (treated as immutable)

    Returns:
        tuple[list, dict | None]: (new_output, metadata)
        - new_output: The updated output list.
        - metadata: Metadata to emit (e.g. usage), {} if update occurred, None if skip.
    """
    # Default: no change
    # Note: treating current_output as immutable, but avoiding full deepcopy for perf.
    # We will shallow copy only if we need to modify the list structure or items.

    event_type = data.get('type', '')

    if event_type == 'response.output_item.added':
        item = data.get('item', {})
        if item:
            new_output = list(current_output)
            output_index = data.get('output_index', len(new_output))
            existing_index = next(
                (
                    idx
                    for idx, existing in enumerate(new_output)
                    if (item.get('id') and existing.get('id') == item.get('id'))
                    or (item.get('call_id') and existing.get('call_id') == item.get('call_id'))
                ),
                None,
            )
            if existing_index is not None:
                new_output[existing_index] = item
            elif 0 <= output_index < len(new_output):
                new_output.insert(output_index, item)
            else:
                new_output.append(item)
            return new_output, None
        return current_output, None

    elif event_type == 'response.content_part.added':
        part = data.get('part', {})
        output_index = data.get('output_index', len(current_output) - 1)

        if current_output and 0 <= output_index < len(current_output):
            new_output = list(current_output)
            # Copy the item to mutate it
            item = new_output[output_index].copy()
            new_output[output_index] = item

            if 'content' not in item:
                item['content'] = []
            else:
                # Copy content list
                item['content'] = list(item['content'])

            if item.get('type') == 'reasoning':
                # Reasoning items should not have content parts
                pass
            else:
                item['content'].append(part)
            return new_output, None
        return current_output, None

    elif event_type == 'response.reasoning_summary_part.added':
        part = data.get('part', {})
        output_index = data.get('output_index', len(current_output) - 1)

        if current_output and 0 <= output_index < len(current_output):
            new_output = list(current_output)
            item = new_output[output_index].copy()
            new_output[output_index] = item

            if 'summary' not in item:
                item['summary'] = []
            else:
                item['summary'] = list(item['summary'])

            item['summary'].append(part)
            return new_output, None
        return current_output, None

    elif event_type.startswith('response.') and event_type.endswith('.delta'):
        # Generic Delta Handling
        parts = event_type.split('.')
        if len(parts) >= 3:
            delta_type = parts[1]
            delta = data.get('delta', '')

            output_index = data.get('output_index', len(current_output) - 1)

            if current_output and 0 <= output_index < len(current_output):
                new_output = list(current_output)
                item = new_output[output_index].copy()
                new_output[output_index] = item
                item_type = item.get('type', '')

                # Determine target field and object based on delta_type and item_type
                if delta_type == 'function_call_arguments':
                    key = 'arguments'
                    if item_type == 'function_call':
                        # Function call args are usually strings
                        item[key] = item.get(key, '') + str(delta)
                else:
                    # Generic handling, refined by item type below
                    pass

                    if item_type == 'message':
                        # Message items: "text"/"output_text" -> "text"
                        # "reasoning_text" -> Skipped (should use reasoning item)
                        if delta_type in ['text', 'output_text']:
                            key = 'text'
                        elif delta_type in ['reasoning_text', 'reasoning_summary_text']:
                            # Skip reasoning updates for message items
                            return new_output, None
                        else:
                            key = delta_type

                        content_index = data.get('content_index', 0)
                        if 'content' not in item:
                            item['content'] = []
                        else:
                            item['content'] = list(item['content'])
                        content_list = item['content']

                        while len(content_list) <= content_index:
                            content_list.append({'type': 'text', 'text': ''})

                        # Copy the part to mutate it
                        part = content_list[content_index].copy()
                        content_list[content_index] = part

                        current_val = part.get(key)
                        if current_val is None:
                            # Initialize based on delta type
                            current_val = {} if isinstance(delta, dict) else ''

                        part[key] = deep_merge(current_val, delta)

                    elif item_type == 'reasoning':
                        # Reasoning items: "reasoning_text"/"reasoning_summary_text" -> "text"
                        # "text"/"output_text" -> Skipped (should use message item)
                        if delta_type == 'reasoning_summary_text':
                            # Summary updates -> item['summary']
                            key = 'text'
                            summary_index = data.get('summary_index', 0)
                            if 'summary' not in item:
                                item['summary'] = []
                            else:
                                item['summary'] = list(item['summary'])
                            summary_list = item['summary']

                            while len(summary_list) <= summary_index:
                                summary_list.append({'type': 'summary_text', 'text': ''})

                            part = summary_list[summary_index].copy()
                            summary_list[summary_index] = part

                            target_val = part.get(key, '')
                            part[key] = deep_merge(target_val, delta)

                        elif delta_type == 'reasoning_text':
                            # Reasoning body updates -> item['content']
                            key = 'text'
                            content_index = data.get('content_index', 0)
                            if 'content' not in item:
                                item['content'] = []
                            else:
                                item['content'] = list(item['content'])
                            content_list = item['content']

                            while len(content_list) <= content_index:
                                # Reasoning content parts default to text
                                content_list.append({'type': 'text', 'text': ''})

                            part = content_list[content_index].copy()
                            content_list[content_index] = part

                            target_val = part.get(key, '')
                            part[key] = deep_merge(target_val, delta)

                        elif delta_type in ['text', 'output_text']:
                            return new_output, None
                        else:
                            # Fallback just in case other deltas target reasoning?
                            pass

                    else:
                        # Fallback for other item types
                        if delta_type in ['text', 'output_text']:
                            key = 'text'
                        else:
                            key = delta_type

                        current_val = item.get(key)
                        if current_val is None:
                            current_val = {} if isinstance(delta, dict) else ''
                        item[key] = deep_merge(current_val, delta)

                return new_output, None

        return current_output, None

    elif event_type == 'response.output_item.done':
        # Delta Event: Output item complete
        item = data.get('item')
        output_index = data.get('output_index', len(current_output) - 1)

        new_output = list(current_output)
        if item and 0 <= output_index < len(current_output):
            new_output[output_index] = item
        elif item:
            new_output.append(item)
        return new_output, {}

    elif event_type.startswith('response.') and event_type.endswith('.done'):
        # Delta Events: response.content_part.done, response.text.done, etc.
        parts = event_type.split('.')
        if len(parts) >= 3:
            type_name = parts[1]

            # 1. Handle specific Delta "done" signals
            if type_name == 'content_part':
                # "Signaling that no further changes will occur to a content part"
                # If payloads contains the full part, we could update it.
                # Usually purely signaling in standard implementation, but we check payload.
                part = data.get('part')
                output_index = data.get('output_index', len(current_output) - 1)

                if part and current_output and 0 <= output_index < len(current_output):
                    new_output = list(current_output)
                    item = new_output[output_index].copy()
                    new_output[output_index] = item

                    if 'content' in item:
                        item['content'] = list(item['content'])
                        content_index = data.get('content_index', len(item['content']) - 1)
                        if 0 <= content_index < len(item['content']):
                            item['content'][content_index] = part
                            return new_output, {}
                return current_output, None

            elif type_name == 'reasoning_summary_part':
                part = data.get('part')
                output_index = data.get('output_index', len(current_output) - 1)

                if part and current_output and 0 <= output_index < len(current_output):
                    new_output = list(current_output)
                    item = new_output[output_index].copy()
                    new_output[output_index] = item

                    if 'summary' in item:
                        item['summary'] = list(item['summary'])
                        summary_index = data.get('summary_index', len(item['summary']) - 1)
                        if 0 <= summary_index < len(item['summary']):
                            item['summary'][summary_index] = part
                            return new_output, {}
                return current_output, None

            # 2. Generic Field Done (text.done, audio.done)
            if type_name not in ['completed', 'failed']:
                output_index = data.get('output_index', len(current_output) - 1)
                if current_output and 0 <= output_index < len(current_output):
                    key = (
                        'text'
                        if type_name
                        in [
                            'text',
                            'output_text',
                            'reasoning_text',
                            'reasoning_summary_text',
                        ]
                        else type_name
                    )
                    if type_name == 'function_call_arguments':
                        key = 'arguments'

                    if key in data:
                        final_value = data[key]
                        new_output = list(current_output)
                        item = new_output[output_index].copy()
                        new_output[output_index] = item
                        item_type = item.get('type', '')

                        if type_name == 'function_call_arguments':
                            if item_type == 'function_call':
                                item['arguments'] = final_value
                        elif item_type == 'message':
                            content_index = data.get('content_index', 0)
                            if 'content' in item:
                                item['content'] = list(item['content'])
                                if len(item['content']) > content_index:
                                    part = item['content'][content_index].copy()
                                    item['content'][content_index] = part
                                    part[key] = final_value
                        elif item_type == 'reasoning':
                            item['status'] = 'completed'
                        else:
                            item[key] = final_value

                        return new_output, {}

        return current_output, None

    elif event_type == 'response.completed':
        # State Machine Event: Completed
        response_data = data.get('response', {})
        final_output = response_data.get('output')

        # Some providers send an empty output on response.completed despite having streamed items
        new_output = final_output if final_output else current_output

        # Ensure reasoning items are marked as completed in the final output
        if new_output:
            for item in new_output:
                if item.get('type') == 'reasoning' and item.get('status') != 'completed':
                    item['status'] = 'completed'

        return new_output, {
            'usage': response_data.get('usage'),
            'done': True,
            'response_id': response_data.get('id'),
        }

    elif event_type == 'response.in_progress':
        # State Machine Event: In Progress
        # We could extract metadata if needed, but for now just acknowledge iteration
        return current_output, None

    elif event_type == 'response.failed':
        # State Machine Event: Failed
        error = data.get('response', {}).get('error', {})
        return current_output, {'error': error}

    else:
        return current_output, None


def get_source_context(sources: list, source_ids: dict = None, include_content: bool = True) -> str:
    """
    Build <source> tag context string from citation sources.
    """
    context_string = ''
    if source_ids is None:
        source_ids = {}
    for source in sources:
        for doc, meta in zip(source.get('document', []), source.get('metadata', [])):
            source_id = meta.get('source') or source.get('source', {}).get('id') or 'N/A'
            if source_id not in source_ids:
                source_ids[source_id] = len(source_ids) + 1
            src_name = source.get('source', {}).get('name')
            src_type = source.get('source', {}).get('type')
            src_rid = source.get('source', {}).get('id')
            body = doc if include_content else ''
            context_string += (
                f'<source id="{source_ids[source_id]}"'
                + (f' name="{src_name}"' if src_name else '')
                + (f' resource-type="{src_type}"' if src_type else '')
                + (f' resource-id="{src_rid}"' if src_rid else '')
                + f'>{body}</source>\n'
            )
    return context_string


async def apply_source_context_to_messages(
    request: Request,
    messages: list,
    sources: list,
    user_message: str,
    include_content: bool = True,
) -> list:
    """
    Build source context from citation sources and apply to messages.
    Uses RAG template to format context for model consumption.

    When include_content is False, emit <source> tags with id/name but no
    document body — useful when the content is already present elsewhere
    (e.g. in a tool result message) and only citation markers are needed.
    """
    if not sources or not user_message:
        return messages

    context = get_source_context(sources, include_content=include_content)

    context = context.strip()
    if not context:
        return messages

    if RAG_SYSTEM_CONTEXT:
        return add_or_update_system_message(
            await rag_template(await Config.get('rag.template'), context, user_message),
            messages,
            append=True,
        )
    else:
        return add_or_update_user_message(
            await rag_template(await Config.get('rag.template'), context, user_message),
            messages,
            append=False,
        )


async def process_tool_result(
    request,
    tool_function_name,
    tool_result,
    tool_type,
    direct_tool=False,
    metadata=None,
    user=None,
):
    tool_result_embeds = []
    EXTERNAL_TOOL_TYPES = ('external', 'action', 'terminal')

    # Support (HTMLResponse, result_context) tuples: the optional second
    # element lets tool authors provide the LLM with actionable context
    # about the generated embed instead of the generic fallback message.
    result_context = None
    if isinstance(tool_result, tuple) and len(tool_result) == 2 and isinstance(tool_result[0], HTMLResponse):
        tool_result, result_context = tool_result

    if isinstance(tool_result, HTMLResponse):
        content_disposition = tool_result.headers.get('Content-Disposition', '')
        if 'inline' in content_disposition:
            content = tool_result.body.decode('utf-8', 'replace')
            tool_result_embeds.append(content)

            if 200 <= tool_result.status_code < 300:
                if result_context is not None and isinstance(result_context, (str, dict, list)):
                    tool_result = result_context
                else:
                    tool_result = {
                        'status': 'success',
                        'code': 'ui_component',
                        'message': f'{tool_function_name}: Embedded UI result is active and visible to the user.',
                    }
            elif 400 <= tool_result.status_code < 500:
                tool_result = {
                    'status': 'error',
                    'code': 'ui_component',
                    'message': f'{tool_function_name}: Client error {tool_result.status_code} from embedded UI result.',
                }
            elif 500 <= tool_result.status_code < 600:
                tool_result = {
                    'status': 'error',
                    'code': 'ui_component',
                    'message': f'{tool_function_name}: Server error {tool_result.status_code} from embedded UI result.',
                }
            else:
                tool_result = {
                    'status': 'error',
                    'code': 'ui_component',
                    'message': f'{tool_function_name}: Unexpected status code {tool_result.status_code} from embedded UI result.',
                }
        else:
            tool_result = tool_result.body.decode('utf-8', 'replace')

    elif (tool_type in EXTERNAL_TOOL_TYPES and isinstance(tool_result, tuple)) or (
        direct_tool and isinstance(tool_result, list) and len(tool_result) == 2
    ):
        tool_result, tool_response_headers = tool_result

        try:
            if not isinstance(tool_response_headers, dict):
                tool_response_headers = dict(tool_response_headers)
        except Exception as e:
            tool_response_headers = {}
            log.debug(e)

        if tool_response_headers and isinstance(tool_response_headers, dict):
            content_disposition = tool_response_headers.get(
                'Content-Disposition',
                tool_response_headers.get('content-disposition', ''),
            )

            if 'inline' in content_disposition:
                content_type = tool_response_headers.get(
                    'Content-Type',
                    tool_response_headers.get('content-type', ''),
                )
                location = tool_response_headers.get(
                    'Location',
                    tool_response_headers.get('location', ''),
                )

                if 'text/html' in content_type:
                    # Support (html_content, result_context) nested tuple
                    result_context = None
                    html_content = tool_result
                    if isinstance(tool_result, (tuple, list)) and len(tool_result) == 2:
                        html_content, result_context = tool_result

                    # Display as iframe embed
                    tool_result_embeds.append(html_content)
                    if result_context is not None and isinstance(result_context, (str, dict, list)):
                        tool_result = result_context
                    else:
                        tool_result = {
                            'status': 'success',
                            'code': 'ui_component',
                            'message': f'{tool_function_name}: Embedded UI result is active and visible to the user.',
                        }
                elif location:
                    # Support (html_content, result_context) nested tuple for location embeds
                    result_context = None
                    if isinstance(tool_result, (tuple, list)) and len(tool_result) == 2:
                        _, result_context = tool_result

                    tool_result_embeds.append(location)
                    if result_context is not None and isinstance(result_context, (str, dict, list)):
                        tool_result = result_context
                    else:
                        tool_result = {
                            'status': 'success',
                            'code': 'ui_component',
                            'message': f'{tool_function_name}: Embedded UI result is active and visible to the user.',
                        }

    tool_result_files = []

    # Detect base64 image data URIs from tool results (e.g. binary image
    # responses from execute_tool_server).  Move the data URI to
    # tool_result_files and replace tool_result with a text summary.
    if isinstance(tool_result, str) and tool_result.startswith('data:image/'):
        tool_result_files.append({'type': 'image', 'url': tool_result})
        tool_result = f'{tool_function_name}: Image file read successfully.'

    if isinstance(tool_result, list):
        if tool_type == 'mcp':  # MCP
            tool_response = []
            for item in tool_result:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text = item.get('text', '')
                        if isinstance(text, str):
                            try:
                                text = JSONCodec.loads(text)
                            except JSONCodec.JSONDecodeError:
                                pass
                        tool_response.append(text)
                    elif item.get('type') in ['image', 'audio']:
                        file_url = await get_file_url_from_base64(
                            request,
                            f'data:{item.get("mimeType")};base64,{item.get("data", item.get("blob", ""))}',
                            {
                                'chat_id': metadata.get('chat_id', None),
                                'message_id': metadata.get('message_id', None),
                                'session_id': metadata.get('session_id', None),
                                'result': item,
                            },
                            user,
                        )

                        tool_result_files.append(
                            {
                                'type': item.get('type', 'data'),
                                'url': file_url,
                            }
                        )
                    elif item.get('type') == 'resource':
                        resource = item.get('resource', {})
                        text = resource.get('text', '')
                        if isinstance(text, str) and text:
                            try:
                                text = JSONCodec.loads(text)
                            except JSONCodec.JSONDecodeError:
                                pass
                            tool_response.append(text)
                        elif resource.get('blob'):
                            resource_mime_type = resource.get('mimeType') or 'application/octet-stream'
                            resource_blob = resource.get('blob', '')
                            if resource_mime_type.startswith('image/'):
                                tool_result_files.append(
                                    {
                                        'type': 'image',
                                        'url': f'data:{resource_mime_type};base64,{resource_blob}',
                                    }
                                )
                            else:
                                resource_uri = resource.get('uri', 'resource')
                                tool_response.append(
                                    f'[Resource: {resource_uri}] (binary data, mimeType: {resource_mime_type})'
                                )
                        elif resource.get('uri'):
                            tool_response.append(resource.get('uri'))
            tool_result = tool_response[0] if len(tool_response) == 1 else tool_response
        else:  # OpenAPI
            for item in tool_result:
                if isinstance(item, str) and item.startswith('data:'):
                    tool_result_files.append(
                        {
                            'type': 'data',
                            'content': item,
                        }
                    )
                    tool_result.remove(item)

    if isinstance(tool_result, list):
        tool_result = {'results': tool_result}

    if isinstance(tool_result, dict) or isinstance(tool_result, list):
        tool_result = json.dumps(tool_result, indent=2, ensure_ascii=False)

    # Safety: ensure tool_result is always a string (or None) to prevent
    # downstream TypeError when concatenating (e.g. if an upstream callable
    # returned a tuple that was not unpacked by the branches above).
    if tool_result is not None and not isinstance(tool_result, str):
        if isinstance(tool_result, tuple):
            # execute_tool_server returns (data, headers); unpack the data part
            tool_result = json.dumps(tool_result[0], indent=2, ensure_ascii=False) if len(tool_result) > 0 else ''
        else:
            tool_result = str(tool_result)

    return tool_result, tool_result_files, tool_result_embeds


async def terminal_event_handler(
    tool_function_name: str,
    tool_function_params: dict,
    tool_result,
    event_emitter,
):
    """Emit terminal:* events for Open Terminal tools.

    - display_file  → emits 'terminal:display_file' to open the file preview.
    - write_file / replace_file_content → emits 'terminal:write_file' to refresh.
    - run_command → emits 'terminal:run_command' with cwd to refresh if relevant.
    """
    if not event_emitter:
        return

    if tool_function_name == 'display_file':
        if tool_function_params.get('inline') is True:
            return
        path = tool_function_params.get('path', '')
        if not path:
            return
        # Only emit if the file actually exists
        parsed = tool_result
        if isinstance(parsed, str):
            try:
                parsed = JSONCodec.loads(parsed)
            except (JSONCodec.JSONDecodeError, TypeError):
                pass
        if isinstance(parsed, dict) and parsed.get('exists') is False:
            return
        page = tool_function_params.get('page')

        await event_emitter(
            {
                'type': f'terminal:{tool_function_name}',
                'data': {
                    'path': path,
                    **({'page': page} if page else {}),
                },
            }
        )
    elif tool_function_name in ('write_file', 'replace_file_content'):
        path = tool_function_params.get('path', '')
        if not path:
            return
        await event_emitter(
            {
                'type': f'terminal:{tool_function_name}',
                'data': {'path': path},
            }
        )
    elif tool_function_name == 'run_command':
        await event_emitter(
            {
                'type': 'terminal:run_command',
                'data': {},
            }
        )


async def chat_completion_tools_handler(
    request: Request, body: dict, extra_params: dict, user: UserModel, models, tools
) -> tuple[dict, dict]:
    async def get_content_from_response(response) -> Optional[str]:
        content = None
        if hasattr(response, 'body_iterator'):
            async for chunk in response.body_iterator:
                data = JSONCodec.loads(chunk.decode('utf-8', 'replace'))
                content = data['choices'][0]['message']['content']

            # Cleanup any remaining background tasks if necessary
            if response.background is not None:
                await response.background()
        else:
            content = response['choices'][0]['message']['content']
        return content

    def get_tools_function_calling_payload(messages, task_model_id, content):
        user_message = get_last_user_message(messages)

        if user_message and messages and messages[-1]['role'] == 'user':
            # Remove the last user message to avoid duplication
            messages = messages[:-1]

        recent_messages = messages[-4:] if len(messages) > 4 else messages
        chat_history = '\n'.join(
            f'{message["role"].upper()}: """{get_content_from_message(message)}"""' for message in recent_messages
        )

        prompt = f'History:\n{chat_history}\nQuery: {user_message}' if chat_history else f'Query: {user_message}'

        return {
            'model': task_model_id,
            'messages': [
                {'role': 'system', 'content': content},
                {'role': 'user', 'content': prompt},
            ],
            'stream': False,
            'metadata': {'task': str(TASKS.FUNCTION_CALLING)},
        }

    event_caller = extra_params['__event_call__']
    event_emitter = extra_params['__event_emitter__']
    metadata = extra_params['__metadata__']

    # One batched SELECT instead of four sequential round trips.
    task_config = await Config.get_many(
        'task.model.default',
        'task.model.external',
        'task.tools.prompt_template',
    )
    task_model_id = get_task_model_id(
        body['model'],
        task_config.get('task.model.default'),
        task_config.get('task.model.external'),
        models,
    )

    skip_files = False
    sources = []

    specs = [tool['spec'] for tool in tools.values()]
    tools_specs = JSONCodec.dumps(specs, ensure_ascii=False)

    tools_prompt_template = task_config.get('task.tools.prompt_template')
    if tools_prompt_template != '':
        template = tools_prompt_template
    else:
        template = DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE

    tools_function_calling_prompt = tools_function_calling_generation_template(template, tools_specs)
    payload = get_tools_function_calling_payload(body['messages'], task_model_id, tools_function_calling_prompt)

    try:
        response = await generate_chat_completion(request, form_data=payload, user=user)
        log.debug('response=%r', response)
        content = await get_content_from_response(response)
        log.debug('content=%r', content)

        if not content:
            return body, {}

        try:
            content = content[content.find('{') : content.rfind('}') + 1]
            if not content:
                raise Exception('No JSON object found in the response')

            result = JSONCodec.loads(content)

            async def tool_call_handler(tool_call):
                nonlocal skip_files

                log.debug('tool_call=%r', tool_call)

                tool_function_name = tool_call.get('name', None)
                if tool_function_name not in tools:
                    log.warning(f'Tool "{tool_function_name}" not found')
                    return

                tool_function_params = tool_call.get('parameters', {})

                tool = None
                tool_type = ''
                direct_tool = False

                try:
                    tool = tools[tool_function_name]
                    tool_type = tool.get('type', '')
                    direct_tool = tool.get('direct', False)

                    spec = tool.get('spec', {})
                    allowed_params = spec.get('parameters', {}).get('properties', {}).keys()
                    tool_function_params = {k: v for k, v in tool_function_params.items() if k in allowed_params}

                    if tool.get('direct', False):
                        tool_result = await event_caller(
                            {
                                'type': 'execute:tool',
                                'data': {
                                    'id': str(uuid4()),
                                    'name': tool_function_name,
                                    'params': tool_function_params,
                                    'server': tool.get('server', {}),
                                    'session_id': metadata.get('session_id', None),
                                },
                            }
                        )
                    else:
                        tool_function = tool['callable']
                        tool_result = await tool_function(**tool_function_params)

                except Exception as e:
                    tool_result = {'error': str(e)}

                tool_result, tool_result_files, tool_result_embeds = await process_tool_result(
                    request,
                    tool_function_name,
                    tool_result,
                    tool_type,
                    direct_tool,
                    metadata,
                    user,
                )

                if event_emitter:
                    await terminal_event_handler(
                        tool_function_name,
                        tool_function_params,
                        tool_result,
                        event_emitter,
                    )

                    if tool_result_files:
                        await event_emitter(
                            {
                                'type': 'files',
                                'data': {
                                    'files': tool_result_files,
                                },
                            }
                        )

                    if tool_result_embeds:
                        await event_emitter(
                            {
                                'type': 'embeds',
                                'data': {
                                    'embeds': tool_result_embeds,
                                },
                            }
                        )

                if tool_result:
                    tool = tools[tool_function_name]
                    tool_id = tool.get('tool_id', '')

                    tool_name = f'{tool_id}/{tool_function_name}' if tool_id else f'{tool_function_name}'

                    # Citation is enabled for this tool
                    sources.append(
                        {
                            'source': {
                                'name': (f'{tool_name}'),
                            },
                            'document': [str(tool_result)],
                            'metadata': [
                                {
                                    'source': (f'{tool_name}'),
                                    'parameters': tool_function_params,
                                }
                            ],
                            'tool_result': True,
                        }
                    )

                    if tools[tool_function_name].get('metadata', {}).get('file_handler', False):
                        skip_files = True

            # check if "tool_calls" in result
            if result.get('tool_calls'):
                for tool_call in result.get('tool_calls'):
                    await tool_call_handler(tool_call)
            else:
                await tool_call_handler(result)

        except Exception as e:
            log.debug('Error: %s', e)
            content = None
    except Exception as e:
        log.debug('Error: %s', e)
        content = None

    log.debug('tool_contexts: %s', sources)

    if skip_files and 'files' in body.get('metadata', {}):
        del body['metadata']['files']

    return body, {'sources': sources}


async def chat_web_search_handler(request: Request, form_data: dict, extra_params: dict, user):
    event_emitter = extra_params['__event_emitter__']
    await event_emitter(
        {
            'type': 'status',
            'data': {
                'action': 'web_search',
                'description': 'Searching the web',
                'done': False,
            },
        }
    )

    messages = form_data['messages']
    user_message = get_last_user_message(messages)

    queries = []
    try:
        res = await generate_queries(
            request,
            {
                'model': form_data['model'],
                'messages': messages,
                'prompt': user_message,
                'type': 'web_search',
                'chat_id': extra_params.get('__chat_id__'),
            },
            user,
        )

        # generate_queries returns a JSONResponse on error (e.g. model not
        # found, chat completion failure).  Extract the error detail and
        # re-raise so the outer except block falls back to using the raw
        # user message as the search query.
        if isinstance(res, JSONResponse):
            try:
                error_body = JSONCodec.loads(res.body)
                detail = error_body.get('detail', 'Query generation failed')
            except Exception:
                detail = 'Query generation failed'
            raise Exception(detail)

        response = res['choices'][0]['message']['content']

        try:
            bracket_start = response.rfind('{')
            bracket_end = response.rfind('}') + 1

            if bracket_start == -1 or bracket_end == -1:
                raise Exception('No JSON object found in the response')

            response = response[bracket_start:bracket_end]
            queries = JSONCodec.loads(response)
            queries = queries.get('queries', [])
        except Exception as e:
            queries = [response]

        if ENABLE_QUERIES_CACHE:
            request.state.cached_queries = queries

    except Exception as e:
        log.exception(e)
        queries = [user_message or '']

    # Check if generated queries are empty
    if len(queries) == 1 and queries[0].strip() == '':
        queries = [user_message or '']

    # Check if queries are not found
    if len(queries) == 0:
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'web_search',
                    'description': 'No search query generated',
                    'done': True,
                },
            }
        )
        return form_data

    await event_emitter(
        {
            'type': 'status',
            'data': {
                'action': 'web_search_queries_generated',
                'queries': queries,
                'done': False,
            },
        }
    )

    try:
        results = await process_web_search(
            request,
            SearchForm(queries=queries),
            user=user,
        )

        if results:
            files = form_data.get('files', [])

            if results.get('collection_names'):
                for col_idx, collection_name in enumerate(results.get('collection_names')):
                    files.append(
                        {
                            'collection_name': collection_name,
                            'name': ', '.join(queries),
                            'type': 'web_search',
                            'urls': results['filenames'],
                            'queries': queries,
                        }
                    )
            elif results.get('docs'):
                # Invoked when bypass embedding and retrieval is set to True
                docs = results['docs']
                files.append(
                    {
                        'docs': docs,
                        'name': ', '.join(queries),
                        'type': 'web_search',
                        'urls': results['filenames'],
                        'queries': queries,
                    }
                )

            form_data['files'] = files

            await event_emitter(
                {
                    'type': 'status',
                    'data': {
                        'action': 'web_search',
                        'description': 'Searched {{count}} sites',
                        'urls': results['filenames'],
                        'items': results.get('items', []),
                        'done': True,
                    },
                }
            )
        else:
            await event_emitter(
                {
                    'type': 'status',
                    'data': {
                        'action': 'web_search',
                        'description': 'No search results found',
                        'done': True,
                        'error': True,
                    },
                }
            )

    except Exception as e:
        log.exception(e)
        detail = e.detail if isinstance(e, HTTPException) else None
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'web_search',
                    'description': (str(detail) if detail else 'An error occurred while searching the web'),
                    'queries': queries,
                    'done': True,
                    'error': True,
                },
            }
        )

    return form_data


def get_images_from_messages(message_list):
    images = []

    for message in reversed(message_list):
        message_images = []
        for file in message.get('files', []):
            if file.get('type') == 'image':
                message_images.append(file.get('url'))
            elif file.get('content_type', '').startswith('image/'):
                message_images.append(file.get('url'))

        if message_images:
            images.append(message_images)

    return images


async def get_image_urls(delta_images, request, metadata, user) -> list[str]:
    if not isinstance(delta_images, list):
        return []

    image_urls = []
    for img in delta_images:
        if not isinstance(img, dict) or img.get('type') != 'image_url':
            continue

        url = img.get('image_url', {}).get('url')
        if not url:
            continue

        if url.startswith('data:image/png;base64'):
            url = await get_image_url_from_base64(request, url, metadata, user)

        image_urls.append(url)

    return image_urls


async def add_file_context(messages: list, chat_id: str, user) -> list:
    """
    Add file URLs to messages for native function calling.
    """
    if not is_saved_chat_id(chat_id):
        return messages

    chat = await Chats.get_chat_by_id_and_user_id(chat_id, user.id)
    if not chat:
        return messages

    history = chat.chat.get('history', {})
    stored_messages = get_message_list(history.get('messages', {}), history.get('currentId'))

    def format_file_tag(file):
        # Every file reaching here has a url or a chat id, so id is always set.
        attrs = f'type="{file.get("type", "file")}" id="{file.get("id") or file.get("url")}"'
        if file.get('url'):
            attrs += f' url="{file["url"]}"'
        if file.get('content_type'):
            attrs += f' content_type="{file["content_type"]}"'
        if file.get('name'):
            attrs += f' name="{file["name"]}"'
        return f'<file {attrs}/>'

    # Pair only user-role messages from both lists to avoid misalignment.
    # After process_messages_with_output(), assistant messages with tool calls
    # are expanded into multiple messages (assistant + tool results), making
    # the payload message list longer than the stored message list. A naive
    # positional zip() would pair user messages with wrong stored messages,
    # causing later images to lose their file context (see #21878).
    user_messages = [m for m in messages if m.get('role') == 'user']
    stored_user_messages = [m for m in stored_messages if m.get('role') == 'user']

    for message, stored_message in zip(user_messages, stored_user_messages):
        # Chat references carry no url - they are addressed by id via view_chat.
        attached_files = [
            file
            for file in stored_message.get('files', [])
            if (file.get('url') and not file.get('url').startswith('data:'))
            or (file.get('type') == 'chat' and file.get('id'))
        ]
        if not attached_files:
            continue

        file_tags = [format_file_tag(file) for file in attached_files]
        file_context = '<attached_files>\n' + '\n'.join(file_tags) + '\n</attached_files>\n\n'

        content = message.get('content', '')
        if isinstance(content, list):
            message['content'] = [{'type': 'text', 'text': file_context}] + content
        else:
            message['content'] = file_context + content

    return messages


async def chat_image_generation_handler(request: Request, form_data: dict, extra_params: dict, user):
    metadata = extra_params.get('__metadata__', {})
    chat_id = metadata.get('chat_id', None)
    __event_emitter__ = extra_params.get('__event_emitter__', None)

    if not chat_id or not isinstance(chat_id, str) or not __event_emitter__:
        return form_data

    is_channel_chat = chat_id.startswith('channel:')
    image_metadata = {
        'message_id': metadata.get('message_id', None),
        **({'channel_id': chat_id.removeprefix('channel:')} if is_channel_chat else {'chat_id': chat_id}),
    }

    if not is_saved_chat_id(chat_id):
        message_list = form_data.get('messages', [])
    else:
        chat = await Chats.get_chat_by_id_and_user_id(chat_id, user.id)

        messages_map = chat.chat.get('history', {}).get('messages', {})
        message_id = chat.chat.get('history', {}).get('currentId')
        message_list = get_message_list(messages_map, message_id)

    user_message = get_last_user_message(message_list)

    prompt = user_message
    message_images = get_images_from_messages(message_list)

    # Limit to first 2 sets of images
    # We may want to change this in the future to allow more images
    input_images = []
    for idx, images in enumerate(message_images):
        if idx >= 2:
            break
        for image in images:
            input_images.append(image)

    # Called directly, bypassing the /images routes that enforce these switches.
    editing = len(input_images) > 0 and await Config.get('images.edit.enable')
    if not editing and not await Config.get('image_generation.enable'):
        return form_data

    if is_saved_chat_id(chat_id):
        await __event_emitter__(
            {
                'type': 'status',
                'data': {'description': 'Creating image', 'done': False},
            }
        )

    system_message_content = ''

    if editing:
        # Edit image(s)
        try:
            images = await image_edits(
                request=request,
                form_data=EditImageForm(**{'prompt': prompt, 'image': input_images}),
                metadata=image_metadata,
                user=user,
            )

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {'description': 'Image created', 'done': True},
                }
            )

            await __event_emitter__(
                {
                    'type': 'files',
                    'data': {
                        'files': [
                            {
                                'type': 'image',
                                **image,
                            }
                            for image in images
                        ]
                    },
                }
            )

            system_message_content = '<context>The requested image has been edited and created and is now being shown to the user. Let them know that it has been generated.</context>'
        except Exception as e:
            log.debug(e)

            error_message = ''
            if isinstance(e, HTTPException):
                if e.detail and isinstance(e.detail, dict):
                    error_message = e.detail.get('message', str(e.detail))
                else:
                    error_message = str(e.detail)

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {
                        'description': f'An error occurred while generating an image',
                        'done': True,
                    },
                }
            )

            system_message_content = f'<context>Image generation was attempted but failed. The system is currently unable to generate the image. Tell the user that the following error occurred: {error_message}</context>'

    elif not await Config.get('image_generation.enable'):
        await __event_emitter__(
            {
                'type': 'status',
                'data': {
                    'description': 'Image generation is disabled',
                    'done': True,
                },
            }
        )

        system_message_content = '<context>Image generation was requested but the feature is currently disabled by the administrator, so no image was created. Let the user know that image generation is currently unavailable.</context>'

    else:
        # Create image(s)
        if await Config.get('image_generation.prompt.enable'):
            try:
                res = await generate_image_prompt(
                    request,
                    {
                        'model': form_data['model'],
                        'messages': form_data['messages'],
                        'chat_id': metadata.get('chat_id'),
                    },
                    user,
                )

                # Handle JSONResponse from error paths
                if isinstance(res, JSONResponse):
                    try:
                        error_body = JSONCodec.loads(res.body)
                        detail = error_body.get('detail', 'Image prompt generation failed')
                    except Exception:
                        detail = 'Image prompt generation failed'
                    raise Exception(detail)

                response = res['choices'][0]['message']['content']

                try:
                    bracket_start = response.rfind('{')
                    bracket_end = response.rfind('}') + 1

                    if bracket_start == -1 or bracket_end == -1:
                        raise Exception('No JSON object found in the response')

                    response = response[bracket_start:bracket_end]
                    response = JSONCodec.loads(response)
                    prompt = response.get('prompt', [])
                except Exception as e:
                    prompt = user_message

            except Exception as e:
                log.exception(e)
                prompt = user_message

        try:
            images = await image_generations(
                request=request,
                form_data=CreateImageForm(**{'prompt': prompt}),
                metadata=image_metadata,
                user=user,
            )

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {'description': 'Image created', 'done': True},
                }
            )

            await __event_emitter__(
                {
                    'type': 'files',
                    'data': {
                        'files': [
                            {
                                'type': 'image',
                                **image,
                            }
                            for image in images
                        ]
                    },
                }
            )

            system_message_content = '<context>The requested image has been created by the system successfully and is now being shown to the user. Let the user know that the image they requested has been generated and is now shown in the chat.</context>'
        except Exception as e:
            log.debug(e)

            error_message = ''
            if isinstance(e, HTTPException):
                if e.detail and isinstance(e.detail, dict):
                    error_message = e.detail.get('message', str(e.detail))
                else:
                    error_message = str(e.detail)

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {
                        'description': f'An error occurred while generating an image',
                        'done': True,
                    },
                }
            )

            system_message_content = f'<context>Image generation was attempted but failed because of an error. The system is currently unable to generate the image. Tell the user that the following error occurred: {error_message}</context>'

    if system_message_content:
        form_data['messages'] = add_or_update_system_message(system_message_content, form_data['messages'])

    return form_data


async def chat_completion_files_handler(
    request: Request, body: dict, extra_params: dict, user: UserModel
) -> tuple[dict, dict[str, list]]:
    __event_emitter__ = extra_params['__event_emitter__']
    sources = []

    files = [item for item in (body.get('metadata', {}).get('files', None) or []) if item.get('type') != 'filesystem']
    if files:
        # Check if all files are in full context mode
        all_full_context = all(item.get('context') == 'full' for item in files)

        queries = []
        if not all_full_context:
            try:
                queries_response = await generate_queries(
                    request,
                    {
                        'model': body['model'],
                        'messages': body['messages'],
                        'type': 'retrieval',
                        'chat_id': body.get('metadata', {}).get('chat_id'),
                    },
                    user,
                )
                queries_response = queries_response['choices'][0]['message']['content']

                try:
                    bracket_start = queries_response.rfind('{')
                    bracket_end = queries_response.rfind('}') + 1

                    if bracket_start == -1 or bracket_end == -1:
                        raise Exception('No JSON object found in the response')

                    queries_response = queries_response[bracket_start:bracket_end]
                    queries_response = JSONCodec.loads(queries_response)
                except Exception as e:
                    queries_response = {'queries': [queries_response]}

                queries = queries_response.get('queries', [])
            except Exception:
                pass

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {
                        'action': 'queries_generated',
                        'queries': queries,
                        'done': False,
                    },
                }
            )

        if len(queries) == 0:
            queries = [get_last_user_message(body['messages']) or '']

        try:
            # One batched SELECT instead of six sequential round trips.
            rag_config = await Config.get_many(
                'rag.top_k',
                'rag.top_k_reranker',
                'rag.relevance_threshold',
                'rag.hybrid_bm25_weight',
                'rag.enable_hybrid_search',
                'rag.full_context',
            )
            # Directly await async get_sources_from_items (no thread needed - fully async now)
            sources = await get_sources_from_items(
                request=request,
                items=files,
                queries=queries,
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=rag_config.get('rag.top_k'),
                reranking_function=(
                    (lambda query, documents: request.app.state.RERANKING_FUNCTION(query, documents, user=user))
                    if request.app.state.RERANKING_FUNCTION
                    else None
                ),
                k_reranker=rag_config.get('rag.top_k_reranker'),
                r=rag_config.get('rag.relevance_threshold'),
                hybrid_bm25_weight=rag_config.get('rag.hybrid_bm25_weight'),
                hybrid_search=rag_config.get('rag.enable_hybrid_search'),
                full_context=all_full_context or rag_config.get('rag.full_context'),
                user=user,
            )
        except Exception as e:
            log.exception(e)

        log.debug('rag_contexts:sources: %s', sources)

        unique_ids = set()
        for source in sources or []:
            if not source or len(source.keys()) == 0:
                continue

            documents = source.get('document') or []
            metadatas = source.get('metadata') or []
            src_info = source.get('source') or {}

            for index, _ in enumerate(documents):
                metadata = metadatas[index] if index < len(metadatas) else None
                _id = (metadata or {}).get('source') or (src_info or {}).get('id') or 'N/A'
                unique_ids.add(_id)

        sources_count = len(unique_ids)
        await __event_emitter__(
            {
                'type': 'status',
                'data': {
                    'action': 'sources_retrieved',
                    'count': sources_count,
                    'done': True,
                },
            }
        )

    return body, {'sources': sources}


async def convert_url_images_to_base64(form_data, user=None):
    messages = form_data.get('messages', [])

    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue

        new_content = []

        for item in content:
            if not isinstance(item, dict) or item.get('type') != 'image_url':
                new_content.append(item)
                continue

            image_url_data = item.get('image_url', {})
            if isinstance(image_url_data, dict):
                image_url = image_url_data.get('url') or ''
            elif isinstance(image_url_data, str):
                image_url = image_url_data
            else:
                image_url = ''
            if image_url.startswith('data:image/'):
                new_content.append(item)
                continue

            try:
                base64_data = await get_image_base64_from_url(image_url, user=user)
                if base64_data:
                    image_url_payload = {'url': base64_data}
                    if isinstance(image_url_data, dict) and image_url_data.get('detail'):
                        image_url_payload['detail'] = image_url_data['detail']
                    new_content.append(
                        {
                            'type': 'image_url',
                            'image_url': image_url_payload,
                        }
                    )
                else:
                    new_content.append(item)
            except Exception as e:
                log.debug('Error converting image URL to base64: %s', e)
                new_content.append(item)

        message['content'] = new_content

    return form_data


MESSAGE_REPLAY_KEYS = ('id', 'role', 'content', 'output', 'files', 'contextSummary', 'usage', 'model')


async def load_messages_from_db(chat_id: str, message_id: str) -> Optional[list[dict]]:
    """
    Load the message chain from DB up to message_id,
    keeping only fields needed to rebuild the LLM payload.
    """
    messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
    if not messages_map:
        return None

    db_messages = get_message_list(messages_map, message_id)
    if not db_messages:
        return None

    return [{k: v for k, v in msg.items() if k in MESSAGE_REPLAY_KEYS} for msg in db_messages]


def get_reasoning_format(model: dict) -> str | None:
    """
    Determine how reasoning should be included in reconstructed messages.

    Returns:
        'thinking': Ollama expects reasoning in the native thinking field.
        'think_tags': wrap reasoning in <think> tags inside content.
        'reasoning_content': llama.cpp supports reasoning_content as a top-level field.
        None: skip reasoning (safe default for strict providers).
    """
    provider = model.get('provider', '')
    if model.get('owned_by') == 'ollama':
        return 'thinking'
    if provider == 'llama.cpp':
        return 'reasoning_content'
    return None


def strip_reasoning_details(output: list) -> list:
    return [
        {key: value for key, value in item.items() if key != 'reasoning_details'} if isinstance(item, dict) else item
        for item in output
    ]


def process_messages_with_output(
    messages: list[dict],
    reasoning_format: str | None = None,
) -> list[dict]:
    """
    Process messages with OR-aligned output items for LLM consumption.

    For assistant messages with 'output' field, produces properly formatted
    OpenAI-style messages (tool_calls + tool results). Strips 'output' before LLM.
    """
    processed = []

    for message in messages:
        if message.get('role') == 'assistant' and message.get('output'):
            # Use output items for clean OpenAI-format messages
            output_messages = convert_output_to_messages(
                message['output'],
                raw=True,
                reasoning_format=reasoning_format,
                flatten_tool_images=True,
            )
            if output_messages:
                processed.extend(output_messages)
                continue

        clean_message = dict(message)
        for key in ('id', 'files', 'output', 'model', 'contextSummary', 'context_summary', 'usage'):
            clean_message.pop(key, None)
        processed.append(clean_message)

    return processed


def sanitize_tool_pairs(messages: list[dict]) -> list[dict]:
    tool_result_ids = {
        message.get('tool_call_id')
        for message in messages
        if message.get('role') == 'tool' and message.get('tool_call_id')
    }

    tool_call_ids = {
        tool_call.get('id')
        for message in messages
        for tool_call in (message.get('tool_calls') or [])
        if message.get('role') == 'assistant' and tool_call.get('id')
    }

    sanitized = []
    for message in messages:
        if message.get('role') == 'assistant' and message.get('tool_calls'):
            kept = [
                tool_call for tool_call in message.get('tool_calls') or [] if tool_call.get('id') in tool_result_ids
            ]
            if kept:
                sanitized.append({**message, 'tool_calls': kept})
            else:
                clean = dict(message)
                clean.pop('tool_calls', None)
                clean.pop('reasoning_items', None)
                if clean.get('content'):
                    sanitized.append(clean)
        elif message.get('role') != 'tool' or message.get('tool_call_id') in tool_call_ids:
            sanitized.append(message)

    return sanitized


# Ids are validated as [a-z0-9_-]+ on create; matching that keeps ordinary "<$..." text intact.
SKILL_MENTION_RE = re.compile(r'<(?:\$([a-z0-9_-]+)(?:\|[^>]*)?|/([a-z0-9_-]+)\|[^>]*)>')


def _get_text_parts(message: dict) -> list[str]:
    """Return all text segments from a message's content."""
    content = message.get('content')
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text']
    return []


def extract_skill_ids_from_messages(messages: list[dict]) -> set[str]:
    """Extract skill IDs from <$skillId|label> and </skillId|label> mention tags."""
    ids: set[str] = set()
    for message in messages:
        for text in _get_text_parts(message):
            ids.update(m.group(1) or m.group(2) for m in SKILL_MENTION_RE.finditer(text))
    return ids


SKILL_MENTION_STRIP_RE = re.compile(r'<(?:\$[a-z0-9_-]+(?:\|([^>]*))?|/[a-z0-9_-]+\|([^>]*))>')


def strip_skill_mentions(messages: list[dict]) -> None:
    """Replace <$skillId|label> and </skillId|label> mention tags with the label in-place."""

    def label(match):
        return match.group(1) or match.group(2) or ''

    for message in messages:
        content = message.get('content')
        if isinstance(content, str) and SKILL_MENTION_STRIP_RE.search(content):
            message['content'] = SKILL_MENTION_STRIP_RE.sub(label, content).strip()
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text = part.get('text', '')
                    if SKILL_MENTION_STRIP_RE.search(text):
                        part['text'] = SKILL_MENTION_STRIP_RE.sub(label, text).strip()


async def connect_mcp_server(
    request,
    server_id: str,
    user,
    metadata: dict,
    extra_params: dict,
) -> tuple[MCPClient, list[dict]] | None:
    """Resolve an MCP server connection, authenticate, and return (client, tool_specs).

    Returns None if the server is not found or access is denied.
    """
    mcp_server_connection = None
    for server_connection in await Config.get('tool_server.connections', []):
        if server_connection.get('type', '') == 'mcp' and (server_connection.get('info') or {}).get('id') == server_id:
            mcp_server_connection = server_connection
            break

    if not mcp_server_connection:
        log.error(f'MCP server with id {server_id} not found')
        return None

    if not await has_connection_access(user, mcp_server_connection):
        log.warning(f'Access denied to MCP server {server_id} for user {user.id}')
        return None

    headers, _ = await build_tool_server_headers(
        mcp_server_connection,
        request,
        user,
        server_id=server_id,
        metadata=metadata,
        extra_params=extra_params,
    )

    client = MCPClient()
    await client.connect(
        url=mcp_server_connection.get('url', ''),
        headers=headers if headers else None,
    )

    function_name_filter_list = mcp_server_connection.get('config', {}).get('function_name_filter_list', '')
    if isinstance(function_name_filter_list, str):
        function_name_filter_list = function_name_filter_list.split(',')

    tool_specs = await client.list_tool_specs()
    if function_name_filter_list:
        tool_specs = [spec for spec in tool_specs if is_string_allowed(spec['name'], function_name_filter_list)]

    return client, tool_specs


