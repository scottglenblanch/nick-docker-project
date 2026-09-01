import json
import logging
import random
import sys

from fastapi import HTTPException
from open_webui.config import CODE_INTERPRETER_PYODIDE_PROMPT, DEFAULT_CODE_INTERPRETER_PROMPT, DEFAULT_VOICE_MODE_PROMPT_TEMPLATE
from open_webui.env import ENABLE_PLUGINS, GLOBAL_LOG_LEVEL
from open_webui.models.access_grants import AccessGrants
from open_webui.models.chats import Chats
from open_webui.models.config import Config
from open_webui.models.folders import Folders
from open_webui.models.notes import Notes
from open_webui.models.users import UserModel
from open_webui.routers.pipelines import process_pipeline_inlet_filter
from open_webui.socket.main import (
    get_event_call,
    get_event_emitter,
)
from open_webui.utils.access_control import has_permission
from open_webui.utils.access_control.files import get_owner_accessible_folder_files
from open_webui.utils.access_control.folders import has_folder_access
from open_webui.utils.chat_id import is_saved_chat_id
from open_webui.utils.context_compaction import compact_messages_for_request
from open_webui.utils.filter import get_filter_context, get_filter_functions, process_filter_functions
from open_webui.utils.memory import add_memory_context
from open_webui.utils.misc import add_or_update_system_message, add_or_update_user_message, get_content_from_message, get_last_user_message, get_system_message, set_last_user_message_content
from open_webui.utils.payload import apply_params_to_form_data, apply_system_prompt_to_body, resolve_system_prompt
from open_webui.utils.task import get_task_model_id
from open_webui.utils.tools import get_attached_knowledge, get_builtin_tools, get_terminal_tools, get_tools

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)



from open_webui.utils.middleware_helpers import MESSAGE_REPLAY_KEYS, add_file_context, apply_source_context_to_messages, chat_completion_files_handler, chat_completion_tools_handler, chat_image_generation_handler, chat_web_search_handler, connect_mcp_server, convert_url_images_to_base64, extract_skill_ids_from_messages, get_reasoning_format, load_messages_from_db, normalize_messages_for_model, process_messages_with_output, sanitize_tool_pairs, strip_reasoning_details, strip_skill_mentions
from open_webui.utils.middleware_runtime import get_system_oauth_token

async def process_chat_payload(request, form_data, user, metadata, model):
    # Ensure chat_id is always a string — external API clients may omit it.
    if not isinstance(metadata.get('chat_id'), str):
        metadata['chat_id'] = ''

    # Pipeline Inlet -> Filter Inlet -> Chat Memory -> Chat Web Search -> Chat Image Generation
    # -> Chat Code Interpreter (Form Data Update) -> (Default) Chat Tools Function Calling
    # -> Chat Files

    # Arena model resolution — pick the sub-model now so all downstream
    # processing (knowledge, capabilities, tools, params) uses its settings
    # instead of the empty arena wrapper.
    if model.get('owned_by') == 'arena':
        arena_model_ids = model.get('info', {}).get('meta', {}).get('model_ids')
        arena_filter_mode = model.get('info', {}).get('meta', {}).get('filter_mode')
        if arena_model_ids and arena_filter_mode == 'exclude':
            arena_model_ids = [
                available_model['id']
                for available_model in request.app.state.MODELS.values()
                if available_model.get('owned_by') != 'arena' and available_model['id'] not in arena_model_ids
            ]

        if isinstance(arena_model_ids, list) and arena_model_ids:
            selected_model_id = random.choice(arena_model_ids)
        else:
            arena_model_ids = [
                available_model['id']
                for available_model in request.app.state.MODELS.values()
                if available_model.get('owned_by') != 'arena'
            ]
            selected_model_id = random.choice(arena_model_ids)

        selected_model = request.app.state.MODELS.get(selected_model_id)
        if selected_model:
            model = selected_model
            form_data['model'] = selected_model_id
            metadata['selected_model_id'] = selected_model_id

    # Captured before apply_params_to_form_data pops 'params'; populates metadata['system_prompt'] below
    model_system_prompt = (form_data.get('params') or {}).get('system')

    form_data = apply_params_to_form_data(form_data, model)
    log.debug('form_data: %s', form_data)

    # Guided regeneration: extract before it reaches the LLM provider
    regeneration_prompt = form_data.pop('regeneration_prompt', None)

    # Load messages from DB when available — DB preserves structured 'output' items
    # which the frontend strips, causing tool calls to be merged into content.
    chat_id = metadata.get('chat_id')
    user_message_id = metadata.get('user_message_id')

    if is_saved_chat_id(chat_id) and user_message_id:
        db_messages = await load_messages_from_db(chat_id, user_message_id)
        if db_messages:
            # Continue: frontend sends assistant_message_id when continuing
            # an existing response. Load its content so the LLM sees prior output.
            assistant_message_id = metadata.get('assistant_message_id')
            if assistant_message_id:
                assistant_message = await Chats.get_message_by_id_and_message_id(chat_id, assistant_message_id)
                if assistant_message and (assistant_message.get('content') or assistant_message.get('output')):
                    db_messages.append({k: v for k, v in assistant_message.items() if k in MESSAGE_REPLAY_KEYS})

            system_message = get_system_message(form_data.get('messages', []))
            form_data['messages'] = [system_message, *db_messages] if system_message else db_messages

            # Inject image files into content as image_url parts (mirrors frontend logic)
            for message in form_data['messages']:
                image_files = [
                    f
                    for f in message.get('files', [])
                    if f.get('type') == 'image' or (f.get('content_type') or '').startswith('image/')
                ]
                if message.get('role') == 'user' and image_files:
                    text_content = message.get('content', '')
                    if isinstance(text_content, str):
                        message['content'] = [
                            {'type': 'text', 'text': text_content},
                            *[
                                {
                                    'type': 'image_url',
                                    'image_url': {'url': f['url']},
                                }
                                for f in image_files
                                if f.get('url')
                            ],
                        ]
                # Strip files field — it's been incorporated into content
                message.pop('files', None)

    if regeneration_prompt:
        form_data['messages'].append({'role': 'user', 'content': regeneration_prompt})

    if is_saved_chat_id(chat_id) and user_message_id:
        if getattr(request.state, 'direct', False) and hasattr(request.state, 'model'):
            compaction_models = {
                **dict(request.app.state.MODELS.items()),
                request.state.model['id']: request.state.model,
            }
        else:
            compaction_models = request.app.state.MODELS

        system_message = get_system_message(form_data.get('messages', []))
        system_prompt = get_content_from_message(system_message) if system_message else ''

        try:
            form_data['messages'], context_summary, _ = await compact_messages_for_request(
                request,
                user,
                form_data.get('messages', []),
                metadata,
                form_data.get('model'),
                compaction_models,
                system_prompt,
            )
            if context_summary:
                form_data['messages'] = add_or_update_system_message(
                    f'[CONVERSATION SUMMARY]\n{context_summary}',
                    form_data['messages'],
                    append=True,
                )
        except Exception:
            log.exception('Context compaction failed; continuing with full chat history')

    # Process messages with OR-aligned output items for clean LLM messages
    for message in form_data.get('messages', []):
        output = message.get('output')
        # reasoning_details can be model/provider-bound, so only replay them
        # for output produced by the same model.
        if message.get('role') == 'assistant' and message.get('model') != model['id'] and isinstance(output, list):
            message['output'] = strip_reasoning_details(output)

    form_data['messages'] = process_messages_with_output(
        form_data.get('messages', []),
        reasoning_format=get_reasoning_format(model),
    )
    form_data['messages'] = sanitize_tool_pairs(form_data['messages'])

    system_message = get_system_message(form_data.get('messages', []))
    if system_message:  # Chat Controls/User Settings
        try:
            form_data = await apply_system_prompt_to_body(
                system_message.get('content'), form_data, metadata, user, replace=True
            )  # Required to handle system prompt variables
        except Exception:
            pass

    form_data = await convert_url_images_to_base64(form_data, user=user)

    event_emitter = await get_event_emitter(metadata)
    event_caller = await get_event_call(metadata)

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
    # Initialize events to store additional event to be sent to the client
    # Initialize contexts and citation
    if getattr(request.state, 'direct', False) and hasattr(request.state, 'model'):
        models = {
            request.state.model['id']: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    task_model_id = get_task_model_id(
        form_data['model'],
        await Config.get('task.model.default'),
        await Config.get('task.model.external'),
        models,
    )

    events = []
    sources = []

    # Folder "Project" handling
    # Check if the request has chat_id and is inside of a folder
    # Uses lightweight column query — only fetches folder_id, not the full chat JSON blob
    chat_id = metadata.get('chat_id', None)
    folder_id = None
    if user and is_saved_chat_id(chat_id):
        folder_id = await Chats.get_chat_folder_id(chat_id, user.id)

    # Fallback: use folder_id from metadata (temporary chats have no DB record)
    if not folder_id:
        folder_id = metadata.get('folder_id', None)

    if folder_id and user:
        folder = await Folders.get_folder_by_id(folder_id)
        if folder and user.role != 'admin' and not await has_folder_access(user.id, folder, 'read', db=None):
            folder = None

        if folder and folder.data:
            if 'system_prompt' in folder.data:
                form_data = await apply_system_prompt_to_body(folder.data['system_prompt'], form_data, metadata, user)
            if 'files' in folder.data:
                if metadata.get('params', {}).get('function_calling') == 'legacy':
                    form_data['files'] = [
                        {'type': 'folder', 'id': folder.id},
                        *form_data.get('files', []),
                    ]
                else:
                    # Native FC: skip RAG injection, builtin tools
                    # will read folder knowledge from metadata.
                    metadata['folder_knowledge'] = await get_owner_accessible_folder_files(folder)

    # Model "Knowledge" handling
    user_message = get_last_user_message(form_data['messages'])
    model_knowledge = model.get('info', {}).get('meta', {}).get('knowledge', False)

    if model_knowledge and metadata.get('params', {}).get('function_calling') == 'legacy':
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'knowledge_search',
                    'query': user_message,
                    'done': False,
                },
            }
        )

        knowledge_files = []
        for item in model_knowledge:
            if item.get('collection_name'):
                knowledge_files.append(
                    {
                        'id': item.get('collection_name'),
                        'name': item.get('name'),
                        'legacy': True,
                    }
                )
            elif item.get('collection_names'):
                knowledge_files.append(
                    {
                        'name': item.get('name'),
                        'type': 'collection',
                        'collection_names': item.get('collection_names'),
                        'legacy': True,
                    }
                )
            else:
                knowledge_files.append(item)

        files = form_data.get('files', [])
        files.extend(knowledge_files)
        form_data['files'] = files

    variables = form_data.pop('variables', None)
    payload_tools = form_data.get('tools', None)  # snapshot before filters

    # Process the form_data through the pipeline
    try:
        form_data = await process_pipeline_inlet_filter(request, form_data, user, models)
    except Exception as e:
        raise e

    filter_functions = []
    filter_context = get_filter_context(request) if ENABLE_PLUGINS else None
    if ENABLE_PLUGINS:
        try:
            filter_functions = await get_filter_functions(request, model, metadata.get('filter_ids', []))

            form_data, flags = await process_filter_functions(
                request=request,
                filter_context=filter_context,
                filter_functions=filter_functions,
                filter_type='inlet',
                form_data=form_data,
                extra_params=extra_params,
            )
        except Exception as e:
            raise Exception(f'{e}')

    features = form_data.pop('features', None) or {}
    # Some clients/chats do not send a features payload. Default to global
    # web-search enablement so retrieval still works in those cases.
    if 'web_search' not in features and await Config.get('web.search.enable'):
        features['web_search'] = True

    extra_params['__features__'] = features
    if features:
        if 'voice' in features and features['voice']:
            if await Config.get('task.voice.prompt.enable'):
                template = await Config.get('task.voice.prompt_template')
                if not template:
                    template = DEFAULT_VOICE_MODE_PROMPT_TEMPLATE

                form_data['messages'] = add_or_update_system_message(
                    template,
                    form_data['messages'],
                )

        if 'memory' in features and features['memory'] and await Config.get('memories.system_context.enable'):
            # features is client-supplied; re-check the permission the native FC path enforces.
            if getattr(user, 'role', None) == 'admin' or await has_permission(
                getattr(user, 'id', ''),
                'features.memories',
                await Config.get('user.permissions'),
            ):
                form_data = await add_memory_context(request, form_data, user, model)

        if 'web_search' in features and features['web_search'] and await Config.get('web.search.enable'):
            # features is client-supplied; re-check the permission the native FC path enforces.
            if getattr(user, 'role', None) == 'admin' or await has_permission(
                getattr(user, 'id', ''),
                'features.web_search',
                await Config.get('user.permissions'),
            ):
                form_data = await chat_web_search_handler(request, form_data, extra_params, user)

        if 'image_generation' in features and features['image_generation']:
            # features is client-supplied; re-check the permission the direct /images routes enforce.
            if getattr(user, 'role', None) == 'admin' or await has_permission(
                getattr(user, 'id', ''),
                'features.image_generation',
                await Config.get('user.permissions'),
            ):
                # Skip forced image generation when native FC is enabled - model can use generate_image tool
                if metadata.get('params', {}).get('function_calling') == 'legacy':
                    form_data = await chat_image_generation_handler(request, form_data, extra_params, user)

        if 'code_interpreter' in features and features['code_interpreter']:
            engine = await Config.get('code_interpreter.engine', 'pyodide')

            # Skip XML-tag prompt injection when native FC is enabled —
            # execute_code will be injected as a builtin tool instead
            if metadata.get('params', {}).get('function_calling') == 'legacy':
                ci_prompt_template = await Config.get('code_interpreter.prompt_template')
                prompt = ci_prompt_template if ci_prompt_template != '' else DEFAULT_CODE_INTERPRETER_PROMPT

                # Append filesystem awareness only for pyodide engine
                if engine != 'jupyter':
                    prompt += CODE_INTERPRETER_PYODIDE_PROMPT

                form_data['messages'] = add_or_update_user_message(
                    prompt,
                    form_data['messages'],
                )
            else:
                # Native FC: tool docstring can't be dynamic, so inject
                # filesystem context into the system message for pyodide
                # engine.  Appending to the system prompt (instead of the
                # user message) keeps it in the stable cached prefix so
                # providers with prefix caching don't re-bill the full
                # conversation on every turn.
                if engine != 'jupyter':
                    form_data['messages'] = add_or_update_system_message(
                        CODE_INTERPRETER_PYODIDE_PROMPT,
                        form_data['messages'],
                        append=True,
                    )

    tool_ids = form_data.pop('tool_ids', None)
    terminal_id = form_data.pop('terminal_id', None)
    files = form_data.pop('files', None)
    form_data.pop('folder_id', None)

    # If the original caller provided tools, use them as-is (skip resolution).
    # Otherwise, save any tools that filter inlets added for merging later.
    inlet_filter_tools = None if payload_tools is not None else form_data.get('tools', None)

    # Mentioned skills get full content; selected/default skills can be loaded through view_skill.
    mentioned_skill_ids = extract_skill_ids_from_messages(form_data.get('messages', []))
    skill_ids = sorted(
        set(form_data.pop('skill_ids', None) or [])
        | set(model.get('info', {}).get('meta', {}).get('skillIds', []))
        | mentioned_skill_ids
    )
    available_skills = []
    view_skill_ids = []
    chat = None
    if is_saved_chat_id(metadata.get('chat_id')):
        chat = await Chats.get_chat_by_id(metadata['chat_id'])

    is_note_chat = bool(chat and (chat.meta or {}).get('internal') is True and (chat.meta or {}).get('type') == 'note')

    if is_note_chat:
        note_id = (chat.meta or {}).get('note_id')
        note = await Notes.get_note_by_id(note_id) if note_id else None
        if note and (
            user.role == 'admin'
            or note.user_id == user.id
            or await AccessGrants.has_access(
                user_id=user.id,
                resource_type='note',
                resource_id=note.id,
                permission='read',
            )
        ):
            note_files = [
                file
                for file in ((note.data or {}).get('files') or [])
                if isinstance(file, dict)
                and file.get('type') != 'image'
                and not (file.get('content_type') or '').startswith('image/')
            ]
            if note_files:
                files = [*(files or []), *note_files]

    use_builtin_tools = is_note_chat or (
        bool(metadata.get('session_id'))
        and metadata.get('params', {}).get('function_calling') != 'legacy'
        and (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get('builtin_tools', True)
    )

    if skill_ids:
        from open_webui.models.skills import Skills as SkillsModel

        accessible_skills = {s.id: s for s in await SkillsModel.get_skills(user_id=user.id, ids=skill_ids)}
        for sid in skill_ids:
            s = accessible_skills.get(sid)
            if s and s.is_active:
                available_skills.append(s)

        skill_manifest = ''
        for skill in available_skills:
            if skill.id in mentioned_skill_ids or not use_builtin_tools:
                form_data['messages'] = add_or_update_system_message(
                    f'<skill name="{skill.name}">\n{skill.content}\n</skill>',
                    form_data['messages'],
                    append=True,
                )
            else:
                view_skill_ids.append(skill.id)
                skill_manifest += (
                    f'<skill>\n<id>{skill.id}</id>\n<name>{skill.name}</name>\n'
                    f'<description>{skill.description or ""}</description>\n</skill>\n'
                )

        if skill_manifest:
            form_data['messages'] = add_or_update_system_message(
                f'<available_skills>\n{skill_manifest}</available_skills>',
                form_data['messages'],
                append=True,
            )

    # Strip <$skillId|label> mention tags so the model doesn't see raw markup.
    strip_skill_mentions(form_data.get('messages', []))

    prompt = get_last_user_message(form_data['messages'])

    # Guard against empty user message after skill mention stripping.
    # When a user selects a skill ($skill-name) without typing additional text,
    # the stripped result is an empty string which causes 400 errors on providers
    # that reject empty content blocks (e.g. AWS Bedrock ConverseStream).
    if not prompt or not prompt.strip():
        fallback = ', '.join(s.name for s in available_skills)
        if fallback:
            set_last_user_message_content(fallback, form_data['messages'])
            prompt = fallback
    # TODO: re-enable URL extraction from prompt
    # urls = []
    # if prompt and len(prompt or "") < 500 and (not files or len(files) == 0):
    #     urls = extract_urls(prompt)

    if files:
        # files = [*files, *[{"type": "url", "url": url, "name": url} for url in urls]]
        # Remove duplicate files based on their content
        files = list({json.dumps(f, sort_keys=True): f for f in files}.values())

    metadata.update(
        {
            'model_id': form_data.get('model'),
            'tool_ids': tool_ids,
            'skill_ids': skill_ids,
            'terminal_id': terminal_id,
            'files': files,
            'features': features,
        }
    )
    form_data['metadata'] = metadata

    # When the caller provides an explicit `tools` key in the request body,
    # skip all server-side tool resolution and pass the caller's tools through
    # unchanged.  Sending `tools: []` explicitly opts out of builtin injection.
    if payload_tools is None:
        # Server side tools
        tool_ids = metadata.get('tool_ids', None)
        # Client side tools
        direct_tool_servers = metadata.get('tool_servers', None)

        log.debug('tool_ids=%r', tool_ids)
        log.debug('direct_tool_servers=%r', direct_tool_servers)

        tools_dict = {}

        mcp_clients = {}
        mcp_tools_dict = {}

        if tool_ids:
            db_tool_ids = []
            for tool_id in tool_ids:
                if tool_id.startswith('server:mcp:'):
                    try:
                        server_id = tool_id[len('server:mcp:') :]

                        result = await connect_mcp_server(
                            request,
                            server_id,
                            user,
                            metadata,
                            extra_params,
                        )
                        if result is None:
                            continue

                        client, tool_specs = result
                        mcp_clients[server_id] = client

                        for tool_spec in tool_specs:

                            async def make_tool_function(client, function_name):
                                async def tool_function(**kwargs):
                                    return await client.call_tool(
                                        function_name,
                                        function_args=kwargs,
                                    )

                                return tool_function

                            tool_function = await make_tool_function(client, tool_spec['name'])

                            mcp_tools_dict[f'{server_id}_{tool_spec["name"]}'] = {
                                'spec': {
                                    **tool_spec,
                                    'name': f'{server_id}_{tool_spec["name"]}',
                                },
                                'callable': tool_function,
                                'type': 'mcp',
                                'client': client,
                                'direct': False,
                            }
                    except Exception as e:
                        log.debug(e)
                        if event_emitter:
                            await event_emitter(
                                {
                                    'type': 'chat:message:error',
                                    'data': {'error': {'content': f"Failed to connect to MCP server '{server_id}'"}},
                                }
                            )
                        continue
                elif ENABLE_PLUGINS:
                    db_tool_ids.append(tool_id)

            if db_tool_ids:
                tools_dict = await get_tools(
                    request,
                    db_tool_ids,
                    user,
                    {
                        **extra_params,
                        '__model__': models[task_model_id],
                        '__messages__': form_data['messages'],
                        '__files__': metadata.get('files', []),
                    },
                )

            if mcp_tools_dict:
                tools_dict = {**tools_dict, **mcp_tools_dict}

        # Resolve terminal tools if terminal_id is set (outside tool_ids check
        # so system terminals work even when no other tools are selected)
        terminal_capability = (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get('terminal', True)
        if terminal_id and terminal_capability:
            try:
                terminal_result = await get_terminal_tools(
                    request,
                    terminal_id,
                    user,
                    extra_params,
                )
                if isinstance(terminal_result, tuple):
                    terminal_tools, system_prompt = terminal_result
                else:
                    terminal_tools = terminal_result
                    system_prompt = None
                if terminal_tools:
                    tools_dict = {**tools_dict, **terminal_tools}
                if system_prompt:
                    form_data['messages'] = add_or_update_system_message(
                        system_prompt,
                        form_data['messages'],
                        append=True,
                    )
            except Exception as e:
                log.exception(e)
                raise HTTPException(status_code=503, detail=f'Terminal unavailable: {e}') from e

        if direct_tool_servers:
            for tool_server in direct_tool_servers:
                system_prompt = tool_server.pop('system_prompt', None)
                if system_prompt:
                    form_data['messages'] = add_or_update_system_message(
                        system_prompt,
                        form_data['messages'],
                        append=True,
                    )

                tool_specs = tool_server.pop('specs', [])

                for tool in tool_specs:
                    tools_dict[tool['name']] = {
                        'spec': tool,
                        'direct': True,
                        'server': tool_server,
                    }

        if mcp_clients:
            metadata['mcp_clients'] = mcp_clients

        # Inject builtin tools for native function calling based on enabled features and model capability.
        # Only inject when the request originates from the UI (identified by session_id).
        # API callers don't expect hidden tools; they can explicitly request tools via tool_ids.
        if use_builtin_tools:
            # Add file context to user messages
            chat_id = metadata.get('chat_id')
            form_data['messages'] = await add_file_context(form_data.get('messages', []), chat_id, user)

            if (model.get('info', {}).get('meta', {}).get('builtinTools') or {}).get('knowledge', True):
                from html import escape

                knowledge_tags = []
                for item in get_attached_knowledge(model, metadata):
                    if not item.get('id') or not item.get('type'):
                        continue
                    attrs = f'type="{escape(str(item["type"]), quote=True)}" id="{escape(str(item["id"]), quote=True)}"'
                    if item.get('name'):
                        attrs += f' name="{escape(str(item["name"]), quote=True)}"'
                    if item.get('source'):
                        attrs += f' source="{escape(str(item["source"]), quote=True)}"'
                    knowledge_tags.append(f'<knowledge {attrs}/>')

                if knowledge_tags:
                    form_data['messages'] = add_or_update_system_message(
                        '<attached_knowledge>\n' + '\n'.join(knowledge_tags) + '\n</attached_knowledge>',
                        form_data['messages'],
                        append=True,
                    )

            builtin_tools = await get_builtin_tools(
                request,
                {
                    **extra_params,
                    '__event_emitter__': event_emitter,
                    '__skill_ids__': view_skill_ids,
                },
                features,
                model,
                is_note_chat=is_note_chat,
            )
            for name, tool_dict in builtin_tools.items():
                if name not in tools_dict:
                    tools_dict[name] = tool_dict

        if tools_dict:
            # Always store resolved tools in metadata so downstream consumers
            # (e.g. pipe functions) can access all tools including MCP and builtins.
            metadata['tools'] = tools_dict

            if metadata.get('params', {}).get('function_calling') != 'legacy':
                # If the function calling is native, then call the tools function calling handler
                form_data['tools'] = [
                    {'type': 'function', 'function': tool.get('spec', {})} for tool in tools_dict.values()
                ]
                if inlet_filter_tools:
                    form_data['tools'].extend(inlet_filter_tools)
            else:
                # If the function calling is not native, then call the tools function calling handler
                try:
                    form_data, flags = await chat_completion_tools_handler(
                        request, form_data, extra_params, user, models, tools_dict
                    )
                    sources.extend(flags.get('sources', []))
                except Exception as e:
                    log.exception(e)

    # Check if file context extraction is enabled for this model (default True)
    file_context_enabled = (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get('file_context', True)

    if file_context_enabled:
        try:
            form_data, flags = await chat_completion_files_handler(request, form_data, extra_params, user)
            sources.extend(flags.get('sources', []))
        except Exception as e:
            log.exception(e)

    # Save the pre-RAG message state so the native tool call loop can
    # restore to the true original (before file-source injection) rather
    # than a snapshot that already has the RAG template baked in.
    system_message = get_system_message(form_data['messages'])
    system_content = get_content_from_message(system_message) if system_message else ''
    resolved_model_system_prompt = await resolve_system_prompt(
        model_system_prompt,
        metadata,
        user,
    )
    if resolved_model_system_prompt:
        system_content = (
            f'{resolved_model_system_prompt}\n{system_content}' if system_content else resolved_model_system_prompt
        )
    metadata['system_prompt'] = system_content or None
    metadata['user_prompt'] = get_last_user_message(form_data['messages'])
    metadata['sources'] = sources[:] if sources else []

    # If context is not empty, insert it into the messages
    if sources and prompt:
        form_data['messages'] = await apply_source_context_to_messages(request, form_data['messages'], sources, prompt)

    # If there are citations, add them to the data_items
    sources = [
        source
        for source in sources
        if source.get('source', {}).get('name', '') or source.get('source', {}).get('id', '')
    ]

    if len(sources) > 0:
        events.append({'sources': sources})

    if model_knowledge:
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'knowledge_search',
                    'query': user_message,
                    'done': True,
                    'hidden': True,
                },
            }
        )

    if ENABLE_PLUGINS:
        try:
            form_data, _ = await process_filter_functions(
                request=request,
                filter_context=filter_context,
                filter_functions=filter_functions,
                filter_type='request',
                form_data=form_data,
                extra_params=extra_params,
            )
        except Exception as e:
            raise Exception(f'{e}')

    form_data = normalize_messages_for_model(form_data)

    return form_data, metadata, events


