# ruff: noqa: F821, I001
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import pathlib
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from prompt_toolkit.history import InMemoryHistory
from rich.table import Table
from . import capabilities as capability_packs
from ui import json_output
from . import runtime_f_integration as f_integration
from ui.chat_ui import ChatTranscript
from platform_support import (
    detect_platform,
    is_windows_mount_path,
)

from config import (
    AGENT_PERMISSION_MODE,
    APP_CONFIG,
    APPROVED_EXTRA_FOLDERS,
    SANDBOX_NETWORK_DEFAULT,
    CODE_OUTPUT_ROOT,
    CONFIG_FILE_PATH,
    CONTEXT_COMPACTION_THRESHOLD,
    CONTEXT_KEEP_RECENT_MESSAGES,
    CONTEXT_SUMMARY_CHARS,
    CONVERSATION_RETENTION_DAYS,
    DEFAULT_DB_PATH,
    ENABLE_WEB_RESEARCH,
    EPISODE_RETENTION_DAYS,
    MAX_CONTEXT_TOKENS,
    MAX_HISTORY_MESSAGES,
    MAX_OUTPUT_TOKENS,
    MAX_TASK_SECONDS,
    MAX_TOOL_CALLS,
    MAX_TOOL_ITERATIONS,
    MAX_TOOL_OUTPUT_CHARS,
    MCP_AUTO_APPROVE_TOOLS,
    MCP_SERVERS,
    ONLINE_PERMISSION_OVERRIDE,
    OVERRIDE_TOOL_LIST,
    PERMISSION_ALLOW_RULES,
    PERMISSION_DENY_RULES,
    PREFERRED_MODEL,
    PROJECT_INSTRUCTIONS_FILE,
    RAG_AUTO_REFRESH_SECONDS,
    RAG_DOCS_DEFAULT,
    RAG_DOCUMENTS_ONLY,
    RAG_INDEX_PATH,
    REGISTERED_SKILL_RETENTION_DAYS,
    SKILLS_FOLDER_DEFAULT,
    GROUPED_HELP,
    SHOW_TURN_DIVIDER,
    SHOW_TURN_STATS,
    SHOW_WELCOME_SCREEN,
    STATUS_BAR,
    SYSTEM_PROMPT,
    TASK_STALE_AFTER_DAYS,
    THINKING_EFFORT_DEFAULT,
    THINKING_TOGGLE_DEFAULT,
    WEB_RESEARCH_CONSENT,
    WORKSPACE_ROOT_DEFAULT,
    load_or_create_config,
    merge_project_config,
)
from database import PersistentMemory
from core.events import Event, ToolFinished, ToolStarted, get_event_bus
from core.hooks import (
    collect_post_tool_hook_output,
    evaluate_pre_tool_hooks,
    load_hooks,
    safe_run_session_start_hooks,
)
from rag.citations import format_retrieved_citations
from utils.run_logging import RUN_LOGGER, log_event, logging_context
from sandbox import SandboxManager
from ui import SessionUI
from .compaction import (
    compact_history,
    compact_loop_messages,
    message_text,
    messages_tokens,
    needs_history_compaction,
    summary_system_message,
)
from .context_engine import (
    ContextEngine,
    is_enabled as context_engine_is_enabled,
)
from .context_telemetry import build_runtime_context_report
from .console import console
from .commands import mission_runtime as _mission_runtime
from .commands.mission_runtime import mission_slash_command as _mission_slash_command
from .steering_session import begin_steering_turn
from .lazy_imports import install_lazy_globals
from .memory_context import (
    _build_learned_context,
    _build_relevant_episode_context,
    _build_startup_memory_context,
    _initialize_rag_for_docs,
    _is_coding_guidance_request,
    _prune_memory_retention,
    _retrieve_rag_documents,
)
from .permissions import (
    authorize_network_research as _authorize_network_research,
)
from .permissions import (
    coding_allows_command_network as _coding_allows_command_network,
)
from .permissions import (
    coding_permission_mode as _coding_permission_mode,
)
from .permissions import (
    coding_uses_isolated_copy as _coding_uses_isolated_copy,
)
from .permissions import (
    evaluate_permission_rules as _evaluate_permission_rules,
)
from .permissions import (
    prompt_tool_approval as _prompt_tool_approval,
)
from .permissions import (
    select_permission_mode as _select_permission_mode_impl,
)
from .permissions import (
    tool_needs_permission as _tool_needs_permission,
)
from .project_paths import discover_project_resources
from .prompt_prefix import build_system_messages
from .prompts import (
    build_bounded_user_input as _build_bounded_user_input,
)
from .prompts import (
    calculate_prompt_budgets,
    estimate_context_window,
    trim_history_to_context_budget,
)
from .prompts import (
    current_datetime_context as _current_datetime_context,
)
from .provider_fallback import _bind_tools, _invoke_initial_model_with_fallback
from .provider_resume import (
    always_ask_provider,
    auto_select_single_provider,
    match_runtime_by_address,
    persist_provider_choice,
    provider_record_for_resume,
)
from .providers import (
    has_saved_online_config as _has_saved_online_config,
)
from .providers import (
    online_request_failure as _online_request_failure,
)
from .session_resume import resolve_session_id, take_over_session_task_lease
from .session_state import SESSION_INFO as _SESSION_INFO
from .session_state import SESSION_STATS as _SESSION_STATS
from .session_summary import (
    _acknowledge_interrupt,
    _history_with_summary,
    _print_context_state,
    _print_session_statistics,
    _summarize_and_save_session,
)
from .session_types import ProviderPreference, ProviderSelection, SessionSetup
from .streaming import (
    _invoke_with_status,
    _visible_chunk_text,
)
from .task_commands import (
    _approved_coding_task_scope,
    _planning_search_citations,
)
from .task_lease import (
    TASK_LEASE_SECONDS,
    _maintain_task_lease,
)
from .undo import undo_last_turn

_ALWAYS_ALLOWED_TOOLS: set[str] = set()
_ACTIVE_MEMORY: PersistentMemory | None = None
_ACTIVE_RAG_TASK: Any | None = None
_CONTEXT_REPORT_STATE: dict[str, Any] = {"value": {}}


def initialize_knowledge_base(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper so importing this module does not pull in Chroma."""
    from rag import initialize_knowledge_base as _initialize

    return _initialize(*args, **kwargs)


def knowledge_base_signature(*args: Any, **kwargs: Any) -> Any:
    """Lazy wrapper so importing this module does not pull in Chroma."""
    from rag import knowledge_base_signature as _signature

    return _signature(*args, **kwargs)


_LAST_HEADLESS_RESPONSE: str = ""
_TASK_LEASE_HEARTBEAT: asyncio.Task | None = None
OS_ISOLATED_TOOL_NAMES = frozenset(
    {
        "run_shell_command",
        "run_project_unit_tests",
        "checkpoint_code_task",
        "finalize_code_task",
    }
)


def _local_location_refusal(name: str, args: Any, local_session: bool) -> str | None:
    """Explain why a location-revealing call must not run, or None if allowed."""
    if local_session:
        return None
    if name in LOCAL_ONLY_NETWORK_TOOL_NAMES:
        return "Error: Current-location lookup requires the Local provider."
    if name == "get_weather" and not (
        isinstance(args, dict) and str(args.get("location") or "").strip()
    ):
        return (
            "Error: Weather for the current location requires the Local "
            "provider; ask the user for a place name and pass it as 'location'."
        )
    return None


def _device_tools_allowed(provider_type: str) -> bool:
    """Local-only (location/media) tools: Local provider, or the explicit override."""
    return provider_type == "local" or ONLINE_PERMISSION_OVERRIDE


def _tool_unavailable_reason(
    name: str,
    *,
    provider_type: str,
    capabilities: dict,
    isolation_warning: str | None,
) -> str | None:
    if name in OS_ISOLATED_TOOL_NAMES and isolation_warning:
        return isolation_warning
    if name in NETWORK_TOOL_NAMES and (
        not ENABLE_WEB_RESEARCH or WEB_RESEARCH_CONSENT == "never"
    ):
        return "Online research is disabled by configuration."
    if name in LOCAL_ONLY_NETWORK_TOOL_NAMES and not _device_tools_allowed(
        provider_type
    ):
        return "Current-location lookup requires the Local provider."
    if name in LOCAL_MEDIA_TOOL_NAMES:
        if not _device_tools_allowed(provider_type):
            return "Local media tools require the Local provider."
        if (
            name
            in {
                "capture_webcam_image",
                "load_workspace_image",
                "load_workspace_video",
            }
            and capabilities.get("vision") is not True
        ):
            return "The selected local model does not declare vision support."
    return None


def _tools_for_model(
    *,
    provider_type: str,
    capabilities: dict,
    isolation_warning: str | None,
) -> list:
    return [
        tool_object
        for name, tool_object in AVAILABLE_TOOLS.items()
        if _tool_unavailable_reason(
            name,
            provider_type=provider_type,
            capabilities=capabilities,
            isolation_warning=isolation_warning,
        )
        is None
        or name in OVERRIDE_TOOL_LIST
    ]


def _print_tool_catalog(
    entries,
    *,
    unavailable_names=frozenset(),
    override_tool_list=OVERRIDE_TOOL_LIST,
):
    unavailable_names = set(unavailable_names)
    override_tool_list = set(override_tool_list)
    overridden_unavailable = unavailable_names & override_tool_list
    available_count = len(entries) - len(
        unavailable_names & {entry["name"] for entry in entries}
    )
    overridden_count = len(
        overridden_unavailable & {entry["name"] for entry in entries}
    )
    table = Table(
        title=(
            f"Tool Catalog ({available_count} available, "
            f"{len(entries) - available_count} unavailable, "
            f"{overridden_count} overridden)"
        ),
        box=None,
        expand=True,
        padding=(0, 1),
        show_lines=False,
    )
    table.add_column(
        "Tool / source",
        style="bold cyan",
        no_wrap=True,
        overflow="ellipsis",
    )
    table.add_column(
        "Description",
        ratio=3,
        no_wrap=True,
        overflow="ellipsis",
    )
    table.add_column(
        "Permission / effects",
        ratio=2,
        style="yellow",
        no_wrap=True,
        overflow="ellipsis",
    )
    for entry in entries:
        description = " ".join(entry["description"].split()) or "No description"
        access = f"{entry['permission']}; {entry['effects']}"
        unavailable = entry["name"] in unavailable_names
        overridden = entry["name"] in overridden_unavailable
        tool_label = f"{entry['name']} ({entry['origin']})"
        if overridden:
            tool_label += " (override)"
        table.add_row(
            tool_label,
            description,
            access,
            style=("red strike" if overridden else "strike") if unavailable else None,
        )
    console.print(table)


async def authorize_network_research(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    permission_mode: str = "manual",
    session_consent_state: dict[str, bool] | None = None,
) -> tuple[bool, str]:
    return await _authorize_network_research(
        tool_name,
        tool_args,
        permission_mode=permission_mode,
        enabled=ENABLE_WEB_RESEARCH,
        consent_policy=WEB_RESEARCH_CONSENT,
        internet_check=has_internet_connection,
        console=console,
        is_interactive=lambda: get_tool_context().interactive and sys.stdin.isatty(),
        session_consent_state=session_consent_state,
    )


@dataclass
class _PreflightState:
    tool_tasks: list
    blocked_tool_messages: list
    declined_media_calls: list
    planning_tool_seen: bool
    authorized_planning_searches: dict
    task_action_seen: bool
    tool_call_count: int
    capability_approved: bool = False


async def _preflight_tool_calls(
    *,
    response,
    state: _PreflightState,
    network_consent_state,
    provider_type,
    capabilities,
    planning_only,
    turn_permission_mode,
    tool_call_limit,
    start_time,
    deadline,
    memory,
    planning_research_citations,
    console,
) -> None:
    schema_read_in_batch = any(
        call.get("name") == "read_table_schema" for call in response.tool_calls
    )
    step_deadline = deadline
    for tool_call in response.tool_calls:
        call_name = tool_call.get("name")
        if get_tool_context().task_id and state.task_action_seen:
            state.blocked_tool_messages.append(
                ToolMessage(
                    content=(
                        "Error: Only one action for the active task "
                        "may run per model step. Wait for this "
                        "action's result before proposing another."
                    ),
                    tool_call_id=tool_call.get("id"),
                )
            )
            continue
        if schema_read_in_batch and call_name in {
            "create_skill",
        }:
            state.blocked_tool_messages.append(
                ToolMessage(
                    content=(
                        "Error: Schema reads and database writes "
                        "must be separate model steps. Read the "
                        "destination schema first, then prepare and "
                        "submit the write in a later step."
                    ),
                    tool_call_id=tool_call.get("id"),
                )
            )
            continue
        effective_deadline = step_deadline
        if time.monotonic() >= effective_deadline:
            state.blocked_tool_messages.append(
                ToolMessage(
                    content="Error: Task time budget exhausted; this tool was not run.",
                    tool_call_id=tool_call.get("id"),
                )
            )
            continue
        if (
            call_name == "update_todo_step"
            and str(tool_call.get("args", {}).get("status", "")).lower()
            == "in_progress"
        ):
            step_deadline = time.monotonic() + MAX_TASK_SECONDS
        if tool_call_limit is not None and state.tool_call_count >= tool_call_limit:
            state.blocked_tool_messages.append(
                ToolMessage(
                    content=(
                        "Error: Maximum tool-call budget exhausted; "
                        "this tool was not run."
                    ),
                    tool_call_id=tool_call.get("id"),
                )
            )
            continue
        state.tool_call_count += 1
        _SESSION_STATS.record_tool_call(str(tool_call.get("name") or "unknown"))
        network_authorized = False
        media_capture_authorized = False
        # Planning mode no longer limits the model to one planner call per
        # model round; multiple read-only/planner operations may run together.
        if call_name in {"create_task_plan", "revise_task_plan"} and not planning_only:
            state.blocked_tool_messages.append(
                ToolMessage(
                    content=(
                        "Error: Task plans may only be created or "
                        "revised in the user's explicit /plan mode."
                    ),
                    tool_call_id=tool_call.get("id"),
                )
            )
            continue
        if planning_only and call_name not in {
            "read_table_schema",
            "create_task_plan",
            "inspect_task_plan",
            "revise_task_plan",
            "web_search",
            "fetch_webpage",
            "read_local_file",
            "list_directory",
            "search_workspace_files",
            "search_workspace_text",
            "search_workspace_symbols",
            "inspect_workspace_git",
            "ask_user",
        }:
            state.blocked_tool_messages.append(
                ToolMessage(
                    content=(
                        "Error: Planning mode permits planner tools "
                        "and consented web searches only; no other "
                        "research or execution tool ran."
                    ),
                    tool_call_id=tool_call.get("id"),
                )
            )
            continue
        if call_name in LOCAL_MEDIA_TOOL_NAMES:
            if not _device_tools_allowed(provider_type):
                state.blocked_tool_messages.append(
                    ToolMessage(
                        content=(
                            "Error: Media device tools are local-only and "
                            "cannot be used with an Online provider."
                        ),
                        tool_call_id=tool_call.get("id"),
                    )
                )
                continue
            if (
                call_name
                in {
                    "capture_webcam_image",
                    "load_workspace_image",
                    "load_workspace_video",
                }
                and capabilities["vision"] is not True
            ):
                state.blocked_tool_messages.append(
                    ToolMessage(
                        content=(
                            "Error: The selected local model does not "
                            "declare vision support."
                        ),
                        tool_call_id=tool_call.get("id"),
                    )
                )
                continue
            if call_name != "list_microphone_devices":
                media_capture_authorized = (
                    turn_permission_mode == "full"
                    or approve_local_capture(
                        call_name,
                        tool_call.get("args", {}),
                    )
                )
                if not media_capture_authorized:
                    state.declined_media_calls.append(call_name)
                    state.blocked_tool_messages.append(
                        ToolMessage(
                            content=(
                                "Error: Media capture was declined; "
                                "the device was not accessed."
                            ),
                            tool_call_id=tool_call.get("id"),
                        )
                    )
                    continue
        location_refusal = _local_location_refusal(
            call_name,
            tool_call.get("args", {}),
            _device_tools_allowed(provider_type),
        )
        if location_refusal:
            state.blocked_tool_messages.append(
                ToolMessage(
                    content=location_refusal,
                    tool_call_id=tool_call.get("id"),
                )
            )
            continue
        if call_name == "enable_capability" and not f_integration.gate_capability(
            tool_call, turn_permission_mode, console, state
        ):
            continue
        if call_name in NETWORK_TOOL_NAMES:
            tool_args = tool_call.get("args", {})
            planning_search = planning_only and call_name == "web_search"
            (
                network_authorized,
                refusal,
            ) = await authorize_network_research(
                call_name,
                tool_args,
                permission_mode=("manual" if planning_search else turn_permission_mode),
                session_consent_state=network_consent_state,
            )
            if not network_authorized:
                state.blocked_tool_messages.append(
                    ToolMessage(
                        content=refusal,
                        tool_call_id=tool_call.get("id"),
                    )
                )
                continue
            if planning_search:
                state.authorized_planning_searches[str(tool_call.get("id") or "")] = (
                    str(tool_args.get("query") or "")
                )
        execution_options = {
            "network_authorized": network_authorized,
            "local_media_allowed": _device_tools_allowed(provider_type),
            "media_capture_authorized": media_capture_authorized,
            "vision_supported": capabilities["vision"] is True,
            "permission_mode": turn_permission_mode,
        }
        execution_tool_call = tool_call
        if planning_only and call_name in {
            "create_task_plan",
            "revise_task_plan",
        }:
            prior_references = []
            if call_name == "revise_task_plan":
                active_plan = (
                    memory.get_task(get_tool_context().task_id)
                    if get_tool_context().task_id
                    else None
                )
                if active_plan:
                    prior_references = active_plan["plan"].get(
                        "research_references",
                        [],
                    )
            execution_tool_call = {
                **tool_call,
                "args": {
                    **tool_call.get("args", {}),
                    "research_references": list(
                        dict.fromkeys(prior_references + planning_research_citations)
                    )[:10],
                },
            }
        if get_tool_context().task_id:
            state.task_action_seen = True
            state.tool_tasks.append(
                _execute_task_tool_call(
                    memory,
                    get_tool_context().task_id,
                    get_tool_context().runtime_owner_id,
                    execution_tool_call,
                    **execution_options,
                )
            )
        else:
            state.tool_tasks.append(
                execute_tool_call(
                    execution_tool_call,
                    **execution_options,
                )
            )


async def _execute_tool_batch(
    *,
    response,
    session_id,
    tool_tasks,
    blocked_tool_messages,
    declined_media_calls,
    authorized_planning_searches,
    planning_research_citations,
    messages,
    provider_type,
    planning_only,
    deadline,
    console,
):
    if tool_tasks:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            tool_messages = [
                ToolMessage(
                    content="Error: Task time budget exhausted; completion is unknown.",
                    tool_call_id=call.get("id"),
                )
                for call in response.tool_calls
                if call.get("name") not in NETWORK_TOOL_NAMES
                or call.get("id")
                not in {message.tool_call_id for message in blocked_tool_messages}
            ]
        else:
            tool_batch_started = time.monotonic()
            console.print(
                f"[dim][Processing][/dim] Running "
                f"{len(tool_tasks)} tool action(s); approval "
                "prompts may appear."
            )
            try:
                tool_messages = await asyncio.wait_for(
                    asyncio.gather(*tool_tasks),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                timed_out_calls = [
                    {
                        "tool": str(call.get("name") or "unknown"),
                        "call_id": str(call.get("id") or ""),
                    }
                    for call in response.tool_calls
                    if call.get("id")
                    not in {message.tool_call_id for message in blocked_tool_messages}
                ]
                log_event(
                    RUN_LOGGER,
                    "tool.batch_timeout",
                    level=logging.ERROR,
                    session_id=session_id,
                    turn_index=_SESSION_STATS.turns + 1,
                    tool_calls=timed_out_calls,
                )
                tool_messages = [
                    ToolMessage(
                        content=(
                            "Error: Tool execution exceeded the task time budget. "
                            "Its final side effects may be unknown; verify before retrying."
                        ),
                        tool_call_id=call.get("id"),
                    )
                    for call in response.tool_calls
                    if call.get("id")
                    not in {message.tool_call_id for message in blocked_tool_messages}
                ]
            tool_batch_elapsed = time.monotonic() - tool_batch_started
            if tool_batch_elapsed >= 2:
                console.print(
                    f"[dim][Completed][/dim] Tool actions in {tool_batch_elapsed:.1f}s."
                )
    else:
        tool_messages = []
    for tool_message in tool_messages:
        result_text = _visible_chunk_text(tool_message.content)
        if not _tool_result_is_error(result_text):
            tool_call = next(
                (
                    call
                    for call in response.tool_calls
                    if call.get("id") == tool_message.tool_call_id
                ),
                None,
            )
            if tool_call is not None:
                _display_tool_result(
                    str(tool_call.get("name", "unknown")),
                    result_text,
                )
    if planning_only and authorized_planning_searches:
        for tool_result in tool_messages:
            if tool_result.tool_call_id not in authorized_planning_searches:
                continue
            result_text = _visible_chunk_text(tool_result.content)
            if result_text.casefold().startswith(("error", "web search failed")):
                continue
            for citation in _planning_search_citations(result_text):
                if (
                    citation not in planning_research_citations
                    and len(planning_research_citations) < 10
                ):
                    planning_research_citations.append(citation)
    messages.extend(blocked_tool_messages + tool_messages)
    if declined_media_calls:
        console.print(
            "Local media access was not permitted; no capture or file load occurred."
        )
        if len(declined_media_calls) == len(response.tool_calls):
            devices = " and ".join(
                sorted(
                    {
                        {
                            "capture_webcam_image": "webcam",
                            "load_workspace_image": "workspace image",
                            "load_workspace_video": "workspace video",
                            "transcribe_workspace_audio": "workspace audio",
                            "record_microphone_audio": "microphone",
                        }.get(name, "local media")
                        for name in declined_media_calls
                    }
                )
            )
            response = AIMessage(
                content=(
                    f"I did not access the {devices} because "
                    "permission was not granted. No capture was made."
                )
            )
            return response, True
    webcam_error = _local_image_input_error(
        response.tool_calls,
        blocked_tool_messages + tool_messages,
    )
    if webcam_error is not None:
        response = AIMessage(content=f"Local image input failed: {webcam_error}")
        return response, True
    if provider_type == "local":
        _append_captured_images_from_tool_results(
            messages,
            tool_messages,
        )
    if time.monotonic() >= deadline:
        response = AIMessage(
            content="The task execution budget was exhausted. "
            "Tool results may be incomplete; the requested outcome "
            "was not verified."
        )
        return response, True
    return response, False


def _select_permission_mode(interactive: bool = True) -> str:
    return _select_permission_mode_impl(
        AGENT_PERMISSION_MODE,
        console,
        lambda: interactive,
    )


async def execute_tool_call(
    tool_call: dict,
    *,
    network_authorized: bool = False,
    local_media_allowed: bool = False,
    media_capture_authorized: bool = False,
    vision_supported: bool = False,
    permission_mode: str = "auto",
    max_output_chars: int = MAX_TOOL_OUTPUT_CHARS,
) -> ToolMessage:
    tool_started = time.monotonic()
    name, args, call_id = (
        tool_call.get("name"),
        tool_call.get("args", {}),
        tool_call.get("id"),
    )
    safe_args = _redact_tool_arguments(args)
    log_event(
        RUN_LOGGER,
        "tool.started",
        tool=str(name or "unknown"),
        call_id=str(call_id or ""),
    )
    console.print(
        f"[warning][Tool Execution][/warning] Calling '{name}' "
        f"with argument summary: {safe_args if isinstance(safe_args, dict) else type(args).__name__}"
    )
    get_event_bus().emit(
        ToolStarted(
            call_id=str(call_id or ""),
            tool_name=str(name or ""),
            args_summary=str(safe_args),
            started_at=tool_started,
        )
    )
    _pre_hook_block_reason = evaluate_pre_tool_hooks(
        name, args, SandboxManager.get_root_or_none()
    )[0]
    if _pre_hook_block_reason:
        return ToolMessage(
            content=f"Error: Tool call blocked by hook: {_pre_hook_block_reason}",
            tool_call_id=call_id,
        )
    tool_func = AVAILABLE_TOOLS.get(name)
    _permission_rule_decision, _permission_rule = _evaluate_permission_rules(
        PERMISSION_ALLOW_RULES, PERMISSION_DENY_RULES, name, args
    )
    _permission_rule_denied = _permission_rule_decision == "deny"
    _permission_rule_allowed = _permission_rule_decision == "allow"
    if _permission_rule_denied:
        result = f"Error: Tool call denied by permission rule: {_permission_rule}"
    elif name in LOCAL_MEDIA_TOOL_NAMES and not local_media_allowed:
        result = (
            "Error: Local media tools are available only in an explicitly "
            "selected local-model session."
        )
    elif (
        name
        in {
            "capture_webcam_image",
            "load_workspace_image",
            "load_workspace_video",
        }
        and not vision_supported
    ):
        result = (
            "Error: Image input is unavailable because this model does not "
            "declare vision support."
        )
    elif (
        name in LOCAL_MEDIA_TOOL_NAMES
        and name != "list_microphone_devices"
        and not media_capture_authorized
    ):
        result = "Error: Explicit media capture authorization was not provided."
    elif _local_location_refusal(name, args, local_media_allowed):
        result = _local_location_refusal(name, args, local_media_allowed)
    elif name in NETWORK_TOOL_NAMES and (
        not ENABLE_WEB_RESEARCH
        or WEB_RESEARCH_CONSENT == "never"
        or not network_authorized
    ):
        result = "Error: Network tool call was not authorized; no request was made."
    elif name in {
        "delete_chat_history_from_sqlite",
        "delete_chat_history_entry_from_sqlite",
    }:
        if not (get_tool_context().interactive and sys.stdin.isatty()):
            result = "Error: Deleting SQLite history requires interactive confirmation."
        else:
            if name == "delete_chat_history_entry_from_sqlite":
                entry_id = args.get("entry_id")
                scope = args.get("session_id", "")
                approval = (
                    console.input(
                        f"[error]Delete the reviewed SQLite history entry {entry_id} "
                        f"from session '{scope}'? Type 'delete {entry_id}' to confirm: [/error]"
                    )
                    .strip()
                    .lower()
                )
                confirmed = approval == f"delete {entry_id}"
            else:
                scope = args.get("session_id") or "all sessions"
                approval = (
                    console.input(
                        f"[error]Type 'delete' to delete chat history for {scope}: [/error]"
                    )
                    .strip()
                    .lower()
                )
                confirmed = approval == "delete"
            if not confirmed:
                result = (
                    "Error: Chat history deletion was declined; no data was removed."
                )
            else:
                with approved_tool_invocation():
                    result = await asyncio.to_thread(tool_func.invoke, args)
    elif (
        _tool_needs_permission(
            name,
            permission_mode,
            network_authorized=network_authorized,
            network_tool_names=NETWORK_TOOL_NAMES,
            mcp_tool_names=MCP_TOOL_NAMES,
            mcp_auto_approve_tools=MCP_AUTO_APPROVE_TOOLS,
        )
        and not (name in LOCAL_MEDIA_TOOL_NAMES and media_capture_authorized)
        and not _permission_rule_allowed
        and name not in _ALWAYS_ALLOWED_TOOLS
    ):
        if not (get_tool_context().interactive and sys.stdin.isatty()):
            result = (
                "Error: This tool requires interactive approval and was not run. "
                "Re-run with --permission-mode full to allow it."
            )
        else:
            description = getattr(tool_func, "description", "") if tool_func else ""
            approved = _prompt_tool_approval(
                console=console,
                name=name,
                description=description,
                safe_args=safe_args,
                on_always_allow=_ALWAYS_ALLOWED_TOOLS.add,
            )
            if not approved:
                result = "Error: Tool invocation was declined; no action was taken."
            else:
                try:
                    with approved_tool_invocation():
                        result = (
                            await tool_func.ainvoke(args)
                            if hasattr(tool_func, "ainvoke")
                            else await asyncio.to_thread(tool_func.invoke, args)
                        )
                except (ValueError, OSError, RuntimeError) as exc:
                    reference_id = uuid.uuid4().hex[:8]
                    log_event(
                        RUN_LOGGER,
                        "tool.execution_failed",
                        level=logging.ERROR,
                        reference_id=reference_id,
                        tool=str(name or "unknown"),
                        error_type=type(exc).__name__,
                        exc_info=True,
                        redactions=_tool_log_redactions(args),
                    )
                    result = (
                        f"Error executing tool {name}: {exc} "
                        f"(reference {reference_id}; verify possible side effects "
                        "before retrying)."
                    )
    elif tool_func:
        try:
            with approved_tool_invocation():
                if hasattr(tool_func, "ainvoke"):
                    result = await tool_func.ainvoke(args)
                else:
                    result = await asyncio.to_thread(tool_func.invoke, args)
        except (ValueError, OSError, RuntimeError) as exc:
            reference_id = uuid.uuid4().hex[:8]
            log_event(
                RUN_LOGGER,
                "tool.execution_failed",
                level=logging.ERROR,
                reference_id=reference_id,
                tool=str(name or "unknown"),
                error_type=type(exc).__name__,
                exc_info=True,
                redactions=_tool_log_redactions(args),
            )
            result = (
                f"Error executing tool {name}: {exc!s} "
                f"(reference {reference_id}; verify possible side effects "
                "before retrying)."
            )
    else:
        result = f"Error: Tool {name} not found."
    result_text = _redact_sensitive_values(str(result), args)
    has_error = _tool_result_is_error(result_text)
    if len(result_text) > max_output_chars:
        result_text = result_text[:max_output_chars] + (
            "\n[Tool output truncated by configured limit.]"
        )
    if has_error:
        reflection_suffix = (
            "\n[System Reflection Prompt]: Your execution encountered an "
            "error/traceback. Analyze why it failed, correct your approach, "
            "and try again."
        )
        reflection_suffix = reflection_suffix[:max_output_chars]
        if len(result_text) + len(reflection_suffix) > max_output_chars:
            truncation_note = "\n[Tool output truncated.]"
            available = max(0, max_output_chars - len(reflection_suffix))
            if available >= len(truncation_note):
                result_text = (
                    result_text[: available - len(truncation_note)] + truncation_note
                )
            else:
                result_text = result_text[:available]
        _display_tool_error(str(name), result_text)
        console.print(
            "[warning][Reflection Loop] Tool error details were returned to the "
            "model for recovery.[/warning]"
        )
        result_text += reflection_suffix
    tool_event = "tool.failed" if has_error else "tool.completed"
    log_event(
        RUN_LOGGER,
        tool_event,
        level=logging.WARNING if has_error else logging.INFO,
        tool=name,
        call_id=str(call_id or ""),
        outcome="error" if has_error else "success",
        elapsed_seconds=round(time.monotonic() - tool_started, 3),
        output_character_count=len(result_text),
    )
    _hook_output = collect_post_tool_hook_output(
        tool_name=name, tool_args=args, workspace_root=SandboxManager.get_root_or_none()
    )
    if _hook_output:
        result_text += "\n\n[Post-tool checks]\n" + _hook_output
    get_event_bus().emit(
        ToolFinished(
            call_id=str(call_id or ""),
            status="ok",
            duration=time.monotonic() - tool_started,
            output_size=len(result_text),
        )
    )
    return ToolMessage(content=result_text, tool_call_id=call_id)


async def _execute_task_tool_call(
    memory: PersistentMemory,
    task_id: str,
    owner_id: str,
    tool_call: dict,
    **execution_options: Any,
) -> ToolMessage:
    """Journal a task-bound action before execution and persist only digests."""
    call_id = str(tool_call.get("id") or "")
    try:
        action_id = await asyncio.to_thread(
            memory.begin_task_action,
            task_id,
            owner_id,
            call_id,
            str(tool_call.get("name") or ""),
            tool_call.get("args", {}),
            lease_seconds=TASK_LEASE_SECONDS,
        )
    except (ValueError, sqlite3.Error, OSError) as exc:
        error_details = (
            "Task action was not run because journaling failed: "
            f"{type(exc).__name__}: {exc}"
        )
        _display_tool_error(str(tool_call.get("name") or ""), error_details)
        return ToolMessage(
            content=f"Error: {error_details}",
            tool_call_id=call_id,
        )
    try:
        with logging_context(task_id=task_id):
            result = await execute_tool_call(tool_call, **execution_options)
    except BaseException:  # noqa: BLE001 - journals an unknown outcome for any failure, including cancellation
        try:
            await asyncio.shield(
                asyncio.to_thread(
                    memory.finish_task_action,
                    action_id,
                    "outcome_unknown",
                )
            )
        except (ValueError, sqlite3.Error, OSError) as journal_error:
            log_event(
                RUN_LOGGER,
                "task.action_outcome_journaling_failed",
                level=logging.ERROR,
                task_id=task_id,
                action_id=action_id,
                error_type=type(journal_error).__name__,
                exc_info=True,
            )
        raise
    result_text = str(result.content)
    lowered_result = result_text.casefold()
    outcome_status = (
        "outcome_unknown"
        if "error executing tool" in lowered_result
        or "completion is unknown" in lowered_result
        or "side effects may be unknown" in lowered_result
        else "outcome_observed"
    )
    outcome_digest = hashlib.sha256(result_text.encode("utf-8")).hexdigest()
    try:
        await asyncio.to_thread(
            memory.finish_task_action,
            action_id,
            outcome_status,
            outcome_digest,
        )
    except (ValueError, sqlite3.Error, OSError) as exc:
        warning = (
            f"Warning: the tool returned, but its outcome could not be journaled "
            f"({type(exc).__name__}: {exc}). Treat the action as unresolved and "
            "verify before retrying."
        )
        _display_tool_error(str(tool_call.get("name") or ""), warning)
        return ToolMessage(
            content=f"{result_text}\n{warning}",
            tool_call_id=call_id,
        )
    return result


async def _prepare_session(
    *,
    verbose_startup: bool = False,
    resume_session: str | None = None,
    continue_session: bool = False,
    non_interactive: bool = False,
    interactive: bool = True,
    input_provider=None,
    permission_mode: str | None = None,
) -> SessionSetup | None:
    """Discover local resources, open memory and build the session state."""
    global _ACTIVE_MEMORY, _SESSION_INFO, _TASK_LEASE_HEARTBEAT
    _SESSION_STATS.reset()
    _SESSION_INFO.clear()
    if not non_interactive and (not interactive or not sys.stdin.isatty()):
        console.print(
            "[error]Private Agent requires an interactive terminal. "
            "Run it from a terminal session.[/error]"
        )
        return
    try:
        return await _build_session_setup(
            verbose_startup=verbose_startup,
            resume_session=resume_session,
            continue_session=continue_session,
            non_interactive=non_interactive,
            interactive=interactive,
            input_provider=input_provider,
            permission_mode=permission_mode,
        )
    except BaseException:  # noqa: BLE001 - startup cleanup must also run for cancellation, then re-raises
        await _discard_session_startup()
        raise


def _apply_sandbox_network_permission(
    configured_default: object = None,
    environ=None,
) -> bool:
    """Apply the sandbox network policy during startup, failing closed."""
    default = (
        SANDBOX_NETWORK_DEFAULT if configured_default is None else configured_default
    )
    environment = os.environ if environ is None else environ
    try:
        if "PRIVATE_AGENT_SANDBOX_NETWORK" in environment:
            # An explicitly set but empty/ambiguous value must fail closed.
            enabled = SandboxManager._coerce_network_permission(
                environment["PRIVATE_AGENT_SANDBOX_NETWORK"]
            )
        else:
            enabled = SandboxManager._coerce_network_permission(default)
        return SandboxManager.set_network_allowed(enabled)
    except (TypeError, ValueError, RuntimeError, OSError) as exc:
        console.print(
            "[warning][Sandbox] Invalid sandbox network configuration; "
            f"network access remains disabled: {exc}[/warning]"
        )
        return SandboxManager.disable_network()


async def _build_session_setup(
    *,
    verbose_startup: bool = False,
    resume_session: str | None = None,
    continue_session: bool = False,
    non_interactive: bool = False,
    interactive: bool = True,
    input_provider=None,
    permission_mode: str | None = None,
) -> SessionSetup:
    """Create the memory, RAG index, skills and provider state for a session."""
    global _ACTIVE_MEMORY, _TASK_LEASE_HEARTBEAT
    startup_started = time.monotonic()
    _input = input_provider if input_provider is not None else console.input
    invocation_root = pathlib.Path.cwd()
    project_resources = discover_project_resources(invocation_root)
    rag_docs_default = (
        str(project_resources.rag)
        if project_resources.rag is not None
        else RAG_DOCS_DEFAULT
    )
    skills_folder_default = (
        str(project_resources.skills)
        if project_resources.skills is not None
        else SKILLS_FOLDER_DEFAULT
    )
    if any(
        (
            project_resources.workspace,
            project_resources.rag,
            project_resources.skills,
        )
    ):
        discovered = []
        for label, folder in (
            ("workspace", project_resources.workspace),
            ("RAG", project_resources.rag),
            ("skills", project_resources.skills),
        ):
            if folder is not None:
                discovered.append(f"{label}: {folder}")
        console.print(
            "[success][Project folders][/success] Auto-detected "
            + " | ".join(discovered)
        )
    local_runtimes = _discover_available_local_runtimes()
    log_event(
        RUN_LOGGER,
        "providers.local_discovery_completed",
        runtime_count=len(local_runtimes),
        runtime_kinds=[runtime.kind for runtime in local_runtimes],
    )
    if local_runtimes:
        names = ", ".join(runtime.name for runtime in local_runtimes)
        console.print(
            f"[bold success][Local runtimes][/bold success] "
            f"{len(local_runtimes)} found ({names})."
        )
        if verbose_startup:
            for runtime_index, runtime in enumerate(local_runtimes, 1):
                console.print(
                    f"  [info]{runtime_index}.[/info] {runtime.name} "
                    f"({len(runtime.models)} chat model(s)) — {runtime.base_url}"
                )
                if runtime.models:
                    console.print("     Models: " + ", ".join(runtime.models))
                else:
                    console.print("     No chat models are currently advertised.")
    else:
        console.print(
            "[warning]No local runtimes found to proceed with local inference. "
            "Online configuration is available.[/warning]"
        )
    if verbose_startup:
        console.print(
            "[bold success][Config Status][/bold success] Loaded configuration "
            f"from: [info]{CONFIG_FILE_PATH.resolve()}[/info]"
        )
        console.print(
            "[bold success][Configured paths][/bold success] "
            f"RAG documents: [info]{rag_docs_default or 'not configured'}[/info] | "
            f"RAG index: [info]{RAG_INDEX_PATH}[/info] | "
            f"Skills: [info]{skills_folder_default or 'not configured'}[/info]"
        )
    import code_tasks as code_tasks_module

    isolation_warning = code_tasks_module.isolation_unavailable_reason()
    if verbose_startup:
        if isolation_warning:
            console.print(
                "[warning][Platform limitation][/warning] Shell execution, isolated "
                f"project tests, and stdio MCP servers are disabled: {isolation_warning}"
            )
        else:
            console.print(
                "[success][Platform capability][/success] OS-isolated command "
                "execution is available."
            )
        console.print(
            f"[bold success][Online Research][/bold success] "
            f"{'enabled; consent policy: ' + WEB_RESEARCH_CONSENT if ENABLE_WEB_RESEARCH else 'disabled'}"
        )
    memory = PersistentMemory(db_path=DEFAULT_DB_PATH)
    _ACTIVE_MEMORY = memory
    recovered_actions = memory.reconcile_task_actions_after_restart()
    if recovered_actions:
        console.print(
            "[warning][Task recovery][/warning] Marked "
            f"{recovered_actions} orphaned action(s) as outcome unknown. "
            "Review and verify them before retrying."
        )
    if verbose_startup:
        saved_messages = memory.count_conversation_messages()
        saved_sessions = memory.count_conversation_sessions()
        saved_episodes = memory.count_episodic_memories()
        saved_skill_files = memory.count_registered_skill_files()
        if any((saved_messages, saved_sessions, saved_episodes, saved_skill_files)):
            console.print(
                f"[info][SQLite memory][/info] File: {memory.db_path}; "
                f"{saved_messages} saved conversation message(s) "
                f"across {saved_sessions} session(s); "
                f"{saved_episodes} episodic summary/summaries; "
                f"{saved_skill_files} session-linked skill file(s)."
            )
    stale_task_ids = memory.mark_stale_tasks(TASK_STALE_AFTER_DAYS)
    if stale_task_ids:
        console.print(
            "[warning][Task recovery][/warning] Marked "
            f"{len(stale_task_ids)} stale task(s) for review and cleared their "
            "old approvals. Nothing was deleted."
        )
    if verbose_startup:
        task_plans, approved_task_plans = memory.count_task_plans()
        if task_plans:
            console.print(
                "[info][Saved plans][/info] "
                f"{task_plans} total | {approved_task_plans} approved | "
                f"{task_plans - approved_task_plans} not approved — "
                "use `/tasks` to review or proceed."
            )
    _tool_context = get_tool_context()
    _tool_context.db_path = os.path.abspath(DEFAULT_DB_PATH)
    set_tool_context(_tool_context)
    for server_name, server_config in MCP_SERVERS.items():
        if server_config.get("transport", "stdio") == "stdio" and isolation_warning:
            console.print(
                f"[warning][MCP warning] Skipping stdio server '{server_name}': "
                f"{isolation_warning}[/warning]"
            )
            continue
        mcp_started = time.monotonic()
        try:
            with console.status(
                f"[bold info]Connecting to MCP server {server_name}...[/bold info]"
            ):
                discovered_tools = await load_configured_mcp_tools(
                    server_name,
                    server_config,
                )
            mcp_elapsed = time.monotonic() - mcp_started
            if mcp_elapsed >= 2:
                console.print(
                    f"[dim][Completed][/dim] MCP discovery for {server_name} "
                    f"in {mcp_elapsed:.1f}s."
                )
            for mcp_tool in discovered_tools:
                tool_name = getattr(mcp_tool, "name", None)
                if not tool_name:
                    console.print(
                        f"[warning][MCP warning] Server '{server_name}' returned a tool without a name; skipped.[/warning]"
                    )
                elif tool_name in AVAILABLE_TOOLS:
                    console.print(
                        f"[warning][MCP warning] Tool name '{tool_name}' from '{server_name}' "
                        "conflicts with an existing tool; skipped.[/warning]"
                    )
                else:
                    AVAILABLE_TOOLS[tool_name] = mcp_tool
                    MCP_TOOL_NAMES.add(tool_name)
                    MCP_TOOL_SOURCES[tool_name] = server_name
                    MCP_TOOL_TRANSPORTS[tool_name] = server_config.get(
                        "transport", "stdio"
                    )
            if verbose_startup:
                console.print(
                    f"[success][MCP][/success] Loaded {len(discovered_tools)} "
                    f"tool(s) from '{server_name}'."
                )
        except Exception as exc:  # noqa: BLE001 - third-party MCP transport errors vary; isolate one server
            reference_id = uuid.uuid4().hex[:8]
            console.print(
                f"[error][MCP error][/error] Could not load {server_name} "
                f"({type(exc).__name__}; reference {reference_id}). "
                "Check its transport, command or endpoint configuration."
            )
            log_event(
                RUN_LOGGER,
                "mcp.server_load_failed",
                level=logging.ERROR,
                reference_id=reference_id,
                server=server_name,
                elapsed_seconds=round(time.monotonic() - mcp_started, 3),
                error_type=type(exc).__name__,
                exc_info=True,
            )
        else:
            log_event(
                RUN_LOGGER,
                "mcp.server_loaded",
                server=server_name,
                tool_count=len(discovered_tools),
                elapsed_seconds=round(time.monotonic() - mcp_started, 3),
            )
    from providers.selection import requested_workspace

    flagged_workspace = requested_workspace()
    if flagged_workspace:
        workspace_input = flagged_workspace
        console.print(f"[success][Workspace flag][/success] Using {workspace_input}.")
    elif project_resources.workspace is not None:
        workspace_input = str(project_resources.workspace)
        if is_windows_mount_path(workspace_input):
            console.print(
                "[warning][WSL][/warning] Workspace is under a Windows drive mount "
                f"({workspace_input}); file operations may be slow. Prefer a "
                "Linux filesystem path such as ~/projects."
            )
    else:
        workspace_input = (
            WORKSPACE_ROOT_DEFAULT
            if WORKSPACE_ROOT_DEFAULT != "."
            else _input(
                "[info]Enter workspace root path for security sandbox "
                "(Press Enter for current dir): [/info]"
            ).strip()
        )
    if not workspace_input:
        workspace_input = "."
    if is_windows_mount_path(workspace_input):
        console.print(
            "[warning][WSL][/warning] Workspace is under a Windows drive mount "
            f"({workspace_input}); file operations may be slow. Prefer a "
            "Linux filesystem path such as ~/projects."
        )
    SandboxManager.set_root(workspace_input)
    SandboxManager.set_approved_extra_folders(APPROVED_EXTRA_FOLDERS)
    _apply_sandbox_network_permission()
    if verbose_startup:
        console.print(
            f"[bold success][Security][/bold success] Workspace root locked to: "
            f"{SandboxManager.get_root()}"
        )
    from providers.selection import requested_rag

    flagged_rag = requested_rag()
    if flagged_rag:
        docs_input = flagged_rag
        console.print(f"[success][RAG flag][/success] Using {docs_input}.")
    elif project_resources.rag is not None:
        docs_input = str(project_resources.rag)
        if is_windows_mount_path(docs_input):
            console.print(
                "[warning][WSL][/warning] RAG documents are under a Windows drive "
                f"mount ({docs_input}); indexing may be slow. Prefer a Linux "
                "filesystem path."
            )
    elif rag_docs_default:
        rag_override = _input(
            "[warning]RAG documents directory "
            f"(Enter to use configured path '{rag_docs_default}', or enter a "
            "project-specific path): [/warning]"
        ).strip()
        docs_input = rag_override or rag_docs_default
    else:
        docs_input = _input(
            "[warning]Enter knowledge base (KB) files directory path "
            "(Press Enter to skip): [/warning]"
        ).strip()
    if docs_input and is_windows_mount_path(docs_input):
        console.print(
            "[warning][WSL][/warning] RAG documents are under a Windows drive "
            f"mount ({docs_input}); indexing may be slow. Prefer a Linux "
            "filesystem path."
        )
    import config as _rag_config

    rag_embedding_provider = _rag_config.EMBEDDING_PROVIDER
    rag_embedding_model = _rag_config.EMBEDDING_MODEL
    rag_embedding_base_url = _rag_config.EMBEDDING_BASE_URL
    ollama_runtimes = [
        runtime
        for runtime in local_runtimes
        if getattr(runtime, "kind", "") == "ollama"
    ]
    openai_runtimes = [
        runtime
        for runtime in local_runtimes
        if getattr(runtime, "kind", "") == "openai-compatible"
    ]
    if rag_embedding_provider == "ollama":
        embedding_models = []
        for runtime in ollama_runtimes:
            embedding_models.extend(getattr(runtime, "embedding_models", ()) or ())
        if embedding_models:
            rag_embedding_model = str(embedding_models[0])
        elif ollama_runtimes:
            pass
        elif openai_runtimes:
            selected_runtime = None
            selected_model = None
            for runtime in openai_runtimes:
                models = getattr(runtime, "embedding_models", ()) or ()
                if models:
                    selected_runtime = runtime
                    selected_model = str(models[0])
                    break
            if selected_runtime is not None:
                rag_embedding_provider = "openai-compatible"
                rag_embedding_base_url = str(
                    getattr(selected_runtime, "base_url", rag_embedding_base_url)
                )
                rag_embedding_model = selected_model
                console.print(
                    "[info][RAG][/info] Ollama is unavailable; using "
                    f"{getattr(selected_runtime, 'name', 'local runtime')} "
                    f"embeddings ({selected_model})."
                )
            elif local_runtimes:
                rag_embedding_provider = "none"
                console.print(
                    "[warning][RAG][/warning] No embedding-capable runtime is "
                    "available; continuing without local RAG."
                )

    def _initialize_knowledge_base_with_embeddings(
        docs_path,
        *,
        index_path=None,
        confirm_rebuild=None,
    ):
        return initialize_knowledge_base(
            docs_path,
            index_path=index_path,
            confirm_rebuild=confirm_rebuild,
            embedding_provider=rag_embedding_provider,
            embedding_model=rag_embedding_model,
            embedding_base_url=rag_embedding_base_url,
        )

    # Start RAG indexing (background when no existing index needs an
    # interactive rebuild) so provider selection is not blocked by it
    global _ACTIVE_RAG_TASK
    (
        vectorstore,
        is_rag_active,
        rag_signature,
        rag_task,
        rag_last_check,
    ) = await _initialize_rag_for_docs(
        docs_input,
        rag_index_path=RAG_INDEX_PATH,
        console=console,
        input_fn=_input,
        log_event=log_event,
        run_logger=RUN_LOGGER,
        uuid=uuid,
        logging=logging,
        initialize_knowledge_base_fn=_initialize_knowledge_base_with_embeddings,
        knowledge_base_signature_fn=knowledge_base_signature,
    )
    if rag_task is not None:
        _ACTIVE_RAG_TASK = rag_task
    skills_folder_input = (
        skills_folder_default
        if skills_folder_default
        else _input(
            "[info]Enter path to skills folder containing .md files "
            "(Press Enter for none): [/info]"
        ).strip()
    )
    if (
        CONVERSATION_RETENTION_DAYS
        or EPISODE_RETENTION_DAYS
        or REGISTERED_SKILL_RETENTION_DAYS
    ):
        await asyncio.to_thread(
            _prune_memory_retention,
            memory,
            conversation_days=CONVERSATION_RETENTION_DAYS,
            episode_days=EPISODE_RETENTION_DAYS or CONVERSATION_RETENTION_DAYS,
            skill_days=REGISTERED_SKILL_RETENTION_DAYS,
            skills_directory=skills_folder_input or None,
        )
    loaded_skills = (
        load_skills_from_folder(skills_folder_input) if skills_folder_input else {}
    )
    _tool_context = get_tool_context()
    _tool_context.skill_directory = (
        os.path.abspath(os.path.expanduser(skills_folder_input))
        if skills_folder_input
        else None
    )
    _tool_context.skill_registry = loaded_skills
    set_tool_context(_tool_context)
    if loaded_skills and verbose_startup:
        console.print(
            f"[bold success][Skills Loaded][/bold success] Successfully loaded {len(loaded_skills)} markdown skill profile(s)."
        )
    elif not loaded_skills and verbose_startup:
        console.print(
            "[warning][Skills Mode][/warning] No skills folder specified or found. Running in plain agent mode."
        )
    has_skills = len(loaded_skills) > 0
    if has_skills and is_rag_active:
        mode_label = "[info]Skilled Private Agent[/info]"
    elif has_skills:
        mode_label = "[info]Skilled Agent[/info]"
    elif is_rag_active:
        mode_label = "[info]Private Agent[/info]"
    else:
        mode_label = "[info]Agent[/info]"
    if verbose_startup:
        console.print(
            f"[bold success][Runtime Mode][/bold success] "
            f"Active Mode Indicator: {mode_label}"
        )
    log_event(
        RUN_LOGGER,
        "agent.startup_completed",
        elapsed_seconds=round(time.monotonic() - startup_started, 3),
        mcp_server_count=len(MCP_SERVERS),
        rag_enabled=is_rag_active,
    )
    resume_prompt = (
        (
            lambda latest: _input(
                f"[info]Resume latest session '{latest}'? \\[Y/n]: [/info]"
            )
        )
        if interactive
        else None
    )
    session_id = resolve_session_id(
        memory,
        resume_session=resume_session,
        continue_session=continue_session,
        prompt=resume_prompt,
    )
    _tool_context = get_tool_context()
    _tool_context.skill_session_id = session_id
    _tool_context.session_id = session_id
    set_tool_context(_tool_context)
    _tool_context = get_tool_context()
    _tool_context.runtime_owner_id = uuid.uuid4().hex
    _tool_context.task_session_id = session_id
    _tool_context.task_id = None
    set_tool_context(_tool_context)
    if (
        take_over_session_task_lease(memory, session_id, _tool_context.runtime_owner_id)
        == "held"
    ):
        console.print(
            "[warning][Task lease warning] This session's active plan is still "
            "owned by another running process; task actions wait until it "
            "releases or expires.[/warning]"
        )
    network_consent_state: dict[str, bool] = {}
    _TASK_LEASE_HEARTBEAT = asyncio.create_task(
        _maintain_task_lease(memory, session_id)
    )
    thinking_enabled = THINKING_TOGGLE_DEFAULT
    thinking_effort = THINKING_EFFORT_DEFAULT
    if permission_mode is not None:
        _permission_mode = permission_mode
    else:
        from providers.selection import requested_mode

        flagged_mode = requested_mode()
        if flagged_mode and flagged_mode in {"manual", "auto", "full", "monitored"}:
            _permission_mode = flagged_mode
            console.print(
                f"[success][Mode flag][/success] Using {flagged_mode} permission mode."
            )
        else:
            if flagged_mode:
                console.print(
                    f"[warning][Mode flag][/warning] '{flagged_mode}' is not a valid "
                    "permission mode; choosing interactively."
                )
            _permission_mode = _select_permission_mode(interactive=interactive)
    permission_mode = _permission_mode
    if verbose_startup:
        console.print(f"[info][Agent Permission Mode][/info] {permission_mode.title()}")
    safe_run_session_start_hooks(
        hooks=load_hooks(
            merge_project_config(load_or_create_config(), SandboxManager.get_root())
        ),
        workspace_root=SandboxManager.get_root(),
        session_id=session_id,
    )
    return SessionSetup(
        memory=memory,
        local_runtimes=local_runtimes,
        isolation_warning=isolation_warning,
        vectorstore=vectorstore,
        docs_input=docs_input,
        is_rag_active=is_rag_active,
        rag_signature=rag_signature,
        rag_last_check=rag_last_check,
        rag_task=rag_task,
        skills_folder_input=skills_folder_input,
        loaded_skills=loaded_skills,
        mode_label=mode_label,
        session_id=session_id,
        network_consent_state=network_consent_state,
        thinking_enabled=thinking_enabled,
        thinking_effort=thinking_effort,
        permission_mode=permission_mode,
    )


async def _discard_session_startup() -> None:
    """Release whatever a failed startup already opened."""
    global _ACTIVE_MEMORY, _TASK_LEASE_HEARTBEAT, _ACTIVE_RAG_TASK
    heartbeat, _TASK_LEASE_HEARTBEAT = _TASK_LEASE_HEARTBEAT, None
    rag_task, _ACTIVE_RAG_TASK = _ACTIVE_RAG_TASK, None
    if rag_task is not None and not rag_task.done():
        rag_task.cancel()
        await asyncio.gather(rag_task, return_exceptions=True)
    if heartbeat is not None:
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
    memory, _ACTIVE_MEMORY = _ACTIVE_MEMORY, None
    if memory is not None:
        memory.close()


async def _choose_provider(
    *,
    verbose_startup: bool,
    preference: ProviderPreference,
    memory,
    local_runtimes,
    isolation_warning,
    thinking_enabled,
    thinking_effort,
    permission_mode,
    loaded_skills,
    mode_label,
    is_rag_active,
    vectorstore,
    session_id,
    skills_folder_input,
    non_interactive: bool = False,
    interactive: bool = True,
    input_provider=None,
) -> ProviderSelection | None:
    """Ask for a provider and model, initialise it and print the ready summary."""
    _input = input_provider if input_provider is not None else console.input
    available_models: list = []
    console.print("\n[bold underline]Model Provider[/bold underline]")
    from providers.selection import (
        load_last_provider,
        match_runtime,
        optional_dependency_hint,
        pick_default_model,
        requested_model,
        requested_provider,
    )

    last_used = load_last_provider() or {}
    resumed_runtime = None
    if resumed_record := provider_record_for_resume(memory, session_id):
        resumed_runtime = match_runtime(
            resumed_record.display_name, local_runtimes
        ) or match_runtime_by_address(resumed_record.base_url, local_runtimes)
    online_options: list[tuple[int, str, bool]] = []
    next_choice = len(local_runtimes) + 1
    if preference.online_preference is not None:
        label = (
            "Online — configured in config (session preference)"
            if preference.online_preference
            else "Online — manual details (session preference)"
        )
        online_options.append((next_choice, label, preference.online_preference))
    elif _has_saved_online_config(APP_CONFIG):
        online_options.extend(
            [
                (
                    next_choice,
                    "Online — configured in config",
                    True,
                ),
                (
                    next_choice + 1,
                    "Online — enter details manually",
                    False,
                ),
            ]
        )
    else:
        online_options.append(
            (
                next_choice,
                "Online — enter details manually",
                False,
            )
        )
    if preference.online_preference is not None or _has_saved_online_config(APP_CONFIG):
        default_choice = str(online_options[0][0])
    else:
        default_choice = "1" if local_runtimes else str(online_options[0][0])
    for _index, _runtime in enumerate(local_runtimes, 1):
        if last_used.get("display_name") == _runtime.name:
            default_choice = str(_index)
            break
    last_choice = online_options[-1][0]
    _arrow_choices: list[tuple[str, str]] = (
        [
            (
                str(_i),
                f"Local ({_rt.name}) — {len(_rt.models)} model(s)"
                + (" (last used)" if last_used.get("display_name") == _rt.name else ""),
            )
            for _i, _rt in enumerate(local_runtimes, 1)
        ]
        + [(str(_cn), _label) for _cn, _label, _ in online_options]
        + [("0", "Exit")]
    )
    flagged_provider = requested_provider()
    flagged_runtime = match_runtime(flagged_provider, local_runtimes)
    if flagged_provider and flagged_runtime is None:
        console.print(
            f"[warning][Provider flag][/warning] '{flagged_provider}' is not "
            "reachable; choose from the list below."
        )
    preselected_model: str | None = None
    if flagged_runtime is not None:
        provider_choice = str(local_runtimes.index(flagged_runtime) + 1)
        console.print(
            f"[success][Provider flag][/success] Using {flagged_runtime.name} "
            "(from --provider)."
        )
    elif resumed_runtime is not None:
        provider_choice = str(local_runtimes.index(resumed_runtime) + 1)
        resumed_models = list(resumed_runtime.models)
        if resumed_record.model in resumed_models:
            preselected_model = resumed_record.model
            console.print(
                f"[success][Resume][/success] Continuing with {resumed_runtime.name} "
                f"· {resumed_record.model} (recorded provider)."
            )
        elif resumed_models:
            console.print(
                f"[warning][Resume][/warning] {resumed_runtime.name} no longer has "
                f"'{resumed_record.model}'; choosing the closest available model."
            )
        else:
            console.print(
                f"[warning][Resume][/warning] {resumed_runtime.name} is running but "
                "advertises no chat models."
            )
    elif (
        len(local_runtimes) == 1
        and not always_ask_provider(APP_CONFIG)
        and auto_select_single_provider(APP_CONFIG)
    ):
        provider_choice = "1"
        console.print(
            f"[success][Local runtime][/success] Using {local_runtimes[0].name} (only reachable local provider)."
        )
    else:
        try:
            from .interactive import arrow_select as _arrow_select

            _chosen = _arrow_select(
                "Choose provider:", _arrow_choices, default=default_choice
            )
        except Exception:  # noqa: BLE001 - fall back to typed prompt
            _chosen = None
        if _chosen is not None:
            provider_choice = _chosen
        else:
            for runtime_index, runtime in enumerate(local_runtimes, 1):
                marker = (
                    " [success](last used)[/success]"
                    if last_used.get("display_name") == runtime.name
                    else ""
                )
                console.print(
                    f"  [info]{runtime_index}.[/info] Local ({runtime.name}) "
                    f"— {len(runtime.models)} model(s){marker}"
                )
            for choice_number, label, _use_saved_details in online_options:
                console.print(f"  [info]{choice_number}.[/info] {label}")
            console.print("  [info]0.[/info] Exit")
            provider_choice = (
                _input(
                    f"[info]Choose provider [1-{last_choice}] "
                    f"(Default: {default_choice}): [/info]"
                ).strip()
                or default_choice
            )
    if provider_choice == "0":
        preference.exit_requested = True
        return None
    selected_online_option = next(
        (option for option in online_options if provider_choice == str(option[0])),
        None,
    )
    if selected_online_option is not None:
        selected_local_runtime = None
        selected_online_preference = selected_online_option[2]
        if preference.online_selection is None:
            preference.online_selection = select_online_model(
                selected_online_preference
            )
        if preference.online_selection is None:
            return None
        preference.online_preference = selected_online_preference
        provider_type = "online"
        selected_model = preference.online_selection["model_name"]
        capabilities = preference.online_selection["capabilities"]
        llm = preference.online_selection["model"]
        tools_list = (
            _tools_for_model(
                provider_type=provider_type,
                capabilities=capabilities,
                isolation_warning=isolation_warning,
            )
            if preference.online_selection["allow_tools"]
            else []
        )
        if (
            not preference.online_selection["share_context"]
            and not ONLINE_PERMISSION_OVERRIDE
        ):
            tools_list = [
                tool
                for tool in tools_list
                if tool.name
                not in {
                    "create_task_plan",
                    "inspect_task_plan",
                    "update_todo_step",
                    "revise_task_plan",
                }
            ]
        if preference.online_selection["allow_tools"]:
            try:
                llm_with_tools = _bind_tools(llm, tools_list)
            except Exception as exc:  # noqa: BLE001 - bind_tools errors depend on the endpoint; continue without tools
                console.print(
                    f"[warning]The selected endpoint/model does not accept tool calls "
                    f"({type(exc).__name__}); continuing without tools.[/warning]"
                )
                llm_with_tools = llm
                capabilities["tools"] = False
                preference.online_selection["allow_tools"] = False
        else:
            llm_with_tools = llm
        if verbose_startup:
            console.print(
                f"[success][Online model][/success] {selected_model} at "
                f"{preference.online_selection['base_url']}"
            )
    elif provider_choice.isdigit() and 1 <= int(provider_choice) <= len(local_runtimes):
        local_runtime = local_runtimes[int(provider_choice) - 1]
        selected_local_runtime = local_runtime
        provider_type = "local"
        available_models = list(local_runtime.models)
        if PREFERRED_MODEL in available_models:
            available_models.remove(PREFERRED_MODEL)
            available_models.insert(0, PREFERRED_MODEL)
        model_preference = requested_model() or preselected_model or ""
        flagged_model = pick_default_model(available_models, model_preference)
        if model_preference and flagged_model in available_models:
            available_models.remove(flagged_model)
            available_models.insert(0, flagged_model)
        if not available_models:
            console.print(
                f"[warning]{local_runtime.name} is running, but it advertises "
                "no chat models. Choose another runtime or online mode.[/warning]"
            )
            return None
        if PREFERRED_MODEL and PREFERRED_MODEL not in available_models:
            console.print(
                f"[warning]Configured preferred model '{PREFERRED_MODEL}' is not "
                f"installed in {local_runtime.name}.[/warning]"
            )
        requested_exact = next(
            (
                model_name
                for model_name in available_models
                if model_name.strip().lower() == requested_model().lower()
            ),
            None,
        )
        if requested_exact is not None:
            preselected_model = requested_exact
        try:
            from .interactive import arrow_select as _arrow_select

            _model_choice = (
                None
                if preselected_model is not None
                else _arrow_select(
                    "Select model:",
                    [(str(i), m) for i, m in enumerate(available_models, 1)],
                    default="1",
                )
            )
        except Exception:  # noqa: BLE001 - arrow UI unavailable; fall back to typed prompt
            _model_choice = None
        if preselected_model is not None:
            selected_model = preselected_model
            console.print(f"[success][Model][/success] Using {preselected_model}.")
        elif _model_choice is not None and _model_choice.isdigit():
            _mc = int(_model_choice) - 1
            if 0 <= _mc < len(available_models):
                selected_model = available_models[_mc]
            else:
                selected_model = available_models[0]
        else:
            console.print(
                f"\n[bold underline]Available {local_runtime.name} Chat Models:[/bold underline]"
            )
            for idx, model_name in enumerate(available_models, 1):
                marker = (
                    " [success](configured, preferred)[/success]"
                    if PREFERRED_MODEL and model_name == PREFERRED_MODEL
                    else ""
                )
                console.print(f"  [info]{idx}.[/info] {model_name}{marker}")
            while True:
                choice_input = _input(
                    f"[info]Select model choice [1-{len(available_models)}] (Default: 1): [/info]"
                ).strip()
                if not choice_input:
                    selected_model = available_models[0]
                    break
                try:
                    choice_idx = int(choice_input) - 1
                    if 0 <= choice_idx < len(available_models):
                        selected_model = available_models[choice_idx]
                        break
                except ValueError:
                    pass
                console.print("[error][Error] Invalid selection.[/error]")
        console.print(
            f"[success][Info] Initializing local model client for: {selected_model}[/success]"
        )
        capabilities = inspect_openai_compatible_model_capabilities(
            local_runtime, selected_model
        )
        if capabilities["tools"] is False:
            tool_capable_model = next(
                (
                    candidate
                    for candidate in available_models
                    if candidate != selected_model
                    and inspect_openai_compatible_model_capabilities(
                        local_runtime, candidate
                    )["tools"]
                    is True
                ),
                None,
            )
            if tool_capable_model:
                console.print(
                    f"[warning][Capability routing] {selected_model} does not support tools; "
                    f"using compatible local model {tool_capable_model} for agent tasks.[/warning]"
                )
                selected_model = tool_capable_model
                capabilities = inspect_openai_compatible_model_capabilities(
                    local_runtime, selected_model
                )
        try:
            llm = _make_selected_local_chat_model(
                local_runtime,
                selected_model,
                thinking_enabled=thinking_enabled,
                thinking_effort=thinking_effort,
                capabilities=capabilities,
            )
        except Exception as exc:  # noqa: BLE001 - provider constructors raise library-specific errors
            log_event(
                RUN_LOGGER,
                "provider.local_model_initialization_failed",
                level=logging.ERROR,
                runtime=local_runtime.name,
                model=selected_model,
                error_type=type(exc).__name__,
                exc_info=True,
            )
            hint = optional_dependency_hint(local_runtime.kind, exc)
            console.print(
                f"[error]Could not initialize {local_runtime.name} "
                f"model '{selected_model}' ({type(exc).__name__}). "
                "Choose another runtime or online mode.[/error]"
            )
            if hint:
                console.print(f"[warning]{hint}[/warning]")
            return None
        tools_list = _tools_for_model(
            provider_type=provider_type,
            capabilities=capabilities,
            isolation_warning=isolation_warning,
        )
        if capabilities["tools"] is False:
            llm_with_tools = llm
        else:
            try:
                llm_with_tools = _bind_tools(llm, tools_list)
            except Exception as exc:  # noqa: BLE001 - bind_tools errors depend on the runtime; continue without tools
                fallback_model = next(
                    (
                        candidate
                        for candidate in available_models
                        if candidate != selected_model
                        and (
                            inspect_openai_compatible_model_capabilities(
                                local_runtime, candidate
                            )
                        )["tools"]
                        is True
                    ),
                    None,
                )
                if fallback_model is None:
                    console.print(
                        f"[warning][Model limitation] Tool binding failed for {selected_model} "
                        f"({type(exc).__name__}); continuing without tools.[/warning]"
                    )
                    llm_with_tools = llm
                else:
                    console.print(
                        f"[warning][Model fallback] Switching to tool-capable model {fallback_model}.[/warning]"
                    )
                    selected_model = fallback_model
                    capabilities = inspect_openai_compatible_model_capabilities(
                        local_runtime, selected_model
                    )
                    llm = _make_selected_local_chat_model(
                        local_runtime,
                        selected_model,
                        thinking_enabled=thinking_enabled,
                        thinking_effort=thinking_effort,
                        capabilities=capabilities,
                    )
                    llm_with_tools = _bind_tools(llm, tools_list)
        log_event(
            RUN_LOGGER,
            "providers.local_runtime_selected",
            runtime=local_runtime.name,
            runtime_kind=local_runtime.kind,
            model=selected_model,
        )
        persist_provider_choice(
            console=console,
            memory=memory,
            session_id=session_id,
            runtime=local_runtime,
            model=selected_model,
            capabilities=capabilities,
            previous=resumed_record,
        )
    else:
        console.print(f"[error]Select a provider from 1 to {last_choice}.[/error]")
        return None
    if provider_type == "local":
        webcam_status = (
            "webcam, workspace image, and sampled workspace video input enabled"
            if capabilities["vision"] is True
            else (
                "webcam, workspace image, and workspace video input unavailable "
                "(model lacks vision support)"
            )
        )
        missing_media = [
            package
            for package, module in (
                ("opencv-python-headless", "cv2"),
                ("Pillow", "PIL"),
                ("sounddevice", "sounddevice"),
                ("faster-whisper", "faster_whisper"),
            )
            if importlib.util.find_spec(module) is None
        ]
        if missing_media:
            console.print(
                "[warning][Local media][/warning] "
                f"{webcam_status}; optional media packages missing "
                f"({', '.join(missing_media)}); install with "
                '`pip install ".[media]"`. Workspace PCM WAV transcription '
                "is local-only and requires per-use approval."
            )
        elif capabilities["vision"] is not True:
            console.print(f"[warning][Local media][/warning] {webcam_status}.")
    console.print(
        "[info][Capabilities][/info] "
        + ", ".join(
            f"{name}={'yes' if value else 'no' if value is False else 'unknown'}"
            for name, value in capabilities.items()
            if name != "context_window"
        )
    )
    include_private_context = (
        provider_type == "local" or preference.online_selection["share_context"]
    )
    startup_memory_context = (
        await asyncio.to_thread(
            _build_startup_memory_context,
            memory,
            skills_folder_input or None,
            exclude_session_id=session_id,
        )
        if include_private_context
        else ""
    )
    chat_history = (
        await asyncio.to_thread(
            memory.load_history, session_id, limit=MAX_HISTORY_MESSAGES
        )
        if include_private_context
        else []
    )
    context_summary = ""
    session_summary_saved = await asyncio.to_thread(
        memory.ends_with_summary, session_id
    )
    _SESSION_INFO.update(
        {
            "SQL session ID": session_id,
            "Session mode": re.sub(r"\[/?[^\]]*\]", "", mode_label),
            "Agent (permission) mode": permission_mode,
            "Model provider": (
                f"Local ({selected_local_runtime.name})"
                if provider_type == "local" and selected_local_runtime
                else "Online"
            ),
            "Model": selected_model,
            "Thinking": (
                f"{'on' if thinking_enabled else 'off'} (effort: {thinking_effort})"
            ),
            "Workspace": str(SandboxManager.get_root()),
            "Local RAG": "active" if is_rag_active else "off",
            "Skills loaded": len(loaded_skills),
            "Resumed history messages": len(chat_history),
        }
    )
    startup_summary = Table(
        title="Ready",
        box=None,
        show_header=False,
    )
    startup_summary.add_column("Setting", style="cyan")
    startup_summary.add_column("Value", overflow="fold")
    _platform_info = detect_platform()
    os_name = _platform_info.name or _platform_info.system or sys.platform
    architecture = _platform_info.machine
    startup_summary.add_row(
        "Platform",
        f"{os_name} ({architecture})" if architecture else os_name,
    )
    startup_summary.add_row(
        "Isolation",
        (
            f"Disabled — {isolation_warning}"
            if isolation_warning
            else "Available — bubblewrap + prlimit"
        ),
    )
    startup_summary.add_row(
        "Provider",
        (
            f"Local ({selected_local_runtime.name})"
            if provider_type == "local"
            else "Online"
        ),
    )
    startup_summary.add_row("Model", selected_model)
    startup_summary.add_row("Permission", permission_mode.title())
    startup_summary.add_row(
        "Workspace",
        str(SandboxManager.get_root()),
    )
    startup_summary.add_row(
        "Local RAG",
        "active" if is_rag_active else "inactive",
    )
    startup_summary.add_row("Skills", str(len(loaded_skills)))
    console.print(startup_summary)
    console.print(
        "[dim]Version: dev "
        f"\u00b7 Workspace: {SandboxManager.get_root()} \u00b7 Type /help for commands.[/dim]"
    )
    console.print(
        f"\n[bold success]--- {mode_label} Ready! Type 'exit' or 'switch' ---[/bold success]"
    )
    return ProviderSelection(
        provider_type=provider_type,
        selected_model=selected_model,
        capabilities=capabilities,
        llm=llm,
        llm_with_tools=llm_with_tools,
        tools_list=tools_list,
        selected_local_runtime=selected_local_runtime,
        include_private_context=include_private_context,
        startup_memory_context=startup_memory_context,
        chat_history=chat_history,
        context_summary=context_summary,
        session_summary_saved=session_summary_saved,
        available_models=available_models,
    )


async def _run_agent_cli_session(
    *,
    verbose_startup: bool = False,
    resume_session: str | None = None,
    continue_session: bool = False,
    prompt: str | None = None,
    output: str = "text",
    permission_mode: str | None = None,
    non_interactive: bool = False,
    interactive: bool = True,
    input_provider=None,
):
    global _ACTIVE_MEMORY, _TASK_LEASE_HEARTBEAT, _LAST_HEADLESS_RESPONSE
    setup = await _prepare_session(
        verbose_startup=verbose_startup,
        resume_session=resume_session,
        continue_session=continue_session,
        non_interactive=non_interactive,
        interactive=interactive,
        input_provider=input_provider,
        permission_mode=permission_mode,
    )
    if setup is None:
        return
    _tc = get_tool_context()
    _tc.interactive = interactive
    set_tool_context(_tc)
    memory = setup.memory
    local_runtimes = setup.local_runtimes
    isolation_warning = setup.isolation_warning
    vectorstore = setup.vectorstore
    docs_input = setup.docs_input
    is_rag_active = setup.is_rag_active
    rag_signature = setup.rag_signature
    rag_last_check = setup.rag_last_check
    rag_task = setup.rag_task
    skills_folder_input = setup.skills_folder_input
    loaded_skills = setup.loaded_skills
    custom_slash_commands = load_custom_slash_commands(SandboxManager.get_root())
    mode_label = setup.mode_label
    session_id = setup.session_id
    network_consent_state = setup.network_consent_state
    thinking_enabled = setup.thinking_enabled
    thinking_effort = setup.thinking_effort
    permission_mode = setup.permission_mode
    provider_preference = ProviderPreference()
    while True:
        selection = await _choose_provider(
            verbose_startup=verbose_startup,
            preference=provider_preference,
            memory=memory,
            local_runtimes=local_runtimes,
            isolation_warning=isolation_warning,
            thinking_enabled=thinking_enabled,
            thinking_effort=thinking_effort,
            permission_mode=permission_mode,
            loaded_skills=loaded_skills,
            mode_label=mode_label,
            is_rag_active=is_rag_active,
            vectorstore=vectorstore,
            session_id=session_id,
            skills_folder_input=skills_folder_input,
            non_interactive=non_interactive,
            interactive=interactive,
            input_provider=input_provider,
        )
        if selection is None:
            if provider_preference.exit_requested:
                return
            if non_interactive:
                get_event_bus().subscribe(Event, json_output.emit_event)
                return
            continue
        # Collect the RAG index that was built in the background during
        # provider selection (J2, step 13.5).
        if rag_task is not None:
            try:
                if not rag_task.done():
                    with console.status(
                        "[bold info]Finishing RAG index...[/bold info]"
                    ):
                        vectorstore = await rag_task
                else:
                    vectorstore = rag_task.result()
            except Exception as exc:  # noqa: BLE001 - keep startup working without RAG
                vectorstore = None
                reference_id = uuid.uuid4().hex[:8]
                console.print(
                    f"[warning][RAG warning] Knowledge base is unavailable; continuing "
                    f"without RAG ({type(exc).__name__}; reference {reference_id}).[/warning]"
                )
            is_rag_active = vectorstore is not None
            if is_rag_active:
                rag_signature = knowledge_base_signature(docs_input)
            else:
                rag_signature = None
            rag_last_check = time.monotonic()
            _ACTIVE_RAG_TASK = None
            rag_task = None
            with contextlib.suppress(Exception):  # best effort
                _SESSION_INFO["Local RAG"] = "active" if is_rag_active else "off"
        provider_type = selection.provider_type
        selected_model = selection.selected_model
        capabilities = selection.capabilities
        llm = selection.llm
        llm_with_tools = selection.llm_with_tools
        tools_list = selection.tools_list
        from .subagent_runner import make_subagent_runner

        _tool_context = get_tool_context()
        _tool_context.subagent_runner = make_subagent_runner(llm)
        set_tool_context(_tool_context)
        selected_local_runtime = selection.selected_local_runtime
        include_private_context = selection.include_private_context
        startup_memory_context = selection.startup_memory_context
        chat_history = selection.chat_history
        context_summary = selection.context_summary
        session_summary_saved = selection.session_summary_saved
        available_models = selection.available_models
        online_selection = provider_preference.online_selection
        model_selection_failed = False
        active_code_workspace: CodeTaskWorkspace | None = None
        previous_workspace_root = SandboxManager.get_root()
        base_tools_list = list(tools_list)
        planning_mode_persistent = False
        coding_mode_armed = False
        selected_skill_key: str | None = None
        pending_folder_context = ""
        pending_folder_context_files: tuple[str, ...] = ()
        pending_folder_context_bytes = 0
        cli_input_history = InMemoryHistory()
        last_user_request = ""

        def clear_context_after_data_reset(scopes: frozenset[str]) -> None:
            nonlocal chat_history, context_summary, startup_memory_context
            nonlocal session_summary_saved
            if "sqlite" in scopes:
                chat_history = []
                context_summary = ""
                startup_memory_context = ""
                session_summary_saved = False

        # see the current llm/capabilities after a mid-session model switch, so its
        # late binding of loop-assigned names (B023) is intentional.
        def cleanup_code_task(*, finalize: bool) -> str | None:
            nonlocal active_code_workspace, tools_list, llm_with_tools
            if active_code_workspace is None:
                return None
            result = active_code_workspace.finalize() if finalize else None
            try:
                from .learning import _record_code_task_validation

                _record_code_task_validation(
                    memory,
                    get_tool_context().task_id,
                    result,
                    get_tool_context().runtime_owner_id,
                )
            except (ValueError, sqlite3.Error, OSError) as exc:
                console.print(
                    "[warning][Task validation warning] Code checks passed, "
                    "but the evidence could not be persisted: "
                    f"{type(exc).__name__}: {exc}[/warning]"
                )
            SandboxManager.set_root(str(previous_workspace_root))  # noqa: B023
            _tool_context = get_tool_context()
            _tool_context.code_task = None
            set_tool_context(_tool_context)
            AVAILABLE_TOOLS.pop("run_project_unit_tests", None)
            AVAILABLE_TOOLS.pop("checkpoint_code_task", None)
            AVAILABLE_TOOLS.pop("finalize_code_task", None)
            tools_list = list(base_tools_list)  # noqa: B023
            if capabilities["tools"] is not False:  # noqa: B023
                try:
                    llm_with_tools = _bind_tools(llm, tools_list)  # noqa: B023
                except Exception as exc:  # noqa: BLE001 - bind_tools errors depend on the runtime; continue without tools
                    console.print(
                        f"[warning]Could not restore the base tool set "
                        f"({type(exc).__name__}); tool calls are disabled for this model.[/warning]"
                    )
                    llm_with_tools = llm  # noqa: B023
            active_code_workspace = None
            return result

        session_ui = SessionUI(
            console=console,
            status_bar=STATUS_BAR,
        )
        session_ui.bind(model=selected_model, permission=permission_mode)
        if SHOW_WELCOME_SCREEN and not non_interactive:
            from rich.panel import Panel

            session_ui.print(
                Panel(
                    f"Private Agent\nModel: {selected_model}\nWorkspace: {SandboxManager.get_root()}",
                    title="Welcome",
                    border_style="cyan",
                )
            )
        _local_model_has_answered = False
        _pending_user_input = None
        try:
            while True:
                try:
                    if _pending_user_input is not None:
                        user_input = _pending_user_input
                        _pending_user_input = None
                    elif non_interactive:
                        user_input = (prompt or "").strip() or "exit"
                        prompt = None
                    else:
                        session_ui._footer.stop()
                        if SHOW_TURN_DIVIDER:
                            session_ui.console.rule(style="dim")
                        user_input = (
                            await prompt_user_input(
                                console,
                                SandboxManager.get_root(),
                                prompt_message="User: ",
                                skills=loaded_skills,
                                history=cli_input_history,
                                bottom_toolbar=session_ui.ribbon_text(),
                            )
                        ).strip()
                    if not non_interactive:
                        session_ui._footer.start()
                        session_ui._footer.update(session_ui._current_state())
                except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
                    _acknowledge_interrupt()
                    user_input = "exit"
                    if not non_interactive:
                        session_ui._footer.start()
                        session_ui._footer.update(session_ui._current_state())
                _custom_command_name = ""
                if user_input.startswith("/"):
                    _custom_command_name = user_input.split(None, 1)[0].lower()
                if _custom_command_name in custom_slash_commands:
                    _custom_command_args = user_input[
                        len(_custom_command_name) :
                    ].strip()
                    _custom_command_content = custom_slash_commands[
                        _custom_command_name
                    ]
                    if _custom_command_args:
                        _custom_command_content = (
                            _custom_command_content
                            + "\n\nArguments: "
                            + _custom_command_args
                        )
                    user_input = _custom_command_content
                    console.print(
                        f"[info][Command][/info] Running {_custom_command_name}"
                    )
                if user_input and not user_input.startswith("/"):
                    last_user_request = user_input
                (
                    _handled,
                    _cont,
                    user_input,
                    chat_history,
                    context_summary,
                    last_user_request,
                ) = handle_session_command(
                    user_input,
                    chat_history,
                    context_summary,
                    last_user_request,
                    SandboxManager.get_root(),
                    console,
                )
                if _handled and _cont:
                    continue
                if user_input.lower() in {"/help", "/commands"}:
                    session_ui.print(_format_interactive_help(GROUPED_HELP))
                    continue
                _mission_input = _mission_slash_command(
                    user_input, memory, session_id, session_ui
                )
                if _mission_input is None:
                    continue
                user_input = _mission_input
                if user_input.lower() == "/doctor":
                    from diagnostics.doctor import doctor_interactive

                    doctor_interactive()
                    continue
                if user_input.lower() == "/context":
                    if not pending_folder_context:
                        session_ui.print(
                            "[info][Context][/info] No folder context is queued."
                        )
                    else:
                        table = Table(
                            title="Queued request context (next prompt only)",
                            box=None,
                            show_header=False,
                        )
                        table.add_column("Field", style="cyan")
                        table.add_column("Value", overflow="fold")
                        table.add_row("Files", str(len(pending_folder_context_files)))
                        table.add_row(
                            "Total size",
                            f"{pending_folder_context_bytes:,} bytes",
                        )
                        for file_path in pending_folder_context_files:
                            table.add_row("Included", file_path)
                        session_ui.print(table)
                        if provider_type == "online":
                            session_ui.print(
                                "[warning]This context will be sent to the selected "
                                "online provider with your next prompt.[/warning]"
                            )
                    continue
                if user_input.lower() in {"/context report", "/context stats", "/context usage"}:
                    try:
                        from .context_telemetry import format_context_report

                        session_ui.print(format_context_report(_CONTEXT_REPORT_STATE.get("value", {}) or {}))
                    except Exception as exc:  # noqa: BLE001 - optional best-effort context feature
                        session_ui.print(f"[warning][Context] {exc}[/warning]")
                    continue
                if user_input.lower() == "/context clear":
                    pending_folder_context = ""
                    pending_folder_context_files = ()
                    pending_folder_context_bytes = 0
                    session_ui.print(
                        "[success][Context][/success] Queued context cleared."
                    )
                    continue
                if user_input.lower().startswith("/context "):
                    session_ui.print(
                        "[warning][Context][/warning] Use `/context` to review "
                        "queued files or `/context clear` to remove them."
                    )
                    continue
                if user_input.lower() == "/status":
                    table = Table(
                        title="Session Status",
                        box=None,
                        show_header=False,
                    )
                    table.add_column("Setting", style="cyan")
                    table.add_column("Current value", overflow="fold")
                    table.add_row(
                        "Provider",
                        (
                            f"Local ({selected_local_runtime.name})"
                            if provider_type == "local"
                            else "Online"
                        ),
                    )
                    table.add_row("Model", selected_model)
                    table.add_row("Permission mode", permission_mode.title())
                    table.add_row("Workspace", str(SandboxManager.get_root()))
                    table.add_row(
                        "RAG",
                        (
                            f"active — {docs_input}"
                            if vectorstore and docs_input
                            else "inactive"
                        ),
                    )
                    table.add_row(
                        "Queued folder context",
                        (
                            f"{len(pending_folder_context_files)} file(s), "
                            f"{pending_folder_context_bytes:,} bytes"
                            if pending_folder_context
                            else "none"
                        ),
                    )
                    session_ui.print(table)
                    continue
                if user_input.lower() == "/undo":
                    undo_outcome = await asyncio.to_thread(
                        undo_last_turn,
                        memory,
                        session_id,
                        workspace_root=SandboxManager.get_root(),
                    )
                    session_ui.print(undo_outcome.message)
                    if undo_outcome.history_changed:
                        chat_history = await asyncio.to_thread(
                            memory.get_history,
                            session_id,
                            MAX_HISTORY_MESSAGES,
                        )
                        context_summary = ""
                        session_summary_saved = await asyncio.to_thread(
                            memory.ends_with_summary, session_id
                        )
                    continue
                lowered_input = user_input.strip().lower()
                if (
                    lowered_input == "/expand"
                    or lowered_input == "/last"
                    or lowered_input.startswith("/last ")
                ):
                    from . import tool_display as _tool_display

                    result_index = -1
                    parts = user_input.split()
                    if len(parts) > 1:
                        try:
                            result_index = int(parts[1])
                        except ValueError:
                            session_ui.print(
                                "[warning][Last][/warning] Usage: `/last [N]` "
                                "where N selects a recent tool result."
                            )
                            continue
                    if len(parts) > 1:
                        expanded = _tool_display.expand_last_tool_result(result_index)
                    else:
                        expanded = _tool_display.expand_last_tool_result()
                    if expanded is None:
                        session_ui.print(
                            "[warning][Last][/warning] No tool output to expand."
                        )
                    else:
                        session_ui.print(expanded)
                    continue
                if user_input.lower() == "/add" or user_input.lower().startswith(
                    "/add "
                ):
                    folder_argument = user_input[len("/add") :].strip()
                    if not folder_argument:
                        session_ui.print(
                            "[warning][Context][/warning] Usage: "
                            "`/add <workspace-relative-folder>`."
                        )
                        continue
                    try:
                        (
                            pending_folder_context,
                            pending_folder_context_files,
                            folder_context_bytes,
                        ) = include_folder_context(folder_argument)
                        pending_folder_context_bytes = folder_context_bytes
                    except (OSError, ValueError) as exc:
                        session_ui.print(f"[error][Folder context error] {exc}[/error]")
                        continue
                    session_ui.print(
                        "[success][Context][/success] Added "
                        f"{len(pending_folder_context_files)} file(s) "
                        f"({folder_context_bytes} bytes) from that folder to the "
                        "next request only. A new `/add` replaces previously "
                        "queued folder context."
                    )
                    continue
                if user_input.lower() == "/maintenance":
                    from .commands import maintenance as _maintenance_command

                    vectorstore = await _maintenance_command.handle_maintenance_command(
                        memory=memory,
                        llm=llm,
                        allow_model_episode_context=include_private_context,
                        local_model_suggestions_allowed=provider_type == "local",
                        rag_docs_path=docs_input or None,
                        vectorstore=vectorstore,
                        on_data_reset=clear_context_after_data_reset,
                    )
                    continue
                if user_input.lower() in {"/plan", "/plan on"}:
                    from .commands import plan as _plan_command

                    planning_mode_persistent = _plan_command.handle_plan_command(
                        user_input, planning_mode_persistent, session_ui
                    )
                    continue
                if user_input.lower() == "/plan off":
                    from .commands import plan as _plan_command

                    planning_mode_persistent = _plan_command.handle_plan_command(
                        user_input, planning_mode_persistent, session_ui
                    )
                    continue
                if user_input.lower() == "/code":
                    from .commands import plan as _plan_command

                    coding_mode_armed = _plan_command.handle_code_command(session_ui)
                    continue
                if user_input.lower() == "/skills" or user_input.lower().startswith(
                    "/skills "
                ):
                    from .commands import skills as _skills_command

                    selected_skill_key, loaded_skills = (
                        _skills_command.handle_skills_command(
                            user_input, loaded_skills, selected_skill_key, session_ui
                        )
                    )
                    continue
                if user_input.lower() == "/tasks":
                    from .commands import tasks as _tasks_command

                    result = await _tasks_command.handle_tasks_command(
                        memory=memory,
                        session_id=session_id,
                        skills_folder_input=skills_folder_input,
                        loaded_skills=loaded_skills,
                        session_ui=session_ui,
                        console=console,
                    )
                    approved_task_id, loaded_skills, new_user_input = result
                    if new_user_input:
                        _pending_user_input = new_user_input
                        continue
                    if approved_task_id is None:
                        continue
                    approved_task = memory.get_task(approved_task_id)
                    if not (
                        approved_task
                        and approved_task["status"] == "active"
                        and approved_task["approved_revision"]
                        == approved_task["revision"]
                        and approved_task["approved_digest"] == approved_task["digest"]
                    ):
                        session_ui.print(
                            "[warning][Task not started][/warning] The task is no "
                            "longer active with current approval; no execution "
                            "was started. Review it with `/tasks`."
                        )
                        continue
                    _pending_user_input = approved_task["goal"]
                    session_ui.print(
                        "[success][Task started][/success] Continuing with the "
                        "approved plan automatically; no need to repeat the "
                        "request. Enter `pause` via `/tasks` to stop it."
                    )
                    continue
                if user_input.lower() == "/list_tools":
                    catalog = describe_tool_catalog()
                    unavailable_names = {
                        entry["name"]
                        for entry in catalog
                        if _tool_unavailable_reason(
                            entry["name"],
                            provider_type=provider_type,
                            capabilities=capabilities,
                            isolation_warning=isolation_warning,
                        )
                    }
                    _print_tool_catalog(
                        catalog,
                        unavailable_names=unavailable_names,
                    )
                    continue
                _thinking_command = user_input.lower()
                if _thinking_command in {
                    "/thinking on",
                    "/thinking off",
                    "/think on",
                    "/think off",
                }:
                    from .commands import think as _think_command

                    thinking_enabled, new_llm, new_llm_with_tools = (
                        _think_command.handle_think_toggle(
                            command=_thinking_command,
                            selected_model=selected_model,
                            selected_local_runtime=selected_local_runtime,
                            thinking_effort=thinking_effort,
                            capabilities=capabilities,
                            tools_list=tools_list,
                            session_ui=session_ui,
                            make_model=_make_selected_local_chat_model,
                            bind_tools=_bind_tools,
                        )
                    )
                    if new_llm is not None:
                        llm = new_llm
                        llm_with_tools = new_llm_with_tools
                    continue
                _effort_command = user_input.lower()
                if _effort_command.startswith(("/effort", "/think-effort")):
                    parts = _effort_command.split()
                    if len(parts) != 2 or parts[1] not in {"low", "medium", "high"}:
                        session_ui.print(
                            "[warning]Usage: /effort low|medium|high[/warning]"
                        )
                        continue
                    thinking_effort = parts[1]
                    if capabilities["thinking"] is not True:
                        session_ui.print(
                            f"[warning]Model '{selected_model}' does not declare thinking support; "
                            "the effort setting is saved but inactive.[/warning]"
                        )
                    else:
                        llm = _make_selected_local_chat_model(
                            selected_local_runtime,
                            selected_model,
                            thinking_enabled=thinking_enabled,
                            thinking_effort=thinking_effort,
                            capabilities=capabilities,
                        )
                        llm_with_tools = (
                            _bind_tools(llm, tools_list)
                            if capabilities["tools"] is not False
                            else llm
                        )
                    continue
                if user_input.lower() == "/think-status":
                    from .commands import think as _think_command

                    _think_command.handle_think_status(
                        selected_model=selected_model,
                        thinking_enabled=thinking_enabled,
                        thinking_effort=thinking_effort,
                        capabilities=capabilities,
                        session_ui=session_ui,
                    )
                    continue
                if user_input.lower() == "/hardware-status":
                    from .commands import think as _think_command

                    _think_command.handle_hardware_status(
                        provider_type=provider_type,
                        selected_local_runtime=selected_local_runtime,
                        session_ui=session_ui,
                    )
                    continue
                if user_input.lower() == "/summarize":
                    if session_summary_saved:
                        session_ui.print(
                            "[info][Memory][/info] Nothing new since the last summary."
                        )
                    else:
                        session_summary_saved = await _summarize_and_save_session(
                            console,
                            memory,
                            llm,
                            session_id,
                            _history_with_summary(context_summary, chat_history),
                            provider_type,
                            reason="on demand",
                            model_name=selected_model,
                        )
                    continue
                if user_input.lower() == "/compact":
                    if not chat_history:
                        session_ui.print(
                            "[info][Context][/info] Nothing to compact yet."
                        )
                        continue
                    chat_history, context_summary = await compact_history(
                        llm,
                        chat_history,
                        context_summary,
                        keep_recent=0,
                        max_chars=CONTEXT_SUMMARY_CHARS,
                        source_token_limit=max(200, MAX_CONTEXT_TOKENS // 4),
                        notify=lambda text, sui=session_ui: sui.print(
                            f"[info][Context][/info] {text}"
                        ),
                        model_name=selected_model,
                    )
                    session_ui.print(
                        "[success][Context][/success] Compacted into a working summary "
                        f"({len(context_summary)} chars); {len(chat_history)} message(s) kept verbatim."
                    )
                    continue
                if user_input.lower() == "exit":
                    if non_interactive:
                        if output == "json":
                            sys.stdout.write(
                                json.dumps({"response": _LAST_HEADLESS_RESPONSE}) + "\n"
                            )
                            sys.stdout.flush()
                        return
                    if session_summary_saved:
                        session_ui.print(
                            "[info][Memory][/info] Session summary already saved; "
                            "no new prompts since."
                        )
                    else:
                        await _summarize_and_save_session(
                            console,
                            memory,
                            llm,
                            session_id,
                            _history_with_summary(context_summary, chat_history),
                            provider_type,
                            model_name=selected_model,
                        )
                    _print_session_statistics()
                    return
                if user_input.lower() == "switch":
                    session_ui.print(
                        "[info][Info] Returning to model selection...[/info]"
                    )
                    model_selection_failed = True
                    break
                if not user_input:
                    continue
                turn_permission_mode = permission_mode
                planning_only = planning_mode_persistent
                _acs = _approved_coding_task_scope(memory)
                coding_task_requested = coding_mode_armed or _acs is not None
                if planning_only:
                    active_task_id = get_tool_context().task_id
                    active_task = (
                        memory.get_task(active_task_id) if active_task_id else None
                    )
                    if (
                        active_task
                        and active_task["status"] == "active"
                        and active_task.get("approved_revision")
                        == active_task.get("revision")
                        and active_task.get("approved_digest")
                        == active_task.get("digest")
                    ):
                        planning_only = False
                planning_mode_persistent = planning_only
                coding_mode_armed = False
                planning_research_citations: list[str] = []
                clear_schema_reads()
                if planning_only:
                    if capabilities["tools"] is False:
                        try:
                            from .text_planning import create_text_plan

                            create_text_plan(
                                memory, session_id, user_input, llm, console
                            )
                        except Exception as exc:  # noqa: BLE001 - text planning is optional; report failure
                            session_ui.print(
                                "[error]Text planning failed: "
                                f"{type(exc).__name__}: {exc}[/error]"
                            )
                        continue
                    planner_names = {
                        "read_table_schema",
                        "create_task_plan",
                        "inspect_task_plan",
                        "revise_task_plan",
                        "read_local_file",
                        "list_directory",
                        "search_workspace_files",
                        "search_workspace_text",
                        "search_workspace_symbols",
                        "inspect_workspace_git",
                        "ask_user",
                    }
                    if ENABLE_WEB_RESEARCH:
                        planner_names.add("web_search")
                        planner_names.add("fetch_webpage")
                    tools_list = [
                        tool for tool in base_tools_list if tool.name in planner_names
                    ]
                    if not tools_list:
                        session_ui.print(
                            "[warning]Planner tools are unavailable. Select Local "
                            "or explicitly enable local-context sharing for this "
                            "Online session.[/warning]"
                        )
                        continue
                    try:
                        llm_with_tools = _bind_tools(llm, tools_list)
                    except Exception as exc:  # noqa: BLE001 - bind_tools errors depend on the model; abort only this turn
                        session_ui.print(
                            f"[error]Could not bind planner tools "
                            f"({type(exc).__name__}); no task was planned.[/error]"
                        )
                        tools_list = list(base_tools_list)
                        continue
                elif (
                    active_code_workspace is None and capabilities["tools"] is not False
                ):
                    tools_list = capability_packs.filter_tools_list(
                        base_tools_list, user_input
                    )
                    try:
                        llm_with_tools = _bind_tools(llm, tools_list)
                    except Exception as exc:  # noqa: BLE001 - bind_tools errors depend on the model; continue without tools
                        session_ui.print(
                            f"[warning]Could not bind the session tools "
                            f"({type(exc).__name__}); continuing without tool calls.[/warning]"
                        )
                        llm_with_tools = llm
                try:
                    user_input, attached_file_context = include_file_context(user_input)
                except (OSError, ValueError) as exc:
                    session_ui.print(f"[error][File context error] {exc}[/error]")
                    continue
                if pending_folder_context:
                    attached_file_context = "\n\n".join(
                        part
                        for part in (
                            pending_folder_context,
                            attached_file_context,
                        )
                        if part
                    )
                    pending_folder_context = ""
                    pending_folder_context_files = ()
                    pending_folder_context_bytes = 0
                if attached_file_context and not user_input:
                    user_input = "Use the attached file context to answer my request."
                clear_captured_images()
                code_related_request = _is_coding_guidance_request(user_input)
                if coding_task_requested and not planning_only:
                    approved_coding_scope = _approved_coding_task_scope(memory)
                    if approved_coding_scope is None:
                        session_ui.print(
                            "[warning][Plan approval required][/warning] Run /plan, "
                            "save the coding checklist from the approved request, "
                            "review and approve it with /tasks (including the "
                            "source project scope), then run /code for the next "
                            "request. The active approved plan selected in "
                            "ToolContext is used; you do not need to repeat the "
                            "original wording. "
                            "No code workspace was created."
                        )
                        continue
                    access_level = approved_coding_scope["access_level"]
                    turn_permission_mode = _coding_permission_mode(access_level)
                    if isolation_warning:
                        session_ui.print(
                            "[error][Code task unavailable][/error] Isolated project "
                            f"execution is disabled on this host: {isolation_warning}"
                        )
                        continue
                    if (
                        provider_type == "online"
                        and not online_selection["allow_tools"]
                    ):
                        session_ui.print(
                            "[error]This online session has local tool access disabled. "
                            "Select Local or explicitly enable tools in a new Online session "
                            "to run a code task.[/error]"
                        )
                        continue
                    if capabilities["tools"] is False:
                        session_ui.print(
                            "[error]The selected model cannot call tools, so it cannot "
                            "safely execute this code task.[/error]"
                        )
                        continue
                    source_path = None
                    if approved_coding_scope is not None:
                        source_path = approved_coding_scope.get("source_path")
                    workspace_action_id = None
                    try:
                        workspace_action_id = memory.begin_task_action(
                            get_tool_context().task_id,
                            get_tool_context().runtime_owner_id,
                            f"runtime-workspace-{uuid.uuid4().hex}",
                            (
                                "create_isolated_code_workspace"
                                if _coding_uses_isolated_copy(access_level)
                                else "attach_direct_code_workspace"
                            ),
                            {
                                "goal": user_input,
                                "output_root": CODE_OUTPUT_ROOT,
                                "source_path": source_path,
                                "access_level": access_level,
                                "network_access": _coding_allows_command_network(
                                    access_level
                                ),
                            },
                            lease_seconds=TASK_LEASE_SECONDS,
                        )
                    except (ValueError, sqlite3.Error, OSError) as exc:
                        session_ui.print(
                            "[error][Task journal error] No code workspace was "
                            f"created because action intent could not be saved: {exc}[/error]"
                        )
                        continue
                    try:
                        active_code_workspace = CodeTaskWorkspace.create(
                            user_input,
                            CODE_OUTPUT_ROOT,
                            source_path=source_path,
                            direct_source=not _coding_uses_isolated_copy(access_level),
                            allow_network=_coding_allows_command_network(access_level),
                        )
                    except Exception as exc:  # noqa: BLE001 - workspace creation mixes git/OS/library failures; resolve the action
                        try:
                            memory.finish_task_action(
                                workspace_action_id,
                                "outcome_unknown",
                            )
                        except (ValueError, sqlite3.Error, OSError) as journal_exc:
                            session_ui.print(
                                "[error][Task journal error] Could not persist "
                                f"workspace failure outcome: {journal_exc}[/error]"
                            )
                        session_ui.print(
                            "[error][Code workspace error] Creation failed; inspect "
                            f"the isolated output location before retrying: {exc}[/error]"
                        )
                        active_code_workspace = None
                        _tool_context = get_tool_context()
                        _tool_context.code_task = None
                        set_tool_context(_tool_context)
                        with contextlib.suppress(ValueError, sqlite3.Error, OSError):
                            memory.set_task_status(
                                _tool_context.task_id,
                                "cancelled",
                                owner_id=getattr(
                                    _tool_context, "runtime_owner_id", None
                                ),
                            )
                    else:
                        workspace_digest = hashlib.sha256(
                            (
                                str(active_code_workspace.root)
                                if active_code_workspace is not None
                                else "no workspace returned"
                            ).encode("utf-8")
                        ).hexdigest()
                        try:
                            memory.finish_task_action(
                                workspace_action_id,
                                "outcome_observed",
                                workspace_digest,
                            )
                        except (ValueError, sqlite3.Error, OSError) as exc:
                            session_ui.print(
                                "[error][Task journal error] Workspace creation "
                                "returned, but its outcome could not be saved; "
                                f"inspect it before retrying: {exc}[/error]"
                            )
                            active_code_workspace = None
                    if active_code_workspace is None:
                        session_ui.print(
                            "[warning]Code task cancelled. Code tools were not started. Inspect task "
                            "actions and the isolated output path before retrying.[/warning]"
                        )
                        continue
                    SandboxManager.set_root(str(active_code_workspace.root))
                    _tool_context = get_tool_context()
                    _tool_context.code_task = active_code_workspace
                    set_tool_context(_tool_context)
                    AVAILABLE_TOOLS["run_project_unit_tests"] = run_project_unit_tests
                    AVAILABLE_TOOLS["checkpoint_code_task"] = checkpoint_code_task
                    AVAILABLE_TOOLS["finalize_code_task"] = finalize_code_task
                    tools_by_name = {tool.name: tool for tool in base_tools_list}
                    tools_by_name.update(
                        {
                            tool.name: tool
                            for tool in (
                                run_project_unit_tests,
                                checkpoint_code_task,
                                finalize_code_task,
                            )
                        }
                    )
                    tools_list = list(tools_by_name.values())
                    try:
                        llm_with_tools = _bind_tools(llm, tools_list)
                    except Exception as exc:  # noqa: BLE001 - bind_tools errors depend on the model; abort the code task
                        session_ui.print(
                            f"[error]Could not bind the code-task tools ({type(exc).__name__}); "
                            "no code was written.[/error]"
                        )
                        cleanup_code_task(finalize=False)
                        continue
                    git_policy = (
                        "A local Git repository is initialized inside this isolated project. "
                        "Use checkpoint_code_task after each completed microtask/subtask; "
                        "it runs unit tests before committing only the specified changed paths. "
                        "Finish with tests and a verified local checkpoint."
                        if active_code_workspace.git_enabled
                        else "This is a user-selected direct workspace. Do not create "
                        "Git commits; run project tests and report concrete validation "
                        "evidence."
                        if active_code_workspace.direct_source
                        else "The user explicitly approved continuing without Git. Do not claim "
                        "commits/checkpoints exist. Create or update a unit-test file, run "
                        "run_project_unit_tests, and report the missing Git history."
                    )
                    session_ui.print(
                        f"[bold info][Code Task][/bold info] Workspace: "
                        f"{active_code_workspace.root} | "
                        f"Access level: {access_level}\n{git_policy}"
                    )
                active_skill = None
                if selected_skill_key:
                    if not include_private_context:
                        session_ui.print(
                            "[warning][Skills][/warning] The selected skill was not "
                            "sent because this session does not permit private "
                            "local context sharing."
                        )
                    else:
                        active_skill = loaded_skills.get(selected_skill_key)
                        if active_skill is None:
                            session_ui.print(
                                "[warning][Skills][/warning] The selected skill is "
                                "no longer available; use `/skills` to choose "
                                "another one."
                            )
                    selected_skill_key = None
                elif loaded_skills and include_private_context:
                    skill_embedder = (
                        getattr(vectorstore, "embeddings", None)
                        if vectorstore is not None
                        else None
                    )
                    if skill_embedder is not None:
                        from skills import match_skill_by_relevancy

                        active_skill = await asyncio.to_thread(
                            match_skill_by_relevancy,
                            user_input,
                            loaded_skills,
                            skill_embedder,
                        )
                    else:
                        from skills import match_skill_by_relevancy

                        active_skill = match_skill_by_relevancy(
                            user_input, loaded_skills
                        )
                enforce_tool_call_limits = (
                    provider_type == "online"
                    and online_selection is not None
                    and online_selection.get("enforce_tool_call_limits", False)
                )
                tool_iteration_limit = (
                    min(
                        MAX_TOOL_ITERATIONS,
                        active_skill.max_iterations
                        if active_skill
                        else MAX_TOOL_ITERATIONS,
                    )
                    if enforce_tool_call_limits
                    else None
                )
                tool_call_limit = MAX_TOOL_CALLS if enforce_tool_call_limits else None
                if active_skill:
                    skill_budget = (
                        f"{tool_iteration_limit} tool rounds"
                        if tool_iteration_limit is not None
                        else "uncapped tool rounds"
                    )
                    session_ui.print(
                        f"[magenta][Skill Active][/magenta] "
                        f"{active_skill.name} ({skill_budget})"
                    )
                status_message = (
                    f"Model thinking is active (effort: {thinking_effort}; "
                    "reasoning is private and not displayed)..."
                    if thinking_enabled and capabilities["thinking"] is True
                    else "Processing your request and generating a response..."
                )
                context_text = ""
                retrieved_citations = []
                # Episode search and learned-preference lookups scan SQLite;
                # keep them off the event loop so streaming and the live
                # overlay stay responsive (J10, 13.8).
                relevant_episodes = (
                    await asyncio.to_thread(
                        _build_relevant_episode_context, memory, user_input
                    )
                    if include_private_context
                    else ""
                )
                learned_task_type = (
                    "coding"
                    if code_related_request
                    else "research"
                    if re.search(
                        r"\b(research|web|online|current information|sources?)\b",
                        user_input,
                        re.IGNORECASE,
                    )
                    else "other"
                )
                if include_private_context:
                    selected_task = (
                        await asyncio.to_thread(
                            memory.get_task, get_tool_context().task_id
                        )
                        if get_tool_context().task_id
                        else None
                    )
                    if selected_task and selected_task["status"] == "active":
                        learned_task_type = selected_task["task_type"]
                learned_context = (
                    await asyncio.to_thread(
                        _build_learned_context, memory, learned_task_type
                    )
                    if include_private_context
                    else ""
                )
                episodic_context = "\n\n".join(
                    part
                    for part in (
                        startup_memory_context,
                        (
                            "Relevant prior task episodes (historical evidence, "
                            "not current verification):\n" + relevant_episodes
                            if relevant_episodes
                            else ""
                        ),
                        (
                            "User-confirmed learned preferences (current instructions "
                            "take precedence):\n" + learned_context
                            if learned_context
                            else ""
                        ),
                    )
                    if part
                )
                if (
                    vectorstore
                    and docs_input
                    and RAG_AUTO_REFRESH_SECONDS
                    and time.monotonic() - rag_last_check >= RAG_AUTO_REFRESH_SECONDS
                ):
                    rag_last_check = time.monotonic()
                    try:
                        current_signature = knowledge_base_signature(docs_input)
                        if current_signature != rag_signature:
                            session_ui.print(
                                "[info][RAG][/info] Knowledge-base changes detected; "
                                "updating the index..."
                            )
                            refresh_started = time.monotonic()
                            with console.status(
                                "[bold info]Refreshing local knowledge index...[/bold info]"
                            ):
                                refreshed = await asyncio.to_thread(
                                    initialize_knowledge_base,
                                    docs_input,
                                    index_path=RAG_INDEX_PATH,
                                    embedding_provider=rag_embedding_provider,
                                    embedding_model=rag_embedding_model,
                                    embedding_base_url=rag_embedding_base_url,
                                )
                            refresh_elapsed = time.monotonic() - refresh_started
                            if refresh_elapsed >= 2:
                                session_ui.print(
                                    f"[dim][Completed][/dim] RAG refresh in "
                                    f"{refresh_elapsed:.1f}s."
                                )
                            if refreshed is not None:
                                vectorstore = refreshed
                                rag_signature = current_signature
                                session_ui.print(
                                    "[success][RAG][/success] Index updated."
                                )
                            else:
                                session_ui.print(
                                    "[warning][RAG] Update failed; keeping the "
                                    "previous index.[/warning]"
                                )
                    except Exception as exc:  # noqa: BLE001 - reindex failures come from Chroma/embeddings; keep the old index
                        reference_id = uuid.uuid4().hex[:8]
                        session_ui.print(
                            f"[warning][RAG] Auto-update skipped "
                            f"({type(exc).__name__}; reference {reference_id}). "
                            "Check the configured documents path and retry "
                            "through `/maintenance` → `rag reindex`.[/warning]"
                        )
                        log_event(
                            RUN_LOGGER,
                            "rag.refresh_failed",
                            level=logging.ERROR,
                            reference_id=reference_id,
                            error_type=type(exc).__name__,
                            exc_info=True,
                        )
                if vectorstore and include_private_context:
                    rag_search_started = time.monotonic()
                    try:
                        with console.status(
                            "[bold info]Searching local knowledge...[/bold info]"
                        ):
                            _rag_budget = None
                            if context_engine_is_enabled():
                                try:
                                    _ctx_engine = ContextEngine.from_config()
                                    _alloc = _ctx_engine.allocate(
                                        MAX_CONTEXT_TOKENS,
                                        source_texts={
                                            "system": "",
                                            "rag": "",
                                            "history": "",
                                            "user": user_input,
                                        },
                                        request_text=user_input,
                                    )
                                    _rag_budget = _alloc.budgets.get("rag")
                                except Exception:  # noqa: BLE001 - optional best-effort context feature
                                    _rag_budget = None
                            relevant_docs = await _retrieve_rag_documents(
                                vectorstore,
                                user_input,
                                include_coding_guidance=code_related_request,
                                budget_tokens=_rag_budget,
                            )
                        rag_search_elapsed = time.monotonic() - rag_search_started
                        if rag_search_elapsed >= 2:
                            session_ui.print(
                                f"[dim][Completed][/dim] Local search in "
                                f"{rag_search_elapsed:.1f}s."
                            )
                        if relevant_docs:
                            retrieved_citations = format_retrieved_citations(
                                relevant_docs
                            )
                            context_text = "\n".join(
                                f"[Source: {d.metadata.get('source', 'local knowledge base')}"
                                f"{', chunk ' + str(d.metadata['chunk']) if 'chunk' in d.metadata else ''}]\n"
                                f"{d.page_content}"
                                for d in relevant_docs
                            )
                            session_ui.print(
                                f"[info][RAG State][/info] Retrieved {len(relevant_docs)} document/episodic chunks."
                            )
                        elif RAG_DOCUMENTS_ONLY:
                            context_text = "No relevant documents were found. Answer that the information is not in the provided documents."
                    except Exception as exc:  # noqa: BLE001 - retrieval failures come from Chroma/embeddings; use no context
                        reference_id = uuid.uuid4().hex[:8]
                        session_ui.print(
                            "[warning][RAG][/warning] Local search failed; "
                            "continuing without retrieved documents "
                            f"(reference {reference_id}). Check the local index "
                            "and configured embedding runtime."
                        )
                        log_event(
                            RUN_LOGGER,
                            "rag.search_failed",
                            level=logging.ERROR,
                            reference_id=reference_id,
                            error_type=type(exc).__name__,
                            exc_info=True,
                        )
                    finally:
                        log_event(
                            RUN_LOGGER,
                            "rag.search_finished",
                            elapsed_seconds=round(
                                time.monotonic() - rag_search_started,
                                3,
                            ),
                        )
                if attached_file_context:
                    context_text = "\n\n".join(
                        part for part in (context_text, attached_file_context) if part
                    )
                context_label = (
                    "[Local memory, knowledge, and explicitly attached file context]"
                    if include_private_context or attached_file_context
                    else "[No local conversation history or retrieved documents are included]"
                )
                provider_privacy_instruction = (
                    "This is the explicitly selected online provider. Only include local context "
                    "if the user approved it for this session."
                    if provider_type == "online"
                    else f"This is the local {selected_local_runtime.name} provider."
                )
                code_task_instructions = (
                    f"""This is a code creation/modification task. Work only inside:
{active_code_workspace.root}
Create or update a relevant unit-test file in this project. Before each subtask checkpoint, run run_project_unit_tests and call checkpoint_code_task with the exact changed project-relative code and test paths. The checkpoint tool reruns tests and refuses commits if tests fail or no test file changed. At task end call finalize_code_task; if tests fail, diagnose and repair the failure before trying again. Do not run git commands manually. Never add a remote, push, or modify the original source project. Report the isolated path and checkpoint commit IDs."""
                    if active_code_workspace is not None
                    else ""
                )
                planning_mode_instruction = (
                    "This turn is planning-only. Before any SQLite planner write, "
                    "call read_table_schema for the destination table. Read the "
                    "reported column types and actual constraints, then tailor "
                    "plan content to those constraints; semantically summarize "
                    "only when a stated limit requires it. Do not use fixed "
                    "truncation or generic static summaries. Then use "
                    "create_task_plan to persist "
                    "a bounded, executable checklist in SQLite. Include "
                    "assumptions, constraints, proposed research questions, and "
                    "for each step dependencies, validation criteria, proof "
                    "points, risks, and relevant edge cases. When up-to-date external "
                    "facts are necessary, you may propose a focused web_search query. "
                    "Runtime checks connectivity and asks for explicit consent before "
                    "sending it; use only returned titles and URLs as citations. "
                    "Citations are saved by runtime as untrusted research references. "
                    "If research is declined/offline, mark the plan incomplete and "
                    "leave the questions unresolved. Do not modify files or run "
                    "commands. The plan remains inactive until the user approves it "
                    "with /tasks."
                    if planning_only
                    else ""
                )
                active_task_instruction = ""
                if include_private_context:
                    active_task_id = get_tool_context().task_id
                    active_task = (
                        memory.get_task(active_task_id) if active_task_id else None
                    )
                    if (
                        active_task
                        and active_task["status"] == "active"
                        and active_task["approved_revision"] == active_task["revision"]
                        and active_task["approved_digest"] == active_task["digest"]
                    ):

                        def _plan_field(step: dict, name: str) -> str:
                            value = step.get(name)
                            if isinstance(value, (list, tuple)):
                                return ", ".join(str(item) for item in value)
                            return str(value or "")

                        active_task_instruction = (
                            "Current user-approved SQLite task plan (reference, "
                            "not authorization beyond its recorded scope): "
                            f"task_id={active_task['task_id']}; "
                            f"revision={active_task['revision']}; "
                            f"type={active_task['task_type']}; "
                            f"goal={active_task['goal']}; "
                            f"assumptions={'; '.join(active_task.get('assumptions') or [])}; "
                            f"constraints={'; '.join(active_task.get('constraints') or [])}; "
                            "steps="
                            + "; ".join(
                                " ".join(
                                    part
                                    for part in (
                                        f"{step['step_id']} [{step['status']}] {_plan_field(step, 'description')}",
                                        (
                                            "deps=" + _plan_field(step, "dependencies")
                                            if step.get("dependencies")
                                            else ""
                                        ),
                                        (
                                            "validation="
                                            + _plan_field(step, "validation")
                                            if step.get("validation")
                                            else ""
                                        ),
                                        (
                                            "proof=" + _plan_field(step, "proof")
                                            if step.get("proof")
                                            else ""
                                        ),
                                        (
                                            "risks=" + _plan_field(step, "risks")
                                            if step.get("risks")
                                            else ""
                                        ),
                                        (
                                            "edge_cases="
                                            + _plan_field(step, "edge_cases")
                                            if step.get("edge_cases")
                                            else ""
                                        ),
                                    )
                                    if part
                                )
                                for step in active_task["steps"]
                            )
                            + ". Persist truthful progress using update_todo_step "
                            "at step boundaries. Model-reported completion is not "
                            "verification; the user/runtime must verify evidence. "
                            "If blocked by a failed tool, unavailable source, or "
                            "invalid assumption, explain what failed, state what "
                            "remains unverified, and ask whether the user wants to "
                            "revise the plan or try a different strategy; never "
                            "silently change approved scope."
                        )
                tool_limits_instruction = (
                    f"At most {tool_iteration_limit} tool rounds and "
                    f"{tool_call_limit} tool calls. "
                    if tool_iteration_limit is not None and tool_call_limit is not None
                    else "Tool rounds and total tool-call count are uncapped. "
                )
                _project_instructions = ""
                if PROJECT_INSTRUCTIONS_FILE:
                    try:
                        _instructions_path = (
                            pathlib.Path(SandboxManager.get_root())
                            / PROJECT_INSTRUCTIONS_FILE
                        )
                        if _instructions_path.is_file():
                            _project_instructions = _instructions_path.read_text(
                                encoding="utf-8"
                            ).strip()
                    except (OSError, ValueError):
                        _project_instructions = ""
                _project_instructions_block = (
                    f"\nProject instructions (from {PROJECT_INSTRUCTIONS_FILE}):\n"
                    f"{_project_instructions}\n"
                    if _project_instructions
                    else ""
                )
                system_prompts = build_system_messages(
                    base_prompt=SYSTEM_PROMPT,
                    tool_limits_instruction=tool_limits_instruction,
                    deadline_seconds=MAX_TASK_SECONDS,
                    max_tool_output_chars=MAX_TOOL_OUTPUT_CHARS,
                    privacy_instruction=provider_privacy_instruction,
                    permission_mode=turn_permission_mode,
                    datetime_context=_current_datetime_context(),
                    planning_mode_instruction=planning_mode_instruction,
                    active_task_instruction=active_task_instruction,
                    code_task_instructions=code_task_instructions,
                    project_instructions_block=_project_instructions_block,
                )
                if active_skill:
                    system_prompts.append(
                        SystemMessage(content=active_skill.system_prompt)
                    )
                effective_context = (
                    _effective_context_window(capabilities)
                    if provider_type == "local"
                    else MAX_CONTEXT_TOKENS
                )
                prompt_budget, output_budget = calculate_prompt_budgets(
                    system_prompts + [SystemMessage(content=context_summary)],
                    effective_context,
                    MAX_OUTPUT_TOKENS,
                )
                _needs_compact = needs_history_compaction(
                    chat_history,
                    messages_tokens([HumanMessage(content=user_input)]),
                    prompt_budget,
                    CONTEXT_COMPACTION_THRESHOLD,
                    MAX_HISTORY_MESSAGES,
                )
                if context_engine_is_enabled():
                    try:
                        _needs_compact = ContextEngine.from_config().needs_compaction(
                            chat_history,
                            messages_tokens([HumanMessage(content=user_input)]),
                            prompt_budget,
                            CONTEXT_COMPACTION_THRESHOLD,
                            MAX_HISTORY_MESSAGES,
                        )
                    except Exception:  # noqa: BLE001 - optional best-effort context feature
                        pass
                if _needs_compact:
                    try:
                        chat_history, context_summary = await compact_history(
                            llm,
                            chat_history,
                            context_summary,
                            keep_recent=CONTEXT_KEEP_RECENT_MESSAGES,
                            max_chars=CONTEXT_SUMMARY_CHARS,
                            source_token_limit=max(200, prompt_budget // 2),
                            notify=lambda text, sui=session_ui: sui.print(
                                f"[info][Context][/info] {text}"
                            ),
                            model_name=selected_model,
                        )
                    except (ValueError, RuntimeError, OSError) as exc:
                        console.print(
                            f"[warning][Context] Compaction failed ({type(exc).__name__}); trimming recent history.[/warning]"
                        )
                        chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
                summary_message = summary_system_message(context_summary)
                if summary_message is not None:
                    system_prompts.append(summary_message)
                    prompt_budget, output_budget = calculate_prompt_budgets(
                        system_prompts,
                        effective_context,
                        MAX_OUTPUT_TOKENS,
                    )
                if prompt_budget < 1:
                    console.print(
                        "[error]The system instructions exceed this model's configured "
                        "context window. Reduce the active skill/system prompt or "
                        "increase agent.max_context_tokens.[/error]"
                    )
                    continue
                enriched_input = _build_bounded_user_input(
                    context_label,
                    episodic_context,
                    context_text,
                    user_input,
                    budget=prompt_budget,
                )
                enriched_input = f_integration.trim_context(
                    enriched_input, prompt_budget
                )
                prompt_history = trim_history_to_context_budget(
                    chat_history,
                    enriched_input,
                    max_context_tokens=prompt_budget,
                )
                context_stats = estimate_context_window(
                    prompt_history,
                    enriched_input,
                    max_context=effective_context - output_budget,
                    system_prompts=system_prompts,
                )
                usage = _print_context_state(
                    context_stats, output_budget, selected_model, turn_permission_mode
                )
                # Store a structured context report for inspection (Phase 6).
                _report = build_runtime_context_report(
                    effective_context, prompt_history, system_prompts, enriched_input, user_input
                )
                if _report is not None:
                    _CONTEXT_REPORT_STATE["value"] = _report
                messages = (
                    system_prompts
                    + prompt_history[-MAX_HISTORY_MESSAGES:]
                    + [HumanMessage(content=enriched_input)]
                )
                turn_id = uuid.uuid4().hex
                _f_state = f_integration.begin_turn(
                    turn_id, user_input, effective_context, non_interactive
                )
                _tool_context = get_tool_context()
                _tool_context.turn_id = turn_id
                set_tool_context(_tool_context)
                _turn_control, _steering_channel = begin_steering_turn(interactive=interactive)
                start_time = time.monotonic()
                chat_transcript = ChatTranscript(console=console)
                chat_transcript.add_message("user", user_input, echo=False)
                deadline = start_time + MAX_TASK_SECONDS
                with f_integration.guard_turn_start(_f_state):
                    (
                        llm,
                        llm_with_tools,
                        selected_model,
                        capabilities,
                        response,
                        selected_local_runtime,
                    ) = await _invoke_initial_model_with_fallback(
                        llm=llm,
                        llm_with_tools=llm_with_tools,
                        tools_list=tools_list,
                        messages=messages,
                        deadline=deadline,
                        selected_model=selected_model,
                        provider_type=provider_type,
                        selected_local_runtime=selected_local_runtime,
                        available_models=available_models,
                        capabilities=capabilities,
                        thinking_enabled=thinking_enabled,
                        thinking_effort=thinking_effort,
                        status_message=status_message,
                        usage=usage,
                        console=console,
                        memory=memory,
                        session_id=session_id,
                        interactive=interactive,
                        input_fn=console.input,
                        discover=_discover_available_local_runtimes,
                        first_local_turn=not _local_model_has_answered,
                        cancel_event=_turn_control.cancel_event,
                    )
                    _local_model_has_answered = (
                        _local_model_has_answered or provider_type == "local"
                    )
                    if selected_local_runtime is not None:
                        available_models = list(
                            selected_local_runtime.models or available_models
                        )

                async def _run_agent_turn(
                    response, session_ui=session_ui, _f_state=_f_state
                ) -> Any:
                    nonlocal tools_list, llm_with_tools
                    iteration = 0
                    tool_call_count = 0
                    while getattr(response, "tool_calls", None) and (
                        tool_iteration_limit is None or iteration < tool_iteration_limit  # noqa: B023
                    ):
                        iteration += 1
                        _iter_start = f_integration.record_iteration(
                            _f_state,  # noqa: B023
                            iteration,
                        )
                        if (
                            iteration == 1
                            and response.content
                            and not isinstance(response, AIMessageChunk)
                        ):
                            plan_text = str(response.content).strip()
                            if plan_text:
                                session_ui.print_markup(
                                    f"[info][Plan][/info] {plan_text}"
                                )
                        iteration_label = (
                            f"{iteration} of {tool_iteration_limit}"  # noqa: B023
                            if tool_iteration_limit is not None  # noqa: B023
                            else str(iteration)
                        )
                        session_ui.print_markup(
                            f"[info][Agent Loop] Plan/act/verify step "
                            f"{iteration_label}...[/info]"
                        )
                        for tool_call in response.tool_calls:
                            _display_tool_request(tool_call)
                        messages.append(response)  # noqa: B023
                        tool_tasks = []
                        blocked_tool_messages = []
                        declined_media_calls = []
                        authorized_planning_searches = {}
                        state = _PreflightState(
                            tool_tasks=[],
                            blocked_tool_messages=[],
                            declined_media_calls=[],
                            planning_tool_seen=False,
                            authorized_planning_searches={},
                            task_action_seen=False,
                            tool_call_count=tool_call_count,
                        )
                        await _preflight_tool_calls(
                            response=response,
                            state=state,
                            network_consent_state=network_consent_state,
                            provider_type=provider_type,  # noqa: B023
                            capabilities=capabilities,  # noqa: B023
                            planning_only=planning_only,  # noqa: B023
                            turn_permission_mode=turn_permission_mode,  # noqa: B023
                            tool_call_limit=tool_call_limit,  # noqa: B023
                            start_time=start_time,  # noqa: B023
                            deadline=deadline,  # noqa: B023
                            memory=memory,
                            planning_research_citations=planning_research_citations,  # noqa: B023
                            console=console,
                        )
                        tool_tasks = state.tool_tasks
                        blocked_tool_messages = state.blocked_tool_messages
                        declined_media_calls = state.declined_media_calls
                        authorized_planning_searches = (
                            state.authorized_planning_searches
                        )
                        tool_call_count = state.tool_call_count
                        if _steering_channel: _steering_channel.pause()
                        response, stop = await _execute_tool_batch(
                            response=response,
                            session_id=session_id,
                            tool_tasks=tool_tasks,
                            blocked_tool_messages=blocked_tool_messages,
                            declined_media_calls=declined_media_calls,
                            authorized_planning_searches=authorized_planning_searches,
                            planning_research_citations=planning_research_citations,  # noqa: B023
                            messages=messages,  # noqa: B023
                            provider_type=provider_type,  # noqa: B023
                            planning_only=planning_only,  # noqa: B023
                            deadline=deadline,  # noqa: B023
                            console=console,
                        )
                        if _steering_channel: _steering_channel.resume()
                        f_integration.finish_iteration(
                            _f_state,  # noqa: B023
                            iteration,
                            _iter_start,
                            "ok",
                        )
                        if getattr(state, "capability_approved", False):
                            tools_list, llm_with_tools = f_integration.rebuild_tools(
                                base_tools_list,  # noqa: B023
                                user_input,  # noqa: B023
                                llm,  # noqa: B023
                            )
                        if stop or _turn_control.stop_requested:
                            break
                        for steer in _turn_control.drain():
                            messages.append(  # noqa: B023
                                HumanMessage(content=steer.prompt_text)
                            )
                        await compact_loop_messages(
                            llm,  # noqa: B023
                            messages,  # noqa: B023
                            context_stats["max"],  # noqa: B023
                            threshold=CONTEXT_COMPACTION_THRESHOLD,
                            max_chars=CONTEXT_SUMMARY_CHARS,
                            notify=lambda text, sui=session_ui: sui.print(
                                f"[info][Context][/info] {text}"
                            ),
                            model_name=selected_model,  # noqa: B023
                        )
                        usage = (messages_tokens(messages), context_stats["max"])  # noqa: B023
                        _plan_summary = f_integration.get_plan_summary(_f_state)  # noqa: B023
                        # Only inject real (non-synthetic) plans into the model context.
                        if _plan_summary and not f_integration.is_synthetic_plan(
                            _f_state
                        ):  # noqa: B023
                            messages.append(  # noqa: B023
                                SystemMessage(content=f"[Task plan]\n{_plan_summary}")
                            )
                        try:
                            response = await _invoke_with_status(
                                llm_with_tools,  # noqa: B023
                                messages,  # noqa: B023
                                deadline,  # noqa: B023
                                model_name=selected_model,  # noqa: B023
                                status_message="Processing tool outputs...",
                                local_model=provider_type == "local",  # noqa: B023
                                first_local_turn=False,  # noqa: B023
                                thinking_enabled=thinking_enabled,  # noqa: B023
                                context_usage=usage,
                                cancel_event=_turn_control.cancel_event,
                            )
                        except Exception as exc:  # noqa: BLE001 - wraps any model error after tool execution; re-raised as RuntimeError
                            if provider_type == "online":  # noqa: B023
                                raise RuntimeError(
                                    "Online model failed after tool execution. "
                                    + _online_request_failure(exc)
                                ) from None
                            raise RuntimeError(
                                f"Model failed after tool execution: {exc}"
                            ) from exc
                    if getattr(response, "tool_calls", None):
                        messages.append(response)  # noqa: B023
                        limit_message = (
                            "Error: The configured tool-call limit was reached. "
                            "The requested task is not verified complete."
                            if enforce_tool_call_limits  # noqa: B023
                            else (
                                "Error: The request time budget was reached. "
                                "The requested task is not verified complete."
                            )
                        )
                        messages.extend(  # noqa: B023
                            ToolMessage(
                                content=limit_message,
                                tool_call_id=call.get("id"),
                            )
                            for call in response.tool_calls
                        )
                        if deadline <= time.monotonic():  # noqa: B023
                            response = AIMessage(
                                content="The task execution budget was exhausted. "
                                "Some work may be complete, but the requested outcome "
                                "was not verified."
                            )
                        else:
                            response = await _invoke_with_status(
                                llm,  # noqa: B023
                                messages  # noqa: B023
                                + [
                                    SystemMessage(
                                        content="Stop requesting tools. Report what was completed, "
                                        "what remains unverified, and that the execution budget ended."
                                    )
                                ],
                                deadline,  # noqa: B023
                                model_name=selected_model,  # noqa: B023
                                status_message="Finalizing response...",
                                local_model=provider_type == "local",  # noqa: B023
                                first_local_turn=False,  # noqa: B023
                                thinking_enabled=thinking_enabled,  # noqa: B023
                                cancel_event=_turn_control.cancel_event,
                            )
                    if (
                        not getattr(response, "tool_calls", None)
                        and not message_text(response).strip()
                        and any(isinstance(m, ToolMessage) for m in messages)  # noqa: B023
                        and deadline > time.monotonic()  # noqa: B023
                    ):
                        session_ui.print_markup(
                            "[info][Context][/info] The model gave no answer after the tool "
                            "results; asking it once to answer from them..."
                        )
                        response = await _invoke_with_status(
                            llm,  # noqa: B023
                            messages  # noqa: B023
                            + [
                                SystemMessage(
                                    content="Answer the user's request now in plain text, "
                                    "using the tool results above."
                                )
                            ],
                            deadline,  # noqa: B023
                            model_name=selected_model,  # noqa: B023
                            status_message="Finalizing response...",
                            local_model=provider_type == "local",  # noqa: B023
                            first_local_turn=False,  # noqa: B023
                            thinking_enabled=thinking_enabled,  # noqa: B023
                            cancel_event=_turn_control.cancel_event,
                        )
                    return response

                session_ui.bind(
                    model=selected_model,
                    permission=turn_permission_mode,
                    output_tokens=output_budget,
                )
                session_ui.set_context(*usage)
                session_ui.turn_started(user_input)
                with contextlib.suppress(Exception):  # noqa: BLE001 - optional best-effort context feature
                    reset_internet_check_cache()
                with contextlib.suppress(Exception):  # noqa: BLE001 - optional best-effort context feature
                    reset_internet_check_for_turn()
                try:
                    response = await _run_agent_turn(response)
                finally:
                    if _steering_channel: _steering_channel.stop()
                    _turn_control.finish_turn()
                    f_integration.finish_turn(_f_state)
                    session_ui.set_generation(
                        getattr(response, "generation_stats", None)
                    )
                if SHOW_TURN_STATS:
                    session_ui.turn_finished()
                else:
                    session_ui._footer.turn_finished()
                content = response.content
                full_output_content = (
                    _visible_chunk_text(content)
                    if isinstance(content, list)
                    else str(content or "")
                )
                if not isinstance(response, AIMessageChunk):
                    chat_transcript.add_message(
                        "assistant", full_output_content, echo=False
                    )
                    for i in range(0, len(full_output_content), 120):
                        chat_transcript.stream_chunk(
                            "assistant", full_output_content[i : i + 120]
                        )
                    if full_output_content:
                        chat_transcript.stream_chunk("assistant", "\n")
                elif not full_output_content and not getattr(
                    response, "tool_calls", None
                ):
                    session_ui.print_markup(
                        "[warning]The model returned no visible response text.[/warning]"
                    )
                if retrieved_citations:
                    session_ui.print_markup(
                        "[bold info][Retrieved Sources][/bold info]"
                    )
                    for citation in retrieved_citations:
                        session_ui.print_markup(f"  - {citation}")
                # Per-turn summary is emitted by session_ui.turn_finished() above.
                if active_code_workspace is not None:
                    verification = cleanup_code_task(finalize=True)
                    session_ui.print_markup(
                        f"[info][Code Task Verification][/info] {verification}"
                    )
                    full_output_content += (
                        f"\n\n[Code task verification]\n{verification}"
                    )
                memory.save_message(session_id, "human", user_input, turn_id=turn_id)
                memory.save_message(
                    session_id, "ai", full_output_content, turn_id=turn_id
                )
                _tool_context = get_tool_context()
                _tool_context.turn_id = None
                set_tool_context(_tool_context)
                chat_history.append(HumanMessage(content=user_input))
                stored_output = (
                    full_output_content[: MAX_CONTEXT_TOKENS * 4]
                    + "\n[truncated for memory]"
                    if len(full_output_content) > MAX_CONTEXT_TOKENS * 4
                    else full_output_content
                )
                chat_history.append(AIMessage(content=stored_output))
                if non_interactive:
                    _LAST_HEADLESS_RESPONSE = full_output_content
                session_summary_saved = False
                _SESSION_STATS.turns += 1
                chat_history = chat_history[-MAX_HISTORY_MESSAGES:]
        except (KeyboardInterrupt, asyncio.CancelledError):
            cleanup_code_task(finalize=False)
            _acknowledge_interrupt()
            if non_interactive:
                return
            console.print(
                "[warning]Turn interrupted; session kept. Press Ctrl+C twice to exit.[/warning]"
            )
            continue
        except RuntimeError as rte:
            cleanup_code_task(finalize=False)
            log_event(
                RUN_LOGGER,
                "agent.turn.failed",
                level=logging.ERROR,
                session_id=session_id,
                turn_index=_SESSION_STATS.turns + 1,
                error_type=type(rte).__name__,
                exc_info=True,
            )
            console.print(f"\n[error][Exception] {rte!s}[/error]")
            console.print(
                "[error][Action Required] Review the error and local model/tool configuration.[/error]\n"
            )
            continue
        except (ValueError, OSError) as ex:
            cleanup_code_task(finalize=False)
            log_event(
                RUN_LOGGER,
                "agent.turn.unexpected_failure",
                level=logging.ERROR,
                session_id=session_id,
                turn_index=_SESSION_STATS.turns + 1,
                error_type=type(ex).__name__,
                exc_info=True,
            )
            console.print(
                f"\n[error][Exception] Unexpected runtime error: {ex!s}[/error]\n"
            )
            continue
        if model_selection_failed: continue
        if (_mission_next := _mission_runtime.after_turn(full_output_content, response)):
            _pending_user_input = _mission_next; continue
        break


async def run_agent_cli_async(
    *,
    verbose_startup: bool = False,
    resume_session: str | None = None,
    continue_session: bool = False,
    prompt: str | None = None,
    output: str = "text",
    permission_mode: str | None = None,
    non_interactive: bool = False,
    interactive: bool = True,
    input_provider=None,
):
    """Own runtime resources and restore dynamically registered global tools."""
    global _ACTIVE_MEMORY, _TASK_LEASE_HEARTBEAT
    original_mcp_tools = set(MCP_TOOL_NAMES)
    original_workspace_root = SandboxManager.get_root_or_none()
    try:
        await _run_agent_cli_session(
            verbose_startup=verbose_startup,
            resume_session=resume_session,
            continue_session=continue_session,
            prompt=prompt,
            output=output,
            permission_mode=permission_mode,
            non_interactive=non_interactive,
            interactive=interactive,
            input_provider=input_provider,
        )
    finally:
        global _ACTIVE_RAG_TASK
        if _ACTIVE_RAG_TASK is not None and not _ACTIVE_RAG_TASK.done():
            _ACTIVE_RAG_TASK.cancel()
            await asyncio.gather(_ACTIVE_RAG_TASK, return_exceptions=True)
        _ACTIVE_RAG_TASK = None
        if _TASK_LEASE_HEARTBEAT is not None:
            _TASK_LEASE_HEARTBEAT.cancel()
            await asyncio.gather(_TASK_LEASE_HEARTBEAT, return_exceptions=True)
            _TASK_LEASE_HEARTBEAT = None
        if _ACTIVE_MEMORY is not None:
            try:
                active_task_id = get_tool_context().task_id
                if active_task_id:
                    try:
                        task = _ACTIVE_MEMORY.get_task(active_task_id)
                        if task and task["status"] == "active":
                            _ACTIVE_MEMORY.set_task_status(
                                active_task_id,
                                "interrupted",
                                owner_id=get_tool_context().runtime_owner_id,
                            )
                    except (ValueError, sqlite3.Error, OSError) as exc:
                        console.print(
                            "[warning][Shutdown warning] Could not persist the "
                            f"task interruption: {exc}[/warning]"
                        )
                _ACTIVE_MEMORY.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise while closing memory
                log_event(
                    RUN_LOGGER,
                    "memory.shutdown_failed",
                    level=logging.ERROR,
                    error_type=type(exc).__name__,
                    exc_info=True,
                )
                console.print(
                    f"[warning][Shutdown warning] Could not close SQLite memory: {exc}[/warning]"
                )
            finally:
                _ACTIVE_MEMORY = None
        for tool_name in set(MCP_TOOL_NAMES) - original_mcp_tools:
            AVAILABLE_TOOLS.pop(tool_name, None)
            MCP_TOOL_NAMES.discard(tool_name)
            MCP_TOOL_SOURCES.pop(tool_name, None)
            MCP_TOOL_TRANSPORTS.pop(tool_name, None)
        for tool_name in (
            "run_project_unit_tests",
            "checkpoint_code_task",
            "finalize_code_task",
        ):
            AVAILABLE_TOOLS.pop(tool_name, None)
        _tool_context = get_tool_context()
        _tool_context.skill_session_id = None
        _tool_context.task_session_id = None
        _tool_context.task_id = None
        _tool_context.code_task = None
        set_tool_context(_tool_context)
        if original_workspace_root is not None:
            SandboxManager.set_root(str(original_workspace_root))
        await close_outbound_http_clients()
        close_mcp_sandbox_dirs()
        clear_captured_images()


def run_agent_cli(
    *,
    verbose_startup: bool = False,
    resume_session: str | None = None,
    continue_session: bool = False,
    prompt: str | None = None,
    output: str = "text",
    permission_mode: str | None = None,
    non_interactive: bool = False,
    interactive: bool = True,
    input_provider=None,
):
    asyncio.run(
        run_agent_cli_async(
            verbose_startup=verbose_startup,
            resume_session=resume_session,
            continue_session=continue_session,
            prompt=prompt,
            output=output,
            permission_mode=permission_mode,
            non_interactive=non_interactive,
            interactive=interactive,
            input_provider=input_provider,
        )
    )


_LAZY_TOOL_NAMES = {
    "AVAILABLE_TOOLS",
    "LOCAL_MEDIA_TOOL_NAMES",
    "LOCAL_ONLY_NETWORK_TOOL_NAMES",
    "MCP_TOOL_NAMES",
    "MCP_TOOL_SOURCES",
    "MCP_TOOL_TRANSPORTS",
    "NETWORK_TOOL_NAMES",
    "approved_tool_invocation",
    "clear_schema_reads",
    "close_mcp_sandbox_dirs",
    "close_outbound_http_clients",
    "describe_tool_catalog",
    "get_tool_context",
    "has_internet_connection",
    "load_configured_mcp_tools",
    "ollama_client_kwargs",
    "set_tool_context",
    "approve_local_capture",
}
_LAZY_MEDIA_NAMES = {"approve_local_capture", "clear_captured_images"}
_LAZY_INTERACTIVE_NAMES = {
    "_format_interactive_help",
    "handle_session_command",
    "include_file_context",
    "include_folder_context",
    "load_custom_slash_commands",
    "prompt_user_input",
}
_LAZY_MODEL_SELECTION_NAMES = {
    "_discover_available_local_runtimes",
    "_effective_context_window",
    "_make_selected_local_chat_model",
    "inspect_openai_compatible_model_capabilities",
    "select_online_model",
}
_LAZY_TASK_COMMAND_NAMES = {"_run_task_management"}
_LAZY_TOOL_DISPLAY_NAMES = {
    "_append_captured_images_from_tool_results",
    "_display_tool_error",
    "_display_tool_request",
    "_display_tool_result",
    "_local_image_input_error",
    "_redact_sensitive_values",
    "_redact_tool_arguments",
    "_tool_log_redactions",
    "_tool_result_is_error",
}


def __getattr__(name: str):
    if name in _LAZY_TOOL_NAMES:
        import tools as _tools

        attr = getattr(_tools, name)
        globals()[name] = attr
        return attr
    if name in _LAZY_MEDIA_NAMES:
        from tools import media as _media

        attr = getattr(_media, name)
        globals()[name] = attr
        return attr
    if name == "_format_interactive_help":
        from . import interactive as _interactive

        attr = _interactive.format_help
        globals()[name] = attr
        return attr
    if name in _LAZY_INTERACTIVE_NAMES:
        from . import interactive as _interactive

        attr = getattr(_interactive, name)
        globals()[name] = attr
        return attr
    if name in _LAZY_MODEL_SELECTION_NAMES:
        from . import model_selection as _model_selection

        attr = getattr(_model_selection, name)
        globals()[name] = attr
        return attr
    if name in _LAZY_TASK_COMMAND_NAMES:
        from . import task_commands as _task_commands

        attr = getattr(_task_commands, name)
        globals()[name] = attr
        return attr
    if name in _LAZY_TOOL_DISPLAY_NAMES:
        from . import tool_display as _tool_display

        attr = getattr(_tool_display, name)
        globals()[name] = attr
        return attr
    if name == "_run_memory_maintenance":
        from . import memory_context as _memory_context

        attr = _memory_context._run_memory_maintenance
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


install_lazy_globals(globals())
