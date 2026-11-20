import contextvars
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import Any

from langchain_core.tools import tool
from rich.console import Console

from ..config import (
    DEFAULT_DB_PATH,
    INTERNET_CHECK_HOST,
    INTERNET_CHECK_PORT,
    INTERNET_CHECK_TIMEOUT,
    MCP_AUTO_APPROVE_TOOLS,
    SKILL_DESCRIPTION_PREVIEW_CHARS,
    SKILLS_FOLDER_DEFAULT,
)
from . import _sqlite_state
from .media import (
    capture_webcam_image,
    list_microphone_devices,
    load_workspace_image,
    load_workspace_video,
    record_microphone_audio,
    transcribe_workspace_audio,
)
from .network import (
    _get_limited_response as _get_limited_response,
)
from .network import (
    _network_slots as _network_slots,
)
from .network import (
    _outbound_http_clients as _outbound_http_clients,
)
from .network import (
    _public_only_async_client as _public_only_async_client,
)
from .network import (
    _PublicOnlyAsyncHTTPTransport as _PublicOnlyAsyncHTTPTransport,
)
from .network import (
    _PublicOnlyNetworkBackend as _PublicOnlyNetworkBackend,
)
from .network import (
    _PublicOnlySyncHTTPTransport as _PublicOnlySyncHTTPTransport,
)
from .network import (
    _PublicOnlySyncNetworkBackend as _PublicOnlySyncNetworkBackend,
)
from .network import (
    _resolve_validated_peer as _resolve_validated_peer,
)
from .network import (
    close_outbound_http_clients as close_outbound_http_clients,
)
from .network import (
    ollama_client_kwargs as ollama_client_kwargs,
)
from .network import (
    ollama_langchain_client_kwargs as ollama_langchain_client_kwargs,
)
from .network import (
    public_only_sync_client as public_only_sync_client,
)
from .network import (
    search_duckduckgo,
    validate_explicit_service_url,
)
from .network import (
    track_ollama_http_clients as track_ollama_http_clients,
)
from .network import (
    validate_outbound_url as validate_outbound_url,
)
from .schemas import (
    CreateSkillInput,
)

console = Console()
_search_duckduckgo = search_duckduckgo
_validate_explicit_service_url = validate_explicit_service_url

_sqlite_lock = _sqlite_state.SQLITE_LOCK
_tool_invocation_approved = contextvars.ContextVar(
    "private_agent_tool_invocation_approved", default=False
)


@contextmanager
def approved_tool_invocation():
    """Mark the current model tool call as approved by the session policy."""
    token = _tool_invocation_approved.set(True)
    try:
        yield
    finally:
        _tool_invocation_approved.reset(token)


@dataclass
class ToolContext:
    """Snapshot of the mutable runtime state tools depend on (B3 step 1)."""

    db_path: str
    session_id: str | None = None
    turn_id: str | None = None
    task_session_id: str | None = None
    task_id: str | None = None
    skill_directory: str | None = None
    skill_registry: dict | None = None
    skill_session_id: str | None = None
    code_task: Any = None
    runtime_owner_id: str | None = None
    schema_read_tables: set | None = None
    interactive: bool = True


_TOOL_CONTEXT = ToolContext(
    db_path=os.path.abspath(DEFAULT_DB_PATH),
    skill_directory=(
        os.path.abspath(os.path.expanduser(SKILLS_FOLDER_DEFAULT))
        if SKILLS_FOLDER_DEFAULT
        else None
    ),
    runtime_owner_id=uuid.uuid4().hex,
    schema_read_tables=set(),
)


def get_tool_context() -> ToolContext:
    """Return the process-local tool context (single source of truth)."""
    return _TOOL_CONTEXT


def set_tool_context(context: ToolContext) -> None:
    """Copy a context object into the process-local tool context."""
    _TOOL_CONTEXT.db_path = context.db_path
    _TOOL_CONTEXT.session_id = context.session_id
    _TOOL_CONTEXT.turn_id = context.turn_id
    _TOOL_CONTEXT.task_session_id = context.task_session_id
    _TOOL_CONTEXT.task_id = context.task_id
    _TOOL_CONTEXT.skill_directory = context.skill_directory
    _TOOL_CONTEXT.skill_registry = context.skill_registry
    _TOOL_CONTEXT.skill_session_id = context.skill_session_id
    _TOOL_CONTEXT.code_task = context.code_task
    _TOOL_CONTEXT.interactive = context.interactive
    if context.runtime_owner_id is not None:
        _TOOL_CONTEXT.runtime_owner_id = context.runtime_owner_id
    if context.schema_read_tables is not None:
        _TOOL_CONTEXT.schema_read_tables = context.schema_read_tables


# --- NETWORK CONNECTIVITY UTILITY ---
def check_internet_connection(
    host: str = INTERNET_CHECK_HOST,
    port: int = INTERNET_CHECK_PORT,
    timeout: float = INTERNET_CHECK_TIMEOUT,
) -> bool:
    """Quick socket probe to check active TCP-level internet reachability."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return True
    except OSError:
        return False


NETWORK_TOOL_NAMES = frozenset(
    {
        "web_search",
        "fetch_webpage",
        "download_web_file",
        "fetch_current_location",
        "get_weather",
    }
)
# Reveal the machine's own location, so they are limited to local sessions.
LOCAL_ONLY_NETWORK_TOOL_NAMES = frozenset({"fetch_current_location"})
LOCAL_MEDIA_TOOL_NAMES = frozenset(
    {
        "capture_webcam_image",
        "load_workspace_image",
        "load_workspace_video",
        "list_microphone_devices",
        "record_microphone_audio",
        "transcribe_workspace_audio",
    }
)
MCP_TOOL_NAMES = set()
MCP_TOOL_SOURCES = {}
MCP_TOOL_TRANSPORTS = {}


def clear_schema_reads() -> None:
    """Require the model to inspect write-target schemas for each user turn."""
    _TOOL_CONTEXT.schema_read_tables.clear()


def describe_tool_catalog() -> list[dict[str, str]]:
    """Return a user-facing inventory of each tool's origin and approval boundary."""
    catalog = []
    auto_approved_names = (
        {str(name) for name in MCP_AUTO_APPROVE_TOOLS}
        if isinstance(MCP_AUTO_APPROVE_TOOLS, (set, list, tuple))
        else set()
    )
    for name, tool_object in sorted(AVAILABLE_TOOLS.items()):
        is_mcp = name in MCP_TOOL_NAMES
        origin = (
            MCP_TOOL_SOURCES.get(name, "configured MCP server")
            if is_mcp
            else "built-in"
        )
        if is_mcp:
            permission = (
                "auto-approved by configuration"
                if name in auto_approved_names
                else "approval required"
            )
        elif name in NETWORK_TOOL_NAMES:
            permission = "network consent required"
        elif name == "create_skill":
            permission = "interactive approval required before creating a skill"
        elif name in LOCAL_MEDIA_TOOL_NAMES:
            permission = (
                "local device metadata only"
                if name == "list_microphone_devices"
                else "explicit per-use local media approval"
            )
        elif name == "get_local_datetime":
            permission = "no approval required; reads local system clock"
        elif name in {
            "create_task_plan",
            "inspect_task_plan",
            "update_todo_step",
            "revise_task_plan",
            "read_table_schema",
        }:
            permission = (
                "read-only configured SQLite schema inspection"
                if name == "read_table_schema"
                else "SQLite task planning; plan execution requires user approval"
            )
        elif name in {
            "read_chat_history_from_sqlite",
            "search_chat_history_in_sqlite",
            "inspect_task_plan",
            "search_workspace_files",
            "search_workspace_text",
            "search_workspace_symbols",
            "inspect_workspace_git",
            "preview_workspace_patch",
        }:
            permission = (
                "no approval required; reads local SQLite history"
                if name
                in {
                    "read_chat_history_from_sqlite",
                    "search_chat_history_in_sqlite",
                    "inspect_task_plan",
                }
                else (
                    "read-only Git inspection"
                    if name == "inspect_workspace_git"
                    else "workspace-bounded read-only search"
                )
            )
        elif name in {
            "run_shell_command",
            "delete_chat_history_from_sqlite",
            "delete_chat_history_entry_from_sqlite",
        }:
            permission = "interactive approval for shell/deletion/network actions"
        else:
            permission = "workspace path validation"

        if name in NETWORK_TOOL_NAMES:
            effects = "network"
        elif name in LOCAL_MEDIA_TOOL_NAMES:
            effects = {
                "list_microphone_devices": "local microphone device metadata",
                "load_workspace_image": "reads an explicitly approved workspace image",
                "load_workspace_video": "samples an explicitly approved workspace video",
                "transcribe_workspace_audio": (
                    "transcribes an explicitly approved workspace audio file locally"
                ),
            }.get(name, "may access a local media device")
        elif name in {
            "edit_local_file",
            "apply_workspace_patch",
            "rename_workspace_file",
            "delete_workspace_file",
            "run_shell_command",
            "download_web_file",
            "delete_chat_history_from_sqlite",
            "delete_chat_history_entry_from_sqlite",
            "create_skill",
            "create_task_plan",
            "revise_task_plan",
            "update_todo_step",
            "read_table_schema",
        }:
            effects = (
                "may modify local SQLite task state"
                if name
                in {
                    "create_task_plan",
                    "revise_task_plan",
                    "update_todo_step",
                }
                else "may modify local state"
            )
        elif name in {
            "read_chat_history_from_sqlite",
            "search_chat_history_in_sqlite",
            "search_workspace_files",
            "search_workspace_text",
            "search_workspace_symbols",
            "inspect_workspace_git",
            "read_table_schema",
        }:
            effects = (
                "read-only configured SQLite schema"
                if name == "read_table_schema"
                else "read-only/local"
            )
        elif is_mcp:
            effects = (
                "network-capable; server-defined effects"
                if MCP_TOOL_TRANSPORTS.get(name) != "stdio"
                else "server-defined; inspect trusted MCP server"
            )
        else:
            effects = "read-only/local"
        catalog.append(
            {
                "name": name,
                "origin": origin,
                "permission": permission,
                "effects": effects,
                "description": str(
                    getattr(tool_object, "description", "") or ""
                ).strip(),
            }
        )
    return catalog


_INTERNET_CHECK_CACHE_SECONDS = 30.0
_internet_check_lock = threading.Lock()
_internet_check_result: tuple[float, bool] | None = None


def reset_internet_check_cache() -> None:
    global _internet_check_result
    with _internet_check_lock:
        _internet_check_result = None


def has_internet_connection() -> bool:
    """Probe connectivity, reusing the last result for a short time.

    Offline probes wait for the full timeout, so repeated web calls would
    otherwise each pay it.
    """
    global _internet_check_result
    with _internet_check_lock:
        cached = _internet_check_result
        if cached is not None and time.monotonic() - cached[0] < (
            _INTERNET_CHECK_CACHE_SECONDS
        ):
            return cached[1]
        reachable = check_internet_connection(
            INTERNET_CHECK_HOST, INTERNET_CHECK_PORT, INTERNET_CHECK_TIMEOUT
        )
        _internet_check_result = (time.monotonic(), reachable)
        return reachable


def require_internet(func: Callable) -> Callable:
    """Decorator that validates active internet connection before executing a web-bound tool."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not has_internet_connection():
            return "Error: Network unavailable. Please check your internet connection and try again."
        return func(*args, **kwargs)

    return wrapper


# --- TOOL IMPLEMENTATIONS ---
@tool
def get_local_datetime() -> str:
    """Get the current local date, time, weekday, month, and year from this machine."""
    now = datetime.now().astimezone()
    return (
        f"Local date: {now:%A, }{now.day} {now:%B %Y}\n"
        f"Local time: {now:%H:%M:%S %Z} (UTC{now:%z})\n"
        f"Weekday: {now:%A}\n"
        f"Month: {now:%B}\n"
        f"Year: {now:%Y}"
    )


@tool(args_schema=CreateSkillInput)
def create_skill(name: str, description: str, instructions: str) -> str:
    """Create a reusable Markdown skill in the configured skills folder after user approval."""
    from ..skills import AgentSkill, create_skill_file

    try:
        context = get_tool_context()
        if context.skill_directory is None:
            return "Error creating skill: No skills directory is configured."
        if context.skill_session_id and "chat_history" not in (
            context.schema_read_tables or set()
        ):
            return (
                "Error creating skill: call read_table_schema for chat_history "
                "before creating a skill so its session registration follows the "
                "destination schema."
            )
        target = create_skill_file(
            context.skill_directory, name, description, instructions
        )
        skill_key = target.stem
        display_name = name.strip()
        skill = AgentSkill(
            name=display_name,
            description=(
                f"{description.strip()} "
                f"{instructions.strip()[:SKILL_DESCRIPTION_PREVIEW_CHARS]}"
            ),
            system_prompt=f"[Skill Markdown Profile: {display_name}]\n"
            f"{target.read_text(encoding='utf-8')}",
        )
        if isinstance(context.skill_registry, dict):
            context.skill_registry[skill_key] = skill
        if context.skill_session_id:
            from ..database import PersistentMemory

            with _sqlite_lock:
                memory = PersistentMemory(context.db_path)
                try:
                    memory.register_skill_file(
                        context.skill_session_id,
                        str(target),
                        description.strip(),
                        source_task_id=context.task_id,
                    )
                finally:
                    memory.close()
        return (
            f"Created skill '{display_name}' at {target}. "
            "It is available in this session and future runs."
        )
    except (OSError, ValueError) as exc:
        return f"Error creating skill: {exc}"


# Import after network helpers are defined; web.py depends on this shared policy layer.
from .filesystem import (
    apply_workspace_patch,
    commit_git_changes,
    create_git_branch,
    create_pull_request,
    delete_workspace_file,
    edit_local_file,
    inspect_workspace_git,
    list_directory,
    preview_workspace_patch,
    read_local_file,
    rename_workspace_file,
    search_workspace_files,
    search_workspace_symbols,
    search_workspace_text,
    stage_git_changes,
)
from .location import fetch_current_location, get_weather
from .memory import (
    delete_chat_history_entry_from_sqlite,
    delete_chat_history_from_sqlite,
    read_chat_history_from_sqlite,
    search_chat_history_in_sqlite,
)
from .planner import (
    create_task_plan,
    inspect_task_plan,
    read_table_schema,
    revise_task_plan,
    update_todo_step,
)
from .shell import (
    get_background_process_status,
    list_background_processes,
    run_shell_command,
    start_background_process,
    stop_background_process,
)
from .web import download_web_file, fetch_webpage, web_search

AVAILABLE_TOOLS = {
    "get_local_datetime": get_local_datetime,
    "read_local_file": read_local_file,
    "search_workspace_files": search_workspace_files,
    "search_workspace_text": search_workspace_text,
    "search_workspace_symbols": search_workspace_symbols,
    "inspect_workspace_git": inspect_workspace_git,
    "create_git_branch": create_git_branch,
    "stage_git_changes": stage_git_changes,
    "commit_git_changes": commit_git_changes,
    "create_pull_request": create_pull_request,
    "edit_local_file": edit_local_file,
    "preview_workspace_patch": preview_workspace_patch,
    "apply_workspace_patch": apply_workspace_patch,
    "rename_workspace_file": rename_workspace_file,
    "delete_workspace_file": delete_workspace_file,
    "run_shell_command": run_shell_command,
    "start_background_process": start_background_process,
    "get_background_process_status": get_background_process_status,
    "stop_background_process": stop_background_process,
    "list_background_processes": list_background_processes,
    "web_search": web_search,
    "list_directory": list_directory,
    "fetch_webpage": fetch_webpage,
    "download_web_file": download_web_file,
    "fetch_current_location": fetch_current_location,
    "get_weather": get_weather,
    "read_chat_history_from_sqlite": read_chat_history_from_sqlite,
    "delete_chat_history_from_sqlite": delete_chat_history_from_sqlite,
    "search_chat_history_in_sqlite": search_chat_history_in_sqlite,
    "delete_chat_history_entry_from_sqlite": delete_chat_history_entry_from_sqlite,
    "create_skill": create_skill,
    "create_task_plan": create_task_plan,
    "inspect_task_plan": inspect_task_plan,
    "read_table_schema": read_table_schema,
    "update_todo_step": update_todo_step,
    "revise_task_plan": revise_task_plan,
    "capture_webcam_image": capture_webcam_image,
    "load_workspace_image": load_workspace_image,
    "load_workspace_video": load_workspace_video,
    "list_microphone_devices": list_microphone_devices,
    "record_microphone_audio": record_microphone_audio,
    "transcribe_workspace_audio": transcribe_workspace_audio,
}


async def load_mcp_tools(*args, **kwargs) -> list:
    """
    Connects to an MCP server, retrieves available tools, and returns them as a list.
    Guarantees that an empty list ([]) is returned upon any connection error,
    timeout, or exception instead of a dictionary or None.
    """
    try:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        # Establishing connection parameters and session
        async with (
            stdio_client(*args, **kwargs) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            response = await session.list_tools()

            tools = []
            for tool in getattr(response, "tools", []):
                tools.append(tool)
            return tools
    except Exception as exc:  # noqa: BLE001 - MCP transports raise library-specific errors; return an empty list
        console.print(f"[red][MCP error] Unable to discover tools: {exc}[/red]")
        return []


from .mcp import (
    close_mcp_sandbox_dirs as close_mcp_sandbox_dirs,
)
from .mcp import (
    load_configured_mcp_tools as load_configured_mcp_tools,
)
