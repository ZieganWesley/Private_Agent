"""Capability packs and turn router (Part F2)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# Plain chat binds only the Core pack (9.2, 13.7); other packs are added by
# the router or by ``enable_capability`` during a turn.
BASELINE_PACKS = {"Core"}

CORE_TOOLS = frozenset(
    {
        "get_local_datetime",
        "read_local_file",
        "search_workspace_text",
        "search_workspace_files",
        "search_workspace_symbols",
        "list_directory",
        "search_documents",
        "read_table_schema",
        "calculate",
        "run_subagent",
        "ask_user",
        "enable_capability",
    }
)

PACKS: dict[str, CapabilityPack] = {}


@dataclass
class CapabilityPack:
    name: str
    tools: frozenset[str]
    prompt_section: str = ""
    limits: dict[str, object] = field(default_factory=dict)


def _register(name: str, tools: Iterable[str], prompt: str = "") -> CapabilityPack:
    pack = CapabilityPack(name=name, tools=frozenset(tools), prompt_section=prompt)
    PACKS[name] = pack
    return pack


_register("Core", CORE_TOOLS, "Base rules, privacy, permission mode.")
_register(
    "Knowledge",
    {"search_documents"},
    "Cite sources; say when documents do not contain the answer.",
)
_register(
    "Research",
    {"web_search", "fetch_webpage", "download_web_file"},
    "Search, read, check gaps, cite only real sources.",
)
_register(
    "Coding",
    {
        "apply_workspace_patch",
        "preview_workspace_patch",
        "edit_local_file",
        "rename_workspace_file",
        "delete_workspace_file",
        "run_shell_command",
        "run_project_unit_tests",
        "inspect_workspace_git",
        "stage_git_changes",
        "commit_git_changes",
        "create_git_branch",
        "create_pull_request",
        "start_background_process",
        "stop_background_process",
        "list_background_processes",
        "get_background_process_status",
        "index_repository",
        "find_symbol",
        "find_definition",
        "find_references",
        "find_callers",
        "find_callees",
        "find_dependencies",
        "trace_execution_flow",
        "get_repository_map",
        "record_architecture_decision",
        "recall_architecture",
        "get_agent_metrics",
    },
    "Project instructions (AGENTS.md), test after edits.",
)
_register(
    "Helper",
    {
        "get_weather",
        "fetch_current_location",
        "create_pdf",
        "create_docx",
        "create_xlsx",
    },
    "Device and privacy rules.",
)
_register(
    "Planning",
    {"create_task_plan", "inspect_task_plan", "revise_task_plan", "update_todo_step"},
    "How to write a plan.",
)
_register("MCP", set(), "Server descriptions.")
_register(
    "Media",
    {
        "capture_webcam_image",
        "load_workspace_image",
        "load_workspace_video",
        "list_microphone_devices",
        "record_microphone_audio",
        "transcribe_workspace_audio",
    },
    "Media capture and transcription rules.",
)
_register(
    "Memory",
    {
        "read_chat_history_from_sqlite",
        "search_chat_history_in_sqlite",
        "delete_chat_history_from_sqlite",
        "delete_chat_history_entry_from_sqlite",
        "create_skill",
    },
    "Memory and skill rules.",
)


_URL_RE = re.compile(r"https?://\S+")


def _mentions(text: str, pattern: re.Pattern[str]) -> bool:
    return pattern.search(text) is not None


_CODING_RE = re.compile(
    r"\b(fix|edit|implement|refactor|test|tests|patch|commit|branch|git|shell|"
    r"command|build|lint|format|rename|delete|code|script)\b",
    re.IGNORECASE,
)
_RESEARCH_RE = re.compile(
    r"\b(research|search|latest|web|online|sources?|news|internet)\b",
    re.IGNORECASE,
)
_PLANNING_RE = re.compile(r"\b(plans?|planning|checklist|steps)\b", re.IGNORECASE)
_MEDIA_RE = re.compile(
    r"\b(image|images|photo|picture|webcam|camera|video|audio|microphone|"
    r"record|transcribe|speech)\b",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(
    r"\b(history|remember|recall|discuss(?:ed)?|earlier|previous|past|"
    r"skill|skills|sessions?|conversations?)\b",
    re.IGNORECASE,
)
_HELPER_RE = re.compile(
    r"\b(weather|forecast|temperature|location|climate|pdf|docx|xlsx|"
    r"spreadsheet|document)\b",
    re.IGNORECASE,
)


def route_packs(
    user_input: str = "",
    *,
    has_attachment: bool = False,
    has_url: bool = False,
    has_active_plan: bool = False,
    has_image: bool = False,
    forced: str | None = None,
) -> set[str]:
    """Return the set of pack names needed for a turn."""
    packs: set[str] = set(BASELINE_PACKS)
    text = (user_input or "").lower()

    if forced:
        packs.add(forced)

    explicit = {
        "/code": "Coding",
        "/research": "Research",
        "/plan": "Planning",
        "/ask": None,
    }
    for cmd, pack in explicit.items():
        if text.strip().startswith(cmd):
            if pack:
                packs.add(pack)
            # /ask keeps baseline packs only
            if cmd == "/ask":
                return set(BASELINE_PACKS)

    if has_attachment:
        packs.add("Coding")
    if has_url or _URL_RE.search(user_input or ""):
        packs.add("Research")
    if has_active_plan:
        packs.add("Coding")
        packs.add("Planning")
    if has_image:
        packs.add("Media")

    # Keyword hints; each pack is only sent when the turn mentions it.
    if _mentions(text, _CODING_RE):
        packs.add("Coding")
    if _mentions(text, _RESEARCH_RE):
        packs.add("Research")
    if _mentions(text, _PLANNING_RE):
        packs.add("Planning")
    if _mentions(text, _MEDIA_RE):
        packs.add("Media")
    if _mentions(text, _MEMORY_RE):
        packs.add("Memory")
    if _mentions(text, _HELPER_RE):
        packs.add("Helper")

    return packs


def tools_for_packs(packs: Iterable[str]) -> set[str]:
    """Return the set of tool names allowed by the given packs."""
    names: set[str] = set(CORE_TOOLS)
    for pack_name in packs:
        pack = PACKS.get(pack_name)
        if pack:
            names |= pack.tools
    return names


def enable_capability(name: str) -> str:
    """Enable a capability pack for the current turn."""
    if name not in PACKS:
        return f"Unknown capability: {name}"
    try:
        from .tools import get_tool_context

        get_tool_context().enabled_packs.add(name)
    except Exception:  # noqa: BLE001 - context may be unavailable in tests
        pass
    return f"Enabled capability: {name}"


def _all_pack_tools() -> set[str]:
    names: set[str] = set()
    for pack in PACKS.values():
        names |= pack.tools
    names |= CORE_TOOLS
    return names


def unknown_pack_tools(registered: Iterable[str]) -> dict[str, list[str]]:
    """Return pack name -> sorted tool names the registry does not provide.

    ``filter_tools_list`` preserves any tool that no pack mentions, so a pack
    naming a tool that is not registered silently leaves the real tool always
    available instead of gated. Callers pass the live registry so a rename on
    either side is caught instead of being maintained by hand.
    """
    available = set(registered)
    return {
        name: sorted(pack.tools - available)
        for name, pack in PACKS.items()
        if pack.tools - available
    }


def filter_tools_list(tools_list, user_input=""):
    """Filter ``tools_list`` to the packs needed for ``user_input`` plus any
    packs enabled for the current turn."""
    try:
        from .tools import get_tool_context

        enabled = set(getattr(get_tool_context(), "enabled_packs", ()) or ())
    except Exception:  # noqa: BLE001 - context may be unavailable in tests
        enabled = set()
    allowed = tools_for_packs(route_packs(user_input))
    allowed |= tools_for_packs(enabled)
    all_known = _all_pack_tools()
    result = []
    for t in tools_list:
        name = getattr(t, "name", None)
        if name is None or name in allowed or name not in all_known:
            result.append(t)
    # A fixed order keeps the tool definitions (and therefore the prompt
    # prefix) identical between turns that bind the same packs.
    result.sort(key=lambda tool: str(getattr(tool, "name", "")))
    return result
