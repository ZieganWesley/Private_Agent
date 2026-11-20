"""Capability packs and turn router (Part F2)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

CORE_TOOLS = frozenset(
    {
        "get_current_datetime",
        "read_local_file",
        "search_workspace_text",
        "list_directory",
        "search_documents",
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
        "edit_local_file",
        "write_local_file",
        "rename_path",
        "delete_path",
        "run_shell_command",
        "run_project_unit_tests",
        "inspect_workspace_git",
    },
    "Project instructions (AGENTS.md), test after edits.",
)
_register(
    "Helper",
    {"get_weather", "fetch_current_location"},
    "Device and privacy rules.",
)
_register(
    "Planning",
    {"create_task_plan", "revise_task_plan", "update_todo_step"},
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
    packs: set[str] = {"Core", "Knowledge", "Media", "Memory", "MCP", "Helper"}
    text = (user_input or "").lower()

    if forced:
        packs.add(forced)
        if forced in ("Coding", "Research"):
            pass

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
            # /ask keeps only Core+Knowledge
            if cmd == "/ask":
                return {"Core", "Knowledge"}

    if has_attachment:
        packs.add("Coding")
    if has_url or _URL_RE.search(user_input or ""):
        packs.add("Research")
    if has_active_plan:
        packs.add("Coding")
        packs.add("Planning")
    if has_image:
        packs.add("Helper")

    # keyword hints
    if any(w in text for w in ("fix", "edit", "implement", "refactor", "test")):
        packs.add("Coding")
    if any(w in text for w in ("research", "search", "latest", "web")):
        packs.add("Research")
    if "plan" in text:
        packs.add("Planning")

    return packs


def tools_for_packs(packs: Iterable[str], available: set[str] | None = None) -> set[str]:
    """Return the set of tool names allowed by the given packs."""
    names: set[str] = set()
    for pack_name in packs:
        pack = PACKS.get(pack_name)
        if pack:
            names |= pack.tools
    names |= CORE_TOOLS
    if available is not None:
        names &= available
    return names


def classify_turn(user_input: str = "") -> tuple[set[str], str]:
    """Return (packs, complexity) using rules (no LLM). Complexity: simple/multi_step/long."""
    packs = route_packs(user_input)
    text = (user_input or "").lower()
    complexity = "simple"
    if any(w in text for w in ("then", "and then", "after", "step", "plan")):
        complexity = "multi_step"
    if len(text.split()) > 40:
        complexity = "long"
    return packs, complexity


def enable_capability(name: str) -> str:
    """Escalate to an additional capability pack (placeholder)."""
    if name not in PACKS:
        return f"Unknown capability: {name}"
    return f"Enabled capability: {name}"


def _all_pack_tools() -> set[str]:
    names: set[str] = set()
    for pack in PACKS.values():
        names |= pack.tools
    names |= CORE_TOOLS
    return names


def filter_tools_list(tools_list, user_input="", **kw):
    allowed = tools_for_packs(route_packs(user_input, **kw), None)
    all_known = _all_pack_tools()
    result = []
    for t in tools_list:
        name = getattr(t, "name", None)
        if name is None or name in allowed:
            result.append(t)
        elif name not in all_known:
            # Preserve tools not explicitly listed in any pack.
            result.append(t)
    return result
