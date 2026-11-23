"""Permission policy helpers for agent tool calls."""

import contextlib
import difflib
import fnmatch
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

MUTATING_TOOL_NAMES = frozenset(
    {
        "edit_local_file",
        "apply_workspace_patch",
        "rename_workspace_file",
        "delete_workspace_file",
        "run_shell_command",
        "download_web_file",
        "delete_chat_history_from_sqlite",
        "delete_chat_history_entry_from_sqlite",
        "create_skill",
        "run_project_unit_tests",
        "checkpoint_code_task",
        "finalize_code_task",
    }
)
MONITORED_COMMAND_TOOL_NAMES = frozenset(
    {
        "run_shell_command",
        "run_project_unit_tests",
        "checkpoint_code_task",
        "finalize_code_task",
    }
)
CODING_ACCESS_LEVELS = {
    "full_hitl": (
        "Full workspace access — network-enabled commands; ask before every tool action"
    ),
    "full_governed": (
        "Full workspace access — network-enabled commands; ask before edits and commands"
    ),
    "full_monitored": (
        "Full workspace access — network-enabled commands; approve and monitor each command"
    ),
    "isolated_governed": (
        "Isolated workspace copy — read freely; ask before edits and commands"
    ),
    "isolated_hitl": ("Fully isolated workspace copy — ask before every tool action"),
}


def tool_needs_permission(
    name: str,
    permission_mode: str,
    *,
    network_authorized: bool,
    network_tool_names: frozenset[str],
    mcp_tool_names: set[str],
    mcp_auto_approve_tools: Any,
) -> bool:
    if permission_mode == "monitored":
        return (
            name in MONITORED_COMMAND_TOOL_NAMES
            or name in mcp_tool_names
            and name not in mcp_auto_approve_tools
        )
    if permission_mode == "full":
        return False
    if permission_mode == "manual":
        return name not in network_tool_names or not network_authorized
    return (
        name in MUTATING_TOOL_NAMES
        or name in mcp_tool_names
        and name not in mcp_auto_approve_tools
    )


def select_coding_access_level(
    console: Any,
    is_interactive: Callable[[], bool],
) -> str:
    """Ask how a coding plan may access its selected workspace."""
    if not is_interactive():
        return "isolated_governed"
    choices = tuple(CODING_ACCESS_LEVELS.items())
    try:
        from .interactive import arrow_select as _arrow_select

        _chosen = _arrow_select(
            "Coding workspace access:",
            [(str(i), desc) for i, (_, desc) in enumerate(choices, 1)],
            default="4",
        )
    except Exception:  # noqa: BLE001 - fall back to typed prompt
        _chosen = None
    if _chosen is not None and _chosen.isdigit():
        choice = int(_chosen)
        if not 1 <= choice <= len(choices):
            choice = 4
    else:
        console.print("[bold info][Coding Workspace Access][/bold info]")
        for index, (_, description) in enumerate(choices, 1):
            console.print(f"  {index}. {description}")
        selection = console.input(
            "Choose workspace access [1-5] (default: 4, isolated): "
        ).strip()
        try:
            choice = int(selection) if selection else 4
        except ValueError:
            console.print("[warning]Invalid choice; using isolated workspace.[/warning]")
            choice = 4
        if not 1 <= choice <= len(choices):
            console.print("[warning]Invalid choice; using isolated workspace.[/warning]")
            choice = 4
    return choices[choice - 1][0]


def coding_permission_mode(access_level: str) -> str:
    """Translate a reviewed workspace level into the per-task tool policy."""
    return {
        "full_hitl": "manual",
        "full_governed": "auto",
        "full_monitored": "monitored",
        "isolated_governed": "auto",
        "isolated_hitl": "manual",
    }.get(access_level, "auto")


def coding_uses_isolated_copy(access_level: str) -> bool:
    return access_level not in {"full_hitl", "full_governed", "full_monitored"}


def coding_allows_command_network(access_level: str) -> bool:
    return access_level in {"full_hitl", "full_governed", "full_monitored"}


def _permission_rule_for_tool(name: str, safe_args: Any) -> str:
    """Build a permission rule that matches this tool call."""
    target = _permission_rule_target(name, safe_args if isinstance(safe_args, dict) else {})
    target = target.strip()
    if target:
        return f"{name}({target})"
    return name


def _approval_diff(name: str, safe_args: Any) -> str | None:
    """Return a coloured unified diff for edit-like tool calls, if possible."""
    if not isinstance(safe_args, dict):
        return None
    old_key = next(
        (key for key in ("old_text", "old_string", "before") if key in safe_args),
        None,
    )
    new_key = next(
        (key for key in ("new_text", "new_string", "after", "content") if key in safe_args),
        None,
    )
    if old_key is None or new_key is None:
        return None
    old_text = str(safe_args.get(old_key, ""))
    new_text = str(safe_args.get(new_key, ""))
    if old_text == new_text:
        return None
    diff_lines = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=f"{name}: before",
        tofile=f"{name}: after",
        lineterm="",
    )
    rendered = []
    for line in diff_lines:
        if line.startswith("+++") or line.startswith("---"):
            rendered.append(f"[bold]{line}[/bold]")
        elif line.startswith("@@"):
            rendered.append(f"[info]{line}[/info]")
        elif line.startswith("+"):
            rendered.append(f"[success]{line}[/success]")
        elif line.startswith("-"):
            rendered.append(f"[error]{line}[/error]")
        else:
            rendered.append(line)
    return "\n".join(rendered) if rendered else None


def add_project_permission_rule(rule: str, workspace_root: Any = None) -> bool:
    """Persist an allow rule in the project config and the active rule list."""
    try:
        from config import discover_project_config_paths

        paths = discover_project_config_paths(workspace_root)
        path = next((item for item in paths if item.is_file()), None)
        if path is None:
            path = paths[0] if paths else Path.cwd() / ".private_agent" / "settings.json"
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except (json.JSONDecodeError, OSError, ValueError):
                existing = {}
        permissions = existing.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            permissions = {}
            existing["permissions"] = permissions
        allow_rules = permissions.setdefault("allow", [])
        if not isinstance(allow_rules, list):
            allow_rules = []
            permissions["allow"] = allow_rules
        if rule not in allow_rules:
            allow_rules.append(rule)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=4) + "\n", encoding="utf-8")
        try:
            import config as config_module

            if rule not in config_module.PERMISSION_ALLOW_RULES:
                config_module.PERMISSION_ALLOW_RULES.append(rule)
        except Exception:  # noqa: BLE001 - in-memory rule refresh is best effort
            pass
        return True
    except Exception:  # noqa: BLE001 - persisting a project allow rule must not crash approval
        return False


def prompt_tool_approval(
    *,
    console: Any,
    name: str,
    description: str,
    safe_args: Any,
    on_always_allow: Any = None,
    on_always_allow_session: Any = None,
    on_always_allow_project: Any = None,
    diff: str | None = None,
) -> bool:
    """Prompt the user for approval of a tool call.

    Returns ``True`` when the user approves.  ``a``/``session`` approves and
    registers the tool for the rest of the session; ``p``/``project`` also
    persists an allow rule for the project.
    """
    session_callback = on_always_allow_session or on_always_allow
    if on_always_allow_project is None:
        on_always_allow_project = lambda rule: add_project_permission_rule(rule)
    rule = _permission_rule_for_tool(name, safe_args)

    console.print("[bold info][Tool Approval][/bold info]")
    console.print(f"Tool: {name}")
    if description:
        console.print(f"Purpose: {description}")
    if isinstance(safe_args, dict) and safe_args:
        console.print("Arguments:")
        for key, value in safe_args.items():
            console.print(f"  {key}: {value}")
    rendered_diff = diff if diff is not None else _approval_diff(name, safe_args)
    if rendered_diff:
        console.print(rendered_diff)
    approval = (
        console.input(
            "[warning]Approve tool '"
            + name
            + "'? "
            + (description or "")
            + "\nRun? [y/N/a (session)/p (project)]: [/warning]"
        )
        .strip()
        .lower()
    )
    if approval in {"a", "session", "s", "always"}:
        if session_callback is not None:
            with contextlib.suppress(Exception):
                session_callback(name)
        return True
    if approval in {"p", "project", "pr"}:
        with contextlib.suppress(Exception):
            on_always_allow_project(rule)
        if session_callback is not None:
            with contextlib.suppress(Exception):
                session_callback(name)
        return True
    if approval in {"y", "yes", "run"}:
        return True
    return False


def select_permission_mode(
    configured: str,
    console: Any,
    is_interactive: Callable[[], bool],
) -> str:
    labels = {
        "manual": "Manual — ask before every tool action",
        "auto": "Auto — read/search freely; ask before edits and command execution",
        "full": "Full — approve all tool calls for this session",
    }
    if not is_interactive():
        return configured
    try:
        from .interactive import arrow_select

        chosen = arrow_select(
            "Permission mode:",
            [(m, labels[m]) for m in ("manual", "auto", "full")],
            default=configured,
        )
        if chosen is not None:
            return chosen
    except Exception:  # noqa: BLE001 - fall back to typed prompt
        pass
    console.print("[bold info][Agent Permission Mode][/bold info]")
    for number, mode in enumerate(("manual", "auto", "full"), 1):
        console.print(f"  {number}. {labels[mode]}")
    selection = (
        console.input(f"Choose permission mode [1/2/3] (default: {configured}): ")
        .strip()
        .lower()
    )
    if not selection:
        return configured
    modes = {"1": "manual", "2": "auto", "3": "full"}
    return modes.get(selection, configured)


def _request_summary(tool_name: str, tool_args: dict[str, Any]) -> str:
    if tool_name == "fetch_current_location":
        return "approximate location lookup from this machine's public IP address"
    if tool_name == "get_weather":
        place = tool_args.get("location") or "current location (public IP lookup)"
        return f"weather for {place} on {tool_args.get('date') or 'today'}"
    return (
        tool_args.get("query") or tool_args.get("url") or "request details unavailable"
    )


async def authorize_network_research(
    tool_name: str,
    tool_args: dict[str, Any],
    *,
    permission_mode: str,
    enabled: bool,
    consent_policy: str,
    internet_check: Callable[[], bool],
    console: Any,
    is_interactive: Callable[[], bool],
    session_consent_state: dict[str, bool] | None = None,
) -> tuple[bool, str]:
    """Require live connectivity and consent before an external request."""
    if not enabled:
        return False, "Online research is disabled in configuration."

    if not internet_check():
        if not is_interactive():
            return False, (
                "Internet connectivity is unavailable. Offline evidence may be "
                "incomplete; no web request was made."
            )
        choice = (
            console.input(
                "[warning]Internet is unavailable. Restore access and retry (r), "
                "continue with incomplete offline data (c), or cancel (x)? [/warning]"
            )
            .strip()
            .lower()
        )
        if choice == "r":
            if not internet_check():
                return False, (
                    "Internet connectivity is still unavailable. No web request "
                    "was made; offline evidence may be incomplete."
                )
        elif choice == "c":
            return False, (
                "The user chose to continue without online research. Offline "
                "evidence may be incomplete; do not claim current facts were verified."
            )
        else:
            return False, "Online research was cancelled by the user."

    if consent_policy == "never":
        return False, "Online research is disabled by the configured consent policy."
    session_consent_state = (
        session_consent_state if session_consent_state is not None else {}
    )
    # Full mode already approves every tool call for the session.
    full_mode = permission_mode == "full"
    needs_session_consent = (
        not full_mode
        and consent_policy == "session"
        and not session_consent_state.get("granted", False)
    )
    needs_per_request_consent = not full_mode and consent_policy == "ask"
    if needs_session_consent or needs_per_request_consent:
        if not is_interactive():
            return False, "Online search requires interactive user consent."
        request_summary = _request_summary(tool_name, tool_args)
        choice = (
            console.input(
                f"[warning]'{tool_name}' will send this to an external service: "
                f"{request_summary}\nProceed? \\[y/N]: [/warning]"
            )
            .strip()
            .lower()
        )
        if choice != "y":
            return False, "Online research was not approved; use offline evidence only."
        if needs_session_consent:
            session_consent_state["granted"] = True
    elif permission_mode == "manual":
        if not is_interactive():
            return False, "Online search requires interactive user approval."
        request_summary = _request_summary(tool_name, tool_args)
        choice = (
            console.input(
                f"[warning]Approve this '{tool_name}' action for the current request: "
                f"{request_summary}\nProceed? \\[y/N]: [/warning]"
            )
            .strip()
            .lower()
        )
        if choice != "y":
            return False, "Online research was not approved; no request was made."
    return True, ""


def _permission_rule_target(name: str, args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    if name == "run_shell_command":
        return str(args.get("command", ""))
    for key in ("file_path", "path", "dir_path", "target_path"):
        if key in args and args.get(key) is not None:
            return str(args.get(key))
    return " ".join(str(value) for value in args.values())


def permission_rule_matches(rule: str, name: str, args: Any) -> bool:
    """Return True if a permission rule string matches a tool call."""
    rule = (rule or "").strip()
    match = re.match(r"^([A-Za-z0-9_]+)(?:\((.*)\))?$", rule)
    if not match:
        return False
    tool_name = match.group(1)
    if tool_name != name:
        return False
    arg_pattern = match.group(2)
    if arg_pattern is None:
        return True
    arg_pattern = arg_pattern.strip()
    if not arg_pattern:
        return True
    target = _permission_rule_target(name, args)
    return fnmatch.fnmatch(target, arg_pattern)


def evaluate_permission_rules(
    allow_rules: Any, deny_rules: Any, name: str, args: Any
) -> tuple[str | None, str | None]:
    """Return ("deny"/"allow", rule) if a rule matches, else (None, None)."""
    for rule in list(deny_rules or []):
        if permission_rule_matches(str(rule), name, args):
            return "deny", str(rule)
    for rule in list(allow_rules or []):
        if permission_rule_matches(str(rule), name, args):
            return "allow", str(rule)
    return None, None
