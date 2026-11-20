"""Hooks that run around tool calls and at session start."""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any

DEFAULT_HOOK_TIMEOUT = 30


@dataclass
class HookResult:
    event: str
    command: str
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    blocked: bool = False
    reason: str = ""
    timed_out: bool = False


def _normalize_one(item: Any, matcher: str | None = None) -> dict[str, Any] | None:
    if isinstance(item, str):
        item = {"command": item}
    if not isinstance(item, dict):
        return None
    command = item.get("command") or item.get("cmd") or item.get("script")
    if not command:
        return None
    hook_type = item.get("type", "command")
    if hook_type not in (None, "command"):
        return None
    timeout = item.get("timeout", item.get("timeout_seconds"))
    try:
        timeout = int(timeout) if timeout is not None else None
    except (TypeError, ValueError):
        timeout = None
    return {
        "command": str(command),
        "matcher": item.get("matcher", matcher),
        "timeout": timeout,
    }


def _normalize_hooks_value(
    value: Any, matcher: str | None = None
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = _normalize_one({"command": value}, matcher)
        return [normalized] if normalized else []
    if isinstance(value, dict):
        if isinstance(value.get("hooks"), list):
            result: list[dict[str, Any]] = []
            for child in value["hooks"]:
                result.extend(
                    _normalize_hooks_value(child, value.get("matcher", matcher))
                )
            return result
        normalized = _normalize_one(value, matcher)
        return [normalized] if normalized else []
    if isinstance(value, list):
        result: list[dict[str, Any]] = []
        for child in value:
            result.extend(_normalize_hooks_value(child, matcher))
        return result
    return []


def load_hooks(config: Any = None) -> dict[str, list[dict[str, Any]]]:
    """Load and normalize hooks from a config mapping."""
    if config is None:
        try:
            from .config import load_or_create_config

            config = load_or_create_config()
        except Exception:  # noqa: BLE001 - config loading is best-effort; fall back to empty hooks
            config = {}
    if not isinstance(config, dict):
        return {}
    raw = config.get("hooks")
    if raw is None:
        raw = (
            config.get("agent", {}).get("hooks")
            if isinstance(config.get("agent"), dict)
            else None
        )
    if not isinstance(raw, dict):
        return {}
    hooks: dict[str, list[dict[str, Any]]] = {}
    for event, value in raw.items():
        normalized = _normalize_hooks_value(value)
        if normalized:
            hooks[str(event)] = normalized
    return hooks


def _match_tool(hook: dict[str, Any], tool_name: str | None) -> bool:
    matcher = hook.get("matcher")
    if not matcher or matcher == "*":
        return True
    if tool_name is None:
        return False
    try:
        return re.search(str(matcher), str(tool_name)) is not None
    except re.error:
        return str(matcher) == str(tool_name)


def _hook_timeout(hook: dict[str, Any], default: int) -> int:
    timeout = hook.get("timeout")
    if timeout is None:
        return default
    try:
        value = int(timeout)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _parse_decision(stdout: str) -> tuple[bool, str]:
    text = (stdout or "").strip()
    if not text:
        return False, ""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False, ""
    if not isinstance(data, dict):
        return False, ""
    decision = str(data.get("decision") or data.get("action") or "").lower()
    reason = str(data.get("reason") or data.get("message") or "")
    if decision in {"block", "deny", "reject"}:
        return True, reason or "Blocked by hook"
    return False, reason


def run_hooks(
    event: str,
    *,
    hooks: dict[str, list[dict[str, Any]]] | None = None,
    config: Any = None,
    tool_name: str | None = None,
    tool_args: Any = None,
    workspace_root: Any = None,
    session_id: str | None = None,
    timeout: int = DEFAULT_HOOK_TIMEOUT,
    stop_on_block: bool = True,
) -> list[HookResult]:
    """Run all hooks for an event and return their results."""
    if hooks is None:
        hooks = load_hooks(config)
    if not isinstance(hooks, dict):
        return []
    results: list[HookResult] = []
    for hook in hooks.get(event, []):
        if not _match_tool(hook, tool_name):
            continue
        payload = {
            "event": event,
            "tool_name": tool_name,
            "tool_args": tool_args or {},
            "workspace_root": str(workspace_root)
            if workspace_root is not None
            else None,
            "session_id": session_id,
        }
        env = dict(os.environ)
        env["PRIVATE_AGENT_EVENT"] = event
        if tool_name is not None:
            env["PRIVATE_AGENT_TOOL_NAME"] = str(tool_name)
        env["PRIVATE_AGENT_TOOL_ARGS"] = json.dumps(tool_args or {}, default=str)
        if workspace_root is not None:
            env["PRIVATE_AGENT_WORKSPACE"] = str(workspace_root)
        if session_id is not None:
            env["PRIVATE_AGENT_SESSION_ID"] = str(session_id)
        hook_timeout = _hook_timeout(hook, timeout)
        command = str(hook["command"])
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(workspace_root) if workspace_root is not None else None,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=hook_timeout,
                env=env,
            )
            blocked_by_json, reason = _parse_decision(completed.stdout)
            blocked = completed.returncode != 0 or blocked_by_json
            if completed.returncode != 0 and not reason:
                reason = (
                    completed.stderr.strip()
                    or f"Hook exited with {completed.returncode}"
                )
            if blocked_by_json and not reason:
                reason = "Blocked by hook"
            result = HookResult(
                event=event,
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                blocked=blocked
                if event in {"PreToolUse", "pre_tool_use", "preToolUse"}
                else False,
                reason=reason if blocked else "",
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            result = HookResult(
                event=event,
                command=command,
                returncode=None,
                stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr or "" if isinstance(exc.stderr, str) else "",
                blocked=event in {"PreToolUse", "pre_tool_use", "preToolUse"},
                reason=f"Hook timed out after {hook_timeout} seconds",
                timed_out=True,
            )
        except OSError as exc:
            result = HookResult(
                event=event,
                command=command,
                returncode=None,
                blocked=False,
                reason=f"Hook failed to start: {exc}",
            )
        results.append(result)
        if stop_on_block and result.blocked:
            break
    return results


def run_pre_tool_hooks(**kwargs: Any) -> list[HookResult]:
    return run_hooks("PreToolUse", **kwargs)


def run_post_tool_hooks(**kwargs: Any) -> list[HookResult]:
    return run_hooks("PostToolUse", **kwargs)


def run_session_start_hooks(**kwargs: Any) -> list[HookResult]:
    return run_hooks("SessionStart", **kwargs)


def safe_run_session_start_hooks(**kwargs: Any) -> None:
    """Run session-start hooks, swallowing errors so startup never fails."""
    with contextlib.suppress(Exception):  # noqa: BLE001 - session-start hooks are best-effort
        run_session_start_hooks(**kwargs)


def evaluate_pre_tool_hooks(
    tool_name: str | None,
    tool_args: Any = None,
    workspace_root: Any = None,
) -> tuple[str | None, dict[str, list[dict[str, Any]]]]:
    """Load hooks, run PreToolUse hooks, and return (block_reason, hooks)."""
    if workspace_root is None:
        loaded = load_hooks()
    else:
        try:
            from .config import load_or_create_config, merge_project_config

            loaded = load_hooks(
                merge_project_config(load_or_create_config(), workspace_root)
            )
        except Exception:  # noqa: BLE001 - hook loading must not block tool execution
            loaded = load_hooks()
    reason = hooks_blocked(
        run_pre_tool_hooks(
            hooks=loaded,
            tool_name=tool_name,
            tool_args=tool_args,
            workspace_root=workspace_root,
        )
    )
    return reason, loaded


def safe_run_post_tool_hooks(**kwargs: Any) -> None:
    """Run post-tool hooks, swallowing errors so tool results are unaffected."""
    with contextlib.suppress(Exception):  # noqa: BLE001 - post-tool hooks are best-effort
        run_post_tool_hooks(**kwargs)


def hooks_blocked(results: list[HookResult]) -> str | None:
    for result in results or []:
        if result.blocked:
            return result.reason or "Blocked by hook"
    return None


def collect_post_tool_hook_output(
    tool_name: str | None = None,
    tool_args: Any = None,
    workspace_root: Any = None,
) -> str:
    """Run PostToolUse hooks and return their combined output (empty on failure)."""
    try:
        if workspace_root is None:
            loaded = load_hooks()
        else:
            from .config import load_or_create_config, merge_project_config

            loaded = load_hooks(
                merge_project_config(load_or_create_config(), workspace_root)
            )
        results = run_post_tool_hooks(
            hooks=loaded,
            tool_name=tool_name,
            tool_args=tool_args,
            workspace_root=workspace_root,
        )
    except Exception:  # noqa: BLE001 - post-tool hooks are best-effort
        return ""
    parts: list[str] = []
    for result in results or []:
        if getattr(result, "stdout", ""):
            parts.append(str(result.stdout).strip())
        if getattr(result, "stderr", ""):
            parts.append(str(result.stderr).strip())
        if getattr(result, "reason", ""):
            parts.append(str(result.reason))
    return "\n".join(part for part in parts if part)
