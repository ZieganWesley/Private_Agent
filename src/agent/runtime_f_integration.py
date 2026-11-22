"""Focused integration helpers wiring F-section subsystems into the turn loop."""

from __future__ import annotations

import contextlib
import sqlite3
import time
import uuid
from typing import Any

from config import DEFAULT_DB_PATH
from ui import json_output

from . import capabilities as capability_packs
from . import context_manager, task_orchestrator, timeline


def rebuild_tools(base_tools_list, user_input, llm):
    """Rebuild the tool list after enable_capability changes packs."""
    import contextlib

    tools_list = capability_packs.filter_tools_list(base_tools_list, user_input)
    llm_with_tools = llm
    with contextlib.suppress(Exception):
        from .runtime import _bind_tools

        llm_with_tools = _bind_tools(llm, tools_list)
    return tools_list, llm_with_tools


def gate_capability(tool_call, permission_mode, console, state):
    """Approve an enable_capability call; append a blocked message if denied."""
    import sys

    from langchain_core.messages import ToolMessage

    name = (tool_call.get("args") or {}).get("name")
    from tools import get_tool_context

    interactive = get_tool_context().interactive and sys.stdin.isatty()
    allowed, reason = approve_capability(name, permission_mode, console, interactive)
    if not allowed:
        state.blocked_tool_messages.append(
            ToolMessage(content=reason, tool_call_id=str(tool_call.get("id") or ""))
        )
    elif name in capability_packs.PACKS:
        state.capability_approved = True
    return allowed


def approve_capability(name, permission_mode, console, interactive):
    """Return (allowed, reason) for an enable_capability request."""
    if name not in ("Coding", "Research"):
        return True, ""
    if permission_mode == "full":
        return True, ""
    if not interactive:
        return False, f"Switching to {name} requires interactive approval."
    from .permissions import prompt_tool_approval

    approved = prompt_tool_approval(
        console=console,
        name=f"enable_capability ({name})",
        description=f"switch the session to {name} capability",
        safe_args={"name": name},
    )
    if not approved:
        return (
            False,
            f"Capability escalation to {name} was not approved; no packs changed.",
        )
    return True, ""


def get_plan_summary(state: dict[str, Any]) -> str:
    plan = (state or {}).get("plan")
    if plan is not None and hasattr(plan, "summary"):
        with contextlib.suppress(Exception):
            return str(plan.summary())
    return ""


def is_synthetic_plan(state: dict[str, Any]) -> bool:
    """Return whether the turn's plan is the default synthetic one."""
    state = state or {}
    if "synthetic" in state:
        return bool(state.get("synthetic"))
    plan = state.get("plan")
    return bool(getattr(plan, "synthetic", False))


def is_plan_complete(state: dict[str, Any]) -> bool:
    plan = (state or {}).get("plan")
    if plan is not None and hasattr(plan, "is_complete"):
        with contextlib.suppress(Exception):
            return bool(plan.is_complete())
    return not bool((state or {}).get("pending", False))


def trim_context(text: str, total_tokens: int) -> str:
    """Trim text to the context budget using context_manager."""
    with contextlib.suppress(Exception):
        out = context_manager.build_context({"recent": text}, total_tokens)
        return out.get("recent", text)
    return text


def begin_turn(
    turn_id: str, user_input: str, total_tokens: int, json_mode: bool = False
) -> dict[str, Any]:
    plan = task_orchestrator.build_plan(
        user_input, ["respond to request"], synthetic=True
    )
    run_id = turn_id or uuid.uuid4().hex
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(DEFAULT_DB_PATH))
        timeline.ensure_schema(conn)
    except Exception:  # noqa: BLE001 - timeline unavailable, disable persistence
        conn = None
    turn_context_token = None
    with contextlib.suppress(Exception):
        from tools import begin_turn_tool_context

        turn_context_token = begin_turn_tool_context()
    state = {
        "plan": plan,
        "synthetic": getattr(plan, "synthetic", False),
        "iteration": 0,
        "pending": True,
        "conn": conn,
        "run_id": run_id,
        "started": time.monotonic(),
        "json_mode": json_mode,
        "_context_token": turn_context_token,
    }
    if json_mode:
        with contextlib.suppress(Exception):
            json_output.emit_event({"event": "turn_started", "turn_id": turn_id})
    return state


def record_iteration(
    state: dict[str, Any], step_index: int, pack: str = "Core"
) -> float:
    step_start = time.monotonic()
    state["iteration"] = step_index
    state["pending"] = True
    plan = state.get("plan")
    if plan is not None:
        with contextlib.suppress(Exception):
            if not plan.steps:
                plan.add_step("execute request", {pack})
            plan.mark(0, "running")
    conn = state.get("conn")
    if conn is not None:
        with contextlib.suppress(Exception):
            est = timeline.estimate_step(conn, pack)
            timeline.record_estimate(
                conn, state["run_id"], step_index, pack, est.seconds, est.confidence
            )
    return step_start


def finish_iteration(
    state: dict[str, Any],
    step_index: int,
    step_start: float,
    status: str = "ok",
    complete: bool = False,
) -> None:
    state["pending"] = not bool(complete and status == "ok")
    plan = state.get("plan")
    if plan is not None and getattr(plan, "steps", None):
        with contextlib.suppress(Exception):
            if complete and status == "ok":
                plan.mark(0, "done")
            elif status != "ok":
                plan.mark(0, "failed")
    conn = state.get("conn")
    if conn is not None:
        with contextlib.suppress(Exception):
            timeline.record_actual(
                conn, state["run_id"], step_index, time.monotonic() - step_start
            )
    if state.get("json_mode"):
        with contextlib.suppress(Exception):
            json_output.emit_event(
                {"event": "iteration_finished", "step": step_index, "status": status}
            )


@contextlib.contextmanager
def guard_turn_start(state: dict[str, Any]):
    """Reset the turn tool context if a turn fails before its tool loop starts."""
    try:
        yield
    except BaseException:  # noqa: BLE001 - a failed turn must not leak its context
        finish_turn(state)
        raise


def finish_turn(state: dict[str, Any]) -> None:
    token = (state or {}).pop("_context_token", None)
    if token is not None:
        with contextlib.suppress(Exception):
            from tools import reset_turn_tool_context

            reset_turn_tool_context(token)
    state["pending"] = False
    plan = state.get("plan")
    if plan is not None:
        with contextlib.suppress(Exception):
            plan.mark(0, "done")
    conn = state.get("conn")
    if conn is not None:
        with contextlib.suppress(Exception):
            conn.close()
    if state.get("json_mode"):
        with contextlib.suppress(Exception):
            json_output.emit_event(
                {"event": "turn_finished", "turn_id": state.get("run_id")}
            )
