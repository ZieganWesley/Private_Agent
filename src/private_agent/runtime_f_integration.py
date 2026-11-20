"""Focused integration helpers wiring F-section subsystems into the turn loop."""

from __future__ import annotations

import contextlib
import sqlite3
import time
import uuid
from typing import Any

from . import context_manager, json_output, task_orchestrator, timeline
from .config import DEFAULT_DB_PATH


def begin_turn(turn_id: str, user_input: str, total_tokens: int, json_mode: bool = False) -> dict[str, Any]:
    plan = task_orchestrator.build_plan(user_input, [user_input])
    budget = context_manager.ContextBudget(total_tokens=max(1, int(total_tokens or 1)))
    with contextlib.suppress(Exception):
        budget.allocate()
    run_id = turn_id or uuid.uuid4().hex
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(DEFAULT_DB_PATH))
        timeline.ensure_schema(conn)
    except Exception:  # noqa: BLE001 - timeline unavailable, disable persistence
        conn = None
    state = {"plan": plan, "conn": conn, "run_id": run_id, "started": time.monotonic(), "json_mode": json_mode}
    if json_mode:
        with contextlib.suppress(Exception):
            json_output.emit_event({"event": "turn_started", "turn_id": turn_id})
    return state


def record_iteration(state: dict[str, Any], step_index: int, pack: str = "Core") -> float:
    step_start = time.monotonic()
    plan = state.get("plan")
    if plan is not None:
        with contextlib.suppress(Exception):
            plan.add_step(f"tool iteration {step_index}", {pack})
    conn = state.get("conn")
    if conn is not None:
        with contextlib.suppress(Exception):
            est = timeline.estimate_step(conn, pack)
            timeline.record_estimate(conn, state["run_id"], step_index, pack, est.seconds, est.confidence)
    return step_start


def finish_iteration(state: dict[str, Any], step_index: int, step_start: float, status: str = "ok") -> None:
    plan = state.get("plan")
    if plan is not None and getattr(plan, "steps", None):
        with contextlib.suppress(Exception):
            plan.mark(len(plan.steps) - 1, "done" if status == "ok" else "failed")
    conn = state.get("conn")
    if conn is not None:
        with contextlib.suppress(Exception):
            timeline.record_actual(conn, state["run_id"], step_index, time.monotonic() - step_start)
    if state.get("json_mode"):
        with contextlib.suppress(Exception):
            json_output.emit_event({"event": "iteration_finished", "step": step_index, "status": status})


def finish_turn(state: dict[str, Any]) -> None:
    conn = state.get("conn")
    if conn is not None:
        with contextlib.suppress(Exception):
            conn.close()
    if state.get("json_mode"):
        with contextlib.suppress(Exception):
            json_output.emit_event({"event": "turn_finished", "turn_id": state.get("run_id")})
