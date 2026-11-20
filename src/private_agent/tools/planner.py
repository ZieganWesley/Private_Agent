"""SQLite-backed planner tools for the active session/task."""

import json
import pathlib
import sqlite3

from langchain_core.tools import tool

from ..config import MAX_SUMMARY_CHARS
from ..database import PersistentMemory
from . import get_tool_context, set_tool_context
from .schemas import (
    CreateTaskPlanInput,
    InspectTaskPlanInput,
    ReadTableSchemaInput,
    ReviseTaskPlanInput,
    UpdateTodoStepInput,
)


def _open_memory() -> PersistentMemory:
    if not get_tool_context().db_path:
        raise RuntimeError("Planner database is not configured for this session.")
    return PersistentMemory(get_tool_context().db_path)


def _format_task(task: dict) -> str:
    formatted_task = dict(task)
    if task.get("task_type") == "research":
        validation_current = bool(
            task.get("validation_revision") == task.get("revision")
            and task.get("validation_evidence")
        )
        formatted_task["research_evidence_status"] = (
            "validation evidence recorded; listed citations are untrusted "
            "references and have not been independently verified"
            if validation_current
            else "incomplete; plan approval does not validate research facts or sources"
        )
    return json.dumps(formatted_task, ensure_ascii=False, indent=2)


_PLAN_WRITE_TABLES = {
    "agent_tasks",
    "agent_plan_revisions",
    "agent_todo_steps",
    "agent_task_events",
}
_SCHEMA_WRITE_GROUPS = {
    "agent_tasks": _PLAN_WRITE_TABLES,
    "agent_plan_revisions": _PLAN_WRITE_TABLES,
    "agent_todo_steps": _PLAN_WRITE_TABLES,
    "agent_task_events": _PLAN_WRITE_TABLES,
    "chat_history": {"chat_history"},
    "agent_learned_items": {"agent_learned_items"},
    "agent_learning_settings": {"agent_learning_settings"},
}


@tool(args_schema=ReadTableSchemaInput)
def read_table_schema(table_name: str) -> str:
    """Read approved SQLite table metadata before preparing a database write."""
    if table_name not in _SCHEMA_WRITE_GROUPS:
        return "Error: Schema inspection is restricted to approved agent tables."
    database_path = get_tool_context().db_path
    if not database_path:
        return "Error: Planner database is not configured for this session."
    if database_path == ":memory:":
        return (
            "Error: Schema inspection requires the configured on-disk SQLite "
            "database; an isolated in-memory database cannot be reopened read-only."
        )
    path = pathlib.Path(database_path).expanduser().resolve()
    connection = None
    try:
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro",
            uri=True,
            timeout=2.0,
        )
        table_names = sorted(_SCHEMA_WRITE_GROUPS[table_name])
        schemas = [
            PersistentMemory.describe_table_schema(connection, name)
            for name in table_names
        ]
        get_tool_context().schema_read_tables.update(table_names)
        return json.dumps(
            {
                "database": "configured local SQLite database",
                "requested_table": table_name,
                "schemas": schemas,
                "application_limits": {
                    "agent_tasks.goal": "1–2000 characters",
                    "agent_tasks.approved_scope": "1–1000 characters",
                    "agent_todo_steps.description": "1–1000 characters",
                    "agent_todo_steps.evidence": "1–2000 characters",
                    "agent_plan_revisions.plan_json": (
                        "1–20 steps; each step has bounded description, "
                        "dependency, validation, proof, risk, and edge-case fields"
                    ),
                    "chat_history.summary.content": {
                        "maximum_characters": MAX_SUMMARY_CHARS,
                        "source": "configured application content budget",
                    },
                    "agent_learned_items.statement": "1–1000 characters",
                },
                "write_boundary": (
                    "This tool is read-only. Use the typed application tool/API "
                    "for writes; schema visibility does not grant write approval."
                ),
                "length_policy": (
                    "Do not infer maximum text length from a SQLite TEXT type. "
                    "Use the explicit database constraint or stated application "
                    "limit; if content is too long, create a meaning-preserving "
                    "summary tailored to that limit before submitting the write."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    except (ValueError, OSError, sqlite3.Error) as exc:
        return f"Error reading SQLite schema: {type(exc).__name__}: {exc}"
    finally:
        if connection is not None:
            connection.close()


def _require_schema(*table_names: str) -> None:
    missing = set(table_names) - (get_tool_context().schema_read_tables or set())
    if missing:
        raise ValueError(
            "Read the destination table schema with read_table_schema before "
            "preparing this SQLite write. Missing schema: " + ", ".join(sorted(missing))
        )


@tool(args_schema=CreateTaskPlanInput)
def create_task_plan(
    goal: str,
    task_type: str,
    steps: list[dict],
    assumptions: list[str] | None = None,
    constraints: list[str] | None = None,
    research_questions: list[str] | None = None,
    research_references: list[str] | None = None,
) -> str:
    """Persist an executable plan with dependencies and validation criteria."""
    session_id = get_tool_context().task_session_id
    if not session_id:
        return "Error: Planner session is not initialized."
    memory = _open_memory()
    try:
        task = memory.create_task_plan(
            session_id,
            goal,
            task_type,
            steps,
            assumptions=assumptions,
            constraints=constraints,
            research_questions=research_questions,
            research_references=research_references,
        )
        _context = get_tool_context()
        _context.task_session_id = session_id
        _context.task_id = task["task_id"]
        set_tool_context(_context)
        return (
            "Plan saved in SQLite with status awaiting_approval. It cannot be "
            "executed until the user reviews and approves it with /tasks.\n"
            + _format_task(task)
        )
    finally:
        memory.close()


@tool(args_schema=InspectTaskPlanInput)
def inspect_task_plan(task_id: str | None = None) -> str:
    """Inspect the current SQLite task plan and its persisted todo progress."""
    session_id = get_tool_context().task_session_id
    selected_id = task_id or get_tool_context().task_id
    if not session_id or not selected_id:
        return "No active task plan is selected."
    memory = _open_memory()
    try:
        task = memory.get_task(selected_id)
        if task is None:
            return "Error: Task plan was not found."
        if (
            task["session_id"] != session_id
            and selected_id != get_tool_context().task_id
        ):
            return "Error: Task plan is not linked to the active session."
        return _format_task(task)
    finally:
        memory.close()


@tool(args_schema=UpdateTodoStepInput)
def update_todo_step(step_id: str, status: str, evidence: str = "") -> str:
    """Record progress for a step in the approved active task plan."""
    session_id = get_tool_context().task_session_id
    task_id = get_tool_context().task_id
    if not session_id or not task_id:
        return "Error: No task plan is selected. Create or resume one first."
    memory = _open_memory()
    try:
        task = memory.get_task(task_id)
        if task is None or (
            task["session_id"] != session_id and task_id != get_tool_context().task_id
        ):
            return "Error: Active task plan is unavailable."
        updated = memory.update_task_step(
            task_id,
            task["revision"],
            step_id,
            status,
            evidence,
            owner_id=get_tool_context().runtime_owner_id,
        )
        return "Todo progress persisted in SQLite.\n" + _format_task(updated)
    except (ValueError, OSError) as exc:
        return f"Error updating todo progress: {exc}"
    finally:
        memory.close()


@tool(args_schema=ReviseTaskPlanInput)
def revise_task_plan(
    goal: str,
    task_type: str,
    expected_revision: int,
    steps: list[dict],
    assumptions: list[str] | None = None,
    constraints: list[str] | None = None,
    research_questions: list[str] | None = None,
    research_references: list[str] | None = None,
) -> str:
    """Create a new immutable plan revision; prior approval is invalidated."""
    session_id = get_tool_context().task_session_id
    task_id = get_tool_context().task_id
    if not session_id or not task_id:
        return "Error: No active task plan is selected."
    memory = _open_memory()
    try:
        task = memory.get_task(task_id)
        if task is None:
            return "Error: Active task plan was not found."
        revised = memory.revise_task_plan(
            task_id,
            goal,
            task_type,
            steps,
            expected_revision=expected_revision,
            assumptions=assumptions,
            constraints=constraints,
            research_questions=research_questions,
            research_references=research_references,
        )
        return (
            "Revised plan saved in SQLite; approval was invalidated. Review it "
            "with /tasks before continuing.\n" + _format_task(revised)
        )
    except (ValueError, OSError) as exc:
        return f"Error revising task plan: {exc}"
    finally:
        memory.close()
