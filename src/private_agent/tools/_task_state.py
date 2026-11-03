"""Process-local binding for planner tools; durable task state lives in SQLite."""

import uuid

ACTIVE_TASK_SESSION_ID: str | None = None
ACTIVE_TASK_ID: str | None = None
RUNTIME_OWNER_ID = uuid.uuid4().hex
SCHEMA_READ_TABLES: set[str] = set()


def set_runtime_owner(owner_id: str | None = None) -> str:
    global RUNTIME_OWNER_ID
    RUNTIME_OWNER_ID = owner_id or uuid.uuid4().hex
    return RUNTIME_OWNER_ID


def set_active_task_context(
    session_id: str | None,
    task_id: str | None = None,
) -> None:
    global ACTIVE_TASK_SESSION_ID, ACTIVE_TASK_ID
    ACTIVE_TASK_SESSION_ID = session_id
    ACTIVE_TASK_ID = task_id


def record_schema_read(table_names: set[str]) -> None:
    SCHEMA_READ_TABLES.update(table_names)


def clear_schema_reads() -> None:
    SCHEMA_READ_TABLES.clear()
