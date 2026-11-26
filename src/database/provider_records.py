"""Persistence for the provider record each session or task runs with (I1p)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

from providers.adapters import ProviderRecord

_RECORD_COLUMNS = (
    "adapter_type",
    "display_name",
    "base_url",
    "model",
    "needs_tools",
    "needs_vision",
    "context_window",
)


def _record_from_row(row) -> ProviderRecord:
    return ProviderRecord(
        adapter_type=row[0],
        display_name=row[1],
        base_url=row[2],
        model=row[3],
        needs_tools=bool(row[4]),
        needs_vision=bool(row[5]),
        context_window=row[6],
    )


def _record_values(record: ProviderRecord) -> tuple:
    return (
        record.adapter_type,
        record.display_name,
        record.base_url,
        record.model,
        int(bool(record.needs_tools)),
        int(bool(record.needs_vision)),
        record.context_window,
    )


class ProviderRecordMixin:
    """Store and read the provider a session or task was run with."""

    def record_session_provider(self, session_id: str, record: ProviderRecord) -> None:
        """Insert or replace the provider record for a session."""
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("A session id is required to record a provider.")
        self._write_provider_record(
            "session_providers", "session_id", session_id, record
        )

    def get_session_provider(self, session_id: str) -> ProviderRecord | None:
        """Return the provider record saved for a session, when there is one."""
        return self._read_provider_record("session_providers", "session_id", session_id)

    def record_task_provider(self, task_id: str, record: ProviderRecord) -> None:
        """Insert or replace the provider record for an agent task."""
        task_id = str(task_id or "").strip()
        if not task_id:
            raise ValueError("A task id is required to record a provider.")
        self._write_provider_record("task_providers", "task_id", task_id, record)

    def get_task_provider(self, task_id: str) -> ProviderRecord | None:
        """Return the provider record saved for an agent task, when there is one."""
        return self._read_provider_record("task_providers", "task_id", task_id)

    def forget_task_provider(self, task_id: str) -> None:
        """Drop a task's provider record (used when the task is deleted)."""
        task_id = str(task_id or "").strip()
        if not task_id:
            return
        with self._lock, self.conn:
            self.conn.execute(
                "DELETE FROM task_providers WHERE task_id = ?", (task_id,)
            )

    def _write_provider_record(
        self, table: str, key_column: str, key: str, record: ProviderRecord
    ) -> None:
        with self._lock, self.conn:
            # table/key_column come from fixed internal constants; values stay bound.
            self.conn.execute(
                f"INSERT INTO {table} "  # nosec B608
                f"({key_column}, adapter_type, display_name, base_url, model, "
                "needs_tools, needs_vision, context_window, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                f"ON CONFLICT({key_column}) DO UPDATE SET "
                "adapter_type = excluded.adapter_type, "
                "display_name = excluded.display_name, "
                "base_url = excluded.base_url, "
                "model = excluded.model, "
                "needs_tools = excluded.needs_tools, "
                "needs_vision = excluded.needs_vision, "
                "context_window = excluded.context_window, "
                "updated_at = CURRENT_TIMESTAMP",
                (key, *_record_values(record)),
            )

    def _read_provider_record(
        self, table: str, key_column: str, key: str
    ) -> ProviderRecord | None:
        key = str(key or "").strip()
        if not key:
            return None
        try:
            row = self.conn.execute(
                "SELECT adapter_type, display_name, base_url, model, "
                # table/key_column come from fixed internal constants; values are bound
                # nosec B608
                f"needs_tools, needs_vision, context_window FROM {table} "
                f"WHERE {key_column} = ?",
                (key,),
            ).fetchone()
        except sqlite3.OperationalError:
            # Older databases created before the table existed.
            return None
        return _record_from_row(row) if row is not None else None

    def forget_session_provider(self, session_id: str) -> None:
        """Drop a session's provider record (used when history is deleted)."""
        session_id = str(session_id or "").strip()
        if not session_id:
            return
        with self._lock, self.conn:
            self.conn.execute(
                "DELETE FROM session_providers WHERE session_id = ?", (session_id,)
            )


def provider_record_payload(record: ProviderRecord) -> dict:
    """Return a JSON-safe dict of a provider record for task event details."""
    return json.loads(json.dumps(asdict(record)))
