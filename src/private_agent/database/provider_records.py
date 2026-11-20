"""Persistence for the provider record each session runs with (I1p)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

from ..providers.adapters import ProviderRecord

_RECORD_COLUMNS = (
    "adapter_type",
    "display_name",
    "base_url",
    "model",
    "needs_tools",
    "needs_vision",
    "context_window",
)


class ProviderRecordMixin:
    """Store and read the provider a session (or resumed task) was run with."""

    def record_session_provider(self, session_id: str, record: ProviderRecord) -> None:
        """Insert or replace the provider record for a session."""
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("A session id is required to record a provider.")
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO session_providers "
                "(session_id, adapter_type, display_name, base_url, model, "
                "needs_tools, needs_vision, context_window, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "adapter_type = excluded.adapter_type, "
                "display_name = excluded.display_name, "
                "base_url = excluded.base_url, "
                "model = excluded.model, "
                "needs_tools = excluded.needs_tools, "
                "needs_vision = excluded.needs_vision, "
                "context_window = excluded.context_window, "
                "updated_at = CURRENT_TIMESTAMP",
                (
                    session_id,
                    record.adapter_type,
                    record.display_name,
                    record.base_url,
                    record.model,
                    int(bool(record.needs_tools)),
                    int(bool(record.needs_vision)),
                    record.context_window,
                ),
            )

    def get_session_provider(self, session_id: str) -> ProviderRecord | None:
        """Return the provider record saved for a session, when there is one."""
        session_id = str(session_id or "").strip()
        if not session_id:
            return None
        try:
            row = self.conn.execute(
                "SELECT adapter_type, display_name, base_url, model, "
                "needs_tools, needs_vision, context_window "
                "FROM session_providers WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            # Older databases created before the table existed.
            return None
        if row is None:
            return None
        return ProviderRecord(
            adapter_type=row[0],
            display_name=row[1],
            base_url=row[2],
            model=row[3],
            needs_tools=bool(row[4]),
            needs_vision=bool(row[5]),
            context_window=row[6],
        )

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
