"""Durable missions, append-only events, and a single-runner lease."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

_LOGGER = logging.getLogger("private_agent.database.missions")
MISSION_STATUSES = {
    "planning",
    "active",
    "paused",
    "waiting_user",
    "completed",
    "failed",
    "stopped",
}
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_TOOL_CALL_BUDGET = 60
DEFAULT_LEASE_SECONDS = 300.0


class MissionMixin:
    """Mission state for the /mission controller."""

    def _ensure_mission_tables(self) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS missions (
                    mission_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    iteration INTEGER NOT NULL DEFAULT 0,
                    max_iterations INTEGER NOT NULL DEFAULT 20,
                    deadline REAL,
                    tool_call_budget INTEGER NOT NULL DEFAULT 60,
                    tool_calls_used INTEGER NOT NULL DEFAULT 0,
                    plan_json TEXT NOT NULL DEFAULT '{}',
                    progress_json TEXT NOT NULL DEFAULT '{}',
                    last_error TEXT,
                    completion_evidence TEXT,
                    revision INTEGER NOT NULL DEFAULT 1,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mission_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mission_events_mission "
                "ON mission_events(mission_id, id)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_missions_session "
                "ON missions(session_id, updated_at)"
            )

    @staticmethod
    def _json(value: Any, *, limit: int = 200_000) -> str:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return text[:limit]

    @staticmethod
    def _decode(text: str | None, default: Any) -> Any:
        if not text:
            return default
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return default

    def _mission_from_row(self, row: tuple) -> dict:
        return {
            "mission_id": row[0],
            "session_id": row[1],
            "goal": row[2],
            "status": row[3],
            "iteration": int(row[4]),
            "max_iterations": int(row[5]),
            "deadline": row[6],
            "tool_call_budget": int(row[7]),
            "tool_calls_used": int(row[8]),
            "plan": self._decode(row[9], {}),
            "progress": self._decode(row[10], {}),
            "last_error": row[11],
            "completion_evidence": row[12],
            "revision": int(row[13]),
            "lease_owner": row[14],
            "lease_expires_at": row[15],
            "created_at": row[16],
            "updated_at": row[17],
        }

    def create_mission(
        self,
        session_id: str,
        goal: str,
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tool_call_budget: int = DEFAULT_TOOL_CALL_BUDGET,
        deadline: float | None = None,
        plan: dict | None = None,
        status: str = "planning",
    ) -> dict:
        """Create a mission and its first audit event."""
        self._ensure_mission_tables()
        session_id = str(session_id or "").strip()
        goal = str(goal or "").strip()
        if not session_id:
            raise ValueError("Mission requires a session ID.")
        if not goal or len(goal) > 4000:
            raise ValueError("Mission goal must contain 1 to 4000 characters.")
        if status not in MISSION_STATUSES:
            raise ValueError("Unknown mission status.")
        max_iterations = max(1, min(int(max_iterations), 100))
        tool_call_budget = max(1, min(int(tool_call_budget), 1000))
        mission_id = uuid.uuid4().hex
        plan_json = self._json(plan or {})
        with self._lock, self._write_transaction():
            self.conn.execute(
                "INSERT INTO missions "
                "(mission_id, session_id, goal, status, max_iterations, deadline, "
                "tool_call_budget, plan_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mission_id,
                    session_id,
                    goal,
                    status,
                    max_iterations,
                    deadline,
                    tool_call_budget,
                    plan_json,
                ),
            )
        self.append_mission_event(
            mission_id,
            "created",
            {"goal": goal, "status": status},
        )
        return self.get_mission(mission_id)

    def get_mission(self, mission_id: str) -> dict | None:
        self._ensure_mission_tables()
        with self._lock:
            row = self.conn.execute(
                "SELECT mission_id, session_id, goal, status, iteration, "
                "max_iterations, deadline, tool_call_budget, tool_calls_used, "
                "plan_json, progress_json, last_error, completion_evidence, "
                "revision, lease_owner, lease_expires_at, created_at, updated_at "
                "FROM missions WHERE mission_id = ?",
                (str(mission_id),),
            ).fetchone()
        return self._mission_from_row(row) if row is not None else None

    def list_missions(
        self,
        *,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        self._ensure_mission_tables()
        limit = max(1, min(int(limit), 100))
        clauses = []
        params: list[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(str(session_id))
        if status:
            clauses.append("status = ?")
            params.append(str(status))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._lock:
            # Only fixed clause fragments are interpolated; values stay bound.
            rows = self.conn.execute(
                "SELECT mission_id FROM missions "  # nosec B608
                f"{where} ORDER BY updated_at DESC LIMIT ?",
                params,
            ).fetchall()
        missions = []
        for (mission_id,) in rows:
            mission = self.get_mission(mission_id)
            if mission is not None:
                missions.append(mission)
        return missions

    def update_mission(
        self,
        mission_id: str,
        *,
        status: str | None = None,
        iteration: int | None = None,
        tool_calls_used: int | None = None,
        plan: dict | None = None,
        progress: dict | None = None,
        last_error: str | None = None,
        completion_evidence: str | None = None,
        expected_revision: int | None = None,
    ) -> dict:
        """Update mission state, bumping the revision atomically."""
        self._ensure_mission_tables()
        if status is not None and status not in MISSION_STATUSES:
            raise ValueError("Unknown mission status.")
        assignments = ["updated_at = CURRENT_TIMESTAMP", "revision = revision + 1"]
        params: list[Any] = []
        if status is not None:
            assignments.append("status = ?")
            params.append(status)
        if iteration is not None:
            assignments.append("iteration = ?")
            params.append(max(0, int(iteration)))
        if tool_calls_used is not None:
            assignments.append("tool_calls_used = ?")
            params.append(max(0, int(tool_calls_used)))
        if plan is not None:
            assignments.append("plan_json = ?")
            params.append(self._json(plan))
        if progress is not None:
            assignments.append("progress_json = ?")
            params.append(self._json(progress))
        if last_error is not None:
            assignments.append("last_error = ?")
            params.append(str(last_error)[:4000])
        if completion_evidence is not None:
            assignments.append("completion_evidence = ?")
            params.append(str(completion_evidence)[:8000])
        params.append(str(mission_id))
        where = "mission_id = ?"
        if expected_revision is not None:
            where += " AND revision = ?"
            params.append(int(expected_revision))
        with self._lock, self._write_transaction():
            # assignments/where are column names from a fixed set; values stay bound.
            cursor = self.conn.execute(
                f"UPDATE missions SET {', '.join(assignments)} WHERE {where}",  # nosec B608
                params,
            )
            if cursor.rowcount == 0:
                raise ValueError(
                    "Mission not found or its revision changed; refresh and retry."
                )
        self.append_mission_event(
            mission_id,
            "updated",
            {
                "status": status,
                "iteration": iteration,
                "revision_expected": expected_revision,
            },
        )
        mission = self.get_mission(mission_id)
        if mission is None:
            raise ValueError("Mission disappeared after update.")
        return mission

    def append_mission_event(
        self,
        mission_id: str,
        event_type: str,
        payload: dict | None = None,
    ) -> None:
        self._ensure_mission_tables()
        with self._lock, self._write_transaction():
            self.conn.execute(
                "INSERT INTO mission_events (mission_id, event_type, payload_json) "
                "VALUES (?, ?, ?)",
                (str(mission_id), str(event_type)[:80], self._json(payload or {})),
            )

    def list_mission_events(self, mission_id: str, *, limit: int = 100) -> list[dict]:
        self._ensure_mission_tables()
        limit = max(1, min(int(limit), 500))
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, event_type, payload_json, created_at "
                "FROM mission_events WHERE mission_id = ? ORDER BY id DESC LIMIT ?",
                (str(mission_id), limit),
            ).fetchall()
        return [
            {
                "id": int(row[0]),
                "event_type": row[1],
                "payload": self._decode(row[2], {}),
                "created_at": row[3],
            }
            for row in rows
        ]

    def acquire_mission_lease(
        self,
        mission_id: str,
        owner: str,
        *,
        ttl_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> bool:
        """Take the mission lease when it is free or expired."""
        self._ensure_mission_tables()
        now = time.time()
        expires = now + max(0.01, float(ttl_seconds))
        with self._lock, self._write_transaction():
            cursor = self.conn.execute(
                "UPDATE missions SET lease_owner = ?, lease_expires_at = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE mission_id = ? "
                "AND (lease_owner IS NULL OR lease_expires_at IS NULL "
                "OR lease_expires_at < ?)",
                (str(owner)[:120], expires, str(mission_id), now),
            )
            acquired = cursor.rowcount > 0
        if acquired:
            self.append_mission_event(
                mission_id, "lease_acquired", {"owner": str(owner)[:120]}
            )
        return acquired

    def release_mission_lease(self, mission_id: str, owner: str) -> bool:
        self._ensure_mission_tables()
        with self._lock, self._write_transaction():
            cursor = self.conn.execute(
                "UPDATE missions SET lease_owner = NULL, lease_expires_at = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE mission_id = ? AND lease_owner = ?",
                (str(mission_id), str(owner)[:120]),
            )
            released = cursor.rowcount > 0
        if released:
            self.append_mission_event(
                mission_id, "lease_released", {"owner": str(owner)[:120]}
            )
        return released

    def mission_lease_active(self, mission_id: str) -> bool:
        self._ensure_mission_tables()
        with self._lock:
            row = self.conn.execute(
                "SELECT lease_owner, lease_expires_at FROM missions "
                "WHERE mission_id = ?",
                (str(mission_id),),
            ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return False
        return float(row[1]) >= time.time()


__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TOOL_CALL_BUDGET",
    "MISSION_STATUSES",
    "MissionMixin",
]
