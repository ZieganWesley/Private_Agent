"""Learned-item storage, review and retrieval."""

import re
import uuid
from datetime import datetime

from config import (
    LEARNED_ITEM_EXPIRY_DAYS,
    LEARNED_ITEM_REVIEW_DAYS,
    MAX_LEARNED_ITEM_CHARS,
)


class LearningMixin:
    """Learned-item storage, review and retrieval."""

    @staticmethod
    def _validate_learned_statement(statement: str) -> str:
        statement = str(statement).strip()
        if not statement or len(statement) > MAX_LEARNED_ITEM_CHARS:
            raise ValueError(
                "Learned statement must contain 1 to "
                f"{MAX_LEARNED_ITEM_CHARS} characters."
            )
        sensitive_patterns = (
            r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret|"
            r"private[_-]?key)\b\s*[:=]\s*\S+",
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            r"\bAKIA[0-9A-Z]{16}\b",
            r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b",
            r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})"
            r"[ .-]\d{3}[ .-]\d{4}(?!\w)",
            r"(?<![\w])/(?:home|Users)/[^/\s]+(?:/[^\s]*)?",
            r"(?i)\b[A-Z]:\\Users\\[^\\\s]+(?:\\[^\s]*)?",
        )
        if any(re.search(pattern, statement) for pattern in sensitive_patterns):
            raise ValueError(
                "Learned statements cannot contain credentials, contact details, "
                "government identifiers, or user-specific filesystem paths."
            )
        return statement

    def learning_is_enabled(self) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT setting_value FROM agent_learning_settings "
                "WHERE setting_key = 'enabled'"
            ).fetchone()
        return bool(row and row[0] == "1")

    def set_learning_enabled(self, enabled: bool) -> None:
        with self._lock, self._write_transaction():
            self.conn.execute(
                "INSERT INTO agent_learning_settings "
                "(setting_key, setting_value) VALUES ('enabled', ?) "
                "ON CONFLICT(setting_key) DO UPDATE SET "
                "setting_value = excluded.setting_value",
                ("1" if enabled else "0",),
            )

    def add_learned_item(
        self,
        statement: str,
        scope: str,
        *,
        provenance: str = "user-confirmed",
        source_task_id: str | None = None,
    ) -> dict:
        statement = self._validate_learned_statement(statement)
        scope = str(scope).strip().lower()
        provenance = str(provenance).strip()
        if scope not in {"global", "coding", "research", "other"}:
            raise ValueError(
                "Learning scope must be global, coding, research, or other."
            )
        if not provenance or len(provenance) > 100:
            raise ValueError("Learning provenance must contain 1 to 100 characters.")
        item_id = uuid.uuid4().hex
        with self._lock, self._write_transaction():
            enabled = self.conn.execute(
                "SELECT setting_value FROM agent_learning_settings "
                "WHERE setting_key = 'enabled'"
            ).fetchone()
            if enabled is None or enabled[0] != "1":
                raise ValueError("Adaptive learning is disabled.")
            if source_task_id:
                source = self.conn.execute(
                    "SELECT status FROM agent_tasks WHERE task_id = ?",
                    (source_task_id,),
                ).fetchone()
                if source is None or source[0] != "completed":
                    raise ValueError(
                        "Learned task procedures must reference a completed task."
                    )
            self.conn.execute(
                "INSERT INTO agent_learned_items "
                "(item_id, statement, scope, provenance, source_task_id, "
                "expires_at, review_at) "
                "VALUES (?, ?, ?, ?, ?, "
                "datetime('now', ?), datetime('now', ?))",
                (
                    item_id,
                    statement,
                    scope,
                    provenance,
                    source_task_id,
                    f"+{LEARNED_ITEM_EXPIRY_DAYS} days",
                    f"+{LEARNED_ITEM_REVIEW_DAYS} days",
                ),
            )
        return self.get_learned_item(item_id)

    def list_learned_items(self, *, include_disabled: bool = True) -> list[dict]:
        query = "SELECT item_id FROM agent_learned_items"
        if not include_disabled:
            query += " WHERE status = 'active'"
        query += " ORDER BY updated_at DESC LIMIT 200"
        with self._lock:
            rows = self.conn.execute(query).fetchall()
        return [
            item for row in rows if (item := self.get_learned_item(row[0])) is not None
        ]

    def get_learned_item(self, item_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT item_id, statement, scope, provenance, source_task_id, "
                "status, created_at, updated_at, expires_at, review_at "
                "FROM agent_learned_items "
                "WHERE item_id = ?",
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "item_id": row[0],
            "statement": row[1],
            "scope": row[2],
            "provenance": row[3],
            "source_task_id": row[4],
            "status": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "expires_at": row[8],
            "review_at": row[9],
            "expired": bool(
                row[8] and row[8] <= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
            "review_due": bool(
                row[9] and row[9] <= datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ),
        }

    def update_learned_item(
        self,
        item_id: str,
        statement: str,
        scope: str,
    ) -> dict:
        statement = self._validate_learned_statement(statement)
        scope = str(scope).strip().lower()
        if scope not in {"global", "coding", "research", "other"}:
            raise ValueError(
                "Learning scope must be global, coding, research, or other."
            )
        with self._lock, self._write_transaction():
            cursor = self.conn.execute(
                "UPDATE agent_learned_items SET statement = ?, scope = ?, "
                "provenance = 'user-corrected', status = 'active', "
                "expires_at = datetime('now', ?), "
                "review_at = datetime('now', ?), "
                "updated_at = CURRENT_TIMESTAMP WHERE item_id = ?",
                (
                    statement,
                    scope,
                    f"+{LEARNED_ITEM_EXPIRY_DAYS} days",
                    f"+{LEARNED_ITEM_REVIEW_DAYS} days",
                    item_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Learned item was not found.")
        return self.get_learned_item(item_id)

    def review_learned_item(self, item_id: str) -> dict:
        """Renew an item only after the user explicitly confirms it remains useful."""
        with self._lock, self._write_transaction():
            cursor = self.conn.execute(
                "UPDATE agent_learned_items SET status = 'active', "
                "expires_at = datetime('now', ?), "
                "review_at = datetime('now', ?), "
                "updated_at = CURRENT_TIMESTAMP WHERE item_id = ?",
                (
                    f"+{LEARNED_ITEM_EXPIRY_DAYS} days",
                    f"+{LEARNED_ITEM_REVIEW_DAYS} days",
                    item_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("Learned item was not found.")
        return self.get_learned_item(item_id)

    def set_learned_item_status(self, item_id: str, status: str) -> dict:
        if status not in {"active", "disabled"}:
            raise ValueError("Learned item status must be active or disabled.")
        with self._lock, self._write_transaction():
            cursor = self.conn.execute(
                "UPDATE agent_learned_items SET status = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE item_id = ?",
                (status, item_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Learned item was not found.")
        return self.get_learned_item(item_id)

    def delete_learned_item(self, item_id: str) -> bool:
        with self._lock, self._write_transaction():
            cursor = self.conn.execute(
                "DELETE FROM agent_learned_items WHERE item_id = ?",
                (item_id,),
            )
            return cursor.rowcount == 1

    def get_relevant_learned_items(
        self,
        task_type: str,
        limit: int = 5,
    ) -> list[dict]:
        if task_type not in {"coding", "research", "other"}:
            task_type = "other"
        limit = max(1, min(int(limit), 10))
        with self._lock:
            enabled = self.conn.execute(
                "SELECT setting_value FROM agent_learning_settings "
                "WHERE setting_key = 'enabled'"
            ).fetchone()
            if enabled is None or enabled[0] != "1":
                return []
            rows = self.conn.execute(
                "SELECT item_id FROM agent_learned_items "
                "WHERE status = 'active' "
                "AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP) "
                "AND (review_at IS NULL OR review_at > CURRENT_TIMESTAMP) "
                "AND scope IN ('global', ?) "
                "ORDER BY updated_at DESC LIMIT ?",
                (task_type, limit),
            ).fetchall()
        return [
            item for row in rows if (item := self.get_learned_item(row[0])) is not None
        ]
