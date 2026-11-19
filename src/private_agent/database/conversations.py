"""Conversation history, summaries and retention."""

from datetime import datetime, timedelta, timezone

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..config import (
    DEFAULT_HISTORY_MESSAGES,
    MAX_SUMMARY_CHARS,
)


class ConversationMixin:
    """Conversation history, summaries and retention."""

    def save_message(self, session_id: str, role: str, content: str):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )

    def get_history(
        self,
        session_id: str | None = None,
        limit: int = DEFAULT_HISTORY_MESSAGES,
    ) -> list[BaseMessage]:
        with self._lock:
            cursor = self.conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT session_id, role, content, timestamp FROM chat_history WHERE session_id = ? AND role != 'summary' ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                )
            else:
                cursor.execute(
                    "SELECT session_id, role, content, timestamp FROM chat_history WHERE role != 'summary' ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()

        messages = []
        for r in reversed(rows):
            role = r[1].lower()
            content = r[2]
            if role in ("human", "user"):
                messages.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "tool":
                messages.append(ToolMessage(content=content, tool_call_id="unknown"))
            elif role == "summary":
                continue
            else:
                messages.append(HumanMessage(content=content))
        return messages

    def get_latest_session_id(self) -> str | None:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT session_id FROM chat_history WHERE role != 'summary' "
                "AND session_id IS NOT NULL ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def list_sessions(self) -> list[str]:
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT session_id FROM chat_history "
                "WHERE role NOT IN ('summary', 'skill') "
                "AND session_id IS NOT NULL GROUP BY session_id ORDER BY MAX(id) DESC"
            )
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def session_exists(self, session_id: str) -> bool:
        """Return whether any saved row belongs to this session id."""
        if not session_id:
            return False
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM chat_history WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None

    def ends_with_summary(self, session_id: str) -> bool:
        """Return whether a summary is already the newest saved row."""
        if not session_id:
            return False
        with self._lock:
            row = self.conn.execute(
                "SELECT role FROM chat_history WHERE session_id = ? "
                "AND role != 'skill' ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return bool(row) and row[0] == "summary"

    def list_sessions_with_metadata(self, limit: int = 20) -> list[dict]:
        """Return recent session ids with message counts and last activity."""
        limit = max(1, min(int(limit), 200))
        with self._lock:
            rows = self.conn.execute(
                "SELECT session_id, COUNT(*), MAX(timestamp) FROM chat_history "
                "WHERE role NOT IN ('summary', 'skill') "
                "AND session_id IS NOT NULL "
                "GROUP BY session_id ORDER BY MAX(id) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"session_id": row[0], "messages": int(row[1]), "last_at": row[2]}
            for row in rows
        ]

    def count_conversation_messages(self) -> int:
        """Count saved chat/tool messages, excluding summaries and skill links."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM chat_history "
                "WHERE role NOT IN ('summary', 'skill')"
            ).fetchone()
        return int(row[0])

    def count_conversation_sessions(self) -> int:
        """Count sessions containing saved chat/tool messages."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM chat_history "
                "WHERE role NOT IN ('summary', 'skill') AND session_id IS NOT NULL"
            ).fetchone()
        return int(row[0])

    def load_history(
        self,
        session_id: str | None = None,
        limit: int = DEFAULT_HISTORY_MESSAGES,
    ) -> list[BaseMessage]:
        """Alias for get_history to support test expectations."""
        return self.get_history(session_id, limit)

    def clear_history(self, session_id: str | None = None):
        with self._lock, self.conn:
            if session_id:
                self.conn.execute(
                    "DELETE FROM agent_learned_items WHERE source_task_id IN "
                    "(SELECT task_id FROM agent_tasks "
                    "WHERE originating_session_id = ?)",
                    (session_id,),
                )
                self.conn.execute(
                    "DELETE FROM agent_tasks WHERE originating_session_id = ?",
                    (session_id,),
                )
                self.conn.execute(
                    "DELETE FROM chat_history WHERE session_id = ?", (session_id,)
                )
            else:
                self.conn.execute("DELETE FROM agent_learned_items")
                self.conn.execute("DELETE FROM agent_tasks")
                self.conn.execute("DELETE FROM chat_history")

    def save_summary(self, session_id: str, summary: str):
        """Save an episodic session summary to the database."""
        summary = str(summary).strip()
        if not summary:
            return
        if len(summary) > MAX_SUMMARY_CHARS:
            raise ValueError(
                f"Episode summary exceeds the configured {MAX_SUMMARY_CHARS}-"
                "character limit; provide a schema-informed concise summary."
            )
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, "summary", summary),
            )

    def prune_history(self, retention_days: int) -> int:
        """Delete messages and derived summaries older than the retention window."""
        if retention_days <= 0:
            return 0
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=retention_days)
        ).isoformat(sep=" ", timespec="seconds")
        with self._lock, self.conn:
            cursor = self.conn.execute(
                "DELETE FROM chat_history WHERE timestamp < ?", (cutoff,)
            )
            return cursor.rowcount
