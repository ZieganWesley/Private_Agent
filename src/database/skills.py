"""Registered skill-file bookkeeping."""

from config import (
    DEFAULT_EPISODIC_SUMMARIES,
)


class SkillMixin:
    """Registered skill-file bookkeeping."""

    def register_skill_file(
        self,
        session_id: str,
        skill_file_path: str,
        description: str,
        *,
        source_task_id: str | None = None,
    ) -> int:
        """Associate a successfully created skill artifact with its originating session."""
        with self._lock, self.conn:
            cursor = self.conn.execute(
                "INSERT INTO chat_history "
                "(session_id, role, content, is_skill_file, skill_file_path, "
                "skill_task_id) VALUES (?, 'skill', ?, ?, ?, ?)",
                (
                    session_id,
                    description,
                    True,
                    skill_file_path,
                    source_task_id,
                ),
            )
            return int(cursor.lastrowid)

    def get_recent_skill_files(
        self,
        limit: int = DEFAULT_EPISODIC_SUMMARIES,
    ) -> list[dict[str, object]]:
        """Return recently registered skill artifacts and their source session."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, session_id, timestamp, content, skill_file_path "
                "FROM chat_history WHERE is_skill_file = 1 "
                "ORDER BY id DESC LIMIT ?",
                (max(0, limit),),
            ).fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "description": row[3],
                "path": row[4],
            }
            for row in rows
        ]

    def count_registered_skill_files(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM chat_history WHERE is_skill_file = 1"
            ).fetchone()
        return int(row[0])
