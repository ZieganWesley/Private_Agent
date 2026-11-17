"""Episodic summaries and their search indexes."""

import logging
import re
import sqlite3

from ..config import (
    DEFAULT_EPISODIC_SUMMARIES,
)

_LOGGER = logging.getLogger("private_agent.database.memory")
_SCHEMA_VERSION = 1


class EpisodeMixin:
    """Episodic summaries and their search indexes."""

    @staticmethod
    def _sanitize_episode_text(value: str) -> str:
        value = re.sub(
            r"(?i)\b(api[_-]?key|access[_-]?token|password|secret)\b"
            r"(\s*[:=]\s*)[^\s,;]+",
            r"\1\2[redacted]",
            value,
        )
        value = re.sub(
            r"(?<![\w])/(?:home|Users|tmp|var|opt|mnt)/[^\s,;)\]]+",
            "[local path]",
            value,
        )
        return value.strip()

    def get_all_episodic_summaries(
        self,
        limit: int = DEFAULT_EPISODIC_SUMMARIES,
        session_id: str | None = None,
    ) -> list[str]:
        """Retrieve recent summaries, optionally restricted to one session."""
        with self._lock:
            cursor = self.conn.cursor()
            if session_id is None:
                cursor.execute(
                    "SELECT content FROM ("
                    "SELECT content, id FROM chat_history WHERE role = 'summary' "
                    "ORDER BY id DESC LIMIT ?"
                    ") ORDER BY id ASC",
                    (max(0, limit),),
                )
            else:
                cursor.execute(
                    "SELECT content FROM ("
                    "SELECT content, id FROM chat_history "
                    "WHERE role = 'summary' AND session_id = ? "
                    "ORDER BY id DESC LIMIT ?"
                    ") ORDER BY id ASC",
                    (session_id, max(0, limit)),
                )
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def list_episodic_memories(self) -> list[dict[str, object]]:
        """Return all saved session summaries for user review."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, session_id, timestamp, content FROM chat_history "
                "WHERE role = 'summary' ORDER BY id"
            ).fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "content": row[3],
            }
            for row in rows
        ]

    def count_episodic_memories(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM chat_history WHERE role = 'summary'"
            ).fetchone()
        return int(row[0])

    def get_recent_episodic_memories(
        self,
        limit: int = DEFAULT_EPISODIC_SUMMARIES,
    ) -> list[dict[str, object]]:
        """Return a bounded chronological window of prior episode summaries."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id, session_id, timestamp, content, episode_topic, "
                "episode_description, episode_task_summary, episode_plan_json, "
                "episode_outcome FROM chat_history "
                "WHERE role = 'summary' ORDER BY id DESC LIMIT ?",
                (max(0, limit),),
            ).fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "timestamp": row[2],
                "content": row[3],
                "topic": row[4],
                "description": row[5],
                "task_summary": row[6],
                "plan": self._decode_episode_plan(row[7], row[0]),
                "outcome": row[8],
            }
            for row in reversed(rows)
        ]

    def search_relevant_episodic_memories(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, object]]:
        """Return a small deterministic lexical match set from recent episodes."""
        limit = max(1, min(int(limit), 100))
        stop_words = {
            "about",
            "after",
            "also",
            "and",
            "are",
            "for",
            "from",
            "have",
            "into",
            "its",
            "make",
            "need",
            "that",
            "the",
            "this",
            "with",
            "what",
            "when",
            "where",
            "which",
            "will",
            "would",
            "topic",
            "task",
            "plan",
            "steps",
            "verified",
        }
        query_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", query.casefold())
            if len(term) > 2 and term not in stop_words
        }
        if not query_terms:
            return []
        with self._lock:
            rows = None
            if self._episode_fts_enabled:
                expression = " OR ".join(f'"{term}"' for term in sorted(query_terms))
                try:
                    rows = self.conn.execute(
                        "SELECT history.id, history.session_id, history.timestamp, "
                        "history.content, history.episode_topic, "
                        "history.episode_description, history.episode_task_summary, "
                        "history.episode_plan_json, history.episode_outcome "
                        "FROM agent_episode_fts "
                        "JOIN chat_history AS history "
                        "ON history.id = agent_episode_fts.rowid "
                        "WHERE agent_episode_fts MATCH ? "
                        "AND history.role = 'summary' "
                        "ORDER BY bm25(agent_episode_fts), history.id DESC LIMIT 500",
                        (expression,),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    self._episode_fts_enabled = False
                    _LOGGER.warning(
                        "Episode FTS5 query failed; using lexical fallback: %s",
                        str(exc)[:200],
                    )
            if rows is None:
                rows = self.conn.execute(
                    "SELECT id, session_id, timestamp, content, episode_topic, "
                    "episode_description, episode_task_summary, episode_plan_json, "
                    "episode_outcome FROM chat_history WHERE role = 'summary' "
                    "ORDER BY id DESC LIMIT 500"
                ).fetchall()
        matches = []
        minimum_overlap = (
            1 if len(query_terms) == 1 else max(2, (len(query_terms) + 3) // 4)
        )
        for row in rows:
            searchable = " ".join(
                str(value or "") for value in (row[3], row[4], row[5], row[6])
            ).casefold()
            episode_terms = set(re.findall(r"[a-z0-9]+", searchable))
            overlap = query_terms & episode_terms
            if len(overlap) < minimum_overlap:
                continue
            matches.append(
                (
                    len(overlap) / len(query_terms),
                    len(overlap),
                    row[0],
                    {
                        "id": row[0],
                        "session_id": row[1],
                        "timestamp": row[2],
                        "content": row[3],
                        "topic": row[4],
                        "description": row[5],
                        "task_summary": row[6],
                        "plan": self._decode_episode_plan(row[7], row[0]),
                        "outcome": row[8],
                    },
                )
            )
        matches.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        return [item[3] for item in matches[:limit]]

    def get_episodic_memory(self, entry_id: int) -> dict[str, object] | None:
        """Return one saved session summary by its SQLite row ID."""
        with self._lock:
            row = self.conn.execute(
                "SELECT id, session_id, timestamp, content, episode_topic, "
                "episode_description, episode_task_summary, episode_plan_json, "
                "episode_outcome FROM chat_history "
                "WHERE id = ? AND role = 'summary'",
                (entry_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "session_id": row[1],
            "timestamp": row[2],
            "content": row[3],
            "topic": row[4],
            "description": row[5],
            "task_summary": row[6],
            "plan": self._decode_episode_plan(row[7], row[0]),
            "outcome": row[8],
        }

    def delete_episodic_memory(self, entry_id: int) -> bool:
        """Delete one summary row without deleting the associated conversation."""
        with self._lock, self.conn:
            cursor = self.conn.execute(
                "DELETE FROM chat_history WHERE id = ? AND role = 'summary'",
                (entry_id,),
            )
            return cursor.rowcount == 1
