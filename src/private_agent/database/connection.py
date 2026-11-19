"""SQLite connection, schema creation and lifecycle."""

import logging
import pathlib
import re
import sqlite3
import threading
from contextlib import contextmanager

from ..config import (
    DEFAULT_DB_PATH,
)

_LOGGER = logging.getLogger("private_agent.database.memory")
_SCHEMA_VERSION = 1


class ConnectionMixin:
    """SQLite connection, schema creation and lifecycle."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = (
            str(pathlib.Path(db_path).expanduser())
            if db_path != ":memory:"
            else db_path
        )
        self._lock = threading.Lock()
        if self.db_path != ":memory:":
            pathlib.Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=2.0,
        )
        self._closed = False
        self._episode_fts_enabled = False
        try:
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA busy_timeout = 2000")
            if self.db_path != ":memory:":
                self.conn.execute("PRAGMA journal_mode = WAL")
                self.conn.execute("PRAGMA synchronous = NORMAL")
            self._create_table()
        except BaseException:  # noqa: BLE001 - closes a half-initialised connection for any failure, then re-raises
            self.conn.close()
            self._closed = True
            raise

    @contextmanager
    def _write_transaction(self):
        """Acquire the SQLite write lock up front; caller holds the instance lock."""
        try:
            self.conn.execute("BEGIN IMMEDIATE")
            yield
            self.conn.commit()
        except BaseException:  # noqa: BLE001 - rolls back for any failure, including cancellation, then re-raises
            self.conn.rollback()
            raise

    @classmethod
    def describe_table_schema(
        cls, connection: sqlite3.Connection, table_name: str
    ) -> dict:
        """Describe an allowlisted application table without exposing SQL execution."""
        if table_name not in cls.SCHEMA_TABLE_ALLOWLIST:
            raise ValueError("Schema inspection is limited to approved agent tables.")
        table = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if table is None:
            raise ValueError(f"Approved table {table_name!r} does not exist.")
        columns = connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        foreign_keys = connection.execute(
            f'PRAGMA foreign_key_list("{table_name}")'
        ).fetchall()
        indexes = connection.execute(f'PRAGMA index_list("{table_name}")').fetchall()
        index_details = []
        for index in indexes:
            index_name = str(index[1]).replace('"', '""')
            index_columns = connection.execute(
                f'PRAGMA index_info("{index_name}")'
            ).fetchall()
            index_details.append(
                {
                    "name": index[1],
                    "unique": bool(index[2]),
                    "origin": index[3],
                    "columns": [item[2] for item in index_columns],
                }
            )
        create_sql = table[0] or ""
        checks = re.findall(
            r"\bCHECK\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)",
            create_sql,
            flags=re.IGNORECASE,
        )
        return {
            "table": table_name,
            "columns": [
                {
                    "name": row[1],
                    "type": row[2] or "unspecified",
                    "not_null": bool(row[3]),
                    "default": row[4],
                    "primary_key_position": row[5],
                }
                for row in columns
            ],
            "foreign_keys": [
                {
                    "referenced_table": row[2],
                    "from_column": row[3],
                    "to_column": row[4],
                    "on_update": row[5],
                    "on_delete": row[6],
                }
                for row in foreign_keys
            ],
            "indexes": index_details,
            "declared_check_constraints": checks,
            "create_statement": create_sql,
            "length_limit_note": (
                "SQLite declared types do not imply a maximum string length. "
                "No explicit database maximum applies unless a CHECK constraint "
                "states one; application API limits are reported separately."
            ),
        }

    def get_table_schema(self, table_name: str) -> dict:
        """Describe an allowlisted application table using this connection."""
        with self._lock:
            return self.describe_table_schema(self.conn, table_name)

    def _configure_episode_fts(self) -> None:
        """Install a private FTS index when the SQLite build provides FTS5."""
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS agent_episode_fts USING "
                "fts5(session_id UNINDEXED, timestamp UNINDEXED, content, "
                "topic, description, task_summary)"
            )
            for trigger_sql in (
                """
                CREATE TRIGGER IF NOT EXISTS chat_history_episode_fts_insert
                AFTER INSERT ON chat_history
                WHEN NEW.role = 'summary'
                BEGIN
                    INSERT INTO agent_episode_fts
                    (rowid, session_id, timestamp, content, topic, description, task_summary)
                    VALUES
                    (NEW.id, NEW.session_id, NEW.timestamp, NEW.content,
                     NEW.episode_topic, NEW.episode_description, NEW.episode_task_summary);
                END
                """,
                """
                CREATE TRIGGER IF NOT EXISTS chat_history_episode_fts_delete
                AFTER DELETE ON chat_history
                WHEN OLD.role = 'summary'
                BEGIN
                    DELETE FROM agent_episode_fts WHERE rowid = OLD.id;
                END
                """,
                """
                CREATE TRIGGER IF NOT EXISTS chat_history_episode_fts_update
                AFTER UPDATE ON chat_history
                BEGIN
                    DELETE FROM agent_episode_fts
                    WHERE rowid = OLD.id AND OLD.role = 'summary';
                    INSERT INTO agent_episode_fts
                    (rowid, session_id, timestamp, content, topic, description, task_summary)
                    SELECT NEW.id, NEW.session_id, NEW.timestamp, NEW.content,
                           NEW.episode_topic, NEW.episode_description,
                           NEW.episode_task_summary
                    WHERE NEW.role = 'summary';
                END
                """,
            ):
                self.conn.execute(trigger_sql)
            self.conn.execute(
                "INSERT INTO agent_episode_fts "
                "(rowid, session_id, timestamp, content, topic, description, task_summary) "
                "SELECT id, session_id, timestamp, content, episode_topic, "
                "episode_description, episode_task_summary FROM chat_history "
                "WHERE role = 'summary' AND id NOT IN "
                "(SELECT rowid FROM agent_episode_fts)"
            )
            self._episode_fts_enabled = True
        except sqlite3.OperationalError as exc:
            self._episode_fts_enabled = False
            _LOGGER.warning(
                "Episode FTS5 unavailable; using bounded lexical retrieval: %s",
                str(exc)[:200],
            )

    def _create_table(self):
        with self._lock:  # noqa: SIM117 - body is the large schema DDL
            with self.conn:
                version = self.conn.execute("PRAGMA user_version").fetchone()[0]
                if version > _SCHEMA_VERSION:
                    raise sqlite3.DatabaseError(
                        f"Database schema version {version} is newer than the "
                        f"supported version {_SCHEMA_VERSION}."
                    )
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        is_skill_file BOOLEAN NOT NULL DEFAULT 0,
                        skill_file_path TEXT
                    )
                """)
                columns = {
                    row[1]
                    for row in self.conn.execute(
                        "PRAGMA table_info(chat_history)"
                    ).fetchall()
                }
                if "is_skill_file" not in columns:
                    self.conn.execute(
                        "ALTER TABLE chat_history ADD COLUMN "
                        "is_skill_file BOOLEAN NOT NULL DEFAULT 0"
                    )
                if "skill_file_path" not in columns:
                    self.conn.execute(
                        "ALTER TABLE chat_history ADD COLUMN skill_file_path TEXT"
                    )
                for column, column_type in (
                    ("episode_topic", "TEXT"),
                    ("episode_description", "TEXT"),
                    ("episode_task_summary", "TEXT"),
                    ("episode_plan_json", "TEXT"),
                    ("episode_outcome", "TEXT"),
                    ("episode_task_id", "TEXT"),
                    ("skill_task_id", "TEXT"),
                ):
                    if column not in columns:
                        self.conn.execute(
                            f"ALTER TABLE chat_history ADD COLUMN {column} "
                            f"{column_type}"
                        )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id, id)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_history_episode_task "
                    "ON chat_history(episode_task_id)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_history_skill_task "
                    "ON chat_history(skill_task_id)"
                )
                self._configure_episode_fts()
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_tasks (
                        task_id TEXT PRIMARY KEY,
                        originating_session_id TEXT NOT NULL,
                        goal TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        current_revision INTEGER NOT NULL,
                        approved_revision INTEGER,
                        approved_digest TEXT,
                        approved_scope TEXT,
                        user_rating INTEGER,
                        resume_state_json TEXT,
                        validation_revision INTEGER,
                        validation_evidence TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        checkpoint_at TEXT
                    )
                    """
                )
                task_columns = {
                    row[1]
                    for row in self.conn.execute(
                        "PRAGMA table_info(agent_tasks)"
                    ).fetchall()
                }
                if "user_rating" not in task_columns:
                    self.conn.execute(
                        "ALTER TABLE agent_tasks ADD COLUMN user_rating INTEGER"
                    )
                if "resume_state_json" not in task_columns:
                    self.conn.execute(
                        "ALTER TABLE agent_tasks ADD COLUMN resume_state_json TEXT"
                    )
                if "validation_revision" not in task_columns:
                    self.conn.execute(
                        "ALTER TABLE agent_tasks ADD COLUMN validation_revision INTEGER"
                    )
                if "validation_evidence" not in task_columns:
                    self.conn.execute(
                        "ALTER TABLE agent_tasks ADD COLUMN validation_evidence TEXT"
                    )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_plan_revisions (
                        task_id TEXT NOT NULL REFERENCES agent_tasks(task_id)
                            ON DELETE CASCADE,
                        revision INTEGER NOT NULL,
                        digest TEXT NOT NULL,
                        plan_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (task_id, revision)
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_todo_steps (
                        task_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        step_id TEXT NOT NULL,
                        position INTEGER NOT NULL,
                        description TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        evidence TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (task_id, revision, step_id),
                        FOREIGN KEY (task_id, revision)
                            REFERENCES agent_plan_revisions(task_id, revision)
                            ON DELETE CASCADE
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_task_events (
                        event_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL REFERENCES agent_tasks(task_id)
                            ON DELETE CASCADE,
                        revision INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        details_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_task_leases (
                        task_id TEXT PRIMARY KEY REFERENCES agent_tasks(task_id)
                            ON DELETE CASCADE,
                        owner_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        expires_at REAL NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_task_actions (
                        action_id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL REFERENCES agent_tasks(task_id)
                            ON DELETE CASCADE,
                        revision INTEGER NOT NULL,
                        tool_call_id TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        arguments_digest TEXT NOT NULL,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        outcome_digest TEXT,
                        UNIQUE(task_id, tool_call_id)
                    )
                    """
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_task_actions_status "
                    "ON agent_task_actions(task_id, status, started_at)"
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_learned_items (
                        item_id TEXT PRIMARY KEY,
                        statement TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        provenance TEXT NOT NULL,
                        source_task_id TEXT,
                        status TEXT NOT NULL DEFAULT 'active',
                        expires_at TEXT,
                        review_at TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                learned_columns = {
                    row[1]
                    for row in self.conn.execute(
                        "PRAGMA table_info(agent_learned_items)"
                    ).fetchall()
                }
                for column in ("expires_at", "review_at"):
                    if column not in learned_columns:
                        self.conn.execute(
                            f"ALTER TABLE agent_learned_items ADD COLUMN {column} TEXT"
                        )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_learned_status "
                    "ON agent_learned_items(status, updated_at DESC)"
                )
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_learning_settings (
                        setting_key TEXT PRIMARY KEY,
                        setting_value TEXT NOT NULL
                    )
                    """
                )
                self.conn.execute(
                    "INSERT OR IGNORE INTO agent_learning_settings "
                    "(setting_key, setting_value) VALUES ('enabled', '1')"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_tasks_status_updated "
                    "ON agent_tasks(status, updated_at DESC)"
                )
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_agent_task_events_task "
                    "ON agent_task_events(task_id, created_at)"
                )
                self.conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    def close(self):
        with self._lock:
            if not self._closed:
                self.conn.close()
                self._closed = True
