"""Pruning, integrity checks and backups."""

import os
import pathlib
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone


class MaintenanceMixin:
    """Pruning, integrity checks and backups."""

    def prune_retained_records(
        self,
        *,
        conversation_days: int = 0,
        episode_days: int = 0,
        skill_days: int = 0,
    ) -> dict[str, object]:
        """Prune configured memory classes while retaining recovery-linked records."""
        days_by_class = {
            "conversation": max(0, int(conversation_days)),
            "episode": max(0, int(episode_days)),
            "skill": max(0, int(skill_days)),
        }
        deleted = {"conversation": 0, "episodes": 0, "skills": 0}
        skill_paths: list[str] = []
        resumable_statuses = (
            "draft",
            "awaiting_approval",
            "active",
            "paused",
            "interrupted",
            "stale",
        )
        with self._lock, self._write_transaction():
            if days_by_class["conversation"]:
                cutoff = (
                    datetime.now(timezone.utc)
                    - timedelta(days=days_by_class["conversation"])
                ).strftime("%Y-%m-%d %H:%M:%S")
                cursor = self.conn.execute(
                    "DELETE FROM chat_history WHERE timestamp < ? "
                    "AND role NOT IN ('summary', 'skill') "
                    "AND is_skill_file = 0",
                    (cutoff,),
                )
                deleted["conversation"] = cursor.rowcount
            if days_by_class["episode"]:
                cutoff = (
                    datetime.now(timezone.utc)
                    - timedelta(days=days_by_class["episode"])
                ).strftime("%Y-%m-%d %H:%M:%S")
                cursor = self.conn.execute(
                    "DELETE FROM chat_history WHERE role = 'summary' "
                    "AND timestamp < ? AND (episode_task_id IS NULL OR "
                    "episode_task_id NOT IN (SELECT task_id FROM agent_tasks "
                    "WHERE status IN (?, ?, ?, ?, ?, ?)))",
                    (cutoff, *resumable_statuses),
                )
                deleted["episodes"] = cursor.rowcount
            if days_by_class["skill"]:
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(days=days_by_class["skill"])
                ).strftime("%Y-%m-%d %H:%M:%S")
                skill_paths = [
                    str(row[0])
                    for row in self.conn.execute(
                        "SELECT skill_file_path FROM chat_history "
                        "WHERE is_skill_file = 1 AND timestamp < ? "
                        "AND (skill_task_id IS NULL OR skill_task_id NOT IN "
                        "(SELECT task_id FROM agent_tasks "
                        "WHERE status IN (?, ?, ?, ?, ?, ?))) "
                        "AND skill_file_path IS NOT NULL",
                        (cutoff, *resumable_statuses),
                    ).fetchall()
                ]
                cursor = self.conn.execute(
                    "DELETE FROM chat_history WHERE is_skill_file = 1 "
                    "AND timestamp < ? AND (skill_task_id IS NULL OR "
                    "skill_task_id NOT IN (SELECT task_id FROM agent_tasks "
                    "WHERE status IN (?, ?, ?, ?, ?, ?)))",
                    (cutoff, *resumable_statuses),
                )
                deleted["skills"] = cursor.rowcount
        return {**deleted, "skill_paths": skill_paths}

    def maintain_database(self) -> dict[str, int | str]:
        """Check integrity, optimize query planning, and reclaim free SQLite pages."""
        with self._lock:
            self.conn.commit()
            result = self.conn.execute("PRAGMA integrity_check").fetchone()
            integrity = str(result[0]) if result else "no result"
            if integrity != "ok":
                raise sqlite3.DatabaseError(
                    f"SQLite integrity check failed: {integrity}"
                )
            pages_before = int(self.conn.execute("PRAGMA page_count").fetchone()[0])
            self.conn.execute("PRAGMA optimize")
            self.conn.execute("VACUUM")
            pages_after = int(self.conn.execute("PRAGMA page_count").fetchone()[0])
        return {
            "integrity": integrity,
            "pages_before": pages_before,
            "pages_after": pages_after,
        }

    def backup_to(self, destination: str) -> str:
        """Write a consistent private backup without overwriting an existing file."""
        target = pathlib.Path(destination).expanduser().absolute()
        if (
            self.db_path != ":memory:"
            and target.resolve() == pathlib.Path(self.db_path).resolve()
        ):
            raise ValueError("Backup destination must differ from the active database.")
        if target.exists():
            raise FileExistsError(f"Backup destination already exists: {target}")
        if not target.parent.is_dir():
            raise FileNotFoundError(
                f"Backup parent directory does not exist: {target.parent}"
            )
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        staged = pathlib.Path(temp_name)
        try:
            with self._lock:
                destination_connection = sqlite3.connect(str(staged))
                try:
                    self.conn.backup(destination_connection)
                    result = destination_connection.execute(
                        "PRAGMA integrity_check"
                    ).fetchone()
                    if not result or result[0] != "ok":
                        raise sqlite3.DatabaseError("Backup integrity check failed.")
                finally:
                    destination_connection.close()
            os.chmod(staged, 0o600)
            with staged.open("rb") as backup_file:
                os.fsync(backup_file.fileno())
            os.link(staged, target)
            return str(target)
        finally:
            staged.unlink(missing_ok=True)

    @classmethod
    def restore_backup(cls, source: str, destination: str) -> str:
        """Restore a valid SQLite backup into a new path, never replacing a database."""
        source_path = pathlib.Path(source).expanduser().resolve(strict=True)
        target = pathlib.Path(destination).expanduser().absolute()
        if not source_path.is_file():
            raise ValueError("Backup source must be a regular file.")
        if source_path == target.resolve():
            raise ValueError("Restore destination must differ from the backup source.")
        if target.exists():
            raise FileExistsError(f"Restore destination already exists: {target}")
        if not target.parent.is_dir():
            raise FileNotFoundError(
                f"Restore parent directory does not exist: {target.parent}"
            )
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        staged = pathlib.Path(temp_name)
        source_connection = None
        destination_connection = None
        try:
            source_connection = sqlite3.connect(
                source_path.as_uri() + "?mode=ro",
                uri=True,
            )
            integrity = source_connection.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise sqlite3.DatabaseError(
                    "Backup source failed SQLite integrity validation."
                )
            tables = {
                row[0]
                for row in source_connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "chat_history" not in tables:
                raise sqlite3.DatabaseError(
                    "Backup source is not a Private Agent SQLite database."
                )
            destination_connection = sqlite3.connect(str(staged))
            source_connection.backup(destination_connection)
            restored_integrity = destination_connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()
            if not restored_integrity or restored_integrity[0] != "ok":
                raise sqlite3.DatabaseError(
                    "Restored database failed SQLite integrity validation."
                )
            destination_connection.close()
            destination_connection = None
            source_connection.close()
            source_connection = None
            os.chmod(staged, 0o600)
            with staged.open("rb") as restored_file:
                os.fsync(restored_file.fileno())
            os.link(staged, target)
            return str(target)
        finally:
            if destination_connection is not None:
                destination_connection.close()
            if source_connection is not None:
                source_connection.close()
            staged.unlink(missing_ok=True)
