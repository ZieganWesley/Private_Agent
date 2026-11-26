"""Snapshot manifest and turn-level undo for workspace file edits.

Backups already exist as ``<name>_<uuid>.bak`` files in ``.agent_snapshots/``,
but nothing recorded which file they came from, so they could not be restored.
The manifest below is an append-only JSONL file kept **next to** the backups
instead of in SQLite: a snapshot belongs to the workspace (it must survive a
database reset, a different ``paths.database`` setting or a restored machine),
the backup files already live in that workspace, and a database row would be
useless without them. Old ``.bak`` files with no manifest line stay harmless.

Each recorded edit writes exactly one line with a single ``O_APPEND`` write, so
recording costs one small append per changed file and the manifest is only read
back by ``/undo``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pathlib
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime

_LOGGER = logging.getLogger(__name__)

SNAPSHOT_DIR_NAME = ".agent_snapshots"
MANIFEST_NAME = "index.jsonl"
MANIFEST_VERSION = 1

_EDIT_ACTIONS = frozenset({"modify", "create"})
_HASH_CHUNK_BYTES = 1024 * 1024


class SnapshotError(RuntimeError):
    """Raised when a snapshot cannot be recorded or restored."""


@dataclass(frozen=True)
class SnapshotRecord:
    """One manifest line: a backup or a created file, tied to a turn."""

    action: str
    target: str | None
    backup: str | None
    session_id: str | None
    turn_id: str | None
    created_at: str
    size: int = 0
    sha256: str = ""


@dataclass(frozen=True)
class UndoResult:
    """Outcome of one undo attempt for a session turn."""

    status: str
    turn_id: str | None = None
    restored: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    turn_created_at: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def manifest_path(root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(root) / SNAPSHOT_DIR_NAME / MANIFEST_NAME


def _workspace_relative(root: pathlib.Path, target: str | pathlib.Path) -> str:
    """Return the validated workspace-relative POSIX path for ``target``."""
    resolved_root = pathlib.Path(root).resolve()
    candidate = pathlib.Path(str(target)).expanduser()
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    try:
        resolved = candidate.parent.resolve() / candidate.name
    except OSError as exc:
        raise SnapshotError(f"Cannot resolve '{target}': {exc}") from exc
    if not resolved.is_relative_to(resolved_root):
        raise SnapshotError(f"Path '{target}' is outside the active workspace.")
    return resolved.relative_to(resolved_root).as_posix()


def _hash_file(path: pathlib.Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _snapshot_directory(root: pathlib.Path) -> pathlib.Path:
    resolved_root = pathlib.Path(root).resolve()
    directory = resolved_root / SNAPSHOT_DIR_NAME
    if directory.is_symlink():
        raise SnapshotError("Workspace snapshot directory cannot be a symlink.")
    directory.mkdir(mode=0o700, exist_ok=True)
    if not directory.is_dir():
        raise SnapshotError("Workspace snapshot directory is not a directory.")
    return directory


def _append_record(root: pathlib.Path, record: SnapshotRecord) -> None:
    directory = _snapshot_directory(root)
    payload = {
        "version": MANIFEST_VERSION,
        "action": record.action,
        "target": record.target,
        "backup": record.backup,
        "session_id": record.session_id,
        "turn_id": record.turn_id,
        "created_at": record.created_at,
        "size": record.size,
        "sha256": record.sha256,
    }
    line = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(directory / MANIFEST_NAME, flags, 0o600)
    try:
        view = memoryview(line)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def record_snapshot(
    *,
    root: pathlib.Path,
    target: str | pathlib.Path,
    session_id: str | None,
    turn_id: str | None,
    backup: str | None = None,
    action: str = "modify",
    size: int | None = None,
    sha256: str | None = None,
) -> SnapshotRecord:
    """Append one snapshot record so the edit can later be undone.

    ``action="create"`` records a file the turn brought into existence (no
    backup); undo removes it. Callers that already hashed the copied bytes pass
    ``size``/``sha256`` to avoid a second read of the backup.
    """
    if action not in _EDIT_ACTIONS:
        raise SnapshotError(f"Unsupported snapshot action '{action}'.")
    if not session_id or not turn_id:
        raise SnapshotError("Snapshots require both a session id and a turn id.")
    relative = _workspace_relative(root, target)
    resolved_root = pathlib.Path(root).resolve()
    record = SnapshotRecord(
        action=action,
        target=relative,
        backup=None,
        session_id=session_id,
        turn_id=turn_id,
        created_at=_now(),
    )
    if action == "modify":
        if not backup:
            raise SnapshotError("A modify snapshot requires its backup file name.")
        backup_name = pathlib.Path(str(backup)).name
        if backup_name != str(backup):
            raise SnapshotError("Backup must be a plain file name.")
        backup_path = resolved_root / SNAPSHOT_DIR_NAME / backup_name
        try:
            backup_stat = backup_path.lstat()
        except OSError as exc:
            raise SnapshotError(
                f"Cannot record a snapshot with a missing backup: {exc}"
            ) from exc
        if not stat.S_ISREG(backup_stat.st_mode):
            raise SnapshotError("Backup is not a regular file.")
        if size is None or sha256 is None:
            size, sha256 = _hash_file(backup_path)
        record = SnapshotRecord(
            action="modify",
            target=relative,
            backup=backup_name,
            session_id=session_id,
            turn_id=turn_id,
            created_at=record.created_at,
            size=size,
            sha256=sha256,
        )
    _append_record(resolved_root, record)
    return record


def read_records(root: pathlib.Path) -> list[SnapshotRecord]:
    """Read the manifest, skipping lines written by a newer version or damaged."""
    path = manifest_path(pathlib.Path(root).resolve())
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise SnapshotError(f"Could not read the snapshot manifest: {exc}") from exc
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            _LOGGER.warning("Skipping malformed snapshot manifest line.")
            continue
        if not isinstance(payload, dict):
            _LOGGER.warning("Skipping malformed snapshot manifest line.")
            continue
        if payload.get("version") != MANIFEST_VERSION:
            _LOGGER.info(
                "Skipping snapshot manifest version %s.",
                payload.get("version"),
            )
            continue
        action = payload.get("action")
        if not isinstance(action, str) or action not in {
            "modify",
            "create",
            "undo",
        }:
            _LOGGER.warning("Skipping snapshot record with unknown action.")
            continue
        created_at = payload.get("created_at")
        records.append(
            SnapshotRecord(
                action=action,
                target=(
                    payload.get("target")
                    if isinstance(payload.get("target"), str)
                    else None
                ),
                backup=(
                    payload.get("backup")
                    if isinstance(payload.get("backup"), str)
                    else None
                ),
                session_id=(
                    payload.get("session_id")
                    if isinstance(payload.get("session_id"), str)
                    else None
                ),
                turn_id=(
                    payload.get("turn_id")
                    if isinstance(payload.get("turn_id"), str)
                    else None
                ),
                created_at=created_at if isinstance(created_at, str) else "",
                size=payload.get("size") if isinstance(payload.get("size"), int) else 0,
                sha256=(
                    payload.get("sha256")
                    if isinstance(payload.get("sha256"), str)
                    else ""
                ),
            )
        )
    return records


def undoable_turn_ids(
    records: list[SnapshotRecord],
    session_id: str,
) -> list[str]:
    """Return this session's turns that still have an undo, oldest first."""
    undone = {
        record.turn_id
        for record in records
        if record.session_id == session_id and record.action == "undo"
    }
    turns: list[str] = []
    for record in records:
        if record.session_id != session_id:
            continue
        if record.action not in _EDIT_ACTIONS or not record.turn_id:
            continue
        if record.turn_id in undone or record.turn_id in turns:
            continue
        turns.append(record.turn_id)
    return turns


def _restore_backup(
    root: pathlib.Path,
    relative: str,
    record: SnapshotRecord,
) -> None:
    if not record.backup:
        raise SnapshotError("Snapshot record has no backup file.")
    backup_name = pathlib.Path(record.backup).name
    if backup_name != record.backup:
        raise SnapshotError("Snapshot record has an invalid backup name.")
    backup_path = root / SNAPSHOT_DIR_NAME / backup_name
    try:
        backup_stat = backup_path.lstat()
    except FileNotFoundError as exc:
        raise SnapshotError("The snapshot backup no longer exists.") from exc
    if not stat.S_ISREG(backup_stat.st_mode):
        raise SnapshotError("The snapshot backup is not a regular file.")
    size, digest = _hash_file(backup_path)
    if size != record.size or digest != record.sha256:
        raise SnapshotError("The snapshot backup changed since it was recorded.")
    target_path = root / relative
    if target_path.is_symlink():
        raise SnapshotError("Refusing to restore through a symlink.")
    parent = target_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(parent),
            prefix=".undo-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_name = temporary.name
            with backup_path.open("rb") as source:
                shutil.copyfileobj(source, temporary, _HASH_CHUNK_BYTES)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temp_name, stat.S_IMODE(backup_stat.st_mode))
        os.replace(temp_name, target_path)
        temp_name = None
    finally:
        if temp_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(temp_name)


def _delete_created(root: pathlib.Path, relative: str) -> bool:
    target_path = root / relative
    try:
        target_stat = target_path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(target_stat.st_mode):
        raise SnapshotError("The file this turn created is no longer a regular file.")
    target_path.unlink()
    return True


def undo_turn(
    *,
    root: pathlib.Path,
    session_id: str,
    turn_id: str | None = None,
) -> UndoResult:
    """Restore the pre-turn state of the newest (or given) turn of a session.

    Each file is restored atomically (temporary file plus ``os.replace``), so a
    failed restore leaves the current content untouched. The turn is marked
    undone only when every file was restored, which keeps a failed undo
    retryable.
    """
    if not session_id:
        raise SnapshotError("Undo requires a session id.")
    resolved_root = pathlib.Path(root).resolve()
    records = read_records(resolved_root)
    session_records = [record for record in records if record.session_id == session_id]
    undone = {record.turn_id for record in session_records if record.action == "undo"}
    if turn_id is None:
        turns = undoable_turn_ids(records, session_id)
        if not turns:
            return UndoResult("empty")
        turn_id = turns[-1]
    elif turn_id in undone:
        return UndoResult("already_undone", turn_id=turn_id)
    turn_records = [
        record
        for record in session_records
        if record.turn_id == turn_id and record.action in _EDIT_ACTIONS
    ]
    if not turn_records:
        return UndoResult("empty", turn_id=turn_id)

    # The first record per target is the pre-turn state; later records in the
    # same turn are intermediate states. Undo the targets in reverse order so a
    # rename's created destination is removed before its source is restored.
    first_by_target: dict[str, SnapshotRecord] = {}
    for record in turn_records:
        if record.target:
            first_by_target.setdefault(record.target, record)

    restored: list[str] = []
    deleted: list[str] = []
    errors: list[str] = []
    for relative, record in reversed(list(first_by_target.items())):
        try:
            if record.action == "create":
                if _delete_created(resolved_root, relative):
                    deleted.append(relative)
            else:
                _restore_backup(resolved_root, relative, record)
                restored.append(relative)
        except (OSError, ValueError, SnapshotError) as exc:
            errors.append(f"{relative}: {exc}")
    if errors:
        return UndoResult(
            "partial",
            turn_id=turn_id,
            restored=tuple(restored),
            deleted=tuple(deleted),
            errors=tuple(errors),
            turn_created_at=turn_records[0].created_at,
        )
    _append_record(
        resolved_root,
        SnapshotRecord(
            action="undo",
            target=None,
            backup=None,
            session_id=session_id,
            turn_id=turn_id,
            created_at=_now(),
        ),
    )
    return UndoResult(
        "undone",
        turn_id=turn_id,
        restored=tuple(restored),
        deleted=tuple(deleted),
        turn_created_at=turn_records[0].created_at,
    )
