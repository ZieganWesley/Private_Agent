"""`/undo`: restore the workspace changes of the last agent turn.

The file work lives in :mod:`code_tasks.snapshots`; this module turns a
snapshot result into a user-facing outcome and rewinds the conversation rows
that belong to the same turn.
"""

from __future__ import annotations

import logging
import pathlib
import sqlite3
from dataclasses import dataclass, field

from rich.markup import escape

from code_tasks import snapshots
from database import PersistentMemory
from sandbox import SandboxManager
from utils.run_logging import RUN_LOGGER, log_event

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class UndoOutcome:
    """What `/undo` did, and what the runtime should do about the session."""

    status: str
    message: str
    history_changed: bool = False
    restored: tuple[str, ...] = field(default=())
    deleted: tuple[str, ...] = field(default=())


def _short(turn_id: str | None) -> str:
    return (turn_id or "")[:8] or "unknown"


def _path_line(path: str) -> str:
    """Escape workspace paths so a name like ``a[1].txt`` cannot break markup."""
    return f"  - {escape(path)}"


def _format_undone(result: snapshots.UndoResult, deleted_messages: int) -> str:
    parts = [
        f"[success][Undo][/success] Restored turn {_short(result.turn_id)} "
        f"({result.turn_created_at or 'time unknown'})."
    ]
    if result.restored:
        parts.append(f"[success]Restored {len(result.restored)} file(s):[/success]")
        parts.extend(_path_line(path) for path in result.restored)
    if result.deleted:
        parts.append(
            f"[success]Removed {len(result.deleted)} created file(s):[/success]"
        )
        parts.extend(_path_line(path) for path in result.deleted)
    if not result.restored and not result.deleted:
        parts.append(
            "No workspace file changed in that turn; only the conversation was rewound."
        )
    if deleted_messages:
        parts.append(
            f"[success]Conversation rewound:[/success] removed {deleted_messages} "
            "message(s) saved by that turn."
        )
    else:
        parts.append(
            "The turn had no saved conversation rows, so only the workspace "
            "was restored."
        )
    return "\n".join(parts)


def _format_partial(result: snapshots.UndoResult) -> str:
    lines = [
        f"[error][Undo error][/error] Could not fully restore turn "
        f"{_short(result.turn_id)}. The conversation was left untouched so you "
        "can retry.",
    ]
    if result.restored:
        lines.append(f"Already restored {len(result.restored)} file(s):")
        lines.extend(_path_line(path) for path in result.restored)
    if result.deleted:
        lines.append(f"Already removed {len(result.deleted)} created file(s):")
        lines.extend(_path_line(path) for path in result.deleted)
    lines.append("Files that failed (each left exactly as it was):")
    lines.extend(_path_line(error) for error in result.errors)
    return "\n".join(lines)


def undo_last_turn(
    memory: PersistentMemory,
    session_id: str,
    *,
    workspace_root: pathlib.Path | str | None = None,
    turn_id: str | None = None,
) -> UndoOutcome:
    """Undo the newest (or given) undoable turn of ``session_id``.

    Files are restored first; the conversation is rewound only when every file
    was restored, so a failed undo never loses the turn's messages.
    """
    root = (
        pathlib.Path(workspace_root)
        if workspace_root is not None
        else pathlib.Path(SandboxManager.get_root())
    )
    try:
        result = snapshots.undo_turn(
            root=root,
            session_id=session_id,
            turn_id=turn_id,
        )
    except (OSError, ValueError, snapshots.SnapshotError) as exc:
        log_event(
            RUN_LOGGER,
            "session.undo_failed",
            level=logging.WARNING,
            session_id=session_id,
            error_type=type(exc).__name__,
        )
        return UndoOutcome(
            "error",
            f"[error][Undo error][/error] Could not read the snapshot manifest: "
            f"{type(exc).__name__}: {exc}\nNothing was changed.",
        )

    if result.status == "empty":
        log_event(RUN_LOGGER, "session.undo_empty", session_id=session_id)
        return UndoOutcome(
            "empty",
            "[warning][Undo][/warning] Nothing to undo: this session has no "
            "recorded workspace changes left. `/undo` restores edits made by "
            "agent tools in earlier turns of this workspace.",
        )
    if result.status == "already_undone":
        log_event(
            RUN_LOGGER,
            "session.undo_skipped",
            session_id=session_id,
            turn_id=result.turn_id,
            reason="already_undone",
        )
        return UndoOutcome(
            "already_undone",
            f"[warning][Undo][/warning] Turn {_short(result.turn_id)} was already "
            "undone; nothing was changed.",
        )
    if result.status == "partial":
        log_event(
            RUN_LOGGER,
            "session.undo_failed",
            level=logging.WARNING,
            session_id=session_id,
            turn_id=result.turn_id,
            error_count=len(result.errors),
            restored_count=len(result.restored),
            deleted_count=len(result.deleted),
        )
        return UndoOutcome(
            "partial",
            _format_partial(result),
            restored=result.restored,
            deleted=result.deleted,
        )

    try:
        deleted_messages = memory.delete_turn_messages(session_id, result.turn_id or "")
    except (sqlite3.Error, OSError, ValueError) as exc:
        log_event(
            RUN_LOGGER,
            "session.undo_history_failed",
            level=logging.ERROR,
            session_id=session_id,
            turn_id=result.turn_id,
            error_type=type(exc).__name__,
        )
        return UndoOutcome(
            "partial",
            f"[error][Undo warning][/error] Turn {_short(result.turn_id)} was "
            "restored on disk, but the conversation history could not be "
            f"rewound ({type(exc).__name__}: {exc}). The files are already "
            "back; use `/undo` again to retry only the conversation step or "
            "continue from here.",
            restored=result.restored,
            deleted=result.deleted,
        )

    log_event(
        RUN_LOGGER,
        "session.undone",
        session_id=session_id,
        turn_id=result.turn_id,
        restored_count=len(result.restored),
        deleted_count=len(result.deleted),
        message_count=deleted_messages,
    )
    return UndoOutcome(
        "undone",
        _format_undone(result, deleted_messages),
        history_changed=True,
        restored=result.restored,
        deleted=result.deleted,
    )
