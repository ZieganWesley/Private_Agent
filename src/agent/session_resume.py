"""Session id resolution and resume-time task-lease ownership."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections.abc import Callable

from database import PersistentMemory
from utils.run_logging import RUN_LOGGER, log_event


class SessionResumeError(RuntimeError):
    """Raised when the session the user asked to resume cannot be used."""


def resolve_session_id(
    memory: PersistentMemory,
    *,
    resume_session: str | None = None,
    continue_session: bool = False,
    prompt: Callable[[str], str] | None = None,
) -> str:
    """Return the session id for this run.

    An explicit ``--resume`` id wins, then ``--continue`` (the most recent
    session), then an interactive offer to resume that session, and finally a
    new id. A failed explicit request raises instead of silently starting a new
    session, so a typo never looks like a loaded conversation.
    """
    requested = (resume_session or "").strip()
    if requested:
        if not memory.session_exists(requested):
            raise SessionResumeError(
                f"Session '{requested}' was not found. Run "
                "`private-agent --list-sessions` to see saved session ids."
            )
        log_event(RUN_LOGGER, "session.resumed", session_id=requested, source="flag")
        return requested

    latest = memory.get_latest_session_id()
    if continue_session:
        if not latest:
            raise SessionResumeError(
                "There is no earlier session to continue; run the agent once "
                "before using --continue."
            )
        log_event(RUN_LOGGER, "session.resumed", session_id=latest, source="continue")
        return latest

    if (
        latest
        and prompt is not None
        and prompt(latest).strip().lower() in {"", "y", "yes"}
    ):
        log_event(RUN_LOGGER, "session.resumed", session_id=latest, source="prompt")
        return latest
    return f"session_{uuid.uuid4().hex}"


def take_over_session_task_lease(
    memory: PersistentMemory,
    session_id: str,
    owner_id: str,
) -> str | None:
    """Re-claim the resumed session's active plan lease when it is free.

    Returns ``"acquired"``, ``"held"`` when a live process still owns it,
    ``"failed"`` when the claim could not be made, or ``None`` when the session
    has no active plan. A lease held by a live owner is never stolen.
    """
    try:
        task = memory.get_session_active_task(session_id)
    except (ValueError, sqlite3.Error, OSError) as exc:
        log_event(
            RUN_LOGGER,
            "task.lease_takeover_failed",
            level=logging.ERROR,
            session_id=session_id,
            task_id=None,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        return "failed"
    if task is None:
        return None
    task_id = str(task["task_id"])
    try:
        lease = memory.get_task_lease(task_id)
        if lease and not lease["expired"] and lease["owner_id"] != owner_id:
            log_event(
                RUN_LOGGER,
                "task.lease_held_elsewhere",
                level=logging.WARNING,
                session_id=session_id,
                task_id=task_id,
            )
            return "held"
        acquired = memory.acquire_task_lease(
            task_id, owner_id, session_id, allow_takeover=True
        )
    except (ValueError, sqlite3.Error, OSError) as exc:
        log_event(
            RUN_LOGGER,
            "task.lease_takeover_failed",
            level=logging.ERROR,
            session_id=session_id,
            task_id=task_id,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        return "failed"
    if not acquired:
        return "failed"
    log_event(
        RUN_LOGGER,
        "task.lease_taken_over",
        session_id=session_id,
        task_id=task_id,
    )
    return "acquired"
