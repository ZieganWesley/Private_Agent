"""Task-lease constants and the lease heartbeat loop."""

from __future__ import annotations

import asyncio
import logging
import sqlite3

from ..database import PersistentMemory
from ..run_logging import RUN_LOGGER, log_event
from .console import console

TASK_LEASE_SECONDS = 180
TASK_LEASE_RENEW_INTERVAL_SECONDS = 30


async def _maintain_task_lease(memory: PersistentMemory, session_id: str) -> None:
    from ..tools import get_tool_context

    while True:
        await asyncio.sleep(TASK_LEASE_RENEW_INTERVAL_SECONDS)
        context = get_tool_context()
        task_id = context.task_id
        if not task_id:
            continue
        try:
            task = await asyncio.to_thread(memory.get_task, task_id)
            if not (
                task
                and task["status"] == "active"
                and task["approved_revision"] == task["revision"]
                and task["approved_digest"] == task["digest"]
            ):
                continue
            acquired = await asyncio.to_thread(
                memory.acquire_task_lease,
                task_id,
                context.runtime_owner_id,
                session_id,
                lease_seconds=TASK_LEASE_SECONDS,
            )
            if not acquired:
                log_event(
                    RUN_LOGGER,
                    "task.lease_lost",
                    level=logging.ERROR,
                    session_id=session_id,
                    task_id=task_id,
                )
                console.print(
                    "[yellow][Task lease warning] This process no longer owns "
                    "the active task. No further task actions will run.[/yellow]"
                )
        except (ValueError, sqlite3.Error, OSError) as exc:
            log_event(
                RUN_LOGGER,
                "task.lease_renewal_failed",
                level=logging.ERROR,
                session_id=session_id,
                task_id=task_id,
                error_type=type(exc).__name__,
                exc_info=True,
            )
            console.print(
                "[yellow][Task lease warning] Could not renew the task lease: "
                f"{type(exc).__name__}: {exc}[/yellow]"
            )
