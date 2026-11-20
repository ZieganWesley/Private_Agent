"""Live status overlay for tool activity (Part F1)."""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .events import ToolFinished, ToolStarted, get_event_bus


def _plain_mode() -> bool:
    if os.environ.get("NO_COLOR"):
        return True
    return not os.isatty(1)


class StatusOverlay:
    """Shows running/finished tools in a live panel while a turn runs."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._running: dict[str, str] = {}
        self._finished: list[tuple[str, str, float]] = []
        self._plan_summary: str = ""
        self._eta_deadline: float | None = None
        self._lock = threading.Lock()
        self._live: Live | None = None
        self._started = False

    def set_plan_summary(self, text: str) -> None:
        with self._lock:
            self._plan_summary = text or ""

    def set_eta_deadline(self, deadline: float | None) -> None:
        with self._lock:
            self._eta_deadline = deadline

    def _render(self) -> Any:
        table = Table(show_header=False, box=None, padding=(0, 1))
        with self._lock:
            if self._plan_summary:
                table.add_row("Plan:", "")
                for line in self._plan_summary.splitlines()[:6]:
                    table.add_row(f"  {line}", "")
            if self._eta_deadline is not None:
                import time as _t
                remaining = max(0.0, self._eta_deadline - _t.monotonic())
                table.add_row("ETA:", f"{remaining:.0f}s")
            for _call_id, name in list(self._running.items()):
                table.add_row(f"~ {name}", "")
            for name, status, dur in self._finished[-5:]:
                mark = "ok" if status == "ok" else "err"
                table.add_row(f"{mark} {name}", f"{dur:.1f}s")
        if table.row_count == 0:
            table.add_row("Thinking...", "")
        return Panel(table, title="Working", border_style="cyan")

    def _on_tool_started(self, event: ToolStarted) -> None:
        if self._live is None:
            return
        with self._lock:
            self._running[event.call_id] = event.tool_name
        with contextlib.suppress(Exception):  # noqa: BLE001 - live updates are best-effort
            self._live.update(self._render())

    def _on_tool_finished(self, event: ToolFinished) -> None:
        if self._live is None:
            return
        with self._lock:
            name = self._running.pop(event.call_id, "")
            self._finished.append((name or event.call_id, event.status, event.duration))
        with contextlib.suppress(Exception):  # noqa: BLE001 - live updates are best-effort
            self._live.update(self._render())

    def start(self) -> None:
        if _plain_mode():
            return
        self._live = Live(self._render(), console=self.console, refresh_per_second=4)
        bus = get_event_bus()
        bus.subscribe(ToolStarted, self._on_tool_started)
        bus.subscribe(ToolFinished, self._on_tool_finished)
        try:
            self._live.start()
            self._started = True
        except Exception:  # noqa: BLE001 - fall back to plain mode
            self._live = None

    def stop(self) -> None:
        if not self._started or self._live is None:
            return
        with contextlib.suppress(Exception):  # noqa: BLE001 - best-effort
            self._live.stop()
        bus = get_event_bus()
        bus.unsubscribe(ToolStarted, self._on_tool_started)
        bus.unsubscribe(ToolFinished, self._on_tool_finished)
        self._live = None
        self._started = False
