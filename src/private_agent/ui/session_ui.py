"""SessionUI facade: the single object the runtime talks to for chrome.

Business modules never call ``console.print`` for status, labels or turn
summaries; they go through this facade so the terminal stays coherent.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from .console import console as _shared_console
from .ribbon import status_ribbon
from .state import StatusState
from .status_bar import StatusBar
from .turn import TurnReport


class SessionUI:
    def __init__(
        self,
        console: Any = None,
        *,
        status_bar: bool = True,
        verbose_timings: bool = False,
    ) -> None:
        self.console = console if console is not None else _shared_console
        self.status_bar_enabled = bool(status_bar)
        self.verbose_timings = bool(verbose_timings)
        self._model: Any = None
        self._permission: str = "auto"
        self._output_tokens: Any = None
        self._context_used: int = 0
        self._context_max: int = 0
        self._bg: int = 0
        self._footer = StatusBar(
            self.console, self._current_state, enabled=self.status_bar_enabled
        )
        self._turn_start: float | None = None
        self._generation: Any = None

    def _current_state(self) -> StatusState:
        return StatusState(
            model=self._model,
            context_tokens=self._context_used,
            context_maximum=self._context_max,
            permission_mode=self._permission,
            output_tokens=self._output_tokens,
            background_processes=self._bg,
        )

    def bind(
        self,
        *,
        model: Any = None,
        permission: str | None = None,
        output_tokens: Any = None,
    ) -> None:
        if model is not None:
            self._model = model
        if permission is not None:
            self._permission = permission
        if output_tokens is not None:
            self._output_tokens = output_tokens
        self._footer.update(self._current_state())
        self._footer.start()
        if self._footer.mode == "B" and self._footer.started:
            with contextlib.suppress(Exception):
                self.console.print(
                    f"[dim]{status_ribbon(self._current_state(), accessible=self._footer._accessible())}[/dim]"
                )

    def set_context(self, used: Any, maximum: Any) -> None:
        try:
            self._context_used = int(used)
        except (TypeError, ValueError):
            self._context_used = 0
        try:
            self._context_max = int(maximum)
        except (TypeError, ValueError):
            self._context_max = 0
        self._footer.update(self._current_state())

    def ribbon_text(self, *, width: int | None = None) -> str:
        """Return the full status ribbon for use as a prompt toolbar.

        Width-aware segment dropping keeps it usable on narrow terminals.
        """
        if width is None:
            try:
                width = int(getattr(self.console, "width", 0) or 0)
            except (TypeError, ValueError):
                width = 0
        if not width:
            try:
                import shutil

                width = shutil.get_terminal_size().columns
            except (OSError, ValueError):
                width = 80
        return status_ribbon(self._current_state(), width=width)

    def turn_started(self, user_input: str = "") -> TurnReport:
        self._turn_start = time.monotonic()
        self._generation = None
        report = TurnReport(start_time=self._turn_start, model=self._model)
        self._footer.start()
        return report

    def set_generation(self, stats: Any) -> None:
        self._generation = stats
        if stats is not None:
            with contextlib.suppress(Exception):
                self._output_tokens = getattr(stats, "output_tokens", self._output_tokens)

    def turn_finished(self, report: TurnReport | None = None) -> None:
        if report is not None and getattr(report, "start_time", None) is not None:
            start = report.start_time
        elif self._turn_start is not None:
            start = self._turn_start
        else:
            start = time.monotonic()
        if report is None:
            report = TurnReport(start_time=start, model=self._model, generation=self._generation)
        else:
            if getattr(report, "generation", None) is None:
                report.generation = self._generation
        self.console.print(f"[dim green]{report.summary_line()}[/dim green]")
        self._footer.turn_finished()

    def rule(self) -> None:
        from rich.rule import Rule

        self.console.print(Rule(style="dim"))

    def print_markup(self, text: Any) -> None:
        """Print a pre-formatted (markup) chrome line through the shared console."""
        self.console.print(text)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Passthrough to the shared console so all output has one owner."""
        self.console.print(*args, **kwargs)

    def close(self) -> None:
        self._footer.stop()
