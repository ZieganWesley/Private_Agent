"""SessionUI facade: the single object the runtime talks to for chrome.

Business modules never call ``console.print`` for status, labels or turn
summaries; they go through this facade so the terminal stays coherent.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any

from .console import console as _shared_console
from .ribbon import status_ribbon
from .state import StatusState
from .status_bar import StatusBar
from .theme import ACCESSIBLE_ENV
from .turn import TurnReport


class SessionUI:
    def __init__(
        self,
        console: Any = None,
        *,
        status_bar: bool = True,
    ) -> None:
        self.console = console if console is not None else _shared_console
        self.status_bar_enabled = bool(status_bar)
        self._model: Any = None
        self._permission: str = "auto"
        self._output_tokens: Any = None
        self._provider: str = ""
        self._provider_mode: str = ""
        self._thinking_enabled: bool = False
        self._thinking_effort: str = "med"
        self._skills_count: int = 0
        self._capabilities: dict[str, Any] = {}
        self._rag_enabled: bool = True
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
            provider=self._provider,
            provider_mode=self._provider_mode,
            thinking_enabled=self._thinking_enabled,
            thinking_effort=self._thinking_effort,
            skills_count=self._skills_count,
            capabilities=dict(self._capabilities),
            rag_enabled=self._rag_enabled,
        )

    def bind(
        self,
        *,
        model: Any = None,
        permission: str | None = None,
        output_tokens: Any = None,
        provider: str | None = None,
        provider_mode: str | None = None,
        thinking_enabled: bool | None = None,
        thinking_effort: str | None = None,
        skills_count: int | None = None,
        capabilities: dict[str, Any] | None = None,
        rag_enabled: bool | None = None,
    ) -> None:
        if model is not None:
            self._model = model
        if permission is not None:
            self._permission = permission
        if output_tokens is not None:
            self._output_tokens = output_tokens
        if provider is not None:
            self._provider = str(provider)
        if provider_mode is not None:
            self._provider_mode = str(provider_mode)
        if thinking_enabled is not None:
            self._thinking_enabled = bool(thinking_enabled)
        if thinking_effort is not None:
            self._thinking_effort = str(thinking_effort)
        if skills_count is not None:
            self._skills_count = int(skills_count)
        if capabilities is not None:
            self._capabilities = dict(capabilities)
        if rag_enabled is not None:
            self._rag_enabled = bool(rag_enabled)
        self._footer.update(self._current_state())
        # The footer owns the ribbon: Mode A paints it live and Mode B prints
        # it as each turn's closing line, so no copy is printed here.
        self._footer.start()

    def set_context(self, used: Any, maximum: Any) -> None:
        try:
            self._context_used = int(used)
        except TypeError, ValueError:
            self._context_used = 0
        try:
            self._context_max = int(maximum)
        except TypeError, ValueError:
            self._context_max = 0
        self._footer.update(self._current_state())

    def ribbon_text(
        self, *, width: int | None = None, accessible: bool | None = None
    ) -> str:
        """Return the full status ribbon for use as a prompt toolbar.

        Width-aware segment dropping keeps it usable on narrow terminals.
        ``accessible`` defaults to the ``PRIVATE_AGENT_ACCESSIBLE`` setting.
        """
        if accessible is None:
            accessible = bool(os.environ.get(ACCESSIBLE_ENV))
        if width is None:
            try:
                width = int(getattr(self.console, "width", 0) or 0)
            except TypeError, ValueError:
                width = 0
        if not width:
            try:
                import shutil

                width = shutil.get_terminal_size().columns
            except OSError, ValueError:
                width = 80
        return status_ribbon(self._current_state(), width=width, accessible=accessible)

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
                self._output_tokens = getattr(
                    stats, "output_tokens", self._output_tokens
                )

    def turn_finished(self, report: TurnReport | None = None) -> None:
        if report is not None and getattr(report, "start_time", None) is not None:
            start = report.start_time
        elif self._turn_start is not None:
            start = self._turn_start
        else:
            start = time.monotonic()
        if report is None:
            report = TurnReport(
                start_time=start, model=self._model, generation=self._generation
            )
        else:
            if getattr(report, "generation", None) is None:
                report.generation = self._generation
        self.console.print(f"[dim success]{report.summary_line()}[/dim success]")
        self._footer.turn_finished()

    def print_markup(self, text: Any) -> None:
        """Print a pre-formatted (markup) chrome line through the shared console."""
        self.console.print(text)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Passthrough to the shared console so all output has one owner."""
        self.console.print(*args, **kwargs)

    def close(self) -> None:
        self._footer.stop()
