"""Pinned status footer.

Two modes, one interface:

* Mode A (interactive TTY): a one-line ``rich.live.Live`` footer that is
  repainted whenever the state changes.
* Mode B (non-TTY / ``NO_COLOR`` / dumb terminal): no live region; the
  canonical ribbon is reprinted as the final line of the turn.

``StatusBar`` never branches at call sites: ``start``/``stop``/``update``
behave the same way in both modes.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable

from rich.console import Console
from rich.live import Live

from .ribbon import status_ribbon
from .state import StatusState
from .theme import NO_COLOR_ENV


def _plain_mode() -> bool:
    if os.environ.get(NO_COLOR_ENV):
        return True
    try:
        return not os.isatty(1)
    except (OSError, ValueError):
        return True


class StatusBar:
    """A one-line footer that shows the canonical status ribbon."""

    def __init__(
        self,
        console: Console,
        state_provider: Callable[[], StatusState] | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.console = console
        self._state_provider = state_provider
        self._state: StatusState = (
            state_provider() if state_provider is not None else StatusState()
        )
        self._enabled = bool(enabled)
        if not self._enabled:
            self._mode = "disabled"
        else:
            self._mode = "B" if _plain_mode() else "A"
        self._live: Live | None = None
        self._started = False
        self._lock = threading.Lock()

    def _current_state(self) -> StatusState:
        if self._state_provider is not None:
            try:
                return self._state_provider()
            except Exception:  # noqa: BLE001 - best effort
                pass
        return self._state

    def _render(self) -> str:
        return status_ribbon(self._current_state())

    def start(self) -> None:
        if not self._enabled or self._started:
            return
        with self._lock:
            self._state = self._current_state()
        if self._mode == "A":
            try:
                self._live = Live(
                    self._render(),
                    console=self.console,
                    refresh_per_second=4,
                    transient=False,
                )
                self._live.start()
            except Exception:  # noqa: BLE001 - fall back to plain mode
                self._live = None
                self._mode = "B"
        self._started = True

    def update(self, state: StatusState | None = None) -> None:
        if not self._enabled:
            if state is not None:
                with self._lock:
                    self._state = state
            return
        if state is not None:
            with self._lock:
                self._state = state
        with self._lock:
            current = self._current_state()
            self._state = current
        if self._mode == "A" and self._live is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
                self._live.update(self._render())

    def stop(self) -> None:
        if not self._enabled or not self._started:
            return
        if self._mode == "A" and self._live is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
                self._live.stop()
            self._live = None
        elif self._mode == "B":
            # Reprint the canonical ribbon as the final line of the turn.
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
                self.console.print(f"[dim]{status_ribbon(self._current_state())}[/dim]")
        self._started = False

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def started(self) -> bool:
        return self._started
