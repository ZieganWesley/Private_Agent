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
from typing import Any

from rich.console import Console
from rich.live import Live

from .ribbon import status_ribbon
from .state import StatusState
from .theme import ACCESSIBLE_ENV, NO_COLOR_ENV

_ACTIVE_FOOTERS: dict[int, StatusBar] = {}


def register_active_footer(console: Any, footer: StatusBar) -> None:
    """Remember the live footer associated with ``console``."""
    _ACTIVE_FOOTERS[id(console)] = footer


def unregister_active_footer(console: Any) -> None:
    """Forget the live footer associated with ``console``."""
    _ACTIVE_FOOTERS.pop(id(console), None)


def get_active_footer(console: Any) -> StatusBar | None:
    """Return the live footer for ``console``, if one is currently active."""
    return _ACTIVE_FOOTERS.get(id(console))


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
        self._busy = False
        self._busy_message = ""
        self._busy_phase = "processing"

    def _current_state(self) -> StatusState:
        if self._state_provider is not None:
            try:
                return self._state_provider()
            except Exception:  # noqa: BLE001 - best effort
                pass
        return self._state

    def _render(self) -> Any:
        if self._busy:
            try:
                from rich.spinner import Spinner

                from .spinner import phase_spinner
            except Exception:  # noqa: BLE001 - fall back to plain text
                ribbon = status_ribbon(
                    self._current_state(),
                    accessible=bool(os.environ.get(ACCESSIBLE_ENV)),
                )
                return f"{self._busy_message} | {ribbon}"
            spinner, style = phase_spinner(self._busy_phase)
            ribbon = status_ribbon(
                self._current_state(),
                accessible=bool(os.environ.get(ACCESSIBLE_ENV)),
            )
            return Spinner(
                spinner,
                text=f"{self._busy_message} | {ribbon}",
                style=style,
                speed=1.0,
            )
        return status_ribbon(
            self._current_state(), accessible=bool(os.environ.get(ACCESSIBLE_ENV))
        )

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
                    auto_refresh=True,
                    transient=False,
                )
                self._live.start()
            except Exception:  # noqa: BLE001 - fall back to plain mode
                self._live = None
                self._mode = "B"
        self._started = True
        if self._enabled:
            register_active_footer(self.console, self)

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

    def set_busy(self, message: str, phase: str = "processing") -> None:
        """Show a spinner message in the live footer without hiding the ribbon."""
        if not self._enabled or not self._started:
            return
        with self._lock:
            self._busy = True
            self._busy_message = str(message or "")
            self._busy_phase = phase or "processing"
        if self._mode == "A" and self._live is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
                self._live.update(self._render())

    def clear_busy(self) -> None:
        """Return the live footer to showing only the ribbon."""
        if not self._enabled or not self._started:
            return
        with self._lock:
            self._busy = False
            self._busy_message = ""
            self._busy_phase = "processing"
        if self._mode == "A" and self._live is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
                self._live.update(self._render())

    def turn_finished(self) -> None:
        """End-of-turn handling that keeps a TTY footer alive."""
        if not self._enabled or not self._started:
            return
        if self._mode == "A":
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
                self.update()
            return
        # Mode B: the ribbon was printed once at startup; do not reprint it
        # on every turn.

    def stop(self) -> None:
        if not self._enabled or not self._started:
            return
        if self._mode == "A" and self._live is not None:
            # Erase the paused live footer so it does not linger above the
            # prompt_toolkit input area; the bottom toolbar takes over while
            # input is active.
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
                self._live.transient = True
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
                self._live.stop()
            self._live = None
        self._started = False
        unregister_active_footer(self.console)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def started(self) -> bool:
        return self._started
