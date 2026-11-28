"""Full-screen production chat surface.

The reference UI's main view is a three-pane screen: conversation history, the
active input line, and the live status ribbon.  This module makes that layout a
real session surface while leaving the transcript and status ribbon canonical:
messages are rich renderables and the ribbon comes from ``ribbon_text``.

Interactive TTY sessions enter the alternate screen while waiting for input.
The previous screen (including streamed tool output) is restored as soon as a
line is submitted, so the runtime keeps its existing streaming behaviour.
"""

from __future__ import annotations

import contextlib
import sys
from collections import deque
from collections.abc import Callable
from typing import Any

from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from .console import resolve_style
from .ribbon import ribbon_text
from .state import StatusState

_INPUT_LABEL = "Input: "


def terminal_capable() -> bool:
    """Return whether the full-screen surface can drive this terminal."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except AttributeError, ValueError:
        return False


class ChatScreenUI:
    """Conversation, input and ribbon in one full-screen layout."""

    def __init__(
        self,
        console: Any,
        state_provider: Callable[[], StatusState] | None = None,
        *,
        history_limit: int = 200,
    ) -> None:
        self.console = console
        self._state_provider = state_provider
        self.history: deque[RenderableType] = deque(maxlen=max(1, history_limit))
        self.input_buffer = ""
        self._live: Live | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Transcript
    # ------------------------------------------------------------------

    def _state(self) -> StatusState:
        if self._state_provider is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
                return self._state_provider()
        return StatusState()

    def append(self, item: RenderableType) -> None:
        """Append one conversation block."""
        self.history.append(item)

    def add_user(self, text: str) -> None:
        """Append a user turn without interpreting its text as markup.

        History uses ``You:`` while the live input pane and prompt use ``User:``
        so a terminal harness can distinguish a past turn from the active one.
        """
        line = Text()
        line.append("You: ", style=resolve_style("accent"))
        line.append(text)
        self.append(line)

    def add_assistant(self, text: str) -> None:
        """Append an assistant reply as sanitised Markdown."""
        from .markdown_view import markdown_renderable

        self.append(
            Group(
                Text("Agent:", style=resolve_style("border.success")),
                markdown_renderable(text),
            )
        )

    def reset(self) -> None:
        """Clear the rendered history."""
        self.history.clear()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _render_chat_pane(self) -> Panel:
        height = max(6, self.console.size.height - 8)
        visible = list(self.history)[-max(1, height // 2) :]
        body: RenderableType = (
            Group(*visible) if visible else Text("No messages yet.", style="dim italic")
        )
        return Panel(
            body,
            title=Text("Conversation", style=resolve_style("accent")),
            subtitle=Text(
                "History: "
                f"{len(self.history)} block(s) · ↑/↓ scroll survives in terminal history",
                style=resolve_style("muted"),
            ),
            border_style=resolve_style("border"),
            padding=(0, 1),
        )

    def _render_input_pane(self) -> Panel:
        content = Text.assemble(
            (_INPUT_LABEL, "bold cyan"),
            (self.input_buffer, "bold bright_white"),
            ("█", "bold bright_cyan"),
        )
        return Panel(
            content,
            title=Text("Input", style=resolve_style("accent")),
            subtitle=Text(
                "Enter send · Ctrl+C interrupt · type exit to end", style="dim"
            ),
            border_style=resolve_style("accent"),
            padding=(0, 1),
        )

    def _render_ribbon(self) -> Panel:
        return Panel(
            ribbon_text(self._state()),
            title=Text("Live Status Ribbon", style=resolve_style("muted")),
            border_style=resolve_style("border.info"),
            padding=(0, 1),
        )

    def render(self) -> Layout:
        """Build the three-pane screen without entering it."""
        layout = Layout()
        layout.split_column(
            Layout(self._render_chat_pane(), name="chat", ratio=1),
            Layout(self._render_input_pane(), name="input", size=3),
            Layout(self._render_ribbon(), name="ribbon", size=5),
        )
        return layout

    def show(self) -> None:
        """Render the full screen once, then leave the prompt machinery below it.

        Input remains the canonical prompt so completion, history, Ctrl+C and
        mid-turn permission prompts keep working. The alternate screen is not
        held open, which also keeps streamed tool output intact.
        """
        # Keep prior output in the scrollback; clearing it hid consent prompts
        # from PTY captures and terminal history.
        self.console.print(self.render())

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self.render())

    def start(self) -> None:
        """Enter the alternate screen."""
        if self._started:
            return
        self._live = Live(
            self.render(),
            console=self.console,
            screen=True,
            refresh_per_second=15,
        )
        self._live.start()
        self._started = True

    def stop(self) -> None:
        """Leave the alternate screen and restore the previous terminal."""
        if not self._started:
            return
        self._started = False
        if self._live is not None:
            with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
                self._live.stop()
            self._live = None

    # ------------------------------------------------------------------
    # Rendering only
    # ------------------------------------------------------------------
    # ChatScreenUI is intentionally a simple layout renderer. Interactive
    # terminal input stays with the canonical prompt machinery in runtime.py.

    def __enter__(self) -> ChatScreenUI:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


__all__ = ["ChatScreenUI", "terminal_capable"]
