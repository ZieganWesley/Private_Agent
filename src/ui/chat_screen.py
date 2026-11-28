"""Full-screen production chat surface.

The reference UI's main view is a three-pane screen: a scrollable conversation
history, the active input line, and the live status ribbon. This module makes
that layout a real session surface while leaving the transcript and status
ribbon canonical: messages are rich renderables and the ribbon is composed from
the same immutable ``StatusState`` the rest of the UI uses.

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
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.tree import Tree

from .console import resolve_style
from .startup import capability_icons
from .state import StatusState


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
        self.streaming_renderable: RenderableType | None = None
        self.input_buffer = ""
        # The canonical prompt owns input, so the pane shows the live prompt
        # unless a caller explicitly marks the model as processing.
        self.prompt_active = True
        self._processing = False
        self.scroll_offset = 0
        self._last_max_offset = 0
        self._height_cache: dict[tuple[int, int], int] = {}
        self._cached_width = 0
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
        self.refresh()

    def add_user(self, text: str) -> None:
        """Append a user turn without interpreting its text as markup."""
        line = Text()
        line.append("User: ", style=resolve_style("history.user"))
        line.append(text)
        self.append(line)

    def add_assistant(self, text: str) -> None:
        """Append an assistant reply as sanitised Markdown."""
        from .markdown_view import markdown_renderable

        self.append(
            Group(
                Text("AI:", style=resolve_style("history.assistant")),
                markdown_renderable(text),
            )
        )

    def reset(self) -> None:
        """Clear the rendered history and any scroll state."""
        self.history.clear()
        self._height_cache.clear()
        self.scroll_offset = 0

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    def _pane_height(self) -> int:
        return max(self.console.size.height - 8, 6)

    def _estimate_lines(self, item: RenderableType, effective_width: int) -> int:
        key = (id(item), effective_width)
        cached = self._height_cache.get(key)
        if cached is not None:
            return cached

        if isinstance(item, Text):
            lines = item.plain.splitlines() or [""]
            estimate = sum(
                max(1, (len(line) + effective_width - 1) // effective_width)
                for line in lines
            )
        elif isinstance(item, Rule):
            estimate = 1
        elif isinstance(item, Tree):
            estimate = 5
        elif isinstance(item, Group):
            estimate = sum(
                self._estimate_lines(child, effective_width)
                for child in item.renderables
            )
        elif isinstance(item, Markdown):
            raw = getattr(item, "markup", "")
            lines = raw.splitlines() or [" "]
            estimate = max(
                3,
                sum(
                    max(1, (len(line) + effective_width - 1) // effective_width)
                    for line in lines
                )
                + 1,
            )
        else:
            estimate = 4

        self._height_cache[key] = estimate
        return estimate

    def scroll_by(self, delta: int) -> None:
        """Move the conversation view by ``delta`` blocks."""
        self.scroll_offset = max(
            0, min(self._last_max_offset, self.scroll_offset + delta)
        )
        self.refresh()

    def scroll_to_top(self) -> None:
        """Scroll to the oldest block."""
        self.scroll_offset = self._last_max_offset
        self.refresh()

    def scroll_to_bottom(self) -> None:
        """Return to the live tail."""
        self.scroll_offset = 0
        self.refresh()

    def _compute_visible_items(
        self, items: list[RenderableType], pane_h: int, pane_w: int
    ) -> tuple[list[RenderableType], int, int]:
        total = len(items)
        if total == 0:
            return [], 0, 0

        effective_width = max(pane_w - 4, 20)
        if effective_width != self._cached_width:
            self._height_cache.clear()
            self._cached_width = effective_width

        available_lines = max(4, pane_h - 2)
        max_offset = max(0, total - 1)
        self._last_max_offset = max_offset
        if self.scroll_offset > max_offset:
            self.scroll_offset = max_offset

        end_index = total - self.scroll_offset
        used_lines = 0
        start_index = end_index
        while start_index > 0:
            height = self._estimate_lines(items[start_index - 1], effective_width)
            if used_lines + height > available_lines and start_index < end_index:
                break
            used_lines += height
            start_index -= 1

        return items[start_index:end_index], start_index, end_index

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _render_chat_pane(self) -> Panel:
        pane_h = self._pane_height()
        pane_w = max(self.console.size.width - 2, 24)

        items: list[RenderableType] = list(self.history)
        if self.streaming_renderable is not None and self.scroll_offset == 0:
            items.append(self.streaming_renderable)

        visible, start_index, end_index = self._compute_visible_items(
            items, pane_h, pane_w
        )
        total = len(items)

        if self.scroll_offset > 0:
            subtitle = Text(
                f"Scrolled +{self.scroll_offset} block(s) "
                f"({start_index + 1}-{end_index}/{total}) · "
                "↓/PgDn or /bottom for live tail",
                style=resolve_style("border.warning"),
            )
            border = resolve_style("border.warning")
        else:
            subtitle = Text(
                f"Auto-scroll active · History: {total} block(s) · "
                "↑/↓/PgUp/PgDn or /up /down",
                style=resolve_style("muted"),
            )
            border = resolve_style("border")

        return Panel(
            Group(*visible),
            title=Text("Conversation", style="bold"),
            subtitle=subtitle,
            border_style=border,
            padding=(0, 1),
        )

    def _render_input_pane(self) -> Panel:
        if self.prompt_active and not self._processing:
            content = Text.assemble(
                ("You › ", resolve_style("input.prompt")),
                (self.input_buffer, resolve_style("input.text")),
                ("█", resolve_style("input.cursor")),
            )
            border = resolve_style("border.info")
        else:
            content = Text(
                "Model is responding...", style=resolve_style("input.waiting")
            )
            border = resolve_style("muted")
        return Panel(content, border_style=border, padding=(0, 1))

    def _render_ribbon(self) -> Panel:
        state = self._state()
        busy = self._processing or state.processing
        suffix = "busy" if busy else "ready"

        val = resolve_style(f"ribbon.val.{suffix}")
        accent = resolve_style(f"ribbon.accent.{suffix}")
        group = resolve_style(f"ribbon.group.{suffix}")

        model = escape(str(state.model or ""))
        provider = escape(str(state.provider or state.provider_mode or ""))
        provider_model = f"{provider}/{model}" if provider else model
        mode = escape(str(state.mode))
        thinking = "ON" if state.thinking_enabled else "OFF"
        effort = escape(str(state.thinking_effort))
        permission = escape(str(state.permission_mode))
        tools_used = state.tools_used
        background = state.background_processes
        net_icon = "🌐" if state.internet_online else "📴"
        net_perm = "allow" if state.internet_allowed else "block"
        ctx_used = state.context_tokens
        ctx_max = state.context_maximum
        ctx_pct = int(state.percent())
        workspace = "on" if state.workspace_enabled else "off"
        rag = "on" if state.rag_enabled else "off"
        skills = state.skills_count
        caps = capability_icons(state.capabilities)
        run_badge = escape(f"[{'⏳ busy' if busy else '● ready'}]")

        model_row = (
            f"[{group}][Model & Mode][/{group}] "
            f"[{accent}]{provider_model}[/{accent}] · "
            f"Mode:[{val}]{mode}[/{val}] · "
            f"🧠Think:[{val}]{thinking}({effort})[/{val}]"
        )
        safety_row = (
            f"[{group}][Safety & Execution][/{group}] "
            f"🔐Perm:[{val}]{permission}[/{val}] · "
            f"🧰Tools:[{val}]{tools_used}[/{val}] · "
            f"⚙️BG:[{val}]{background}[/{val}] · "
            f"{net_icon}Net:[{val}]{net_perm}[/{val}]"
        )
        # Emoji are double-width in a terminal, so the reference's spaced
        # separators push the context row past a 100-column panel even though
        # its character count looks short enough. Tighten only when needed.
        sep = " · " if self.console.size.width >= 120 else "·"
        context_row = sep.join(
            (
                f"[{group}][Context & Caps][/{group}] "
                f"📊Ctx:[{val}]{ctx_used}/{ctx_max} ({ctx_pct}%)[/{val}]",
                f"📁WS:[{val}]{workspace}[/{val}]",
                f"🔎RAG:[{val}]{rag}[/{val}]",
                f"📚Skills:[{val}]{skills}[/{val}]",
                f"Caps:{caps}",
                f"[{accent}]{run_badge}[/{accent}]",
            )
        )
        if self.console.size.width >= 160:
            body = f"{model_row}   │   {safety_row}\n{context_row}"
        else:
            # Keep the reference groups readable in ordinary 80-120 column
            # terminals instead of wrapping the safety group mid-label.
            body = f"{model_row}\n{safety_row}\n{context_row}"

        return Panel(
            Text.from_markup(body),
            title=Text("Live Status Ribbon", style=resolve_style("muted")),
            style=resolve_style(f"ribbon.bg.{suffix}"),
            border_style=resolve_style(f"border.ribbon.{suffix}"),
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

    def set_streaming(self, item: RenderableType | None) -> None:
        """Show or clear the in-progress assistant block."""
        self.streaming_renderable = item
        self.refresh()

    def set_processing(self, active: bool) -> None:
        """Switch the input and ribbon between ready and busy states."""
        self._processing = active
        self.refresh()

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
