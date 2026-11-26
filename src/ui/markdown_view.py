"""One Markdown renderer for replies, whether streamed or delivered at once.

Two paths used to render reply Markdown: ``agent.streaming`` kept a ``Live``
view, and ``ui.chat_ui`` parsed each 120-character chunk as its own document.
That split meant a code fence or table arriving in pieces was rendered as
several broken documents. Both now go through this module; live updates are
throttled so a burst of tokens costs one parse per refresh, not one per chunk.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from rich.live import Live
from rich.markdown import Markdown

DEFAULT_REFRESH_PER_SECOND = 8


def markdown_renderable(text: str) -> Markdown:
    """Return the renderable used for every assistant reply."""
    return Markdown(text)


def print_markdown(console: Any, text: str, *, end: str = "") -> None:
    """Print a complete reply through the shared Markdown renderer."""
    console.print(markdown_renderable(text), soft_wrap=True, end=end)


class MarkdownStream:
    """Accumulate streamed text and repaint one Markdown document.

    ``update`` is cheap: it appends text and only rebuilds the renderable when
    a refresh interval has elapsed. ``stop`` paints the final, complete text so
    nothing is lost when the last chunk arrived inside the interval.
    """

    def __init__(
        self,
        console: Any,
        *,
        refresh_per_second: int = DEFAULT_REFRESH_PER_SECOND,
        transient: bool = False,
    ) -> None:
        self.console = console
        self.refresh_per_second = max(int(refresh_per_second), 1)
        self.transient = transient
        self._interval = 1.0 / self.refresh_per_second
        self._live: Live | None = None
        self._text = ""
        self._last_render = 0.0
        self.renders = 0

    @property
    def text(self) -> str:
        return self._text

    def _paint(self) -> None:
        self.renders += 1
        if self._live is None:
            self._live = Live(
                markdown_renderable(self._text),
                console=self.console,
                refresh_per_second=self.refresh_per_second,
                transient=self.transient,
            )
            self._live.start()
        else:
            self._live.update(markdown_renderable(self._text))

    def update(self, chunk: str) -> None:
        """Append ``chunk`` and repaint if the refresh interval has passed."""
        if not chunk:
            return
        self._text += chunk
        now = time.monotonic()
        if self._live is None or now - self._last_render >= self._interval:
            self._paint()
            self._last_render = now

    def stop(self) -> None:
        """Paint the final text and end the live region."""
        if self._live is None:
            if not self._text:
                return
            self._paint()
        elif self._text:
            self._live.update(markdown_renderable(self._text))
            self.renders += 1
        with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
            self._live.stop()
        self._live = None

    def __enter__(self) -> MarkdownStream:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


__all__ = [
    "DEFAULT_REFRESH_PER_SECOND",
    "MarkdownStream",
    "markdown_renderable",
    "print_markdown",
]
