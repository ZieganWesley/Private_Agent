"""The /help command: render the command tree and page it on a terminal.

The tree comes from ``agent.interactive.help_renderable`` so the renderable and
the string form share one source. Paging is a terminal affordance, so a piped
or non-interactive session still gets the whole block in a single print.
"""

from __future__ import annotations

import sys
from typing import Any


def _pageable(console: Any, non_interactive: bool) -> bool:
    """Return whether help should be shown a screen at a time."""
    return (
        not non_interactive
        and sys.stdin.isatty()
        and bool(getattr(console, "is_terminal", False))
    )


def print_help(
    session_ui: Any,
    *,
    grouped: bool = True,
    non_interactive: bool = False,
    console: Any = None,
) -> None:
    """Print the help tree, paging it when the reader is on a terminal."""
    from ..interactive import (
        help_page_lines,
        help_renderable,
        print_help_pages,
        wait_for_next_help_page,
    )

    renderable = help_renderable(grouped)
    target = console if console is not None else session_ui.console
    if not _pageable(target, non_interactive):
        session_ui.print(renderable)
        return

    # Render once, then page the plain lines: a rich renderable cannot be
    # split across screens without re-flowing it.
    with target.capture() as capture:
        target.print(renderable)
    print_help_pages(
        lambda page: session_ui.print(page, markup=False),
        capture.get(),
        page_lines=help_page_lines(),
        pause=wait_for_next_help_page,
    )


__all__ = ["print_help"]
