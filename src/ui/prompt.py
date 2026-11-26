"""The interactive input prompt, styled from the shared theme.

``prompt_toolkit`` draws the prompt outside rich, so the theme is converted to
one ANSI sequence per prompt.  When colour is disabled (``NO_COLOR``,
``PRIVATE_AGENT_ACCESSIBLE``) or the console has no colour system, the plain
message is returned and no escape sequence is emitted.
"""

from __future__ import annotations

import os
from typing import Any

from .console import console as shared_console
from .theme import ACCESSIBLE_ENV, NO_COLOR_ENV, fallback_style


def colour_enabled(console: Any = None) -> bool:
    """Return whether the prompt may carry colour escapes."""
    if os.environ.get(NO_COLOR_ENV) or os.environ.get(ACCESSIBLE_ENV):
        return False
    target = console if console is not None else shared_console
    return getattr(target, "color_system", None) is not None


def _style_codes(style_name: str, console: Any) -> str:
    """Convert one semantic style name to its ANSI SGR parameters."""
    style = console.get_style(style_name, default=fallback_style(style_name))
    codes: list[str] = []
    if style.bold:
        codes.append("1")
    if style.dim:
        codes.append("2")
    if style.italic:
        codes.append("3")
    if style.color is not None:
        codes.extend(style.color.get_ansi_codes())
    return ";".join(codes)


def prompt_text(message: str, *, console: Any = None) -> Any:
    """Return ``message`` as a prompt_toolkit prompt.

    The result is a plain string when colour is unavailable, so screen readers
    and ``NO_COLOR`` terminals get the same words without escapes.
    """
    if not colour_enabled(console):
        return message
    target = console if console is not None else shared_console
    codes = _style_codes("prompt", target)
    if not codes:
        return message
    from prompt_toolkit.formatted_text import ANSI

    return ANSI(f"\x1b[{codes}m{message}\x1b[0m")


__all__ = ["colour_enabled", "prompt_text"]
