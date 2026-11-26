"""Theme tokens for the CLI UI.

Every glyph, width, separator and label used by the status ribbon, the spinner
and the turn summary lives here, so a visual tweak is a one-line edit in a
single file. Every colour used by a rich renderable (panels, tables, trees,
spans) is a named style below, so no module hardcodes a colour name.
"""

from __future__ import annotations

import os

from rich.theme import Theme

# Separators used between ribbon segments and toolbar segments.
SEGMENT_SEPARATOR = " | "

# Prefix used for the per-turn summary line.
TURN_LABEL = "[Turn]"

# Environment variable that disables colour.
NO_COLOR_ENV = "NO_COLOR"

# Environment variable that requests screen-reader-friendly, text-only gauges.
ACCESSIBLE_ENV = "PRIVATE_AGENT_ACCESSIBLE"

# Environment variable set by ``--light``.
LIGHT_ENV = "PRIVATE_AGENT_LIGHT"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Named message levels: one colour and one symbol each.
MESSAGE_LEVELS: dict[str, tuple[str, str]] = {
    "info": ("cyan", "ℹ"),
    "success": ("green", "✓"),
    "warning": ("yellow", "⚠"),
    "error": ("red", "✗"),
}

# Backwards/alternate friendly aliases for the symbols.
MESSAGE_SYMBOLS: dict[str, str] = {
    level: symbol for level, (_colour, symbol) in MESSAGE_LEVELS.items()
}

THEME_NAME = "private_agent"

# Semantic styles shared by every renderable. Keep these names stable: modules
# refer to them by string (for example ``border_style="border.error"``).
DARK_STYLES: dict[str, str] = {
    "accent": "cyan",
    "accent_alt": "magenta",
    "muted": "grey62",
    "border": "blue",
    "border.error": "red",
    "border.warning": "yellow",
    "border.success": "green",
    "key": "cyan",
    "value": "white",
    "ribbon.model": "cyan",
    "ribbon.context.ok": "green",
    "ribbon.context.warn": "yellow",
    "ribbon.context.high": "red",
    "ribbon.permission.readonly": "green",
    "ribbon.permission.ask": "cyan",
    "ribbon.permission.auto": "yellow",
    "ribbon.permission.full": "red",
    "ribbon.meta": "grey54",
    "ribbon.dim": "grey54",
}

# Light terminals: same names, darker hues that stay legible on white.
LIGHT_STYLES: dict[str, str] = {
    "accent": "blue",
    "accent_alt": "magenta",
    "muted": "grey30",
    "border": "blue",
    "border.error": "dark_red",
    "border.warning": "dark_orange3",
    "border.success": "dark_green",
    "key": "blue",
    "value": "black",
    "ribbon.model": "blue",
    "ribbon.context.ok": "dark_green",
    "ribbon.context.warn": "dark_orange3",
    "ribbon.context.high": "dark_red",
    "ribbon.permission.readonly": "dark_green",
    "ribbon.permission.ask": "blue",
    "ribbon.permission.auto": "dark_orange3",
    "ribbon.permission.full": "dark_red",
    "ribbon.meta": "grey30",
    "ribbon.dim": "grey30",
}

LIGHT_MESSAGE_LEVELS: dict[str, str] = {
    "info": "blue",
    "success": "dark_green",
    "warning": "dark_orange3",
    "error": "dark_red",
}


def light_requested(light: bool | None = None) -> bool:
    """Return whether the light palette should be used.

    An explicit argument wins; otherwise ``PRIVATE_AGENT_LIGHT`` decides.
    """
    if light is not None:
        return bool(light)
    return os.environ.get(LIGHT_ENV, "").strip().lower() in _TRUTHY


def build_theme(light: bool | None = None) -> Theme:
    """Build the named theme, dark by default and light on request."""
    styles = dict(LIGHT_STYLES) if light_requested(light) else dict(DARK_STYLES)
    levels = LIGHT_MESSAGE_LEVELS if light_requested(light) else {}
    for level, (colour, _symbol) in MESSAGE_LEVELS.items():
        styles[level] = levels.get(level, colour)
    return Theme(styles, inherit=True)


THEME = build_theme()
