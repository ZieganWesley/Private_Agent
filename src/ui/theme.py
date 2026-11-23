"""Theme tokens for the CLI UI.

Every glyph, width, separator and label used by the status ribbon, the spinner
and the turn summary lives here, so a visual tweak is a one-line edit in a
single file.
"""

from __future__ import annotations

from rich.theme import Theme

# Separators used between ribbon segments and toolbar segments.
SEGMENT_SEPARATOR = " | "

# Prefix used for the per-turn summary line.
TURN_LABEL = "[Turn]"

# Environment variable that disables colour.
NO_COLOR_ENV = "NO_COLOR"

# Environment variable that requests screen-reader-friendly, text-only gauges.
ACCESSIBLE_ENV = "PRIVATE_AGENT_ACCESSIBLE"

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

THEME = Theme(
    {level: colour for level, (colour, _symbol) in MESSAGE_LEVELS.items()},
    inherit=True,
)
