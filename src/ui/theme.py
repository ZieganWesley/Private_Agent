"""Theme tokens for the CLI UI.

Every glyph, width, separator and label used by the status ribbon, the spinner
and the turn summary lives here, so a visual tweak is a one-line edit in a
single file.
"""

from __future__ import annotations

# Context-window gauge glyphs.
GAUGE_FILLED = "\u25b0"
GAUGE_EMPTY = "\u25b1"

# Separators used between ribbon segments and toolbar segments.
SEGMENT_SEPARATOR = " | "

# Prefix used for the per-turn summary line.
TURN_LABEL = "[Turn]"

# Environment variable that disables colour.
NO_COLOR_ENV = "NO_COLOR"

# Environment variable that requests screen-reader-friendly, text-only gauges.
ACCESSIBLE_ENV = "PRIVATE_AGENT_ACCESSIBLE"

