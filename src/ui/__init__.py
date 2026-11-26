"""Terminal UI package: the single home for CLI chrome.

Import from here (or from the focused submodules) instead of re-implementing
rendering in business modules.
"""

from __future__ import annotations

from .console import (
    console,  # re-exported leaf
    error,
    info,
    success,
    warning,
)
from .ribbon import ribbon_text, status_ribbon
from .spinner import (
    LOADING,
    PROCESSING,
    SUMMARIZING,
    THINKING,
    StatusIndicator,
    loading_message,
    phase_spinner,
    processing_message,
    reasoning_text,
    summarizing_message,
    thinking_message,
    waiting_message,
)
from .state import (
    StatusState,
    context_bar,
    context_percent,
    context_state_line,
    context_style,
)
from .theme import (
    MESSAGE_LEVELS,
    MESSAGE_SYMBOLS,
    NO_COLOR_ENV,
    SEGMENT_SEPARATOR,
    THEME,
    THEME_NAME,
    TURN_LABEL,
)


# Lazy imports for heavier modules to avoid import cycles.
def __getattr__(name: str):
    if name == "SessionUI":
        from .session_ui import SessionUI

        return SessionUI
    if name == "StatusBar":
        from .status_bar import StatusBar

        return StatusBar
    if name == "TurnReport":
        from .turn import TurnReport

        return TurnReport
    if name == "GenerationStats":
        from .turn import GenerationStats

        return GenerationStats
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LOADING",
    "PROCESSING",
    "SUMMARIZING",
    "THINKING",
    "NO_COLOR_ENV",
    "SEGMENT_SEPARATOR",
    "TURN_LABEL",
    "StatusIndicator",
    "StatusState",
    "console",
    "context_bar",
    "context_percent",
    "MESSAGE_LEVELS",
    "MESSAGE_SYMBOLS",
    "THEME",
    "THEME_NAME",
    "context_state_line",
    "context_style",
    "error",
    "info",
    "success",
    "warning",
    "loading_message",
    "phase_spinner",
    "processing_message",
    "reasoning_text",
    "ribbon_text",
    "status_ribbon",
    "summarizing_message",
    "thinking_message",
    "waiting_message",
    "SessionUI",
    "StatusBar",
    "TurnReport",
    "GenerationStats",
]
