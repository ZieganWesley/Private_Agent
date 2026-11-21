"""Backward-compatible shim: the terminal chrome moved to ``private_agent.ui``.

Kept so existing ``from .status import ...`` and ``from private_agent.agent import
status`` call sites keep working during the migration. New code should import
from ``private_agent.ui`` directly.
"""

from __future__ import annotations

from private_agent.ui import (
    CONTEXT_BAR_WIDTH,
    LOADING,
    NO_COLOR_ENV,
    PROCESSING,
    RIBBON_GAUGE_CELLS,
    SEGMENT_SEPARATOR,
    SPINNER_GAUGE_CELLS,
    SUMMARIZING,
    THINKING,
    TURN_LABEL,
    StatusIndicator,
    StatusState,
    context_bar,
    context_percent,
    context_state_line,
    context_style,
    loading_message,
    phase_spinner,
    processing_message,
    reasoning_text,
    status_ribbon,
    summarizing_message,
    thinking_message,
    waiting_message,
)
from private_agent.ui.console import console

# Back-compat alias used by older callers.
CONTEXT_BAR_WIDTH = CONTEXT_BAR_WIDTH

# Re-export StatusState-based helpers under the old names.
def status_line(
    model_name=None,
    context_tokens=0,
    context_maximum=0,
    permission_mode="auto",
    output_tokens=None,
    background_processes=None,
):
    from private_agent.tools.shell import background_process_indicator
    from private_agent.ui.ribbon import status_ribbon
    from private_agent.ui.state import StatusState

    if background_processes is None:
        try:
            indicator = background_process_indicator()
            if indicator.startswith("background_processes="):
                background_processes = int(indicator.split("=", 1)[1])
            else:
                background_processes = 0
        except (ImportError, OSError, RuntimeError, ValueError):
            background_processes = 0
    state = StatusState(
        model=model_name,
        context_tokens=context_tokens,
        context_maximum=context_maximum,
        permission_mode=permission_mode,
        output_tokens=output_tokens,
        background_processes=background_processes or 0,
    )
    return status_ribbon(state)


__all__ = [
    "LOADING",
    "THINKING",
    "PROCESSING",
    "SUMMARIZING",
    "CONTEXT_BAR_WIDTH",
    "RIBBON_GAUGE_CELLS",
    "SPINNER_GAUGE_CELLS",
    "SEGMENT_SEPARATOR",
    "TURN_LABEL",
    "NO_COLOR_ENV",
    "phase_spinner",
    "loading_message",
    "processing_message",
    "thinking_message",
    "summarizing_message",
    "waiting_message",
    "reasoning_text",
    "StatusIndicator",
    "StatusState",
    "context_percent",
    "context_style",
    "context_bar",
    "context_state_line",
    "status_line",
    "status_ribbon",
    "console",
]
