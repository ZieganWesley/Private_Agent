"""Start and stop the interactive steering reader for one turn."""

from __future__ import annotations

import sys

from .input_channel import InputChannel
from .turn_control import TurnController


def begin_steering_turn(
    *, interactive: bool, session_ui: object | None = None
) -> tuple[TurnController, InputChannel | None]:
    """Create and start a fresh turn controller and optional TTY reader."""
    controller = TurnController()
    controller.begin_turn()
    return controller, start_steering_channel(
        controller, interactive=interactive, session_ui=session_ui
    )


def start_steering_channel(
    controller: TurnController,
    *,
    interactive: bool,
    session_ui: object | None = None,
) -> InputChannel | None:
    """Return a running TTY reader, or None for non-interactive sessions."""
    if not interactive or not sys.stdin.isatty():
        return None
    channel = InputChannel(controller, session_ui=session_ui)
    channel.start()
    if not channel.running:
        channel.stop()
        return None
    return channel


__all__ = ["begin_steering_turn", "start_steering_channel"]
