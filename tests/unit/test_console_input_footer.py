"""Interactive prompts must stay visible while the live footer is active."""

from __future__ import annotations

import io
from unittest.mock import Mock, call

from ui.console import ThemedConsole
from ui.status_bar import register_active_footer, unregister_active_footer


def _console() -> ThemedConsole:
    return ThemedConsole(file=io.StringIO())


def test_input_stops_and_restarts_the_active_footer():
    console = _console()
    footer = Mock(started=True)
    register_active_footer(console, footer)
    try:
        answer = console.input("Approve? ", stream=io.StringIO("y\n"))
    finally:
        unregister_active_footer(console)

    assert answer == "y\n"
    assert footer.mock_calls == [call.stop(), call.start()]


def test_input_leaves_an_inactive_footer_alone():
    console = _console()
    footer = Mock(started=False)
    register_active_footer(console, footer)
    try:
        answer = console.input("Approve? ", stream=io.StringIO("n\n"))
    finally:
        unregister_active_footer(console)

    assert answer == "n\n"
    assert footer.mock_calls == []


def test_input_without_a_footer_still_reads_the_stream():
    assert _console().input("Answer: ", stream=io.StringIO("hello\n")) == "hello\n"
