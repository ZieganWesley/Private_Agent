from unittest.mock import MagicMock

from rich.console import Console

from private_agent import status_overlay


def test_plain_mode_returns_true_when_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert status_overlay._plain_mode() is True


def test_overlay_start_stop_plain_mode(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    console = MagicMock(spec=Console)
    overlay = status_overlay.StatusOverlay(console)
    overlay.start()
    assert overlay._started is False
    overlay.stop()  # no-op


def test_overlay_context_usage_renders(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    import os

    monkeypatch.setattr(os, "isatty", lambda fd: False)
    from rich.console import Console

    from private_agent import status_overlay

    console = Console(record=True)
    overlay = status_overlay.StatusOverlay(console)
    overlay.set_context_usage(42.0)
    rendered = overlay._render()
    console.print(rendered)
    text = console.export_text()
    assert "42%" in text or "Context" in text


def test_overlay_empty_state_renders_thinking(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    import os

    monkeypatch.setattr(os, "isatty", lambda fd: False)
    from rich.console import Console

    from private_agent import status_overlay

    console = Console(record=True)
    overlay = status_overlay.StatusOverlay(console)
    rendered = overlay._render()
    console.print(rendered)
    assert "Thinking..." in console.export_text()
