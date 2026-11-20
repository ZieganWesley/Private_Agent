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
