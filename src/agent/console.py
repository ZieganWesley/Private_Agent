"""Re-export the shared console from the leaf module."""

from ui.console import build_console, console

__all__ = ["build_console", "console"]
