"""Re-export the shared console from the leaf module."""

from ui.console import build_console, console, resolve_style

__all__ = ["build_console", "console", "resolve_style"]
