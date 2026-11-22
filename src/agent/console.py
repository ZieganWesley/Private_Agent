"""Re-export the shared console from the leaf module."""

from ui.console import (
    DARK_PALETTE,
    LIGHT_PALETTE,
    build_console,
    console,
)

__all__ = ["DARK_PALETTE", "LIGHT_PALETTE", "build_console", "console"]
