"""Re-export the single shared console from the leaf module."""

from private_agent.console import (
    DARK_PALETTE,
    LIGHT_PALETTE,
    build_console,
    console,
)

__all__ = ["DARK_PALETTE", "LIGHT_PALETTE", "build_console", "console"]
