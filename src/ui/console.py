"""The single shared rich console used by the CLI runtime and its helpers."""

from __future__ import annotations

import os

from rich.console import Console
from rich.theme import Theme

DARK_PALETTE = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        "tool": "blue",
        "user": "bold blue",
        "dim": "dim",
    }
)

LIGHT_PALETTE = Theme(
    {
        "info": "blue",
        "success": "dark_green",
        "warning": "dark_orange",
        "error": "dark_red",
        "tool": "dark_blue",
        "user": "bold dark_blue",
        "dim": "grey50",
    }
)


def _light_mode(light: bool | None) -> bool:
    if light is not None:
        return bool(light)
    return os.environ.get("PRIVATE_AGENT_LIGHT", "").lower() in {"1", "true", "yes"}


def build_console(light: bool | None = None) -> Console:
    palette = LIGHT_PALETTE if _light_mode(light) else DARK_PALETTE
    return Console(theme=palette)


console = build_console()
"""Single shared console instance. Import this everywhere instead of Console()."""
