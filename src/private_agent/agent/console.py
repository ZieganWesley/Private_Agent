"""The single shared rich console used by the CLI runtime and its helpers."""

from __future__ import annotations

import os

from rich.console import Console
from rich.theme import Theme

PALETTE = Theme(
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

_LIGHT = os.environ.get("PRIVATE_AGENT_LIGHT", "").lower() in {"1", "true", "yes"}


def build_console(light: bool | None = None) -> Console:
    return Console(theme=PALETTE)


console = build_console()
"""Single shared console instance. Import this everywhere instead of Console()."""
