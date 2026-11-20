"""The single shared rich console used by the CLI runtime and its helpers."""

from __future__ import annotations

import os

from rich.console import Console
from rich.theme import Theme

_LIGHT = os.environ.get("PRIVATE_AGENT_LIGHT_THEME", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_THEME = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red bold",
        "tool": "blue",
        "user": "bold blue",
        "dim": "dim",
        "accent": "magenta",
    }
)

console = Console(theme=_THEME, force_terminal=None)


def is_light_terminal() -> bool:
    """Return True when the user opted into the light-terminal palette."""
    return _LIGHT


def make_console(*, light: bool = False) -> Console:
    """Create a console sharing the project theme, optionally for light terminals."""
    return Console(theme=_THEME, force_terminal=None)
