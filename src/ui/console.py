"""The single shared rich console used by the CLI runtime and its helpers."""

from __future__ import annotations

from rich.console import Console


def build_console(light: bool | None = None) -> Console:
    """Return the shared console.

    Colour theming is intentionally minimal: Rich's default theme is used so
    output stays readable in any terminal.
    """

    return Console()


console = build_console()
"""Single shared console instance. Import this everywhere instead of Console()."""
