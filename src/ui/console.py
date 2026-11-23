"""The single shared rich console used by the CLI runtime and its helpers."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.theme import Theme

from .theme import MESSAGE_SYMBOLS, THEME


class ThemedConsole(Console):
    """A shared console that also exposes one method per message level."""

    def _print_level(self, level: str, *args: Any, **kwargs: Any) -> None:
        symbol = MESSAGE_SYMBOLS.get(level, "")
        style = level
        prefix = f"[{style}]{symbol}[/] " if symbol else ""
        self.print(f"{prefix}", end="", markup=True)
        self.print(*args, **kwargs)

    def info(self, *args: Any, **kwargs: Any) -> None:
        self._print_level("info", *args, **kwargs)

    def success(self, *args: Any, **kwargs: Any) -> None:
        self._print_level("success", *args, **kwargs)

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self._print_level("warning", *args, **kwargs)

    def error(self, *args: Any, **kwargs: Any) -> None:
        self._print_level("error", *args, **kwargs)


def build_console(light: bool | None = None) -> ThemedConsole:
    """Return the shared console.

    A named theme provides one style per message level. Rich still honours
    ``NO_COLOR`` and terminal capability detection.
    """

    theme = Theme(THEME.styles if hasattr(THEME, "styles") else {}, inherit=True)
    # Prefer the module-level theme object, which already contains the levels.
    theme = THEME
    return ThemedConsole(theme=theme)


def _level_function(level: str):
    def printer(*args: Any, console: Any = None, **kwargs: Any) -> None:
        target = console if console is not None else console_global
        target._print_level(level, *args, **kwargs)

    return printer


console = build_console()
console_global = console

info = _level_function("info")
success = _level_function("success")
warning = _level_function("warning")
error = _level_function("error")

# Backwards/alternate friendly aliases.
print_info = info
print_success = success
print_warning = warning
print_error = error
