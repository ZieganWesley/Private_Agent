"""The single shared rich console used by the CLI runtime and its helpers."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.style import Style

from .theme import MESSAGE_SYMBOLS, build_theme, fallback_style


class ThemedConsole(Console):
    """A shared console that also exposes one method per message level."""

    def input(self, *args: Any, **kwargs: Any) -> str:
        """Prompt for input without hiding the prompt under the live footer.

        Rich renders ``Console.input`` through the active ``Live`` region, so a
        prompt issued mid-turn is painted into the footer and scrolls away. Stop
        the footer for the duration of the prompt and repaint it afterwards so
        the cursor and the question are always visible.
        """
        from .status_bar import get_active_footer

        footer = get_active_footer(self)
        if footer is None or not getattr(footer, "started", False):
            return super().input(*args, **kwargs)
        footer.stop()
        try:
            return super().input(*args, **kwargs)
        finally:
            footer.start()

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
    """Return a console with the named theme for the requested palette.

    A named theme provides one style per message level and every semantic
    colour used by rich renderables. Rich still honours ``NO_COLOR`` and
    terminal capability detection. ``light`` overrides ``PRIVATE_AGENT_LIGHT``.
    """

    return ThemedConsole(theme=build_theme(light))


def resolve_style(name: str) -> Style:
    """Resolve a semantic style name against the shared theme.

    Returns a concrete ``Style`` so a renderable still works on a console that
    does not carry the application theme (tests, embedded tools).
    """
    return console.get_style(name, default=fallback_style(name))


def apply_theme(light: bool | None = None) -> None:
    """Re-theme the shared console in place (used after ``--light`` is parsed).

    The console object is imported all over the application, so the palette is
    pushed onto it rather than rebuilt.
    """

    console.push_theme(build_theme(light))


def _level_function(level: str):
    def printer(*args: Any, console: Any = None, **kwargs: Any) -> None:
        target = console if console is not None else console_global
        target._print_level(level, *args, **kwargs)

    return printer


console = build_console()
console_global = console

__all__ = [
    "ThemedConsole",
    "apply_theme",
    "build_console",
    "console",
    "console_global",
    "error",
    "info",
    "print_error",
    "print_info",
    "print_success",
    "print_warning",
    "resolve_style",
    "success",
    "warning",
]

info = _level_function("info")
success = _level_function("success")
warning = _level_function("warning")
error = _level_function("error")

# Backwards/alternate friendly aliases.
print_info = info
print_success = success
print_warning = warning
print_error = error
