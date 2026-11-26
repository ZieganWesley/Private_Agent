"""Phase-aware live status indicators for model loading, processing and thinking.

Moved out of ``agent/status.py`` so the terminal chrome has a single home.
The spinner colour says what the program is waiting for, so a slow
local-model load is not mistaken for a model that is already producing
output.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from .theme import ACCESSIBLE_ENV, REDUCED_MOTION_ENV

LOADING = "loading"
THINKING = "thinking"
PROCESSING = "processing"
SUMMARIZING = "summarizing"

_PHASE_SPINNERS: dict[str, tuple[str, str]] = {
    LOADING: ("dots12", "cyan"),
    THINKING: ("dots2", "magenta"),
    PROCESSING: ("dots", "cyan"),
    SUMMARIZING: ("dots3", "blue"),
}

_REASONING_BLOCK_TYPES = frozenset({"thinking", "reasoning", "thinking_delta"})
_REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def motion_reduced() -> bool:
    """Return True when animated spinners should be replaced by static text.

    Terminals and screen readers that cannot keep up with a redrawn spinner
    (or users who simply do not want motion) set ``REDUCED_MOTION_ENV``; the
    status then prints one line and stays put.
    """
    return os.environ.get(REDUCED_MOTION_ENV, "").strip().lower() in _TRUE_VALUES


def phase_spinner(phase: str) -> tuple[str, str]:
    """Return the ``(spinner, style)`` pair that represents a status phase."""
    return _PHASE_SPINNERS.get(phase, _PHASE_SPINNERS[PROCESSING])


def loading_message(model_name: str) -> str:
    """Describe waiting for a local model that may still need to be loaded."""
    return f"Waiting for {model_name} (loading it into memory first if needed)"


def processing_message(model_name: str) -> str:
    """Describe a request that is in flight with no reasoning output."""
    return f"Processing your query with {model_name}"


def thinking_message(model_name: str) -> str:
    """Describe reasoning tokens that are being generated but not shown."""
    return f"{model_name} is thinking"


def summarizing_message(model_name: str, subject: str = "the conversation") -> str:
    """Describe the model condensing ``subject`` into a working summary."""
    return f"Summarizing {subject} with {model_name}"


def waiting_message(model_name: str, *, local_model: bool) -> str:
    """Pick the waiting message for a local or an online provider."""
    if local_model:
        return loading_message(model_name)
    return processing_message(model_name)


def reasoning_text(message: Any) -> str:
    """Return the reasoning carried by a streamed chunk or message, if any."""
    if message is None:
        return ""
    blocks: list[str] = []
    content = getattr(message, "content", "")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "")).lower()
            text = block.get("text")
            if block_type in _REASONING_BLOCK_TYPES and isinstance(text, str):
                blocks.append(text)
    if blocks:
        return "".join(blocks)
    additional = getattr(message, "additional_kwargs", None)
    if isinstance(additional, dict):
        for field in _REASONING_FIELDS:
            value = additional.get(field)
            if isinstance(value, str) and value:
                return value
    return ""


class StatusIndicator:
    """A restartable spinner whose colour states what the program is waiting for."""

    def __init__(
        self,
        message: str,
        *,
        phase: str = PROCESSING,
        console: Any = None,
        context_usage: tuple[int, int] | None = None,
    ):
        self.message = message
        self.phase = phase
        self.context_usage = context_usage
        self._console = console if console is not None else _default_console()
        self._status: Any = None
        self._footer: Any = None
        self._used_footer = False
        self._static = False

    @property
    def active(self) -> bool:
        return self._status is not None or self._static

    def _render(self) -> str:
        _spinner, style = phase_spinner(self.phase)
        text = f"[bold {style}]{self.message}[/bold {style}]"
        if self.context_usage is not None:
            tokens, maximum = self.context_usage
            try:
                percent = round((int(tokens) / int(maximum)) * 100)
            except (TypeError, ValueError, ZeroDivisionError):
                percent = 0
            text += f" ({percent}%)"
        if os.environ.get(ACCESSIBLE_ENV):
            text += f" [dim]({self.phase})[/dim]"

        return text

    def start(self) -> StatusIndicator:
        if self._status is not None or self._static or not self.message:
            return self
        if motion_reduced():
            with contextlib.suppress(ValueError, OSError, RuntimeError):
                self._console.print(self._render(), highlight=False)
                self._static = True
            return self
        try:
            from .status_bar import get_active_footer

            footer = get_active_footer(self._console)
        except Exception:  # noqa: BLE001 - best effort
            footer = None
        if (
            footer is not None
            and getattr(footer, "started", False)
            and getattr(footer, "mode", "") == "A"
        ):
            self._footer = footer
            self._used_footer = True
            try:
                footer.set_busy(self._render(), self.phase)
            except (ValueError, OSError, RuntimeError):
                self._footer = None
                self._used_footer = False
            else:
                self._status = footer
                return self
        spinner, style = phase_spinner(self.phase)
        try:
            status = self._console.status(
                self._render(), spinner=spinner, spinner_style=style
            )
            status.start()
        except (ValueError, OSError, RuntimeError):
            self._status = None
            return self
        self._status = status
        return self

    def update(self, *, phase: str | None = None, message: str | None = None) -> None:
        if message is not None:
            self.message = message
        if phase is not None:
            self.phase = phase
        if self._static:
            return
        if self._used_footer and self._footer is not None:
            try:
                self._footer.set_busy(self._render(), self.phase)
            except (ValueError, OSError, RuntimeError):
                self._footer = None
                self._used_footer = False
                self._status = None
            return
        if self._status is None:
            return
        spinner, style = phase_spinner(self.phase)
        try:
            self._status.update(self._render(), spinner=spinner, spinner_style=style)
        except (ValueError, OSError, RuntimeError):
            self._status = None

    def stop(self) -> None:
        if self._static:
            self._static = False
            return
        if self._used_footer and self._footer is not None:
            with contextlib.suppress(ValueError, OSError, RuntimeError):
                self._footer.clear_busy()
            self._footer = None
            self._used_footer = False
            self._status = None
            return
        status, self._status = self._status, None
        if status is None:
            return
        try:
            status.stop()
        except (ValueError, OSError, RuntimeError):
            return

    def __enter__(self) -> StatusIndicator:
        return self.start()

    def __exit__(self, *_exc_info: Any) -> bool:
        self.stop()
        return False


def _default_console() -> Any:
    from ui.console import console as _shared_console

    return _shared_console


__all__ = [
    "LOADING",
    "THINKING",
    "PROCESSING",
    "SUMMARIZING",
    "phase_spinner",
    "loading_message",
    "processing_message",
    "thinking_message",
    "summarizing_message",
    "waiting_message",
    "reasoning_text",
    "motion_reduced",
    "StatusIndicator",
]
