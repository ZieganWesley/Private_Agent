"""Phase-aware live status indicators for model loading, processing and thinking.

The CLI shows one spinner while it waits for the model. The spinner colour says
what the program is waiting for, so a slow local-model load is not mistaken for
a model that is already producing output:

- ``LOADING`` (cyan): the local model is being read into memory.
- ``THINKING`` (magenta): the model is emitting reasoning tokens that are not
  shown as text.
- ``PROCESSING`` (cyan): a request is in flight without reasoning output, for
  example an online model that declares no thinking support.

When ``visible_reasoning`` is enabled the reasoning itself is streamed as dimmed
text instead of a spinner.
"""

from __future__ import annotations

from typing import Any

from .console import console

LOADING = "loading"
THINKING = "thinking"
PROCESSING = "processing"

_PHASE_SPINNERS: dict[str, tuple[str, str]] = {
    LOADING: ("dots12", "cyan"),
    THINKING: ("dots2", "magenta"),
    PROCESSING: ("dots", "cyan"),
}

_REASONING_BLOCK_TYPES = frozenset({"thinking", "reasoning", "thinking_delta"})
_REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")


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


def waiting_message(model_name: str, *, local_model: bool) -> str:
    """Pick the waiting message for a local or an online provider."""
    if local_model:
        return loading_message(model_name)
    return processing_message(model_name)


def reasoning_text(message: Any) -> str:
    """Return the reasoning carried by a streamed chunk or message, if any.

    Providers report reasoning either as thinking content blocks or through
    ``additional_kwargs``; blocks win when a message carries both, so the same
    tokens are never shown twice.
    """
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
    """A restartable spinner whose colour states what the program is waiting for.

    UI failures never interrupt a turn: an unavailable console disables the
    indicator instead of raising. ``start`` and ``stop`` are idempotent, and the
    indicator is usable either directly or as a context manager.
    """

    def __init__(
        self,
        message: str,
        *,
        phase: str = PROCESSING,
        console: Any = None,
    ):
        self.message = message
        self.phase = phase
        self._console = console if console is not None else _default_console()
        self._status: Any = None

    @property
    def active(self) -> bool:
        """Return whether a spinner is currently running."""
        return self._status is not None

    def _render(self) -> str:
        _spinner, style = phase_spinner(self.phase)
        return f"[bold {style}]{self.message}[/bold {style}]"

    def start(self) -> StatusIndicator:
        """Start the spinner, ignoring a console that cannot animate."""
        if self._status is not None or not self.message:
            return self
        spinner, style = phase_spinner(self.phase)
        try:
            status = self._console.status(
                self._render(), spinner=spinner, spinner_style=style
            )
            status.start()
        except Exception:
            self._status = None
            return self
        self._status = status
        return self

    def update(self, *, phase: str | None = None, message: str | None = None) -> None:
        """Change the phase or message of a running spinner."""
        if message is not None:
            self.message = message
        if phase is not None:
            self.phase = phase
        if self._status is None:
            return
        spinner, style = phase_spinner(self.phase)
        try:
            self._status.update(self._render(), spinner=spinner, spinner_style=style)
        except Exception:
            self._status = None

    def stop(self) -> None:
        """Stop the spinner; safe to call when it is not running."""
        status, self._status = self._status, None
        if status is None:
            return
        try:
            status.stop()
        except Exception:
            return

    def __enter__(self) -> StatusIndicator:
        return self.start()

    def __exit__(self, *_exc_info: Any) -> bool:
        self.stop()
        return False


def _default_console() -> Any:
    """Return the shared CLI console as it is bound right now."""
    return console
