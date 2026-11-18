"""Phase-aware live status indicators for model loading, processing and thinking.

The CLI shows one spinner while it waits for the model. The spinner colour says
what the program is waiting for, so a slow local-model load is not mistaken for
a model that is already producing output:

- ``LOADING`` (cyan): the local model is being read into memory.
- ``THINKING`` (magenta): the model is emitting reasoning tokens that are not
  shown as text.
- ``PROCESSING`` (cyan): a request is in flight without reasoning output, for
  example an online model that declares no thinking support.
- ``SUMMARIZING`` (blue): the model is condensing older context or the session.

When ``visible_reasoning`` is enabled the reasoning itself is streamed as dimmed
text instead of a spinner.

The same module renders the context-window gauge: a bar that fills with the
share of the prompt budget actually in use, shown beside the spinner and in the
``[Context State]`` line printed before each request.
"""

from __future__ import annotations

from typing import Any

from .console import console

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

CONTEXT_BAR_WIDTH = 24
_SPINNER_CONTEXT_WIDTH = 10
_FILLED_CELL = "\u25b0"
_EMPTY_CELL = "\u25b1"

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


def summarizing_message(model_name: str, subject: str = "the conversation") -> str:
    """Describe the model condensing ``subject`` into a working summary."""
    return f"Summarizing {subject} with {model_name}"


def _context_use(tokens: Any, maximum: Any) -> tuple[int, int, float]:
    """Clamp context-window counts to a usable ``(used, maximum, fraction)``."""
    try:
        limit = int(maximum)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0:
        return 0, 0, 0.0
    try:
        used = int(tokens)
    except (TypeError, ValueError):
        used = 0
    used = max(0, min(used, limit))
    return used, limit, used / limit


def context_percent(tokens: Any, maximum: Any) -> float:
    """Return the share of the context window in use, from 0 to 100."""
    _used, _limit, fraction = _context_use(tokens, maximum)
    return round(fraction * 100, 1)


def context_style(tokens: Any, maximum: Any) -> str:
    """Return the colour that warns as the context window fills up."""
    percent = context_percent(tokens, maximum)
    if percent >= 80:
        return "red"
    if percent >= 50:
        return "yellow"
    return "green"


def context_bar(tokens: Any, maximum: Any, *, width: int = CONTEXT_BAR_WIDTH) -> str:
    """Return a filled bar and percentage for the context window in use."""
    used, limit, fraction = _context_use(tokens, maximum)
    if limit <= 0:
        return ""
    cells = max(1, int(width))
    filled = min(cells, int(fraction * cells + 0.5))
    return (
        f"{_FILLED_CELL * filled}{_EMPTY_CELL * (cells - filled)} "
        f"{used / limit * 100:.0f}%"
    )


def context_state_line(
    stats: Any,
    output_tokens: Any,
    *,
    width: int = CONTEXT_BAR_WIDTH,
) -> str:
    """Return the styled context-window gauge printed before a request.

    ``stats`` is the mapping returned by ``estimate_context_window``.
    """
    values = stats if isinstance(stats, dict) else {}
    tokens, maximum = values.get("tokens", 0), values.get("max", 0)
    used, limit, _fraction = _context_use(tokens, maximum)
    bar = context_bar(tokens, maximum, width=width)
    if not bar:
        return (
            f"[dim][Context State] {used:,} prompt tokens; "
            f"up to {output_tokens} output tokens[/dim]"
        )
    style = context_style(tokens, maximum)
    return (
        f"[dim][Context State][/dim] [{style}]{bar}[/{style}] "
        f"[dim]{used:,} / {limit:,} prompt tokens; up to {output_tokens} "
        "output tokens[/dim]"
    )


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
        context_usage: tuple[int, int] | None = None,
    ):
        self.message = message
        self.phase = phase
        self.context_usage = context_usage
        self._console = console if console is not None else _default_console()
        self._status: Any = None

    @property
    def active(self) -> bool:
        """Return whether a spinner is currently running."""
        return self._status is not None

    def _render(self) -> str:
        _spinner, style = phase_spinner(self.phase)
        text = f"[bold {style}]{self.message}[/bold {style}]"
        if self.context_usage is not None:
            tokens, maximum = self.context_usage
            bar = context_bar(tokens, maximum, width=_SPINNER_CONTEXT_WIDTH)
            if bar:
                bar_style = context_style(tokens, maximum)
                text += f" [dim]\u00b7[/dim] [{bar_style}]{bar}[/{bar_style}]"
        return text

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
        except (ValueError, OSError):
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
        except (ValueError, OSError):
            self._status = None

    def stop(self) -> None:
        """Stop the spinner; safe to call when it is not running."""
        status, self._status = self._status, None
        if status is None:
            return
        try:
            status.stop()
        except (ValueError, OSError):
            return

    def __enter__(self) -> StatusIndicator:
        return self.start()

    def __exit__(self, *_exc_info: Any) -> bool:
        self.stop()
        return False


def _default_console() -> Any:
    """Return the shared CLI console as it is bound right now."""
    return console
