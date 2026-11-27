"""Canonical context-window math and the immutable UI state.

``StatusState`` is the single source of truth for the status ribbon: one
rounding rule, one fill rule, one segment list (with width-aware dropping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.markup import escape

from .theme import SEGMENT_LABELS, SEGMENT_SEPARATOR

# Context-window gauge glyphs (owned here so theme stays minimal).
GAUGE_FILLED = "\u25b0"
GAUGE_EMPTY = "\u25b1"

# Theme tokens for the context gauge. The status ribbon colours its context
# segment with the same tokens, so both read one definition instead of one
# raw colour each.
CONTEXT_STYLE_OK = "ribbon.context.ok"
CONTEXT_STYLE_WARN = "ribbon.context.warn"
CONTEXT_STYLE_HIGH = "ribbon.context.high"


def _context_use(tokens: Any, maximum: Any) -> tuple[int, int, float]:
    """Clamp context-window counts to a usable ``(used, maximum, fraction)``."""
    try:
        limit = int(maximum)
    except TypeError, ValueError:
        limit = 0
    if limit <= 0:
        return 0, 0, 0.0
    try:
        used = int(tokens)
    except TypeError, ValueError:
        used = 0
    used = max(0, min(used, limit))
    return used, limit, used / limit


def context_percent(tokens: Any, maximum: Any) -> float:
    """Return the share of the context window in use, from 0 to 100."""
    _used, _limit, fraction = _context_use(tokens, maximum)
    return round(fraction * 100, 1)


def context_style(tokens: Any, maximum: Any) -> str:
    """Return the theme token that warns as the context window fills up.

    The result is a ``ribbon.context.*`` style name rather than a raw colour so
    the gauge and the status ribbon cannot drift apart: one threshold table and
    one palette entry now drive both.
    """
    percent = context_percent(tokens, maximum)
    if percent >= 80:
        return CONTEXT_STYLE_HIGH
    if percent >= 50:
        return CONTEXT_STYLE_WARN
    return CONTEXT_STYLE_OK


def context_bar(
    tokens: Any, maximum: Any, *, width: int = 24, accessible: bool = False
) -> str:
    """Return a filled bar and percentage for the context window in use."""
    used, limit, fraction = _context_use(tokens, maximum)
    if limit <= 0:
        return ""
    cells = max(1, int(width))
    filled = min(cells, int(fraction * cells + 0.5))
    percent = used / limit * 100
    if accessible:
        return f"{used}/{limit} ({percent:.0f}%)"
    return f"{GAUGE_FILLED * filled}{GAUGE_EMPTY * (cells - filled)} {percent:.0f}%"


def _pair_text(pair: tuple[str, str]) -> str:
    """Render one ribbon segment, escaping markup in its value."""
    key, value = pair
    if not key:
        return value
    return f"{SEGMENT_LABELS.get(key, key)}: {escape(value)}"


def context_state_line(
    stats: Any,
    output_tokens: Any,
    *,
    width: int = 24,
    accessible: bool = False,
) -> str:
    """Return the styled context-window gauge printed before a request."""
    values = stats if isinstance(stats, dict) else {}
    tokens, maximum = values.get("tokens", 0), values.get("max", 0)
    used, limit, _fraction = _context_use(tokens, maximum)
    bar = context_bar(tokens, maximum, width=width, accessible=accessible)
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


@dataclass(frozen=True)
class StatusState:
    """Immutable snapshot of everything the status ribbon can show."""

    model: Any = None
    context_tokens: Any = 0
    context_maximum: Any = 0
    permission_mode: str = "auto"
    output_tokens: Any = None
    background_processes: int = 0
    provider_mode: str = ""

    def percent(self) -> float:
        return context_percent(self.context_tokens, self.context_maximum)

    def segment_pairs(
        self, *, width: int | None = None, accessible: bool = False
    ) -> list[tuple[str, str]]:
        """Build ``(key, value)`` ribbon segments, dropping trailing ones.

        Keys are the stable names used for per-segment styling; the accessible
        gauge has no key because it reads as a value on its own.
        """
        model_name = "" if self.model is None else str(self.model)
        pairs: list[tuple[str, str]] = [
            ("model", model_name),
            ("context", f"{self.percent()}%"),
            ("permission", str(self.permission_mode)),
        ]
        if self.provider_mode:
            pairs.append(("mode", str(self.provider_mode)))
        if accessible and self.context_maximum:
            used, limit, _fraction = _context_use(
                self.context_tokens, self.context_maximum
            )
            if limit > 0:
                pairs.append(("window", f"{used}/{limit} ({self.percent():.0f}%)"))
        if self.output_tokens is not None:
            pairs.append(("output", str(self.output_tokens)))
        pairs.append(("background", str(self.background_processes)))
        if width is not None:
            while (
                len(SEGMENT_SEPARATOR.join(_pair_text(pair) for pair in pairs)) > width
                and len(pairs) > 3
            ):
                pairs.pop()
        return pairs

    def segments(
        self, *, width: int | None = None, accessible: bool = False
    ) -> list[str]:
        """Build the ribbon segments, dropping trailing ones if ``width`` is given."""
        return [
            _pair_text(pair)
            for pair in self.segment_pairs(width=width, accessible=accessible)
        ]
