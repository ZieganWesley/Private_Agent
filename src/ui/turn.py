"""Per-turn report and generation statistics.

Collects the timing/token facts for a single turn and renders the one summary
line that replaces the old ``[Generation]`` / ``[Completed]`` / ``[Latency]``
triple.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .theme import TURN_LABEL


@dataclass
class GenerationStats:
    """Token/timing facts for a single model generation."""

    output_tokens: int = 0
    estimated: bool = True
    first_token_seconds: float = 0.0
    last_token_seconds: float = 0.0

    @property
    def decode_seconds(self) -> float:
        return max(0.0, self.last_token_seconds - self.first_token_seconds)

    @property
    def tokens_per_second(self) -> float:
        if self.decode_seconds <= 0:
            return 0.0
        return self.output_tokens / self.decode_seconds


@dataclass
class TurnReport:
    """Mutable accumulator for a single turn."""

    start_time: float = field(default_factory=time.monotonic)
    model: Any = None
    tool_calls: int = 0
    generation: GenerationStats | None = None

    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def summary_line(self) -> str:
        elapsed = self.elapsed()
        gen = self.generation
        tokens = gen.output_tokens if gen is not None and gen.output_tokens else 0
        model = self.model if self.model is not None else "unknown"
        return f"{TURN_LABEL} {model} | {tokens} tokens | {elapsed:.2f}s"

    def __str__(self) -> str:
        return self.summary_line()
