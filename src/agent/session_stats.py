"""Session statistics collected during a run and printed as a table on exit."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ui.console import resolve_style

from .compaction import message_text
from .prompts import token_count


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def response_usage(response: Any) -> tuple[int | None, int | None]:
    """Return provider-reported (input, output) tokens when available."""
    usage = getattr(response, "usage_metadata", None)
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = _int_or_none(usage.get("input_tokens"))
    output_tokens = _int_or_none(usage.get("output_tokens"))
    if input_tokens is None or output_tokens is None:
        metadata = getattr(response, "response_metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        alt = metadata.get("token_usage", metadata.get("usage", {}))
        alt = alt if isinstance(alt, dict) else {}
        if input_tokens is None:
            input_tokens = _int_or_none(
                alt.get(
                    "prompt_tokens",
                    alt.get("input_tokens", metadata.get("prompt_eval_count")),
                )
            )
        if output_tokens is None:
            output_tokens = _int_or_none(
                alt.get(
                    "completion_tokens",
                    alt.get("output_tokens", metadata.get("eval_count")),
                )
            )
    return input_tokens, output_tokens


@dataclass
class SessionStats:
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    model_constructions: int = 0
    suspected_cold_loads: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_calls: int = 0
    model_seconds: float = 0.0
    turns: int = 0
    tool_calls: int = 0
    tools: Counter = field(default_factory=Counter)
    models: Counter = field(default_factory=Counter)
    summary_tokens: int = 0
    summary_seconds: float = 0.0
    summary_saved: bool = False

    def record_model_call(
        self,
        messages: Iterable[Any],
        response: Any,
        elapsed: float,
        model_name: str = "",
    ) -> None:
        """Count one model call, estimating tokens the provider did not report."""
        reported_in, reported_out = response_usage(response)
        if reported_in is None or reported_out is None:
            self.estimated_calls += 1
        if reported_in is None:
            reported_in = sum(token_count(message_text(m)) for m in messages)
        if reported_out is None:
            reported_out = token_count(message_text(response))
        self.model_calls += 1
        self.input_tokens += reported_in
        self.output_tokens += reported_out
        self.model_seconds += max(0.0, elapsed)
        if model_name:
            self.models[model_name] += 1

    def reset(self) -> None:
        """Clear the accumulators in place so shared references stay valid."""
        fresh = type(self)()
        for spec in fields(self):
            setattr(self, spec.name, getattr(fresh, spec.name))

    def record_tool_call(self, name: str) -> None:
        self.tool_calls += 1
        self.tools[name] += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def session_seconds(self) -> float:
        return time.monotonic() - self.started


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s"


_MODEL_INFO_FIELDS = ("Model provider", "Model", "Thinking")
_MODEL_INFO_FIELD_SET = frozenset(_MODEL_INFO_FIELDS)


def _section(title: str, rows: list[tuple[str, Any]]) -> Group | None:
    if not rows:
        return None
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=resolve_style("key"), no_wrap=True)
    grid.add_column(overflow="fold", style=resolve_style("value"))
    for label, value in rows:
        grid.add_row(label, str(value))
    return Group(Text(title, style=resolve_style("accent")), grid)


def build_stats_table(stats: SessionStats, info: dict[str, Any]) -> Panel:
    """Build the grouped statistics panel.

    ``info`` holds descriptive session fields. Model fields are pulled into
    their own section; unknown fields stay in the Session section so callers
    can add context without changing this layout.
    """
    session_rows = [
        (label, value)
        for label, value in info.items()
        if label not in _MODEL_INFO_FIELD_SET
    ]
    session_rows.extend(
        (
            ("Total session time", format_duration(stats.session_seconds())),
            ("Prompts this session", stats.turns),
        )
    )
    model_rows = [(label, info[label]) for label in _MODEL_INFO_FIELDS if label in info]
    model_rows.extend(
        (
            ("Time in model calls", format_duration(stats.model_seconds)),
            ("Model calls", stats.model_calls),
        )
    )
    if stats.model_constructions:
        model_rows.append(("Model constructions", stats.model_constructions))
    if stats.suspected_cold_loads:
        model_rows.append(("Suspected cold loads", stats.suspected_cold_loads))
    if stats.models:
        model_rows.append(
            (
                "Models used",
                ", ".join(f"{n} ({c} calls)" for n, c in stats.models.items()),
            )
        )
    token_rows = [
        ("Input tokens", f"{stats.input_tokens:,}"),
        ("Output tokens", f"{stats.output_tokens:,}"),
        ("Total tokens", f"{stats.total_tokens:,}"),
    ]
    if stats.estimated_calls:
        token_rows.append(
            (
                "Token accuracy",
                f"{stats.estimated_calls} of {stats.model_calls} call(s) estimated "
                "(provider did not report usage)",
            )
        )
    tool_detail = ", ".join(f"{n} x{c}" for n, c in stats.tools.most_common())
    tool_rows = [
        (
            "Tool uses",
            f"{stats.tool_calls}" + (f" ({tool_detail})" if tool_detail else ""),
        )
    ]
    summary_rows = [
        ("Summary tokens generated", f"{stats.summary_tokens:,}"),
        ("Summary generation time", format_duration(stats.summary_seconds)),
        ("Summary stored in SQLite", "yes" if stats.summary_saved else "no"),
    ]
    sections = [
        section
        for section in (
            _section("Session", session_rows),
            _section("Model", model_rows),
            _section("Tokens", token_rows),
            _section("Tools", tool_rows),
            _section("Summary", summary_rows),
        )
        if section is not None
    ]
    return Panel(
        Group(*sections),
        title=Text("Session Statistics", style=resolve_style("accent")),
        border_style=resolve_style("border"),
    )
