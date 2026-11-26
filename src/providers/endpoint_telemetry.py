"""Bounded, content-free endpoint telemetry (P6).

Counts and latency only: callers must never pass prompts, documents, or
response bodies.  The registry is process-local and safe for threaded use.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

MAX_EVENTS = 50
MAX_ENDPOINTS = 32


def _host(base_url: str) -> str:
    try:
        parsed = urlsplit(str(base_url or ""))
        host = parsed.hostname or "unknown"
        return f"{host}:{parsed.port}" if parsed.port else host
    except ValueError:
        return "unknown"


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value >= 0 else None


def _token_usage(usage: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    usage = usage if isinstance(usage, Mapping) else {}
    return (
        _positive_int(usage.get("prompt_tokens", usage.get("input_tokens"))),
        _positive_int(usage.get("completion_tokens", usage.get("output_tokens"))),
    )


@dataclass
class _EndpointCounters:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    last_error_type: str = ""
    latency: dict[str, int] = field(
        default_factory=lambda: {"under_100ms": 0, "under_1s": 0, "over_1s": 0}
    )


class EndpointTelemetry:
    """Per-endpoint, per-operation counters with a bounded recent-event log."""

    def __init__(self, *, max_events: int = MAX_EVENTS) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, str], _EndpointCounters] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=max(1, int(max_events)))

    def record(
        self,
        operation: str,
        base_url: str,
        *,
        success: bool,
        elapsed_seconds: float,
        error_type: str | None = None,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        host = _host(base_url)
        elapsed = max(0.0, float(elapsed_seconds or 0.0))
        key = (host, str(operation or "unknown"))
        input_tokens, output_tokens = _token_usage(usage)
        with self._lock:
            counters = self._counters.setdefault(key, _EndpointCounters())
            counters.requests += 1
            counters.total_seconds += elapsed
            counters.max_seconds = max(counters.max_seconds, elapsed)
            if elapsed < 0.1:
                counters.latency["under_100ms"] += 1
            elif elapsed < 1.0:
                counters.latency["under_1s"] += 1
            else:
                counters.latency["over_1s"] += 1
            if input_tokens is not None:
                counters.input_tokens += input_tokens
            if output_tokens is not None:
                counters.output_tokens += output_tokens
            if success:
                counters.successes += 1
            else:
                counters.failures += 1
                counters.last_error_type = str(error_type or "unknown")
            if len(self._counters) > MAX_ENDPOINTS:
                for oldest in list(self._counters)[
                    : len(self._counters) - MAX_ENDPOINTS
                ]:
                    self._counters.pop(oldest, None)
            self._events.append(
                {
                    "host": host,
                    "operation": key[1],
                    "success": bool(success),
                    "elapsed_seconds": round(elapsed, 3),
                    "error_type": str(error_type or ""),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = []
            for (host, operation), counters in sorted(self._counters.items()):
                average = (
                    counters.total_seconds / counters.requests
                    if counters.requests
                    else 0.0
                )
                rows.append(
                    {
                        "host": host,
                        "operation": operation,
                        "requests": counters.requests,
                        "successes": counters.successes,
                        "failures": counters.failures,
                        "avg_seconds": round(average, 3),
                        "max_seconds": round(counters.max_seconds, 3),
                        "latency": dict(counters.latency),
                        "input_tokens": counters.input_tokens,
                        "output_tokens": counters.output_tokens,
                        "last_error_type": counters.last_error_type,
                    }
                )
            return rows

    def recent_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._events.clear()


ENDPOINT_TELEMETRY = EndpointTelemetry()


def record_endpoint_request(
    operation: str,
    base_url: str,
    *,
    success: bool,
    elapsed_seconds: float,
    error_type: str | None = None,
    usage: Mapping[str, Any] | None = None,
) -> None:
    ENDPOINT_TELEMETRY.record(
        operation,
        base_url,
        success=success,
        elapsed_seconds=elapsed_seconds,
        error_type=error_type,
        usage=usage,
    )


def endpoint_snapshot() -> list[dict[str, Any]]:
    return ENDPOINT_TELEMETRY.snapshot()


def reset_endpoint_telemetry() -> None:
    ENDPOINT_TELEMETRY.reset()


def format_endpoint_report() -> str:
    rows = ENDPOINT_TELEMETRY.snapshot()
    if not rows:
        return "Endpoint usage: no requests recorded."
    lines = ["Endpoint usage:"]
    for row in rows:
        lines.append(
            f"  {row['host']} [{row['operation']}]: "
            f"requests={row['requests']} ok={row['successes']} "
            f"failed={row['failures']} avg={row['avg_seconds']:.3f}s "
            f"max={row['max_seconds']:.3f}s "
            f"tokens={row['input_tokens']}/{row['output_tokens']}"
            + (
                f" last_error={row['last_error_type']}"
                if row["last_error_type"]
                else ""
            )
        )
    return "\n".join(lines)


def start_timer() -> float:
    return time.perf_counter()


__all__ = [
    "ENDPOINT_TELEMETRY",
    "EndpointTelemetry",
    "endpoint_snapshot",
    "format_endpoint_report",
    "record_endpoint_request",
    "reset_endpoint_telemetry",
    "start_timer",
]
