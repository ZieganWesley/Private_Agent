"""Lightweight in-process event bus for runtime status (Part F)."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="Event")


@dataclass
class Event:
    """Base class for all runtime events."""

    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class TurnStarted(Event):
    turn_id: str = ""
    packs: tuple[str, ...] = ()
    complexity: str = "simple"


@dataclass
class TurnFinished(Event):
    turn_id: str = ""
    duration: float = 0.0
    tokens: int = 0


@dataclass
class ModelCallStarted(Event):
    model: str = ""
    input_tokens: int = 0


@dataclass
class ModelCallFinished(Event):
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0


@dataclass
class ToolStarted(Event):
    call_id: str = ""
    tool_name: str = ""
    pack: str = ""
    args_summary: str = ""
    started_at: float = 0.0


@dataclass
class ToolProgress(Event):
    call_id: str = ""
    message: str = ""


@dataclass
class ToolFinished(Event):
    call_id: str = ""
    status: str = "ok"
    duration: float = 0.0
    output_size: int = 0


@dataclass
class ApprovalRequested(Event):
    call_id: str = ""
    kind: str = ""


@dataclass
class ApprovalResolved(Event):
    call_id: str = ""
    answer: str = ""


@dataclass
class PlanStepChanged(Event):
    task_id: str = ""
    step_id: str = ""
    old_status: str = ""
    new_status: str = ""


@dataclass
class EstimateUpdated(Event):
    task_id: str = ""
    low_seconds: float = 0.0
    high_seconds: float = 0.0


@dataclass
class BudgetWarning(Event):
    message: str = ""


Handler = Callable[[Event], None]


class EventBus:
    """Thread-safe in-process pub/sub. Handler errors are isolated."""

    def __init__(self) -> None:
        self._handlers: dict[type, list[Handler]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: type[Event], handler: Handler) -> None:
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: type[Event], handler: Handler) -> None:
        with self._lock:
            handlers = self._handlers.get(event_type)
            if handlers:
                with contextlib.suppress(ValueError):
                    handlers.remove(handler)

    def emit(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._handlers.get(type(event), []))
            if type(event) is not Event:
                base = list(self._handlers.get(Event, []))
                seen = {id(h) for h in handlers}
                handlers = handlers + [h for h in base if id(h) not in seen]
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001 - handler errors must not break emitters
                logger.exception("Event handler failed for %s", type(event).__name__)

    def clear(self) -> None:
        with self._lock:
            self._handlers.clear()


_default_bus = EventBus()


def get_event_bus() -> EventBus:
    return _default_bus


def reset_event_bus() -> EventBus:
    """Replace the default bus (useful for tests)."""
    global _default_bus
    _default_bus = EventBus()
    return _default_bus
