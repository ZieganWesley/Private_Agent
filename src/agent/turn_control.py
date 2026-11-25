"""Mid-turn steering state and safe-point queueing.

A model call cannot receive text while it is generating, so steering happens at
safe points: before a model call, after a stream, after a tool batch, and before
finalization.  The controller owns the queue, cancellation event, and turn
limits; the runtime decides where to drain it.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

TurnState = Literal[
    "idle",
    "generating",
    "tool_running",
    "awaiting_permission",
    "finalizing",
]
SteerMode = Literal["queue", "interrupt", "stop"]

DEFAULT_MAX_STEERING_MESSAGES = 8
DEFAULT_MAX_STEERING_CHARS = 4000
STEERING_PREFIX = "[Steering instruction from the user during this turn]\n"


@dataclass(frozen=True)
class SteerMessage:
    """One instruction supplied while a turn is in progress."""

    text: str
    mode: SteerMode = "queue"
    source: str = "input"
    created_at: float = field(default_factory=time.monotonic)

    @property
    def prompt_text(self) -> str:
        return f"{STEERING_PREFIX}{self.text.strip()}"


class TurnController:
    """Bound the steering queue and expose cancellation to the turn loop."""

    def __init__(
        self,
        *,
        max_messages: int = DEFAULT_MAX_STEERING_MESSAGES,
        max_chars: int = DEFAULT_MAX_STEERING_CHARS,
    ):
        self.max_messages = max(1, int(max_messages))
        self.max_chars = max(1, int(max_chars))
        self.state: TurnState = "idle"
        self._queue: deque[SteerMessage] = deque()
        self._cancel_event = asyncio.Event()
        self._abort_requested = False
        self._interrupt_requested = False
        self._queued_chars = 0
        self.applied: list[SteerMessage] = []

    @property
    def queued_count(self) -> int:
        return len(self._queue)

    @property
    def queued_chars(self) -> int:
        return self._queued_chars

    @property
    def cancel_event(self) -> asyncio.Event:
        return self._cancel_event

    @property
    def stop_requested(self) -> bool:
        return self._abort_requested

    @property
    def interrupt_requested(self) -> bool:
        return self._interrupt_requested

    def begin_turn(self) -> None:
        self.state = "generating"
        self._queue.clear()
        self._queued_chars = 0
        self._cancel_event = asyncio.Event()
        self._abort_requested = False
        self._interrupt_requested = False
        self.applied.clear()

    def set_state(self, state: TurnState) -> None:
        self.state = state

    def queue_message(
        self,
        text: str,
        *,
        mode: SteerMode = "queue",
        source: str = "input",
    ) -> SteerMessage | None:
        """Queue a bounded instruction; stop always wins over the limits."""
        cleaned = str(text or "").strip()
        if not cleaned:
            return None
        if mode == "stop":
            self.request_stop()
            return None
        if mode == "interrupt":
            message = SteerMessage(cleaned, "interrupt", source)
            self._interrupt_requested = True
            self._cancel_event.set()
            self._append(message)
            return message
        if len(self._queue) >= self.max_messages:
            return None
        if self._queued_chars + len(cleaned) > self.max_chars:
            return None
        message = SteerMessage(cleaned, "queue", source)
        self._append(message)
        return message

    def _append(self, message: SteerMessage) -> None:
        self._queue.append(message)
        self._queued_chars += len(message.text)

    def request_interrupt(self) -> None:
        self._interrupt_requested = True
        self._cancel_event.set()

    def request_stop(self) -> None:
        self._abort_requested = True
        self._cancel_event.set()

    def drain(self) -> list[SteerMessage]:
        """Return queued messages in order and clear the queue."""
        messages = list(self._queue)
        self._queue.clear()
        self._queued_chars = 0
        self.applied.extend(messages)
        return messages

    def clear(self) -> None:
        self._queue.clear()
        self._queued_chars = 0

    def finish_turn(self) -> None:
        self.state = "idle"
        self._abort_requested = False
        self._interrupt_requested = False


__all__ = [
    "DEFAULT_MAX_STEERING_CHARS",
    "DEFAULT_MAX_STEERING_MESSAGES",
    "STEERING_PREFIX",
    "SteerMessage",
    "SteerMode",
    "TurnController",
    "TurnState",
]
