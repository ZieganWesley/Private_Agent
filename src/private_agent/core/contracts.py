"""Provider-neutral contracts shared across agent subsystems."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class ModelCapabilities:
    """Capabilities reported for a selected model; unknown is not support."""

    tools: bool | None = None
    function_calls: bool | None = None
    structured_output: bool | None = None
    thinking: bool | None = None
    vision: bool | None = None
    audio: bool | None = None
    computer_use: bool | None = None
    context_window: int | None = None


@dataclass(frozen=True)
class ModelMessage:
    """Provider-neutral text message sent to or returned by a model."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class ModelRequest:
    """Bounded model invocation input independent of provider SDK classes."""

    messages: tuple[ModelMessage, ...]
    model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Token usage reported by a provider when available."""

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class ToolCall:
    """A normalized model-requested tool invocation."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """Provider-neutral outcome for a normalized tool invocation."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ModelResponse:
    """Provider-neutral model output with optional usage and tool calls."""

    content: str
    model: str | None = None
    usage: TokenUsage | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class Evidence:
    """A traceable source excerpt or location supporting a claim."""

    source: str
    retrieved_at: datetime
    excerpt: str | None = None
    locator: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class MemoryRecord:
    """Provider- and storage-neutral record for attributable agent memory."""

    record_id: str
    kind: Literal["episodic", "semantic", "procedural", "repository"]
    content: str
    created_at: datetime
    scope: str | None = None
    expires_at: datetime | None = None
    evidence: tuple[Evidence, ...] = ()


class ExecutionError(RuntimeError):
    """Structured subsystem failure safe to classify across boundaries."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not code.strip():
            raise ValueError("ExecutionError code must not be empty.")
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = dict(details or {})
