"""Model invocation with budget enforcement, retries and streaming output."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from rich.panel import Panel
from rich.text import Text

from ..config import STREAMING_OUTPUT, VISIBLE_REASONING
from ..run_logging import RUN_LOGGER, log_event
from .console import console
from .prompts import token_count as _token_count
from .session_state import SESSION_STATS as _SESSION_STATS
from .status import (
    LOADING,
    PROCESSING,
    THINKING,
    StatusIndicator,
    reasoning_text,
    thinking_message,
    waiting_message,
)


def _supports_async_streaming(model: Any) -> bool:
    stream_method = getattr(model, "astream", None)
    return (
        STREAMING_OUTPUT
        and callable(stream_method)
        and inspect.isasyncgenfunction(stream_method)
    )


def _visible_chunk_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text", ""), str)
            )
        )
    return ""


def _visible_reasoning_summary(response: Any) -> str:
    """Return only an explicitly labeled summary, never raw reasoning tokens."""
    candidates = [
        getattr(response, "additional_kwargs", None),
        getattr(response, "response_metadata", None),
    ]
    for metadata in candidates:
        if not isinstance(metadata, dict):
            continue
        summary = metadata.get("reasoning_summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()[:2000]
    return ""


def _model_response_is_empty(response: Any) -> bool:
    """Return whether a model response has neither visible content nor tool calls."""
    if response is None:
        return True
    content = _visible_chunk_text(getattr(response, "content", ""))
    return not content.strip() and not getattr(response, "tool_calls", None)


def _display_generation_rate(response: Any, elapsed: float) -> None:
    """Report model output throughput, preferring provider token usage."""
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        usage = {}
    output_tokens = usage.get("output_tokens")
    source = "model-reported"
    if (
        not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens < 0
    ):
        metadata = getattr(response, "response_metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        usage = metadata.get("token_usage", metadata.get("usage", {}))
        usage = usage if isinstance(usage, dict) else {}
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    if (
        not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens < 0
    ):
        output_tokens = _token_count(
            _visible_chunk_text(getattr(response, "content", ""))
        )
        source = "estimated visible"
    if output_tokens <= 0:
        return
    safe_elapsed = max(elapsed, 0.001)
    console.print(
        f"[dim][Generation][/dim] {output_tokens} {source} output tokens "
        f"in {elapsed:.2f}s ({output_tokens / safe_elapsed:.1f} tokens/s)"
    )


def _display_visible_reasoning(response: Any) -> None:
    if not VISIBLE_REASONING:
        return
    summary = _visible_reasoning_summary(response)
    if summary:
        console.print(
            Panel(
                Text(summary),
                title=Text("Model rationale summary"),
                border_style="magenta",
            )
        )


async def _invoke_with_budget(
    model: Any,
    messages: list,
    deadline: float,
    *,
    model_name: str = "model",
    status_message: str = "Processing model response...",
    local_model: bool = False,
    context_usage: tuple[int, int] | None = None,
):
    """Invoke a model within the task deadline, streaming text when supported."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Agent task time budget exhausted before model invocation.")
    started = time.monotonic()
    stream_method: Callable[[list], AsyncIterator[Any]] | None = getattr(
        model, "astream", None
    )
    if _supports_async_streaming(model) and stream_method is not None:
        indicator = StatusIndicator(
            waiting_message(model_name, local_model=local_model),
            phase=LOADING if local_model else PROCESSING,
            context_usage=context_usage,
        ).start()

        async def collect_stream():
            aggregate = None
            first_token_at = None
            emitted_header = False
            reasoning_open = False
            try:
                async for chunk in stream_method(messages):
                    aggregate = chunk if aggregate is None else aggregate + chunk
                    reasoning = reasoning_text(chunk)
                    text = _visible_chunk_text(getattr(chunk, "content", ""))
                    if reasoning:
                        if VISIBLE_REASONING:
                            indicator.stop()
                            if not reasoning_open:
                                console.print(
                                    f"\n[dim]Thinking ({model_name}):[/dim] ",
                                    end="",
                                )
                                reasoning_open = True
                            console.print(
                                reasoning,
                                end="",
                                style="dim",
                                markup=False,
                                highlight=False,
                                soft_wrap=True,
                            )
                        elif indicator.active and indicator.phase != THINKING:
                            indicator.update(
                                phase=THINKING,
                                message=thinking_message(model_name),
                            )
                    if text:
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                        indicator.stop()
                        if reasoning_open:
                            console.print()
                            reasoning_open = False
                        if not emitted_header:
                            console.print(
                                f"\n[bold green]AI ({model_name}):[/bold green] ",
                                end="",
                            )
                            emitted_header = True
                        console.print(
                            text,
                            end="",
                            markup=False,
                            highlight=False,
                            soft_wrap=True,
                        )
            finally:
                indicator.stop()
                if reasoning_open:
                    console.print()
                if emitted_header:
                    console.print()
            return aggregate, first_token_at

        try:
            aggregate, first_token_at = await asyncio.wait_for(
                collect_stream(),
                timeout=remaining,
            )
            if _model_response_is_empty(aggregate):
                _SESSION_STATS.record_model_call(
                    messages,
                    aggregate if aggregate is not None else AIMessageChunk(content=""),
                    time.monotonic() - started,
                    model_name,
                )
                log_event(
                    RUN_LOGGER,
                    "model.stream.empty_retry_started",
                    level=logging.WARNING,
                    model=model_name,
                )
                retry_remaining = deadline - time.monotonic()
                retry_method = getattr(model, "ainvoke", None)
                if retry_remaining <= 0 or not callable(retry_method):
                    log_event(
                        RUN_LOGGER,
                        "model.stream.empty_retry_unavailable",
                        level=logging.WARNING,
                        model=model_name,
                    )
                    return AIMessageChunk(
                        content=(
                            "The model returned no response after an automatic "
                            "retry. Please try the request again in this session."
                        )
                    )
                try:
                    retry_response = retry_method(messages)
                    if not inspect.isawaitable(retry_response):
                        log_event(
                            RUN_LOGGER,
                            "model.stream.empty_retry_unavailable",
                            level=logging.WARNING,
                            model=model_name,
                            reason="retry_not_awaitable",
                        )
                        return AIMessageChunk(
                            content=(
                                "The model could not retry this response. "
                                "Please try again in this session."
                            )
                        )
                    aggregate = await asyncio.wait_for(
                        retry_response,
                        timeout=retry_remaining,
                    )
                except (ValueError, OSError) as retry_error:
                    log_event(
                        RUN_LOGGER,
                        "model.stream.empty_retry_failed",
                        level=logging.ERROR,
                        model=model_name,
                        error_type=type(retry_error).__name__,
                        exc_info=True,
                    )
                    return AIMessageChunk(
                        content=(
                            "The model did not return a response after an "
                            "automatic retry. Please try again; this session "
                            "remains available."
                        )
                    )
                if _model_response_is_empty(aggregate):
                    _SESSION_STATS.record_model_call(
                        messages,
                        aggregate,
                        time.monotonic() - started,
                        model_name,
                    )
                    log_event(
                        RUN_LOGGER,
                        "model.stream.empty_retry_returned_empty",
                        level=logging.WARNING,
                        model=model_name,
                    )
                    return AIMessageChunk(
                        content=(
                            "The model returned an empty response twice. "
                            "Please try rephrasing or retrying in this session."
                        )
                    )
                first_token_at = None
            elapsed = time.monotonic() - started
            _display_visible_reasoning(aggregate)
            _display_generation_rate(aggregate, elapsed)
            _SESSION_STATS.record_model_call(
                messages,
                aggregate if aggregate is not None else AIMessageChunk(content=""),
                elapsed,
                model_name,
            )
            log_event(
                RUN_LOGGER,
                "model.invoke.completed",
                mode="stream_or_retry",
                model=model_name,
                elapsed_seconds=round(elapsed, 3),
                first_token_seconds=(
                    round(first_token_at - started, 3)
                    if first_token_at is not None
                    else None
                ),
            )
            return aggregate if aggregate is not None else AIMessageChunk(content="")
        except asyncio.TimeoutError:
            log_event(
                RUN_LOGGER,
                "model.invoke.timeout",
                level=logging.WARNING,
                mode="stream",
                model=model_name,
                elapsed_seconds=round(time.monotonic() - started, 3),
                exc_info=True,
            )
            raise
        except (ValueError, OSError):
            log_event(
                RUN_LOGGER,
                "model.invoke.failed",
                level=logging.ERROR,
                mode="stream",
                model=model_name,
                elapsed_seconds=round(time.monotonic() - started, 3),
                exc_info=True,
            )
            raise

    try:
        response = await asyncio.wait_for(model.ainvoke(messages), timeout=remaining)
    except (ValueError, OSError):
        log_event(
            RUN_LOGGER,
            "model.invoke.failed",
            level=logging.ERROR,
            mode="invoke",
            model=model_name,
            elapsed_seconds=round(time.monotonic() - started, 3),
            exc_info=True,
        )
        raise
    if _model_response_is_empty(response):
        _SESSION_STATS.record_model_call(
            messages,
            response if response is not None else AIMessageChunk(content=""),
            time.monotonic() - started,
            model_name,
        )
        log_event(
            RUN_LOGGER,
            "model.invoke.empty_retry_started",
            level=logging.WARNING,
            model=model_name,
        )
        retry_remaining = deadline - time.monotonic()
        if retry_remaining <= 0:
            return AIMessage(
                content=(
                    "The model returned no response. The turn budget expired "
                    "before an automatic retry could start."
                )
            )
        try:
            response = await asyncio.wait_for(
                model.ainvoke(messages),
                timeout=retry_remaining,
            )
        except (ValueError, OSError) as retry_error:
            log_event(
                RUN_LOGGER,
                "model.invoke.empty_retry_failed",
                level=logging.ERROR,
                model=model_name,
                error_type=type(retry_error).__name__,
                exc_info=True,
            )
            return AIMessage(
                content=(
                    "The model did not return a response after an automatic "
                    "retry. Please try again; this session remains available."
                )
            )
        if _model_response_is_empty(response):
            _SESSION_STATS.record_model_call(
                messages,
                response if response is not None else AIMessageChunk(content=""),
                time.monotonic() - started,
                model_name,
            )
            log_event(
                RUN_LOGGER,
                "model.invoke.empty_retry_returned_empty",
                level=logging.WARNING,
                model=model_name,
            )
            return AIMessage(
                content=(
                    "The model returned an empty response twice. Please try "
                    "rephrasing or retrying in this session."
                )
            )
    elapsed = time.monotonic() - started
    _display_visible_reasoning(response)
    _display_generation_rate(response, elapsed)
    _SESSION_STATS.record_model_call(messages, response, elapsed, model_name)
    log_event(
        RUN_LOGGER,
        "model.invoke.completed",
        mode="invoke",
        model=model_name,
        elapsed_seconds=round(elapsed, 3),
    )
    return response


async def _invoke_with_status(
    model: Any,
    messages: list,
    deadline: float,
    *,
    model_name: str,
    status_message: str,
    local_model: bool = False,
    thinking_enabled: bool = False,
    context_usage: tuple[int, int] | None = None,
):
    """Show operation progress and report elapsed time for slow model calls."""
    started = time.monotonic()
    if _supports_async_streaming(model):
        try:
            response = await _invoke_with_budget(
                model,
                messages,
                deadline,
                model_name=model_name,
                status_message=status_message,
                local_model=local_model,
                context_usage=context_usage,
            )
        except (ValueError, OSError) as exc:
            reference_id = uuid.uuid4().hex[:8]
            log_event(
                RUN_LOGGER,
                "ui.model_operation_failed",
                level=logging.ERROR,
                reference_id=reference_id,
                model=model_name,
                error_type=type(exc).__name__,
            )
            console.print(
                "[red][Model request][/red] Failed "
                f"(reference {reference_id}). Retry or use `switch` to choose "
                "another provider."
            )
            raise
    else:
        phase = THINKING if thinking_enabled else LOADING if local_model else PROCESSING
        message = (
            thinking_message(model_name)
            if thinking_enabled
            else waiting_message(model_name, local_model=local_model)
        )
        try:
            with StatusIndicator(message, phase=phase, context_usage=context_usage):
                response = await _invoke_with_budget(
                    model,
                    messages,
                    deadline,
                    model_name=model_name,
                    local_model=local_model,
                )
        except (ValueError, OSError) as exc:
            reference_id = uuid.uuid4().hex[:8]
            log_event(
                RUN_LOGGER,
                "ui.model_operation_failed",
                level=logging.ERROR,
                reference_id=reference_id,
                model=model_name,
                error_type=type(exc).__name__,
            )
            console.print(
                "[red][Model request][/red] Failed "
                f"(reference {reference_id}). Retry or use `switch` to choose "
                "another provider."
            )
            raise
    elapsed = time.monotonic() - started
    if elapsed >= 2:
        console.print(
            f"[dim][Completed][/dim] {status_message.rstrip('.')} in {elapsed:.1f}s."
        )
    return response
