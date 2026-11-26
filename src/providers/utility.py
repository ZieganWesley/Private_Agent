"""Internal utility generation over the raw completions route (P4).

Raw completions are opt-in.  When disabled, or when the endpoint refuses the
route, callers keep their existing chat-model path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from typing import Any

from .transports import CompletionsClient, TransportResult

DEFAULT_UTILITY_MAX_TOKENS = 512


def raw_completions_enabled() -> bool:
    from config import RAW_COMPLETIONS_ENABLED

    return bool(RAW_COMPLETIONS_ENABLED)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return str(content or "")


def prompt_text(messages: Iterable[Any]) -> str:
    """Flatten chat messages into a bounded raw-completions prompt."""
    parts: list[str] = []
    for message in messages:
        role = message.__class__.__name__.replace("Message", "")
        parts.append(f"{role}: {_message_text(message)}")
    return "\n\n".join(parts)


def _llm_base_url(llm: Any) -> str | None:
    for attribute in ("openai_api_base", "base_url", "api_base"):
        value = getattr(llm, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def complete_utility(
    messages: Iterable[Any],
    *,
    model: str,
    base_url: str | None = None,
    llm: Any = None,
    max_tokens: int = DEFAULT_UTILITY_MAX_TOKENS,
    temperature: float = 0.0,
    client_factory: Callable[..., Any] | None = None,
) -> TransportResult | None:
    """Return a raw completion, or ``None`` when disabled or unusable."""
    if not raw_completions_enabled():
        return None
    endpoint = base_url or _llm_base_url(llm)
    if not endpoint:
        return None
    client = CompletionsClient(
        endpoint,
        client_factory=client_factory,
        enabled=True,
    )
    result = client.complete(
        prompt_text(messages),
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if result is None or not result.text.strip():
        return None
    return result


async def complete_utility_text(
    messages: Iterable[Any],
    *,
    model: str,
    chat_fallback: Callable[[], Any],
    base_url: str | None = None,
    llm: Any = None,
    max_tokens: int = DEFAULT_UTILITY_MAX_TOKENS,
    client_factory: Callable[..., Any] | None = None,
) -> tuple[str, bool]:
    """Use raw completions when enabled, otherwise the chat fallback.

    Returns ``(text, used_raw)``.  Any transport or shape failure silently
    degrades to the chat fallback so utility work never fails a turn.
    """
    materialized = list(messages)
    result = await asyncio.to_thread(
        complete_utility,
        materialized,
        model=model,
        base_url=base_url,
        llm=llm,
        max_tokens=max_tokens,
        client_factory=client_factory,
    )
    if result is not None:
        return result.text.strip(), True
    fallback = chat_fallback()
    if hasattr(fallback, "__await__"):
        fallback = await fallback
    return str(getattr(fallback, "content", fallback)).strip(), False


__all__ = [
    "DEFAULT_UTILITY_MAX_TOKENS",
    "complete_utility",
    "complete_utility_text",
    "prompt_text",
    "raw_completions_enabled",
]
