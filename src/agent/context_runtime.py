"""Live request-path wiring for context telemetry and KV prefix policy."""

from __future__ import annotations

from typing import Any

from .context_cache import ContextCachePolicy
from .context_telemetry import build_context_telemetry


def context_cache_policy() -> ContextCachePolicy:
    """Build the policy from validated runtime settings."""
    from config import (
        CONTEXT_KV_CACHE_ENABLED,
        CONTEXT_KV_CACHE_MIN_PREFIX_TOKENS,
        CONTEXT_KV_CACHE_PROVIDER_HINTS,
    )

    return ContextCachePolicy(
        enabled=bool(CONTEXT_KV_CACHE_ENABLED),
        provider_hints=str(CONTEXT_KV_CACHE_PROVIDER_HINTS),
        min_prefix_tokens=int(CONTEXT_KV_CACHE_MIN_PREFIX_TOKENS),
    )


def prepare_context_request(
    *,
    system_prompts: list[Any],
    prompt_history: list[Any],
    user_message: Any,
    policy: ContextCachePolicy,
    provider: str,
    previous_hash: str | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Order the request prefix and return cache/telemetry metadata."""
    system_text = "\n".join(
        str(getattr(message, "content", "")) for message in system_prompts
    )
    history_text = "\n".join(
        str(getattr(message, "content", "")) for message in prompt_history
    )
    user_text = str(getattr(user_message, "content", user_message))
    sources = {
        "system": system_text,
        "history": history_text,
        "user": user_text,
    }
    prefix = policy.detect_cacheable_prefix(sources)
    invalidation = policy.track_invalidation(
        previous_hash, str(prefix["prefix_hash"])
    )
    cache = {
        **prefix,
        **invalidation,
        "provider": policy.provider_hint(provider),
    }
    telemetry = build_context_telemetry(
        profile="chat",
        total_tokens=0,
        budgets={name: max(0, len(text) // 4) for name, text in sources.items()},
        sources=sources,
        cacheable_prefix_tokens=int(prefix["prefix_tokens"]),
        invalidations=1 if invalidation["invalidated"] else 0,
    ).to_dict()
    # Keep the existing semantic order while making the cacheable prefix explicit.
    messages = list(system_prompts) + list(prompt_history) + [user_message]
    return messages, {"cache": cache, "telemetry": telemetry}
