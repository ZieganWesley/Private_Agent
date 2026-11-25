"""KV-cache / prompt-prefix optimization helpers (Phase 8 foundation).

This module does not change prompt semantics.  It provides a
``ContextCachePolicy`` that identifies a cacheable prefix, enforces a
deterministic ordering for cacheable sources, tracks invalidation when the
prefix changes, emits optional provider cache-control hints, and produces
telemetry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

# Sources that are stable across turns and therefore cacheable as a prefix.
DEFAULT_CACHEABLE_SOURCES: tuple[str, ...] = (
    "system",
    "tool_definitions",
    "skills",
    "plan",
    "files",
)

# Volatile sources that should follow the cacheable prefix.
DEFAULT_VOLATILE_SOURCES: tuple[str, ...] = (
    "summary",
    "history",
    "episodic",
    "rag",
    "tool_results",
    "user",
)


@dataclass
class ContextCachePolicy:
    enabled: bool = False
    provider_hints: str = "auto"  # auto | on | off
    min_prefix_tokens: int = 1024
    cacheable_sources: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_CACHEABLE_SOURCES)
    )
    volatile_sources: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_VOLATILE_SOURCES)
    )
    stable_order: list[str] = field(default_factory=list)

    def is_cacheable(self, source_name: str) -> bool:
        return str(source_name) in set(self.cacheable_sources)

    def cacheable_source_set(self) -> set[str]:
        return set(self.cacheable_sources)

    def classify_sources(self, sources: Any) -> tuple[list[str], list[str]]:
        """Split source names into (cacheable, volatile) preserving policy order."""
        names: list[str] = []
        if isinstance(sources, (list, tuple)):
            names = [str(s) for s in sources]
        elif isinstance(sources, dict):
            names = [str(k) for k in sources.keys()]
        elif isinstance(sources, str):
            names = [sources]
        cacheable: list[str] = []
        volatile: list[str] = []
        cacheable_set = self.cacheable_source_set()
        for name in names:
            if name in cacheable_set:
                cacheable.append(name)
            else:
                volatile.append(name)
        # apply stable ordering for cacheable
        ordered_cacheable = self._order_cacheable(cacheable)
        return ordered_cacheable, volatile

    def _order_cacheable(self, names: list[str]) -> list[str]:
        if self.stable_order:
            ordered = [n for n in self.stable_order if n in names]
            ordered += [n for n in names if n not in ordered]
            return ordered
        # default: policy cacheable_sources order
        base_order = list(self.cacheable_sources)
        ordered = [n for n in base_order if n in names]
        ordered += sorted(n for n in names if n not in ordered)
        return ordered

    def stable_source_order(self, sources: Any) -> list[str]:
        """Return a deterministic full source ordering: cacheable prefix first."""
        cacheable, volatile = self.classify_sources(sources)
        # volatile in given order, then any unknowns sorted
        return list(cacheable) + list(volatile)

    def cacheable_prefix_sources(self, sources: Any) -> list[str]:
        cacheable, _ = self.classify_sources(sources)
        return cacheable

    def compute_prefix_hash(self, prefix_texts: dict[str, str] | list | None = None, **kwargs: Any) -> str:
        """Compute a stable hash for the cacheable prefix content."""
        if prefix_texts is None:
            prefix_texts = kwargs.get("texts", {}) or {}
        if isinstance(prefix_texts, (list, tuple)):
            items = [(str(i), str(t)) for i, t in enumerate(prefix_texts)]
        elif isinstance(prefix_texts, dict):
            items = sorted((str(k), str(v)) for k, v in prefix_texts.items())
        else:
            items = [("prefix", str(prefix_texts))]
        hasher = hashlib.sha256()
        for key, value in items:
            hasher.update(key.encode("utf-8", "replace"))
            hasher.update(b"\x00")
            hasher.update(value.encode("utf-8", "replace"))
            hasher.update(b"\x01")
        return hasher.hexdigest()

    def detect_cacheable_prefix(
        self, sources: dict[str, str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Identify the cacheable prefix and its estimated token count."""
        if sources is None:
            sources = kwargs.get("sources", {}) or {}
        if not isinstance(sources, dict):
            sources = {}
        cacheable_names = self.cacheable_prefix_sources(sources)
        prefix_tokens = 0
        # char proxy 4 chars/token
        for name in cacheable_names:
            text = sources.get(name, "")
            prefix_tokens += max(0, len(str(text)) // 4)
        prefix_hash = self.compute_prefix_hash({n: sources.get(n, "") for n in cacheable_names})
        qualifies = self.enabled and prefix_tokens >= self.min_prefix_tokens
        return {
            "cacheable_sources": cacheable_names,
            "prefix_tokens": prefix_tokens,
            "prefix_hash": prefix_hash,
            "qualifies": qualifies,
        }

    def invalidation_reason(
        self, previous_hash: str | None, current_hash: str | None
    ) -> str | None:
        """Return a reason string if the prefix was invalidated, else None."""
        if previous_hash is None:
            return None
        if current_hash is None:
            return "missing_current_hash"
        if previous_hash == current_hash:
            return None
        return "prefix_content_changed"

    def track_invalidation(
        self,
        previous_hash: str | None,
        current_hash: str | None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        computed = reason if reason is not None else self.invalidation_reason(previous_hash, current_hash)
        invalidated = computed is not None
        return {
            "previous_hash": previous_hash,
            "current_hash": current_hash,
            "invalidated": invalidated,
            "reason": computed,
        }

    def provider_hint(self, provider: str | None = None) -> dict[str, Any]:
        """Emit an optional provider-specific cache control marker."""
        provider = (provider or "").strip().lower()
        hints_enabled = self.provider_hints in {"on", "auto"}
        if not self.enabled or not hints_enabled:
            return {"enabled": False, "provider": provider, "markers": {}}
        markers: dict[str, Any] = {}
        if provider in {"anthropic", "claude"}:
            markers = {"cache_control": {"type": "ephemeral"}}
        elif provider in {"openai", "openai-compatible"}:
            markers = {"prompt_cache_key": "stable_prefix"}
        elif provider in {"ollama", "llama.cpp", "lm studio", "lm-studio"}:
            markers = {"prefix_cache": True}
        return {"enabled": True, "provider": provider, "markers": markers}

    def telemetry(
        self,
        prefix_tokens: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0,
        invalidations: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_prefix_tokens": self.min_prefix_tokens,
            "prefix_tokens": int(prefix_tokens),
            "qualifies": bool(self.enabled and prefix_tokens >= self.min_prefix_tokens),
            "cache_hits": int(cache_hits),
            "cache_misses": int(cache_misses),
            "invalidations": int(invalidations),
            "hit_rate": (
                cache_hits / (cache_hits + cache_misses)
                if (cache_hits + cache_misses) > 0
                else 0.0
            ),
        }


def detect_cacheable_prefix(sources: dict[str, str], policy: ContextCachePolicy | None = None) -> dict[str, Any]:
    if policy is None:
        policy = ContextCachePolicy()
    return policy.detect_cacheable_prefix(sources)


def stable_ordering(sources: Any, policy: ContextCachePolicy | None = None) -> list[str]:
    if policy is None:
        policy = ContextCachePolicy()
    return policy.stable_source_order(sources)


__all__ = [
    "DEFAULT_CACHEABLE_SOURCES",
    "DEFAULT_VOLATILE_SOURCES",
    "ContextCachePolicy",
    "detect_cacheable_prefix",
    "stable_ordering",
]
