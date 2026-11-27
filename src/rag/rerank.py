"""Optional HTTP reranking with a permanent local cosine fallback.

``/v1/rerank`` is not part of the core OpenAI-compatible surface, so every
failure is treated as "this endpoint does not rerank" and the caller keeps the
existing local embedding similarity order.  A small circuit breaker prevents a
broken endpoint from adding a timeout to every retrieval.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_RERANK_TIMEOUT_SECONDS = 5.0
DEFAULT_RERANK_TOP_N = 20
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 60.0
DEFAULT_CACHE_ENTRIES = 128
DEFAULT_MAX_CANDIDATES = 50
DEFAULT_LOCAL_THRESHOLD = 20
RERANK_MODES = ("api", "local", "auto")


@dataclass
class CircuitBreaker:
    """Stop calling a rerank endpoint after repeated failures."""

    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    failures: int = 0
    opened_at: float | None = None
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if self.clock() - self.opened_at >= self.cooldown_seconds:
            self.failures = 0
            self.opened_at = None
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = self.clock()


def _document_text(document: Any) -> str:
    text = getattr(document, "page_content", None)
    if text is None:
        text = getattr(document, "content", "")
    return str(text or "")


def _score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _results(body: Any) -> list[dict[str, Any]] | None:
    if isinstance(body, dict):
        for key in ("results", "data", "output"):
            value = body.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return None
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    return None


class HttpReranker:
    """Call an OpenAI-style ``/v1/rerank`` endpoint with bounded caching."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = DEFAULT_RERANK_TIMEOUT_SECONDS,
        top_n: int = DEFAULT_RERANK_TOP_N,
        client_factory: Callable[..., Any] | None = None,
        breaker: CircuitBreaker | None = None,
        enabled: bool = True,
        mode: str = "api",
        local_threshold: int = DEFAULT_LOCAL_THRESHOLD,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        cache_entries: int = DEFAULT_CACHE_ENTRIES,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.top_n = max(1, int(top_n))
        self.client_factory = client_factory
        self.breaker = breaker or CircuitBreaker()
        self.enabled = enabled
        self.mode = mode if mode in RERANK_MODES else "api"
        self.local_threshold = max(0, int(local_threshold))
        self.max_candidates = max(1, int(max_candidates))
        self.cache_entries = max(1, int(cache_entries))
        self._cache: OrderedDict[str, list[int]] = OrderedDict()

    def _url(self) -> str:
        return f"{self.base_url}/rerank"

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _key(query: str, documents: Sequence[Any], top_n: int) -> str:
        digest = hashlib.sha256()
        digest.update(query.encode("utf-8", "replace"))
        for document in documents:
            digest.update(b"\0")
            digest.update(_document_text(document).encode("utf-8", "replace"))
        digest.update(f"\0{top_n}".encode())
        return digest.hexdigest()

    def _remember(self, key: str, order: list[int]) -> None:
        self._cache[key] = order
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_entries:
            self._cache.popitem(last=False)

    def rerank(
        self, query: str, documents: Sequence[Any], *, top_n: int | None = None
    ) -> list[Any] | None:
        """Return documents in API order, or ``None`` when unavailable."""
        if not self.enabled or not documents or not self.breaker.allow():
            return None
        if self.mode == "local":
            return None
        if self.mode == "auto" and len(documents) <= self.local_threshold:
            return None
        candidates = list(documents)[: self.max_candidates]
        limit = min(len(candidates), max(1, int(top_n or self.top_n)))
        cache_key = self._key(query, candidates, limit)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return [candidates[index] for index in cached]

        payload: dict[str, Any] = {
            "query": query,
            "documents": [_document_text(document) for document in candidates],
            "top_n": limit,
        }
        if self.model:
            payload["model"] = self.model

        client = None
        try:
            factory = self.client_factory or httpx.Client
            client = factory(
                headers=self._request_headers(),
                timeout=self.timeout,
            )
            response = client.post(self._url(), json=payload)
            response.raise_for_status()
            items = _results(response.json())
            if items is None:
                self.breaker.record_failure()
                return None
            scored: list[tuple[float, int]] = []
            seen: set[int] = set()
            for item in items:
                index = item.get("index")
                score = _score(item.get("relevance_score", item.get("score")))
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(candidates)
                    or index in seen
                    or score is None
                ):
                    continue
                seen.add(index)
                scored.append((score, index))
            if not scored:
                self.breaker.record_failure()
                return None
            scored.sort(key=lambda item: (-item[0], item[1]))
            order = [index for _, index in scored[:limit]]
            self._remember(cache_key, order)
            self.breaker.record_success()
            return [candidates[index] for index in order]
        except httpx.HTTPError, ValueError, TypeError, OSError:
            self.breaker.record_failure()
            return None
        finally:
            if client is not None and hasattr(client, "close"):
                with contextlib.suppress(httpx.HTTPError, OSError):
                    client.close()


def local_cosine_rerank(
    query: str,
    documents: Sequence[Any],
    *,
    embeddings: Any,
) -> list[Any]:
    """Rank documents by embedding cosine similarity; never raises."""
    if not documents:
        return []
    try:
        query_vec = embeddings.embed_query(query)
        doc_vecs = embeddings.embed_documents(
            [_document_text(document) for document in documents]
        )
    except AttributeError, ValueError, RuntimeError, TypeError, OSError:
        return list(documents)
    if not query_vec or not doc_vecs or len(doc_vecs) != len(documents):
        return list(documents)

    def cosine(a: Iterable[float], b: Iterable[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    scored = sorted(
        zip(documents, doc_vecs, strict=False),
        key=lambda item: cosine(query_vec, item[1]),
        reverse=True,
    )
    return [document for document, _ in scored]


def reranker_from_config(
    config: Any, *, preferred_model: str | None = None
) -> HttpReranker | None:
    """Build a reranker for the endpoint that advertises the rerank role.

    ``preferred_model`` is the remembered choice. It is reused when the
    endpoint still advertises it, so the router does not silently switch
    models between launches.
    """
    try:
        from providers.endpoints import EndpointRegistry

        registry = EndpointRegistry.from_config(config)
    except Exception:  # noqa: BLE001 - optional feature boundary
        return None
    endpoint = registry.resolve("rerank")
    if endpoint is None:
        return None
    rag = config.get("rag", {}) if isinstance(config, dict) else {}
    rag = rag if isinstance(rag, dict) else {}

    def _number(key: str, default: float) -> float:
        value = rag.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return float(value) if value > 0 else default

    def _integer(key: str, default: int) -> int:
        value = rag.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return default
        return value

    mode = str(rag.get("rerank_mode", "auto")).strip().lower()
    if mode not in RERANK_MODES:
        mode = "auto"
    if not rag.get("rerank_enabled", True):
        mode = "local"
    model = rag.get("rerank_model")
    if not isinstance(model, str) or not model.strip():
        remembered = str(preferred_model or "").strip()
        if remembered and (not endpoint.models or remembered in endpoint.models):
            model = remembered
        else:
            model = endpoint.models[0] if endpoint.models else None
    breaker = CircuitBreaker(
        failure_threshold=_integer(
            "rerank_breaker_failures", DEFAULT_FAILURE_THRESHOLD
        ),
        cooldown_seconds=_number(
            "rerank_breaker_cooldown_seconds", DEFAULT_COOLDOWN_SECONDS
        ),
    )
    return HttpReranker(
        base_url=endpoint.base_url,
        model=model,
        headers=endpoint.headers(),
        timeout=_number("rerank_timeout_seconds", DEFAULT_RERANK_TIMEOUT_SECONDS),
        top_n=_integer("rerank_top_n", DEFAULT_RERANK_TOP_N),
        breaker=breaker,
        mode=mode,
        local_threshold=_integer("rerank_local_threshold", DEFAULT_LOCAL_THRESHOLD),
        max_candidates=_integer("rerank_max_candidates", DEFAULT_MAX_CANDIDATES),
        cache_entries=_integer("rerank_cache_entries", DEFAULT_CACHE_ENTRIES),
    )


def rerank_with_fallback(
    query: str,
    documents: Sequence[Any],
    *,
    embeddings: Any,
    reranker: HttpReranker | None = None,
    top_n: int | None = None,
) -> list[Any]:
    """Use the API reranker when it works, otherwise local cosine ordering."""
    if reranker is not None:
        reranked = reranker.rerank(query, documents, top_n=top_n)
        if reranked is not None:
            return reranked
    return local_cosine_rerank(query, documents, embeddings=embeddings)


__all__ = [
    "CircuitBreaker",
    "HttpReranker",
    "local_cosine_rerank",
    "rerank_with_fallback",
    "reranker_from_config",
    "RERANK_MODES",
]
