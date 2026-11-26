"""Optional HTTP reranking with a permanent local cosine fallback.

``/v1/rerank`` is not part of the core OpenAI-compatible surface, so every
failure is treated as "this endpoint does not rerank" and the caller keeps the
existing local embedding similarity order.  A small circuit breaker prevents a
broken endpoint from adding a timeout to every retrieval.
"""

from __future__ import annotations

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
MAX_CACHE_ENTRIES = 128


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
        while len(self._cache) > MAX_CACHE_ENTRIES:
            self._cache.popitem(last=False)

    def rerank(
        self, query: str, documents: Sequence[Any], *, top_n: int | None = None
    ) -> list[Any] | None:
        """Return documents in API order, or ``None`` when unavailable."""
        if not self.enabled or not documents or not self.breaker.allow():
            return None
        limit = min(len(documents), max(1, int(top_n or self.top_n)))
        cache_key = self._key(query, documents, limit)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return [documents[index] for index in cached]

        payload: dict[str, Any] = {
            "query": query,
            "documents": [_document_text(document) for document in documents],
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
                score = _score(
                    item.get("relevance_score", item.get("score"))
                )
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(documents)
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
            return [documents[index] for index in order]
        except (httpx.HTTPError, ValueError, TypeError, OSError):
            self.breaker.record_failure()
            return None
        finally:
            if client is not None and hasattr(client, "close"):
                try:
                    client.close()
                except (httpx.HTTPError, OSError):
                    pass


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
    except (AttributeError, ValueError, RuntimeError, TypeError, OSError):
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


def reranker_from_config(config: Any) -> HttpReranker | None:
    """Build a reranker for the endpoint that advertises the rerank role."""
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
    model = rag.get("rerank_model")
    if not isinstance(model, str) or not model.strip():
        model = endpoint.models[0] if endpoint.models else None
    timeout = rag.get("rerank_timeout_seconds", DEFAULT_RERANK_TIMEOUT_SECONDS)
    top_n = rag.get("rerank_top_n", DEFAULT_RERANK_TOP_N)
    enabled = rag.get("rerank_enabled", True)
    return HttpReranker(
        base_url=endpoint.base_url,
        model=model,
        headers=endpoint.headers(),
        timeout=timeout if isinstance(timeout, (int, float)) else DEFAULT_RERANK_TIMEOUT_SECONDS,
        top_n=top_n if isinstance(top_n, int) else DEFAULT_RERANK_TOP_N,
        enabled=bool(enabled),
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
]
