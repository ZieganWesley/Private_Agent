"""Discover embedding and rerank models without naming a provider.

A route is one ``(model, endpoint)`` pair.  Endpoints come from the configured
OpenAI-compatible registry, so the same discovery works for a local server and
for a remote one; locally discovered runtimes contribute the models their
``/models`` response classified as embeddings or rerankers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from endpoint_legacy import with_v1

ROLE_EMBEDDING = "embedding"
ROLE_RERANK = "rerank"


@dataclass(frozen=True)
class ModelRoute:
    """One selectable model on one serving endpoint."""

    role: str
    label: str
    model: str
    base_url: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.base_url.rstrip("/").lower(), self.model)


def _add(
    routes: list[ModelRoute],
    seen: set[tuple[str, str]],
    *,
    role: str,
    label: str,
    model: Any,
    base_url: Any,
) -> None:
    name = str(model or "").strip()
    url = str(base_url or "").strip()
    if not name or not url:
        return
    route = ModelRoute(role, str(label or "endpoint"), name, with_v1(url))
    if route.key in seen:
        return
    seen.add(route.key)
    routes.append(route)


def embedding_routes(
    runtimes: Iterable[Any] | None = None,
    registry: Any | None = None,
) -> tuple[ModelRoute, ...]:
    """Return every advertised embedding model on every usable endpoint."""
    routes: list[ModelRoute] = []
    seen: set[tuple[str, str]] = set()
    for runtime in runtimes or ():
        for model in getattr(runtime, "embedding_models", ()) or ():
            _add(
                routes,
                seen,
                role=ROLE_EMBEDDING,
                label=getattr(runtime, "name", "endpoint"),
                model=model,
                base_url=getattr(runtime, "base_url", ""),
            )
    if registry is not None:
        for endpoint in registry.for_role(ROLE_EMBEDDING):
            for model in endpoint.models:
                _add(
                    routes,
                    seen,
                    role=ROLE_EMBEDDING,
                    label=endpoint.name,
                    model=model,
                    base_url=endpoint.base_url,
                )
    return tuple(routes)


def rerank_routes(
    runtimes: Iterable[Any] | None = None,
    registry: Any | None = None,
) -> tuple[ModelRoute, ...]:
    """Return every advertised reranker on every usable endpoint."""
    routes: list[ModelRoute] = []
    seen: set[tuple[str, str]] = set()
    for runtime in runtimes or ():
        for record in getattr(runtime, "records", ()) or ():
            if getattr(record, "role", "") != "reranker":
                continue
            _add(
                routes,
                seen,
                role=ROLE_RERANK,
                label=getattr(runtime, "name", "endpoint"),
                model=getattr(record, "model_id", ""),
                base_url=getattr(runtime, "base_url", ""),
            )
    if registry is not None:
        for endpoint in registry.for_role(ROLE_RERANK):
            for model in endpoint.models:
                _add(
                    routes,
                    seen,
                    role=ROLE_RERANK,
                    label=endpoint.name,
                    model=model,
                    base_url=endpoint.base_url,
                )
    return tuple(routes)


__all__ = [
    "ModelRoute",
    "ROLE_EMBEDDING",
    "ROLE_RERANK",
    "embedding_routes",
    "rerank_routes",
]
