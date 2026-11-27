"""Select and remember the RAG embedding and reranker models."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from providers.selection import (
    load_embedding_choice,
    load_reranker_choice,
    save_embedding_choice,
    save_reranker_choice,
)
from rag.embedding_resolver import ModelRoute, embedding_routes, rerank_routes


def _default_index(routes: tuple[ModelRoute, ...], remembered: dict | None) -> int:
    if remembered:
        model = str(remembered.get("model") or "")
        base_url = str(remembered.get("base_url") or "").rstrip("/")
        for index, route in enumerate(routes):
            if route.model == model and route.base_url.rstrip("/") == base_url:
                return index
    return 0


def choose_route(
    routes: tuple[ModelRoute, ...],
    *,
    role: str,
    console: Any,
    interactive: bool,
    remembered: dict | None = None,
) -> ModelRoute | None:
    """Return the only route, a remembered one, or the user's menu choice."""
    if not routes:
        return None
    if len(routes) == 1:
        return routes[0]

    default = _default_index(routes, remembered)
    if not interactive:
        return routes[default]

    title = f"{role.capitalize()} model"
    choices = [
        (str(index), f"{route.label} · {route.model}")
        for index, route in enumerate(routes)
    ]
    chosen = None
    try:
        from .interactive import arrow_select

        chosen = arrow_select(title, choices, default=str(default))
    except Exception:  # noqa: BLE001 - fall back to the numbered prompt
        chosen = None
    if chosen is not None:
        try:
            return routes[int(chosen)]
        except IndexError, TypeError, ValueError:
            return routes[default]

    console.print(f"[bold info][RAG][/bold info] Multiple {role} models are available:")
    for index, route in enumerate(routes, 1):
        console.print(f"  {index}. {route.label} · {route.model}")
    answer = console.input(f"Select {role} model [1-{len(routes)}]: ").strip()
    if not answer:
        return routes[default]
    try:
        return routes[int(answer) - 1]
    except IndexError, TypeError, ValueError:
        return routes[default]


def runtime_for_online_selection(selection: dict | None) -> Any | None:
    """Describe the selected online endpoint as a RAG discovery runtime."""
    if not selection:
        return None
    embedding = tuple(selection.get("embedding_models") or ())
    rerank = tuple(selection.get("rerank_models") or ())
    if not embedding and not rerank:
        return None
    return SimpleNamespace(
        name="online endpoint",
        kind="openai-compatible",
        base_url=str(selection.get("base_url") or ""),
        embedding_models=embedding,
        records=tuple(
            SimpleNamespace(model_id=model, role="reranker") for model in rerank
        ),
    )


def resolve_rag_routes(
    *,
    runtimes: Any,
    registry: Any,
    console: Any,
    interactive: bool,
    embedding_enabled: bool = True,
) -> tuple[ModelRoute | None, ModelRoute | None]:
    """Select both RAG routes, disabling a capability nobody serves."""
    embedding = None
    if embedding_enabled:
        embedding = choose_embedding_route(
            embedding_routes(runtimes, registry),
            console=console,
            interactive=interactive,
        )
    if embedding is None:
        console.print(
            "[warning][RAG][/warning] No embedding model is served by a "
            "configured endpoint; semantic search is disabled."
        )
    rerank = choose_rerank_route(
        rerank_routes(runtimes, registry),
        console=console,
        interactive=interactive,
    )
    return embedding, rerank


def _route_from_choice(role: str, choice: dict | None) -> ModelRoute | None:
    if not choice:
        return None
    provider = str(choice.get("provider") or "")
    if provider == "none":
        return None
    return ModelRoute(
        role,
        provider,
        str(choice.get("model") or ""),
        str(choice.get("base_url") or ""),
    )


def remembered_embedding_route() -> ModelRoute | None:
    """Return the embedding route remembered for this user, if any."""
    return _route_from_choice("embedding", load_embedding_choice())


def remembered_rerank_route() -> ModelRoute | None:
    """Return the reranker route remembered for this user, if any."""
    return _route_from_choice("rerank", load_reranker_choice())


def choose_embedding_route(
    routes: tuple[ModelRoute, ...],
    *,
    console: Any,
    interactive: bool,
) -> ModelRoute | None:
    """Select and remember the embedding route, disabling it when absent."""
    route = choose_route(
        routes,
        role="embedding",
        console=console,
        interactive=interactive,
        remembered=load_embedding_choice(),
    )
    if route is not None:
        save_embedding_choice(
            provider=route.label,
            model=route.model,
            base_url=route.base_url,
        )
    else:
        # Remember that this session found no embedding model so a later
        # reindex does not silently fall back to a configured default.
        save_embedding_choice(provider="none", model="disabled", base_url="")
    return route


def choose_rerank_route(
    routes: tuple[ModelRoute, ...],
    *,
    console: Any,
    interactive: bool,
) -> ModelRoute | None:
    """Select and remember the reranker, disabling reranking when absent."""
    route = choose_route(
        routes,
        role="reranker",
        console=console,
        interactive=interactive,
        remembered=load_reranker_choice(),
    )
    if route is not None:
        save_reranker_choice(
            label=route.label, model=route.model, base_url=route.base_url
        )
    else:
        save_reranker_choice(label="none", model="disabled", base_url="")
    return route


__all__ = [
    "choose_embedding_route",
    "choose_rerank_route",
    "choose_route",
    "remembered_embedding_route",
    "remembered_rerank_route",
    "resolve_rag_routes",
    "runtime_for_online_selection",
]
