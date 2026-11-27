"""Remember the RAG embedding and reranker routes across launches."""

from __future__ import annotations

from typing import Any

from providers.selection import (
    load_embedding_choice,
    load_reranker_choice,
    save_embedding_choice,
    save_reranker_choice,
)
from rag.embedding_resolver import EmbeddingChoice, stored_embedding_choice
from rag.embedding_resolver import (
    resolve_embedding_provider as _resolve_embedding_provider,
)


def resolve_embedding_route(
    *,
    configured_provider: str,
    configured_model: str,
    configured_base_url: str,
    runtimes: Any,
    console: Any,
) -> EmbeddingChoice:
    """Resolve the embedding route, reusing the remembered one when possible."""
    choice = None
    if configured_provider == "auto":
        choice = stored_embedding_choice(load_embedding_choice(), runtimes)
        if choice is not None:
            console.print(
                "[info][RAG][/info] Using the remembered embedding route "
                f"({choice.provider}:{choice.model})."
            )
    if choice is None:
        choice = _resolve_embedding_provider(
            runtimes,
            configured_provider,
            configured_model,
            configured_base_url,
        )
    if configured_provider == "ollama" and choice.provider == "openai-compatible":
        console.print(
            "[info][RAG][/info] Ollama is unavailable; using the local "
            f"OpenAI-compatible endpoint for embeddings ({choice.model})."
        )
    if choice.provider == "none" and runtimes and configured_provider != "none":
        console.print(
            "[warning][RAG][/warning] No embedding-capable runtime is "
            "available; continuing without local RAG."
        )
    if choice.provider != "none" and choice.model:
        save_embedding_choice(
            provider=choice.provider,
            model=choice.model,
            base_url=choice.base_url,
        )
    return choice


def remembered_reranker_model() -> str | None:
    """Return the model from the last successful reranker resolution."""
    return (load_reranker_choice() or {}).get("model") or None


def save_reranker_route(model: str, base_url: str) -> None:
    """Remember the resolved reranker endpoint and model."""
    save_reranker_choice(model=model, base_url=base_url)


__all__ = [
    "remembered_reranker_model",
    "resolve_embedding_route",
    "save_reranker_route",
]
