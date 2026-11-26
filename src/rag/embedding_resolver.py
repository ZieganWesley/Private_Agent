"""Resolve which embedding provider, model, and endpoint RAG should use (O4)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

VALID_PROVIDERS = ("ollama", "openai-compatible", "auto", "none")


@dataclass(frozen=True)
class EmbeddingChoice:
    provider: str
    model: str
    base_url: str


def _embedding_models(runtime: Any) -> tuple[str, ...]:
    models = getattr(runtime, "embedding_models", ()) or ()
    return tuple(str(model) for model in models if str(model).strip())


def _kind(runtime: Any) -> str:
    return str(getattr(runtime, "kind", "") or "")


def _base_url(runtime: Any, fallback: str) -> str:
    return str(getattr(runtime, "base_url", "") or fallback)


def _provider_for(runtime: Any) -> str:
    return "ollama" if _kind(runtime) == "ollama" else "openai-compatible"


def resolve_embedding_provider(
    runtimes: Iterable[Any] | None,
    provider: str = "ollama",
    model: str = "",
    base_url: str = "",
) -> EmbeddingChoice:
    """Resolve the embedding provider without making a network request.

    Order for ``auto``: the configured model when an endpoint advertises it,
    then the first Ollama embedding model, then the first OpenAI-compatible
    embedding model, then ``none``. Explicit providers keep their previous
    behavior, including the legacy Ollama -> OpenAI-compatible fallback.
    """
    discovered = list(runtimes or ())
    choice = str(provider or "none").strip().lower()
    if choice not in VALID_PROVIDERS:
        choice = "none"
    if choice == "none":
        return EmbeddingChoice("none", model, base_url)
    if choice == "openai-compatible":
        return EmbeddingChoice("openai-compatible", model, base_url)

    ollama_runtimes = [runtime for runtime in discovered if _kind(runtime) == "ollama"]
    openai_runtimes = [
        runtime for runtime in discovered if _kind(runtime) == "openai-compatible"
    ]
    ollama_embedders = [
        runtime for runtime in ollama_runtimes if _embedding_models(runtime)
    ]
    openai_embedders = [
        runtime for runtime in openai_runtimes if _embedding_models(runtime)
    ]

    if choice == "auto":
        if model:
            for runtime in ollama_embedders + openai_embedders:
                if model in _embedding_models(runtime):
                    return EmbeddingChoice(
                        _provider_for(runtime), model, _base_url(runtime, base_url)
                    )
        for runtime in ollama_embedders + openai_embedders:
            return EmbeddingChoice(
                _provider_for(runtime),
                _embedding_models(runtime)[0],
                _base_url(runtime, base_url),
            )
        return EmbeddingChoice("none", model, base_url)

    # Explicit Ollama: use the discovered runtime's own endpoint (G6), then
    # keep the established fallback to any OpenAI-compatible embedder.
    if ollama_embedders:
        runtime = ollama_embedders[0]
        return EmbeddingChoice(
            "ollama", _embedding_models(runtime)[0], _base_url(runtime, base_url)
        )
    if ollama_runtimes:
        return EmbeddingChoice("ollama", model, _base_url(ollama_runtimes[0], base_url))
    if openai_embedders:
        runtime = openai_embedders[0]
        return EmbeddingChoice(
            "openai-compatible",
            _embedding_models(runtime)[0],
            _base_url(runtime, base_url),
        )
    if discovered:
        return EmbeddingChoice("none", model, base_url)
    return EmbeddingChoice("ollama", model, base_url)


__all__ = ["EmbeddingChoice", "VALID_PROVIDERS", "resolve_embedding_provider"]
