"""Provider adapters for local and online model runtimes.

Each provider type implements the :class:`ProviderAdapter` protocol so that
discovery, capability detection, model construction and embeddings no longer
depend on a single runtime (Ollama).
"""

from .adapters import (
    OllamaAdapter,
    OpenAICompatibleAdapter,
    ProviderAdapter,
    ProviderPreset,
    ProviderRecord,
    all_presets,
    build_adapter,
)

__all__ = [
    "ProviderAdapter",
    "ProviderRecord",
    "ProviderPreset",
    "OllamaAdapter",
    "OpenAICompatibleAdapter",
    "build_adapter",
    "all_presets",
]
