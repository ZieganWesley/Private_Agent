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
from .selection import (
    IdleProvider,
    installed_but_idle,
    load_last_provider,
    match_runtime,
    pick_default_model,
    requested_model,
    requested_provider,
    save_last_provider,
)

__all__ = [
    "ProviderAdapter",
    "ProviderRecord",
    "ProviderPreset",
    "OllamaAdapter",
    "OpenAICompatibleAdapter",
    "build_adapter",
    "all_presets",
    "IdleProvider",
    "installed_but_idle",
    "load_last_provider",
    "match_runtime",
    "pick_default_model",
    "requested_model",
    "requested_provider",
    "save_last_provider",
]
