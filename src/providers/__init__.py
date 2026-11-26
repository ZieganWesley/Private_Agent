"""Provider adapters for configured OpenAI-compatible endpoints.

Every endpoint speaks the same API, so discovery, capability detection, model
construction and embeddings no longer depend on a named runtime vendor.
"""

from .adapters import (
    OpenAICompatibleAdapter,
    ProviderAdapter,
    ProviderRecord,
)
from .endpoints import (
    EndpointConfig,
    EndpointRegistry,
    clamp_limit,
    normalize_base_url,
    resolved_limits,
)
from .native import NativeClient, NativeProfile, detect_native
from .selection import (
    load_last_provider,
    match_runtime,
    pick_default_model,
    requested_model,
    requested_provider,
    save_last_provider,
)
from .transports import CompletionsClient, ResponsesClient, TransportResult

__all__ = [
    "NativeClient",
    "NativeProfile",
    "detect_native",
    "CompletionsClient",
    "ResponsesClient",
    "TransportResult",
    "EndpointConfig",
    "EndpointRegistry",
    "clamp_limit",
    "normalize_base_url",
    "resolved_limits",
    "ProviderAdapter",
    "ProviderRecord",
    "OpenAICompatibleAdapter",
    "load_last_provider",
    "match_runtime",
    "pick_default_model",
    "requested_model",
    "requested_provider",
    "save_last_provider",
]
