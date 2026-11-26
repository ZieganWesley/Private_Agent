"""Provider adapter implementations.

Private Agent talks to one kind of API: the OpenAI-compatible surface exposed by
configured endpoints.  There is no app-install probing and no per-application
preset table, so this module only keeps the small adapter used to build chat and
embedding models for an endpoint.

Optional dependencies stay behind lazy imports so that online-only sessions never
pay for them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from utils.run_logging import RUN_LOGGER, log_event


@dataclass(frozen=True)
class ProviderRecord:
    """The provider/model a session or task was run with."""

    adapter_type: str
    display_name: str
    base_url: str
    model: str
    needs_tools: bool = False
    needs_vision: bool = False
    context_window: int | None = None


@runtime_checkable
class ProviderAdapter(Protocol):
    """Common interface implemented by an OpenAI-compatible endpoint adapter."""

    name: str
    kind: str

    def list_models(self, runtime: Any) -> tuple[str, ...]: ...

    def capabilities(
        self,
        model: str,
        *,
        base_url: str,
        client_factory: Callable[..., Any],
        timeout: float,
    ) -> dict[str, Any]: ...

    def chat_model(
        self, model: str, *, base_url: str, options: Mapping[str, Any] | None = None
    ) -> Any: ...

    def embeddings(self, model: str, *, base_url: str) -> Any | None: ...


class OpenAICompatibleAdapter:
    """Build models for any endpoint that speaks the OpenAI-compatible API."""

    kind = "openai-compatible"

    def __init__(self, *, name: str = "OpenAI-compatible"):
        self.name = name

    def list_models(self, runtime) -> tuple[str, ...]:
        if runtime is None:
            return ()
        return tuple(getattr(runtime, "models", ()) or ())

    def capabilities(self, model, *, base_url, client_factory, timeout):
        try:
            from agent import model_capabilities

            return model_capabilities.detect_openai_compatible_capabilities(
                model,
                base_url=base_url,
                client_factory=client_factory,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            log_event(
                RUN_LOGGER,
                "provider.capabilities_failed",
                level=logging.DEBUG,
                error=str(exc),
            )
            return {
                "tools": None,
                "vision": None,
                "thinking": None,
                "context_window": None,
            }

    def chat_model(self, model, *, base_url, options=None):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model, base_url=base_url, api_key="not-needed", **(options or {})
        )

    def embeddings(self, model, *, base_url):
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=model, base_url=base_url, api_key="not-needed"
            )
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            log_event(
                RUN_LOGGER,
                "provider.embeddings_failed",
                level=logging.DEBUG,
                error=str(exc),
            )
            return None
