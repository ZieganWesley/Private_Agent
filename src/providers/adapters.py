"""Provider adapter implementations.

This module intentionally avoids importing ``ollama`` or ``langchain_ollama`` at
module import time so that online-only sessions never pay for them (I1e).
"""

from __future__ import annotations

import logging
import os
import shutil
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


@dataclass(frozen=True)
class ProviderPreset:
    """A known local provider speaking an OpenAI-compatible API."""

    name: str
    kind: str  # "ollama" or "openai-compatible"
    default_ports: tuple[int, ...]
    install_check: Callable[[], bool]
    start_hint: str
    base_url_template: str = "http://127.0.0.1:{port}/v1"


@runtime_checkable
class ProviderAdapter(Protocol):
    """Common interface implemented by every provider adapter."""

    name: str
    kind: str

    def detect_installed(self) -> bool: ...

    def detect_running(
        self, *, client_factory: Callable[..., Any], timeout: float
    ) -> Any: ...

    def start(self, *, consent: bool) -> bool: ...

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

    def hardware_status(self, *, base_url: str, **kwargs: Any) -> str: ...


def _which(name: str) -> bool:
    return shutil.which(name) is not None


def _ollama_installed() -> bool:
    return _which("ollama")


def _lmstudio_installed() -> bool:
    # LM Studio ships a CLI named lms on some platforms; otherwise it is an app.
    return _which("lms") or os.path.isdir("/Applications/LM Studio.app")


def _llamacpp_installed() -> bool:
    return _which("llama-server") or _which("llama-cli") or _which("llama")


def _vllm_installed() -> bool:
    return _which("vllm")


def _jan_installed() -> bool:
    return os.path.isdir("/Applications/Jan.app") or _which("jan")


def _koboldcpp_installed() -> bool:
    return _which("koboldcpp") or _which("koboldcpp-linux-x86_64")


def _gpt4all_installed() -> bool:
    return os.path.isdir("/Applications/GPT4All.app") or _which("gpt4all")


def _localai_installed() -> bool:
    return _which("local-ai") or _which("localai")


def _llamafile_installed() -> bool:
    return _which("llamafile")


def _mlxlm_installed() -> bool:
    return _which("mlx_lm.server") or _which("mlx-lm")


def _textgenwebui_installed() -> bool:
    return os.path.isdir(os.path.expanduser("~/text-generation-webui"))


_PRESET_TABLE: list[tuple[str, str, tuple[int, ...], Callable[[], bool], str]] = [
    ("Ollama", "ollama", (11434,), _ollama_installed, "Run `ollama serve`."),
    (
        "LM Studio",
        "openai-compatible",
        (1234,),
        _lmstudio_installed,
        "Start LM Studio and enable the local server.",
    ),
    (
        "llama.cpp",
        "openai-compatible",
        (8080,),
        _llamacpp_installed,
        "Run `llama-server -m <model>` (default port 8080).",
    ),
    (
        "vLLM",
        "openai-compatible",
        (8000,),
        _vllm_installed,
        "Run `vllm serve <model>` (default port 8000).",
    ),
    (
        "Jan",
        "openai-compatible",
        (1337,),
        _jan_installed,
        "Start Jan and enable its local API server.",
    ),
    (
        "KoboldCpp",
        "openai-compatible",
        (5001,),
        _koboldcpp_installed,
        "Run KoboldCpp with `--api` (default port 5001).",
    ),
    (
        "GPT4All",
        "openai-compatible",
        (4891,),
        _gpt4all_installed,
        "Start GPT4All and enable the local server.",
    ),
    (
        "LocalAI",
        "openai-compatible",
        (8080,),
        _localai_installed,
        "Run LocalAI (default port 8080).",
    ),
    (
        "llamafile",
        "openai-compatible",
        (8080,),
        _llamafile_installed,
        "Run the llamafile executable (default port 8080).",
    ),
    (
        "MLX-LM",
        "openai-compatible",
        (8080,),
        _mlxlm_installed,
        "Run `mlx_lm.server` (default port 8080).",
    ),
    (
        "text-generation-webui",
        "openai-compatible",
        (5000,),
        _textgenwebui_installed,
        "Start text-generation-webui with `--api` (default port 5000).",
    ),
]


def all_presets() -> list[ProviderPreset]:
    presets: list[ProviderPreset] = []
    for name, kind, ports, check, hint in _PRESET_TABLE:
        presets.append(
            ProviderPreset(
                name=name,
                kind=kind,
                default_ports=ports,
                install_check=check,
                start_hint=hint,
            )
        )
    return presets


class OllamaAdapter:
    name = "Ollama"
    kind = "ollama"

    def detect_installed(self) -> bool:
        return _ollama_installed()

    def detect_running(
        self, *, client_factory: Callable[..., Any], timeout: float
    ) -> Any:
        # Lazy import to keep ollama optional (I1e).
        from agent import local_runtimes

        try:
            runtimes = local_runtimes.discover_local_runtimes(
                "http://localhost:11434",
                [],
                client_factory=client_factory,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            log_event(
                RUN_LOGGER,
                "provider.adapter_detect_failed",
                level=logging.DEBUG,
                error=str(exc),
            )
            return None
        for runtime in runtimes:
            if runtime.kind == "ollama":
                return runtime
        return None

    def start(self, *, consent: bool) -> bool:
        if not consent:
            return False
        import subprocess

        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (OSError, FileNotFoundError) as exc:
            log_event(
                RUN_LOGGER,
                "provider.adapter_start_failed",
                level=logging.WARNING,
                error=str(exc),
            )
            return False

    def list_models(self, runtime: Any) -> tuple[str, ...]:
        if runtime is None:
            return ()
        return tuple(getattr(runtime, "models", ()) or ())

    def capabilities(self, model, *, base_url, client_factory, timeout):
        # Delegate to model_capabilities when available; keep unknown on failure.
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
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, base_url=base_url, **(options or {}))

    def embeddings(self, model, *, base_url):
        try:
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=model, base_url=base_url)
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            log_event(
                RUN_LOGGER,
                "provider.embeddings_failed",
                level=logging.DEBUG,
                error=str(exc),
            )
            return None

    def hardware_status(self, *, base_url, mode="auto", client_factory=None):
        """Return the human-readable hardware report for this Ollama endpoint."""
        try:
            from hardware.status import (
                format_hardware_status,
                inspect_ollama_hardware,
            )

            info = inspect_ollama_hardware(
                base_url, mode, client_factory=client_factory
            )
            return format_hardware_status(info)
        except Exception as exc:  # noqa: BLE001 - hardware probes are best effort
            log_event(
                RUN_LOGGER,
                "provider.hardware_status_failed",
                level=logging.DEBUG,
                provider=self.name,
                error_type=type(exc).__name__,
            )
            return "Hardware status is unavailable for this runtime."


class OpenAICompatibleAdapter:
    def __init__(
        self, preset: ProviderPreset | None = None, *, name: str = "OpenAI-compatible"
    ):
        self._preset = preset
        self.name = preset.name if preset else name
        self.kind = "openai-compatible"

    def detect_installed(self) -> bool:
        if self._preset is not None:
            return bool(self._preset.install_check())
        return True

    def detect_running(self, *, client_factory, timeout):
        from agent import local_runtimes

        base_urls = []
        if self._preset is not None:
            for port in self._preset.default_ports:
                base_urls.append(self._preset.base_url_template.format(port=port))
        if not base_urls:
            return None
        for base_url in base_urls:
            try:
                runtimes = local_runtimes.discover_local_runtimes(
                    "http://localhost:11434",
                    [{"name": self.name, "base_url": base_url}],
                    client_factory=client_factory,
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001 - optional dependency boundary
                log_event(
                    RUN_LOGGER,
                    "provider.adapter_detect_failed",
                    level=logging.DEBUG,
                    error=str(exc),
                )
                runtimes = []
            for runtime in runtimes:
                if runtime.kind == "openai-compatible" and runtime.base_url.rstrip(
                    "/"
                ) == base_url.rstrip("/"):
                    return runtime
        return None

    def start(self, *, consent: bool) -> bool:
        if not consent:
            return False
        # Generic adapter cannot start arbitrary servers; rely on preset hint.
        return False

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

    def hardware_status(self, *, base_url):
        return "no standard API"


def build_adapter(name_or_preset: str | ProviderPreset) -> ProviderAdapter:
    if isinstance(name_or_preset, ProviderPreset):
        preset = name_or_preset
    else:
        preset = next(
            (p for p in all_presets() if p.name.lower() == str(name_or_preset).lower()),
            None,
        )
    if preset is None:
        return OpenAICompatibleAdapter(name=str(name_or_preset))
    if preset.kind == "ollama":
        return OllamaAdapter()
    return OpenAICompatibleAdapter(preset=preset)
