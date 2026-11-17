"""Unit tests for the extracted model-selection helpers."""

import types
from unittest.mock import MagicMock

from private_agent.agent import model_selection
from private_agent.agent.local_runtimes import LocalRuntime


def test_effective_context_window_prefers_declared_value_within_limit(monkeypatch):
    monkeypatch.setattr(model_selection, "MAX_CONTEXT_TOKENS", 8192)

    assert model_selection._effective_context_window({"context_window": 4096}) == 4096
    assert model_selection._effective_context_window({"context_window": 32000}) == 8192
    assert model_selection._effective_context_window({"context_window": 0}) == 8192
    assert model_selection._effective_context_window({"context_window": True}) == 8192
    assert model_selection._effective_context_window({}) == 8192


def test_local_endpoint_capability_overrides_match_configured_endpoint(monkeypatch):
    monkeypatch.setattr(
        model_selection,
        "APP_CONFIG",
        {
            "models": {
                "local_openai_compatible_endpoints": [
                    {
                        "base_url": "http://127.0.0.1:1234/v1/",
                        "capabilities": {"tools": True, "vision": False},
                    },
                    {"base_url": "http://127.0.0.1:8080/v1"},
                ]
            }
        },
    )

    assert model_selection._local_endpoint_capability_overrides(
        "http://127.0.0.1:1234/v1"
    ) == {"tools": True, "vision": False}
    assert (
        model_selection._local_endpoint_capability_overrides("http://127.0.0.1:9999/v1")
        is None
    )


def test_local_endpoint_capability_overrides_tolerates_missing_config(monkeypatch):
    monkeypatch.setattr(model_selection, "APP_CONFIG", {})
    assert (
        model_selection._local_endpoint_capability_overrides("http://127.0.0.1:1234/v1")
        is None
    )


def test_selected_local_chat_model_dispatches_on_runtime_kind(monkeypatch):
    ollama_model = object()
    compatible_model = object()
    make_chat_model = MagicMock(return_value=ollama_model)
    make_compatible = MagicMock(return_value=compatible_model)
    monkeypatch.setattr(model_selection, "_make_chat_model", make_chat_model)
    monkeypatch.setattr(
        model_selection, "_make_local_openai_compatible_chat_model", make_compatible
    )
    ollama_runtime = LocalRuntime("Ollama", "ollama", "http://127.0.0.1:11434", ("m",))
    compatible_runtime = LocalRuntime(
        "LM Studio", "openai-compatible", "http://127.0.0.1:1234/v1", ("m",)
    )

    assert (
        model_selection._make_selected_local_chat_model(
            ollama_runtime,
            "m",
            thinking_enabled=True,
            thinking_effort="high",
            capabilities={"thinking": True, "context_window": 4096},
        )
        is ollama_model
    )
    make_chat_model.assert_called_once_with(
        "m",
        thinking_enabled=True,
        thinking_effort="high",
        supports_thinking=True,
        context_window=4096,
    )

    assert (
        model_selection._make_selected_local_chat_model(
            compatible_runtime,
            "m",
            thinking_enabled=False,
            thinking_effort="medium",
            capabilities={},
        )
        is compatible_model
    )
    make_compatible.assert_called_once_with("http://127.0.0.1:1234/v1", "m")


def test_inspect_model_capabilities_uses_shared_cache_and_client(monkeypatch):
    seen = {}

    def fake_inspect(model_name, **kwargs):
        seen["model_name"] = model_name
        seen["kwargs"] = kwargs
        return {"tools": True}

    monkeypatch.setattr(
        model_selection, "_inspect_local_model_capabilities", fake_inspect
    )

    result = model_selection.inspect_model_capabilities("local-model")

    assert result == {"tools": True}
    assert seen["model_name"] == "local-model"
    assert (
        seen["kwargs"]["capability_cache"] is model_selection._MODEL_CAPABILITIES_CACHE
    )
    assert seen["kwargs"]["close_client"] is model_selection._close_ollama_client


def test_close_ollama_client_swallows_close_failures():
    client = MagicMock()
    client.close.side_effect = RuntimeError("already closed")

    model_selection._close_ollama_client(client)

    client.close.assert_called_once()


def test_model_capability_overrides_for_compatible_runtimes(monkeypatch):
    monkeypatch.setattr(model_selection, "APP_CONFIG", {})
    detect = MagicMock(return_value={"tools": True})
    monkeypatch.setattr(
        model_selection, "detect_openai_compatible_capabilities", detect
    )
    runtime = types.SimpleNamespace(base_url="http://127.0.0.1:1234/v1")

    capabilities = model_selection.inspect_openai_compatible_model_capabilities(
        runtime, "local-model"
    )

    assert capabilities == {"tools": True}
    detect.assert_called_once()
    assert detect.call_args.kwargs["base_url"] == "http://127.0.0.1:1234/v1"
    assert detect.call_args.kwargs["overrides"] is None
