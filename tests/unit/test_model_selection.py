"""Unit tests for the extracted model-selection helpers."""

from unittest.mock import MagicMock

from agent import model_selection
from agent.local_runtimes import LocalRuntime


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


def test_selected_local_chat_model_uses_openai_compatible_api(monkeypatch):
    compatible_model = object()
    make_compatible = MagicMock(return_value=compatible_model)
    monkeypatch.setattr(
        model_selection, "_make_local_openai_compatible_chat_model", make_compatible
    )
    compatible_runtime = LocalRuntime(
        "LM Studio", "openai-compatible", "http://127.0.0.1:1234/v1", ("m",)
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
