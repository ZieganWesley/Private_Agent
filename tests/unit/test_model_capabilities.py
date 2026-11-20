from types import SimpleNamespace

import httpx
import pytest

from private_agent.agent.model_capabilities import (
    capabilities_from_family,
    capabilities_from_llamacpp_props,
    capabilities_from_model_entry,
    detect_openai_compatible_capabilities,
    merge_capabilities,
)

LLAMACPP_URL = "http://127.0.0.1:8080/v1"
LLAMACPP_MODEL = "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Shapes follow the llama.cpp server README (GET /props and GET /v1/models).
LLAMACPP_PROPS = {
    "default_generation_settings": {"id": 0, "n_ctx": 65536},
    "total_slots": 1,
    "model_path": LLAMACPP_MODEL,
    "chat_template": "...",
    "chat_template_caps": {
        "supports_tools": True,
        "supports_tool_calls": True,
        "supports_system_role": True,
        "supports_parallel_tool_calls": True,
        "supports_preserve_reasoning": False,
        "supports_reasoning_effort": False,
    },
    "modalities": {"vision": False},
    "build_info": "b1234-abcdef",
}
LLAMACPP_MODELS = {
    "object": "list",
    "data": [
        {
            "id": LLAMACPP_MODEL,
            "object": "model",
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
            "owned_by": "llamacpp",
            "meta": {"n_ctx_train": 131072, "n_params": 8030261312},
        }
    ],
}


class FakeClient:
    """Serves canned JSON per URL and records every request."""

    def __init__(self, responses):
        self.responses = responses
        self.requested = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get(self, url):
        self.requested.append(url)
        if url not in self.responses:
            raise ConnectionError(f"no route for {url}")
        body = self.responses[url]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)


def _detect(responses, model=LLAMACPP_MODEL, base_url=LLAMACPP_URL, **kwargs):
    client = FakeClient(responses)
    result = detect_openai_compatible_capabilities(
        model,
        base_url=base_url,
        client_factory=lambda **_kwargs: client,
        **kwargs,
    )
    return result, client


def test_llamacpp_server_reports_tools_modalities_and_running_context():
    capabilities, _client = _detect(
        {
            "http://127.0.0.1:8080/props": LLAMACPP_PROPS,
            f"{LLAMACPP_URL}/models": LLAMACPP_MODELS,
        }
    )
    assert capabilities == {
        "tools": True,
        "function_calls": True,
        "structured_output": None,
        "thinking": None,
        "vision": False,
        "audio": False,
        # The running n_ctx from /props wins over the training context.
        "context_window": 65536,
    }


def test_unreachable_openai_compatible_endpoint_is_treated_as_unknown():
    class UnreachableClient(FakeClient):
        def get(self, url):
            raise httpx.ConnectError("connection refused")

    result = detect_openai_compatible_capabilities(
        "local-chat-model",
        base_url="http://127.0.0.1:1234/v1",
        client_factory=lambda **_kwargs: UnreachableClient({}),
    )

    assert all(value is None for value in result.values())


def test_llamacpp_router_props_are_requested_for_the_selected_model():
    model = "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"
    capabilities, client = _detect(
        {
            "http://127.0.0.1:8080/props?model=ggml-org%2Fgemma-3-4b-it-GGUF%3AQ4_K_M": {
                **LLAMACPP_PROPS,
                "modalities": {"vision": True, "audio": False},
            },
        },
        model=model,
    )
    assert client.requested[0].startswith("http://127.0.0.1:8080/props?model=")
    assert capabilities["vision"] is True


def test_llamacpp_template_without_tool_support_disables_tools():
    props = {
        **LLAMACPP_PROPS,
        "chat_template_caps": {"supports_tools": False, "supports_tool_calls": False},
    }
    assert capabilities_from_llamacpp_props(props)["tools"] is False


def test_llamacpp_reasoning_template_reports_thinking():
    props = {
        **LLAMACPP_PROPS,
        "chat_template_caps": {
            "supports_tool_calls": True,
            "supports_reasoning_effort": True,
            "reasoning_effort_levels": ["low", "medium", "high"],
        },
    }
    assert capabilities_from_llamacpp_props(props)["thinking"] is True


def test_llamacpp_empty_template_caps_leave_tools_unknown():
    props = {**LLAMACPP_PROPS, "chat_template_caps": {}}
    assert capabilities_from_llamacpp_props(props)["tools"] is None


def test_llamacpp_listing_alone_gives_modalities_and_training_context():
    entry = {
        "id": "gemma",
        "architecture": {"input_modalities": ["text", "image", "audio"]},
        "meta": {"n_ctx_train": 8192},
    }
    capabilities = capabilities_from_model_entry(entry)
    assert capabilities["vision"] is True
    assert capabilities["audio"] is True
    assert capabilities["context_window"] == 8192
    assert capabilities["tools"] is None


def test_lmstudio_native_listing_is_used_on_loopback():
    capabilities, _client = _detect(
        {
            "http://127.0.0.1:1234/api/v0/models": {
                "data": [
                    {
                        "id": "qwen2-vl-7b-instruct",
                        "type": "vlm",
                        "capabilities": ["tool_use"],
                        "max_context_length": 32768,
                        "loaded_context_length": 8192,
                    }
                ]
            },
            "http://127.0.0.1:1234/v1/models": {
                "data": [{"id": "qwen2-vl-7b-instruct"}]
            },
        },
        model="qwen2-vl-7b-instruct",
        base_url="http://127.0.0.1:1234/v1",
    )
    assert capabilities["tools"] is True
    assert capabilities["vision"] is True
    assert capabilities["context_window"] == 8192


def test_lmstudio_v1_capability_object_is_understood():
    entry = {
        "key": "google/gemma-3-12b",
        "type": "llm",
        "capabilities": {"vision": False, "trained_for_tool_use": True},
        "max_context_length": 131072,
    }
    capabilities = capabilities_from_model_entry(entry)
    assert capabilities["tools"] is True
    assert capabilities["vision"] is False
    assert capabilities["context_window"] == 131072


def test_vllm_max_model_len_becomes_context_window():
    entry = {"id": "Qwen/Qwen2.5-7B-Instruct", "max_model_len": 32768}
    assert capabilities_from_model_entry(entry)["context_window"] == 32768


def test_openrouter_style_supported_parameters():
    entry = {
        "id": "vendor/model",
        "context_length": 200000,
        "architecture": {"input_modalities": ["text", "image"]},
        "supported_parameters": ["tools", "tool_choice", "reasoning", "max_tokens"],
    }
    assert capabilities_from_model_entry(entry) == {
        "tools": True,
        "function_calls": True,
        "structured_output": False,
        "thinking": True,
        "vision": True,
        "audio": False,
        "context_window": 200000,
    }


def test_remote_endpoints_only_read_the_standard_listing():
    base_url = "https://models.example.test/v1"
    capabilities, client = _detect(
        {f"{base_url}/models": {"data": [{"id": "m", "max_model_len": 4096}]}},
        model="m",
        base_url=base_url,
    )
    assert client.requested == [f"{base_url}/models"]
    assert capabilities["context_window"] == 4096


def test_unreachable_server_reports_everything_unknown():
    capabilities, _client = _detect({})
    assert all(value is None for value in capabilities.values())


def test_client_failure_does_not_raise():
    def broken_factory(**_kwargs):
        raise OSError("refused")

    capabilities = detect_openai_compatible_capabilities(
        "m", base_url=LLAMACPP_URL, client_factory=broken_factory
    )
    assert all(value is None for value in capabilities.values())


def test_config_overrides_win_over_detected_values():
    capabilities, _client = _detect(
        {"http://127.0.0.1:8080/props": LLAMACPP_PROPS},
        overrides={"tools": False, "vision": True, "context_window": 4096},
    )
    assert capabilities["tools"] is False
    assert capabilities["function_calls"] is False
    assert capabilities["vision"] is True
    assert capabilities["context_window"] == 4096


@pytest.mark.parametrize(
    "overrides",
    [{"tools": "yes"}, {"context_window": -1}, {"context_window": True}, None],
)
def test_invalid_overrides_are_ignored(overrides):
    capabilities, _client = _detect(
        {"http://127.0.0.1:8080/props": LLAMACPP_PROPS}, overrides=overrides
    )
    assert capabilities["tools"] is True
    assert capabilities["context_window"] == 65536


def test_detection_is_cached_per_endpoint_and_model():
    cache = {}
    responses = {"http://127.0.0.1:8080/props": LLAMACPP_PROPS}
    first, first_client = _detect(responses, capability_cache=cache)
    second, second_client = _detect(responses, capability_cache=cache)
    assert first == second
    assert first_client.requested
    assert second_client.requested == []


def test_merge_keeps_first_known_value_and_mirrors_function_calls():
    merged = merge_capabilities(
        {"tools": None, "vision": True},
        {"tools": True, "vision": False, "context_window": 2048},
    )
    assert merged["tools"] is True
    assert merged["function_calls"] is True
    assert merged["vision"] is True
    assert merged["context_window"] == 2048


def test_family_table_fills_capabilities_for_known_models():
    qwen = capabilities_from_family("qwen3:8b")
    assert qwen["tools"] is True
    assert qwen["thinking"] is True
    assert qwen["vision"] is False

    assert capabilities_from_family("llava:13b")["vision"] is True
    assert capabilities_from_family("gemma3:4b")["tools"] is False
    assert capabilities_from_family("deepseek-r1:7b")["thinking"] is True


def test_family_table_picks_the_most_specific_match():
    coder = capabilities_from_family("qwen2.5-coder:7b")
    assert coder["thinking"] is False
    assert coder["tools"] is True

    vl = capabilities_from_family("qwen2.5-vl:7b")
    assert vl["vision"] is True


def test_family_table_leaves_unknown_models_unknown():
    assert capabilities_from_family("totally-unknown-model") == {
        "tools": None,
        "function_calls": None,
        "structured_output": None,
        "thinking": None,
        "vision": None,
        "audio": None,
        "context_window": None,
    }
    assert capabilities_from_family("")["tools"] is None


def test_server_report_wins_over_the_family_table():
    props = {
        "default_generation_settings": {"id": 0, "n_ctx": 4096},
        "chat_template_caps": {"supports_tools": False},
        "modalities": {"vision": True},
    }
    detected, _client = _detect(
        {"http://127.0.0.1:8080/props": props}, model="qwen3:8b"
    )

    assert detected["tools"] is False  # server says no tools, family table says yes
    assert detected["context_window"] == 4096
    assert detected["vision"] is True
