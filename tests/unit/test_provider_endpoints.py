"""Unit tests for the OpenAI-compatible endpoint registry."""

from providers.endpoints import (
    EndpointConfig,
    EndpointRegistry,
    clamp_limit,
    is_loopback_url,
    normalize_base_url,
    resolved_limits,
    with_v1,
)


def test_normalize_base_url_accepts_http_and_strips_slash():
    assert (
        normalize_base_url(" http://127.0.0.1:8080/v1/ ") == "http://127.0.0.1:8080/v1"
    )


def test_normalize_base_url_rejects_credentials_queries_and_bad_schemes():
    assert normalize_base_url("ftp://example.test/v1") is None
    assert normalize_base_url("http://user:pass@example.test/v1") is None
    assert normalize_base_url("http://example.test/v1?x=1") is None
    assert normalize_base_url("") is None


def test_is_loopback_url():
    assert is_loopback_url("http://127.0.0.1:8080/v1")
    assert is_loopback_url("http://localhost:8080/v1")
    assert is_loopback_url("http://[::1]:8080/v1")
    assert not is_loopback_url("http://192.168.1.20:8080/v1")


def test_with_v1_only_appends_when_missing():
    assert with_v1("http://127.0.0.1:11434") == "http://127.0.0.1:11434/v1"
    assert with_v1("http://127.0.0.1:8080/v1/") == "http://127.0.0.1:8080/v1"


def test_registry_reads_roles_and_limits():
    registry = EndpointRegistry.from_config(
        {
            "models": {
                "endpoints": [
                    {
                        "name": "main",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "roles": ["chat", "rerank"],
                        "max_context_tokens": 8192,
                        "max_output_tokens": 1024,
                        "request_timeout_seconds": 12,
                    }
                ]
            }
        }
    )

    assert len(registry.endpoints) == 1
    endpoint = registry.endpoints[0]
    assert endpoint.name == "main"
    assert endpoint.roles == ("chat", "rerank")
    assert endpoint.max_context_tokens == 8192
    assert endpoint.max_output_tokens == 1024
    assert endpoint.request_timeout_seconds == 12
    assert endpoint.chat_url() == "http://127.0.0.1:8080/v1/chat/completions"


def test_registry_migrates_legacy_endpoints_with_all_roles():
    registry = EndpointRegistry.from_config(
        {
            "models": {
                "local_openai_compatible_endpoints": [
                    {"name": "LM Studio", "base_url": "http://127.0.0.1:1234/v1"}
                ]
            }
        }
    )

    endpoint = registry.endpoints[0]
    assert endpoint.source == "legacy.local_openai_compatible_endpoints"
    assert set(endpoint.roles) == {"chat", "embedding", "rerank", "native"}


def test_registry_migrates_legacy_ollama_url_with_v1():
    registry = EndpointRegistry.from_config(
        {"models": {"ollama_base_url": "http://127.0.0.1:11434"}}
    )

    assert registry.endpoints[0].base_url == "http://127.0.0.1:11434/v1"
    assert registry.endpoints[0].name == "Ollama"


def test_registry_deduplicates_by_base_url_and_warns():
    registry = EndpointRegistry.from_config(
        {
            "models": {
                "endpoints": [
                    {"name": "one", "base_url": "http://127.0.0.1:8080/v1"},
                    {"name": "two", "base_url": "http://127.0.0.1:8080/v1/"},
                ]
            }
        }
    )

    assert [endpoint.name for endpoint in registry.endpoints] == ["one"]
    assert any("duplicate" in warning for warning in registry.warnings)


def test_registry_skips_invalid_entries_and_warns():
    registry = EndpointRegistry.from_config(
        {
            "models": {
                "endpoints": [
                    {"name": "bad", "base_url": "not a url"},
                    "not a table",
                ]
            }
        }
    )

    assert registry.endpoints == ()
    assert any("invalid base_url" in warning for warning in registry.warnings)


def test_resolve_role_prefers_name_then_first():
    registry = EndpointRegistry.from_config(
        {
            "models": {
                "endpoints": [
                    {
                        "name": "chat-a",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "roles": ["chat"],
                    },
                    {
                        "name": "embed-b",
                        "base_url": "http://127.0.0.1:9090/v1",
                        "roles": ["embedding"],
                    },
                ]
            }
        }
    )

    assert registry.resolve("chat").name == "chat-a"
    assert registry.resolve("embedding").name == "embed-b"
    assert registry.resolve("chat", preferred="embed-b").name == "chat-a"
    assert registry.resolve("rerank") is None


def test_disabled_endpoint_is_not_resolved():
    registry = EndpointRegistry.from_config(
        {
            "models": {
                "endpoints": [
                    {
                        "name": "off",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "enabled": False,
                    }
                ]
            }
        }
    )

    assert registry.resolve("chat") is None


def test_headers_are_read_from_environment(monkeypatch):
    endpoint = EndpointConfig(
        name="remote",
        base_url="https://example.test/v1",
        auth_header_env="PRIVATE_AGENT_TEST_KEY",
    )
    monkeypatch.setenv("PRIVATE_AGENT_TEST_KEY", "secret-value")

    assert endpoint.headers() == {"Authorization": "secret-value"}


def test_headers_are_empty_when_env_is_missing(monkeypatch):
    monkeypatch.delenv("PRIVATE_AGENT_TEST_KEY", raising=False)
    endpoint = EndpointConfig(
        name="remote",
        base_url="https://example.test/v1",
        auth_header_env="PRIVATE_AGENT_TEST_KEY",
    )

    assert endpoint.headers() == {}


def test_clamp_limit_never_exceeds_ceiling():
    assert clamp_limit(64000, 32000, 16000) == 16000
    assert clamp_limit(None, 8000, 16000) == 8000
    assert clamp_limit(None, None, 16000) == 16000


def test_resolved_limits_clamps_output_to_context():
    endpoint = EndpointConfig(
        name="main",
        base_url="http://127.0.0.1:8080/v1",
        max_context_tokens=32000,
        max_output_tokens=16000,
    )

    context, output = resolved_limits(
        endpoint,
        ceiling_context=64000,
        ceiling_output=32000,
        discovered_context=48000,
    )

    assert context == 32000
    assert output == 16000


def test_registry_ttl_default_and_override():
    assert EndpointRegistry.from_config({}).probe_ttl_seconds == 300
    registry = EndpointRegistry.from_config(
        {"models": {"endpoint_probe_ttl_seconds": 10}}
    )
    assert registry.probe_ttl_seconds == 10


def test_registry_keeps_legacy_ollama_endpoint_with_v1_roles():
    registry = EndpointRegistry.from_config(
        {"models": {"ollama_base_url": "http://127.0.0.1:11434"}}
    )

    endpoint = registry.endpoints[0]
    assert endpoint.base_url == "http://127.0.0.1:11434/v1"
    assert endpoint.name == "Ollama"
    assert set(endpoint.roles) == {"chat", "embedding", "rerank", "native"}


def test_registry_does_not_duplicate_an_ollama_entry_that_is_already_listed():
    registry = EndpointRegistry.from_config(
        {
            "models": {
                "ollama_base_url": "http://127.0.0.1:11434",
                "local_openai_compatible_endpoints": [
                    {
                        "name": "Ollama",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "roles": ["chat", "embedding"],
                    }
                ],
            }
        }
    )

    assert [endpoint.name for endpoint in registry.endpoints] == ["Ollama"]
    assert registry.warnings == ()
