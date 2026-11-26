"""Unit tests for the configurable online endpoint list."""

from providers.online import (
    DEFAULT_ONLINE_BASE_URL,
    has_configured_online_key,
    normalize_online_base_url,
    online_endpoints_from_config,
    resolve_online_endpoint,
)


def test_normalize_online_base_url_adds_v1_and_requires_https():
    assert (
        normalize_online_base_url("https://api.example.test")
        == "https://api.example.test/v1"
    )
    assert (
        normalize_online_base_url("https://api.example.test/v1/")
        == "https://api.example.test/v1"
    )
    assert normalize_online_base_url("http://localhost:8080") == (
        "http://localhost:8080/v1"
    )
    assert normalize_online_base_url("http://api.example.test/v1") is None
    assert normalize_online_base_url("https://user:key@api.example.test/v1") is None
    assert normalize_online_base_url("https://api.example.test/v1?x=1") is None
    assert normalize_online_base_url("ftp://api.example.test") is None
    assert normalize_online_base_url("") is None


def test_online_endpoints_are_read_from_the_list():
    endpoints = online_endpoints_from_config(
        {
            "models": {
                "online_endpoints": [
                    {
                        "name": "primary",
                        "base_url": "https://primary.example.test",
                        "api_key": "primary-key",
                        "models": ["gpt-4o", "gpt-4o-mini"],
                    },
                    {
                        "name": "gateway",
                        "base_url": "https://gateway.example.test/v1",
                        "api_key_env": "GATEWAY_KEY",
                    },
                ]
            }
        },
        environ={"GATEWAY_KEY": "env-key"},
    )

    assert [endpoint.name for endpoint in endpoints] == ["primary", "gateway"]
    assert endpoints[0].base_url == "https://primary.example.test/v1"
    assert endpoints[0].api_key == "primary-key"
    assert endpoints[0].models == ("gpt-4o", "gpt-4o-mini")
    assert endpoints[1].api_key == "env-key"


def test_legacy_online_pair_becomes_an_endpoint():
    endpoints = online_endpoints_from_config(
        {
            "models": {
                "online_base_url": "https://legacy.example.test",
                "online_api_key": "legacy-key",
            }
        }
    )

    assert len(endpoints) == 1
    assert endpoints[0].name == "OpenAI"
    assert endpoints[0].base_url == "https://legacy.example.test/v1"
    assert endpoints[0].api_key == "legacy-key"
    assert endpoints[0].source == "legacy.online_base_url"


def test_top_level_legacy_keys_are_still_accepted():
    endpoints = online_endpoints_from_config(
        {"online_base_url": "https://flat.example.test/v1", "online_api_key": "flat"}
    )

    assert endpoints[0].base_url == "https://flat.example.test/v1"
    assert endpoints[0].api_key == "flat"


def test_the_legacy_pair_is_not_duplicated_when_the_list_covers_it():
    endpoints = online_endpoints_from_config(
        {
            "models": {
                "online_base_url": "https://same.example.test/v1",
                "online_api_key": "legacy-key",
                "online_endpoints": [
                    {
                        "name": "same",
                        "base_url": "https://same.example.test/v1",
                        "api_key": "list-key",
                    }
                ],
            }
        }
    )

    assert len(endpoints) == 1
    assert endpoints[0].name == "same"
    assert endpoints[0].api_key == "list-key"


def test_invalid_entries_are_skipped_and_the_default_is_used_when_empty():
    endpoints = online_endpoints_from_config(
        {
            "models": {
                "online_endpoints": [
                    {"name": "bad", "base_url": "http://remote.example.test/v1"},
                    "not-a-table",
                ]
            }
        }
    )

    assert len(endpoints) == 1
    assert endpoints[0].base_url == DEFAULT_ONLINE_BASE_URL
    assert endpoints[0].source == "default"


def test_resolve_prefers_the_named_endpoint():
    config = {
        "models": {
            "preferred_online_endpoint": "gateway",
            "online_endpoints": [
                {"name": "primary", "base_url": "https://primary.example.test"},
                {"name": "gateway", "base_url": "https://gateway.example.test"},
                {
                    "name": "off",
                    "base_url": "https://off.example.test",
                    "enabled": False,
                },
            ],
        }
    }

    assert resolve_online_endpoint(config).name == "gateway"
    assert resolve_online_endpoint(config, preferred="PRIMARY").name == "primary"
    assert resolve_online_endpoint(config, preferred="missing").name == "primary"


def test_disabled_endpoints_are_never_resolved():
    config = {
        "models": {
            "online_endpoints": [
                {
                    "name": "off",
                    "base_url": "https://off.example.test",
                    "enabled": False,
                }
            ]
        }
    }

    assert resolve_online_endpoint(config) is None


def test_has_configured_online_key_checks_every_endpoint():
    assert not has_configured_online_key({})
    assert has_configured_online_key(
        {
            "models": {
                "online_endpoints": [
                    {"name": "a", "base_url": "https://a.example.test"},
                    {
                        "name": "b",
                        "base_url": "https://b.example.test",
                        "api_key": "key",
                    },
                ]
            }
        }
    )
