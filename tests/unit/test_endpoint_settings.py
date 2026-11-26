"""Location endpoints, weather limits and the tokenizer name are settings (K6/14.8)."""

from __future__ import annotations

import pytest

import config
from config import DEFAULT_CONFIG
from config.settings import Settings


def _settings(network: dict | None = None, models: dict | None = None) -> Settings:
    return Settings.from_config(
        {"network": network or {}, "models": models or {}},
        default_paths=DEFAULT_CONFIG["paths"],
    )


def test_endpoints_and_limits_default_to_the_previous_literals() -> None:
    settings = _settings()

    assert settings.geocoding_url == "https://geocoding-api.open-meteo.com/v1/search"
    assert settings.forecast_url == "https://api.open-meteo.com/v1/forecast"
    assert settings.archive_url == "https://archive-api.open-meteo.com/v1/archive"
    assert settings.ip_geolocation_urls == [
        "https://ipwho.is/",
        "https://ipapi.co/json/",
    ]
    assert settings.location_max_response_bytes == 1048576
    assert settings.weather_max_days == 16
    assert settings.weather_past_days == 90
    assert settings.tokenizer_encoding == "cl100k_base"


def test_self_hosted_endpoints_replace_the_defaults() -> None:
    settings = _settings(
        {
            "geocoding_url": "http://nominatim.internal:8080/search",
            "forecast_url": "https://weather.internal/forecast",
            "archive_url": "https://weather.internal/archive",
            "ip_geolocation_urls": ["http://geo.internal/json"],
        }
    )

    assert settings.geocoding_url == "http://nominatim.internal:8080/search"
    assert settings.forecast_url == "https://weather.internal/forecast"
    assert settings.archive_url == "https://weather.internal/archive"
    assert settings.ip_geolocation_urls == ["http://geo.internal/json"]


def test_invalid_endpoints_fall_back_to_the_defaults() -> None:
    settings = _settings(
        {
            "geocoding_url": "not a url",
            "forecast_url": "ftp://example.com/forecast",
            "archive_url": "",
            "ip_geolocation_urls": ["", "file:///etc/passwd", 5],
        }
    )

    assert settings.geocoding_url == "https://geocoding-api.open-meteo.com/v1/search"
    assert settings.forecast_url == "https://api.open-meteo.com/v1/forecast"
    assert settings.archive_url == "https://archive-api.open-meteo.com/v1/archive"
    assert settings.ip_geolocation_urls == [
        "https://ipwho.is/",
        "https://ipapi.co/json/",
    ]


def test_location_limits_are_bounded() -> None:
    configured = _settings(
        {
            "location_max_response_bytes": 2 * 1024 * 1024,
            "weather_max_days": 7,
            "weather_past_days": 30,
        }
    )
    assert configured.location_max_response_bytes == 2 * 1024 * 1024
    assert configured.weather_max_days == 7
    assert configured.weather_past_days == 30

    floored = _settings(
        {
            "location_max_response_bytes": 1,
            "weather_max_days": 0,
            "weather_past_days": -1,
        }
    )
    assert floored.location_max_response_bytes == 1048576
    assert floored.weather_max_days == 16
    assert floored.weather_past_days == 90


def test_tokenizer_encoding_is_validated() -> None:
    assert _settings(
        models={"tokenizer_encoding": "o200k_base"}
    ).tokenizer_encoding == ("o200k_base")
    assert _settings(models={"tokenizer_encoding": "bad name!"}).tokenizer_encoding == (
        "cl100k_base"
    )


def test_module_constants_mirror_settings() -> None:
    assert config.SETTINGS.geocoding_url == config.GEOCODING_URL
    assert config.SETTINGS.forecast_url == config.FORECAST_URL
    assert config.SETTINGS.archive_url == config.ARCHIVE_URL
    assert config.SETTINGS.ip_geolocation_urls == config.IP_GEOLOCATION_URLS
    assert config.SETTINGS.tokenizer_encoding == config.TOKENIZER_ENCODING


def test_location_module_uses_the_configured_values() -> None:
    from tools import location

    assert location.GEOCODING_URL == config.GEOCODING_URL
    assert location.FORECAST_URL == config.FORECAST_URL
    assert location.ARCHIVE_URL == config.ARCHIVE_URL
    assert location.MAX_API_RESPONSE_BYTES == config.LOCATION_MAX_RESPONSE_BYTES
    assert location.MAX_WEATHER_DAYS == config.WEATHER_MAX_DAYS
    assert location.FORECAST_API_PAST_DAYS == config.WEATHER_PAST_DAYS
    assert (
        tuple(
            (url, url.split("//", 1)[1].split("/", 1)[0])
            for url in config.IP_GEOLOCATION_URLS
        )
        == location.IP_GEOLOCATION_SERVICES
    )


def test_prompt_tokenizer_uses_the_configured_encoding() -> None:
    tiktoken = pytest.importorskip("tiktoken")
    from agent import prompts

    assert prompts.TOKENIZER_ENCODING == config.TOKENIZER_ENCODING
    assert (
        prompts._get_tokenizer().name
        == tiktoken.get_encoding(config.TOKENIZER_ENCODING).name
    )
