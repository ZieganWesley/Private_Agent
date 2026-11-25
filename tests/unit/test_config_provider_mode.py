"""Focused tests for the provider API-surface configuration mode."""

import pytest

from config.settings import Settings


def test_provider_api_mode_defaults_to_chat():
    assert Settings.from_config({}).provider_api_mode == "chat"


@pytest.mark.parametrize("mode", ["completions", "responses"])
def test_provider_api_mode_is_read_from_models_section(mode):
    settings = Settings.from_config({"models": {"provider_api_mode": mode}})

    assert settings.provider_api_mode == mode


def test_unknown_provider_api_mode_falls_back_to_chat():
    settings = Settings.from_config({"models": {"provider_api_mode": "nope"}})

    assert settings.provider_api_mode == "chat"
