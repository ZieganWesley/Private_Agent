"""Unit coverage for provider selection helpers (Part I)."""

from types import SimpleNamespace

from private_agent.providers.adapters import ProviderPreset
from private_agent.providers.selection import (
    installed_but_idle,
    load_last_provider,
    match_runtime,
    pick_default_model,
    requested_model,
    requested_provider,
    save_last_provider,
    state_path,
)


def _preset(name, installed):
    return ProviderPreset(
        name=name,
        kind="openai-compatible",
        default_ports=(1234,),
        install_check=lambda: installed,
        start_hint=f"start {name}",
    )


def test_state_path_honours_the_environment_override(monkeypatch, tmp_path):
    target = tmp_path / "provider.json"
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(target))

    assert state_path() == target


def test_last_provider_round_trips(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(tmp_path / "state.json"))

    save_last_provider(
        adapter_type="ollama",
        display_name="Ollama",
        base_url="http://127.0.0.1:11434",
        model="qwen3:8b",
    )

    assert load_last_provider() == {
        "adapter_type": "ollama",
        "display_name": "Ollama",
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen3:8b",
    }


def test_last_provider_is_none_for_missing_or_malformed_state(monkeypatch, tmp_path):
    target = tmp_path / "state.json"
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(target))
    assert load_last_provider() is None

    target.write_text("{not json", encoding="utf-8")
    assert load_last_provider() is None

    target.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_last_provider() is None


def test_unwritable_state_does_not_raise(monkeypatch, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(blocker / "state.json"))

    save_last_provider(
        adapter_type="ollama", display_name="Ollama", base_url="http://x", model="m"
    )


def test_installed_but_idle_lists_only_unreachable_installed_providers():
    running = [SimpleNamespace(name="Ollama")]

    idle = installed_but_idle(
        running,
        presets=[
            _preset("Ollama", True),
            _preset("LM Studio", True),
            _preset("vLLM", False),
        ],
    )

    assert [item.preset.name for item in idle] == ["LM Studio"]
    assert idle[0].start_hint == "start LM Studio"


def test_installed_check_failure_is_treated_as_not_installed():
    def exploding():
        raise OSError("no permission")

    preset = ProviderPreset(
        name="Broken",
        kind="openai-compatible",
        default_ports=(1,),
        install_check=exploding,
        start_hint="never",
    )

    assert installed_but_idle([], presets=[preset]) == []


def test_match_runtime_is_case_insensitive_and_tolerates_missing_names():
    runtimes = [SimpleNamespace(name="Ollama"), SimpleNamespace(name="LM Studio")]

    assert match_runtime(" lm studio ", runtimes) is runtimes[1]
    assert match_runtime("", runtimes) is None
    assert match_runtime("vLLM", runtimes) is None


def test_pick_default_model_prefers_exact_then_substring_then_first():
    models = ["qwen3:8b", "llama3:70b", "mistral:7b"]

    assert pick_default_model(models, "llama3:70b") == "llama3:70b"
    assert pick_default_model(models, "mistral") == "mistral:7b"
    assert pick_default_model(models, "llama") == "llama3:70b"
    assert pick_default_model(models, "") == "qwen3:8b"
    assert pick_default_model([], "anything") is None


def test_requested_flags_read_the_environment(monkeypatch):
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER", "  Ollama ")
    monkeypatch.setenv("PRIVATE_AGENT_MODEL", "qwen3:8b")

    assert requested_provider() == "Ollama"
    assert requested_model() == "qwen3:8b"

    monkeypatch.delenv("PRIVATE_AGENT_PROVIDER")
    monkeypatch.delenv("PRIVATE_AGENT_MODEL")
    assert requested_provider() == ""
    assert requested_model() == ""


def test_provider_loss_guidance_lists_reachable_alternatives():
    from private_agent.agent.provider_resume import provider_loss_guidance

    runtimes = [
        SimpleNamespace(name="Ollama", models=("qwen3:8b",)),
        SimpleNamespace(name="LM Studio", models=("a", "b")),
    ]

    message = provider_loss_guidance(
        runtime_name="LM Studio", session_id="s-1", discover=lambda: runtimes
    )

    assert "LM Studio stopped answering" in message
    assert "Ollama (1 model(s))" in message
    assert "--resume s-1 --provider 'Ollama'" in message
    assert "LM Studio (2 model(s))" not in message


def test_provider_loss_guidance_handles_no_alternatives():
    from private_agent.agent.provider_resume import provider_loss_guidance

    message = provider_loss_guidance(
        runtime_name="Ollama", session_id="s-1", discover=lambda: []
    )

    assert "No other local provider is reachable" in message
    assert "ollama serve" in message


def test_provider_loss_guidance_survives_a_failing_discovery():
    from private_agent.agent.provider_resume import provider_loss_guidance

    def exploding():
        raise OSError("probe failed")

    message = provider_loss_guidance(
        runtime_name="Ollama", session_id="s-1", discover=exploding
    )

    assert "No other local provider is reachable" in message


def test_preset_probe_endpoints_cover_installed_presets_only():
    from private_agent.providers.adapters import ProviderPreset
    from private_agent.providers.selection import preset_probe_endpoints

    def installed():
        return True

    def missing():
        return False

    presets = [
        ProviderPreset(
            name="LM Studio",
            kind="openai-compatible",
            default_ports=(1234,),
            install_check=installed,
            start_hint="start it",
        ),
        ProviderPreset(
            name="vLLM",
            kind="openai-compatible",
            default_ports=(8000,),
            install_check=installed,
            start_hint="serve it",
        ),
        ProviderPreset(
            name="Jan",
            kind="openai-compatible",
            default_ports=(1337,),
            install_check=missing,
            start_hint="start Jan",
        ),
    ]

    endpoints = preset_probe_endpoints(
        [{"name": "LM Studio", "base_url": "http://127.0.0.1:1234/v1"}],
        presets=presets,
    )

    assert endpoints == [{"name": "vLLM", "base_url": "http://127.0.0.1:8000/v1"}]


def test_preset_probe_endpoints_deduplicate_shared_default_ports():
    from private_agent.providers.adapters import ProviderPreset
    from private_agent.providers.selection import preset_probe_endpoints

    presets = [
        ProviderPreset(
            name=name,
            kind="openai-compatible",
            default_ports=(8080,),
            install_check=lambda: True,
            start_hint="run it",
        )
        for name in ("llama.cpp", "LocalAI", "llamafile")
    ]

    endpoints = preset_probe_endpoints([], presets=presets)

    assert endpoints == [{"name": "llama.cpp", "base_url": "http://127.0.0.1:8080/v1"}]


def test_preset_probe_endpoints_tolerate_a_failing_install_check():
    from private_agent.providers.adapters import ProviderPreset
    from private_agent.providers.selection import preset_probe_endpoints

    def exploding():
        raise OSError("permission denied")

    presets = [
        ProviderPreset(
            name="Jan",
            kind="openai-compatible",
            default_ports=(1337,),
            install_check=exploding,
            start_hint="start Jan",
        )
    ]

    assert preset_probe_endpoints([], presets=presets) == []
