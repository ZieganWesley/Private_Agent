"""Unit coverage for provider selection helpers (Part I)."""

from types import SimpleNamespace

from providers.selection import (
    load_embedding_choice,
    load_last_provider,
    load_reranker_choice,
    match_runtime,
    pick_default_model,
    requested_model,
    requested_provider,
    save_embedding_choice,
    save_last_provider,
    save_reranker_choice,
    state_path,
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


def test_rag_routes_round_trip_and_survive_a_provider_save(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(tmp_path / "state.json"))

    save_embedding_choice(
        provider="openai-compatible",
        model="bge",
        base_url="http://127.0.0.1:1234/v1",
    )
    save_reranker_choice(
        label="ranker",
        model="bge-reranker",
        base_url="http://127.0.0.1:9000/v1",
    )
    save_last_provider(
        adapter_type="ollama",
        display_name="Ollama",
        base_url="http://127.0.0.1:11434",
        model="qwen3:8b",
    )

    assert load_embedding_choice() == {
        "provider": "openai-compatible",
        "model": "bge",
        "base_url": "http://127.0.0.1:1234/v1",
    }
    assert load_reranker_choice() == {
        "provider": "ranker",
        "model": "bge-reranker",
        "base_url": "http://127.0.0.1:9000/v1",
    }
    assert load_last_provider()["embedding"]["model"] == "bge"


def test_malformed_rag_routes_are_ignored(monkeypatch, tmp_path):
    target = tmp_path / "state.json"
    monkeypatch.setenv("PRIVATE_AGENT_PROVIDER_STATE", str(target))
    target.write_text(
        '{"embedding": {"provider": "ollama"}, "reranker": "bge"}',
        encoding="utf-8",
    )

    assert load_embedding_choice() is None
    assert load_reranker_choice() is None


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
    from agent.provider_resume import provider_loss_guidance

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
    from agent.provider_resume import provider_loss_guidance

    message = provider_loss_guidance(
        runtime_name="Ollama", session_id="s-1", discover=lambda: []
    )

    assert "No other local provider is reachable" in message
    assert "ollama serve" in message


def test_provider_loss_guidance_survives_a_failing_discovery():
    from agent.provider_resume import provider_loss_guidance

    def exploding():
        raise OSError("probe failed")

    message = provider_loss_guidance(
        runtime_name="Ollama", session_id="s-1", discover=exploding
    )

    assert "No other local provider is reachable" in message
