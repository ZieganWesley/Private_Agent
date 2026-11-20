"""Unit coverage for provider resume records and fallback guidance (I1p)."""

from types import SimpleNamespace

from private_agent.agent.provider_resume import (
    persist_provider_choice,
    provider_loss_guidance,
    provider_record_for_resume,
)
from private_agent.database import PersistentMemory
from private_agent.providers.adapters import ProviderRecord


def _memory(tmp_path):
    return PersistentMemory(db_path=str(tmp_path / "memory.sqlite"))


def _approved_task(memory, session_id):
    task = memory.create_task_plan(
        session_id, "Ship the release", "coding", [{"description": "Do it"}]
    )
    memory.approve_task_plan(
        task["task_id"], task["revision"], task["digest"], "coding"
    )
    return task


def _runtime(name="LM Studio", base_url="http://127.0.0.1:1234/v1"):
    return SimpleNamespace(
        name=name, base_url=base_url, kind="openai-compatible", models=("m",)
    )


def test_persist_provider_choice_records_the_session_and_its_active_task(tmp_path):
    memory = _memory(tmp_path)
    task = _approved_task(memory, "session-a")

    record = persist_provider_choice(
        console=SimpleNamespace(print=lambda *_args, **_kwargs: None),
        memory=memory,
        session_id="session-a",
        runtime=_runtime(),
        model="qwen3-8b",
        capabilities={"tools": True, "vision": False, "context_window": 32768},
        previous=None,
    )

    assert memory.get_session_provider("session-a") == record
    assert memory.get_task_provider(task["task_id"]) == record
    assert record.context_window == 32768
    memory.close()


def test_resume_falls_back_to_the_task_provider_when_the_session_has_none(tmp_path):
    memory = _memory(tmp_path)
    task = _approved_task(memory, "session-a")
    memory.record_task_provider(
        task["task_id"],
        ProviderRecord("ollama", "Ollama", "http://127.0.0.1:11434", "qwen3:8b"),
    )

    record = provider_record_for_resume(memory, "session-a")

    assert record is not None
    assert record.display_name == "Ollama"
    memory.close()


def test_session_provider_wins_over_the_task_provider_on_resume(tmp_path):
    memory = _memory(tmp_path)
    task = _approved_task(memory, "session-a")
    memory.record_task_provider(
        task["task_id"],
        ProviderRecord("ollama", "Ollama", "http://127.0.0.1:11434", "qwen3:8b"),
    )
    memory.record_session_provider(
        "session-a",
        ProviderRecord(
            "openai-compatible", "LM Studio", "http://127.0.0.1:1234/v1", "m"
        ),
    )

    record = provider_record_for_resume(memory, "session-a")

    assert record is not None
    assert record.display_name == "LM Studio"
    memory.close()


def test_resume_without_any_record_is_none(tmp_path):
    memory = _memory(tmp_path)

    assert provider_record_for_resume(memory, "session-a") is None
    assert provider_record_for_resume(None, "session-a") is None
    memory.close()


def test_provider_loss_guidance_names_other_reachable_providers():
    guidance = provider_loss_guidance(
        runtime_name="Ollama",
        session_id="session-1",
        discover=lambda: [_runtime()],
    )

    assert "Ollama stopped answering" in guidance
    assert "LM Studio" in guidance
    assert "--resume session-1 --provider 'LM Studio'" in guidance


def test_provider_loss_guidance_handles_no_other_provider_and_discovery_errors():
    assert "No other local provider is reachable" in provider_loss_guidance(
        runtime_name="Ollama", session_id="s", discover=lambda: []
    )

    def exploding():
        raise OSError("cannot scan ports")

    assert "No other local provider" in provider_loss_guidance(
        runtime_name="Ollama", session_id="s", discover=exploding
    )
