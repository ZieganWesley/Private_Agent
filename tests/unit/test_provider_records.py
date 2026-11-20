"""Unit coverage for the per-session provider record (I1p)."""

from private_agent.database import PersistentMemory
from private_agent.database.provider_records import provider_record_payload
from private_agent.providers.adapters import ProviderRecord


def _memory(tmp_path):
    return PersistentMemory(db_path=str(tmp_path / "memory.sqlite"))


def test_provider_record_round_trips(tmp_path):
    memory = _memory(tmp_path)
    record = ProviderRecord(
        adapter_type="ollama",
        display_name="Ollama",
        base_url="http://127.0.0.1:11434",
        model="qwen3:8b",
        needs_tools=True,
        context_window=32768,
    )

    memory.record_session_provider("session-a", record)

    assert memory.get_session_provider("session-a") == record
    memory.close()


def test_provider_record_is_replaced_not_duplicated(tmp_path):
    memory = _memory(tmp_path)
    memory.record_session_provider(
        "session-a",
        ProviderRecord("ollama", "Ollama", "http://127.0.0.1:11434", "qwen3:8b"),
    )
    memory.record_session_provider(
        "session-a",
        ProviderRecord(
            "openai-compatible", "LM Studio", "http://127.0.0.1:1234/v1", "m"
        ),
    )

    stored = memory.get_session_provider("session-a")
    assert stored is not None
    assert stored.display_name == "LM Studio"
    assert stored.adapter_type == "openai-compatible"
    memory.close()


def test_missing_or_blank_provider_record_is_none(tmp_path):
    memory = _memory(tmp_path)

    assert memory.get_session_provider("unknown") is None
    assert memory.get_session_provider("") is None
    memory.close()


def test_recording_without_a_session_id_is_rejected(tmp_path):
    memory = _memory(tmp_path)

    try:
        memory.record_session_provider(
            "", ProviderRecord("ollama", "Ollama", "http://x", "m")
        )
    except ValueError as exc:
        assert "session id" in str(exc)
    else:  # pragma: no cover - the raise above is the expected path
        raise AssertionError("a blank session id must be rejected")
    memory.close()


def test_forgetting_a_provider_record_is_idempotent(tmp_path):
    memory = _memory(tmp_path)
    memory.record_session_provider(
        "session-a", ProviderRecord("ollama", "Ollama", "http://x", "m")
    )

    memory.forget_session_provider("session-a")
    memory.forget_session_provider("session-a")

    assert memory.get_session_provider("session-a") is None
    memory.close()


def test_provider_record_payload_is_json_safe():
    payload = provider_record_payload(
        ProviderRecord("ollama", "Ollama", "http://x", "m", needs_vision=True)
    )

    assert payload == {
        "adapter_type": "ollama",
        "display_name": "Ollama",
        "base_url": "http://x",
        "model": "m",
        "needs_tools": False,
        "needs_vision": True,
        "context_window": None,
    }


def test_task_provider_record_round_trips_and_survives_reopen(tmp_path):
    memory = _memory(tmp_path)
    record = ProviderRecord(
        adapter_type="openai-compatible",
        display_name="LM Studio",
        base_url="http://127.0.0.1:1234/v1",
        model="qwen3-8b",
    )
    task = memory.create_task_plan(
        "session-a", "Ship the release", "coding", [{"description": "Do it"}]
    )

    memory.record_task_provider(task["task_id"], record)

    assert memory.get_task_provider(task["task_id"]) == record
    memory.close()
    memory = PersistentMemory(db_path=str(tmp_path / "memory.sqlite"))
    assert memory.get_task_provider(task["task_id"]) == record
    memory.close()


def test_task_provider_record_needs_a_task_id(tmp_path):
    memory = _memory(tmp_path)

    try:
        memory.record_task_provider(
            "  ", ProviderRecord("ollama", "Ollama", "http://x", "m")
        )
    except ValueError as exc:
        assert "task id" in str(exc)
    else:  # pragma: no cover - the raise above is the expected path
        raise AssertionError("a blank task id must be rejected")
    memory.close()


def test_forgetting_a_task_provider_record_is_idempotent(tmp_path):
    memory = _memory(tmp_path)
    task = memory.create_task_plan(
        "session-a", "Ship the release", "coding", [{"description": "Do it"}]
    )
    memory.record_task_provider(
        task["task_id"], ProviderRecord("ollama", "Ollama", "http://x", "m")
    )

    memory.forget_task_provider(task["task_id"])
    memory.forget_task_provider(task["task_id"])

    assert memory.get_task_provider(task["task_id"]) is None
    memory.close()
