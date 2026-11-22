"""Unit coverage for provider resume records and fallback guidance (I1p)."""

from types import SimpleNamespace

from agent.provider_resume import (
    persist_provider_choice,
    provider_loss_guidance,
    provider_record_for_resume,
)
from database import PersistentMemory
from providers.adapters import ProviderRecord


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


def test_capability_change_warnings_describe_what_a_resumed_model_lost():
    from agent.provider_resume import capability_change_warnings

    previous = ProviderRecord(
        "ollama",
        "Ollama",
        "http://127.0.0.1:11434",
        "qwen3:8b",
        needs_tools=True,
        needs_vision=True,
        context_window=32768,
    )

    warnings = capability_change_warnings(
        previous, {"tools": False, "vision": False, "context_window": 8192}
    )

    assert len(warnings) == 3
    assert any("tool calling" in warning for warning in warnings)
    assert any("vision" in warning for warning in warnings)
    assert any("32768 to 8192" in warning for warning in warnings)


def test_capability_change_warnings_are_quiet_when_nothing_shrank():
    from agent.provider_resume import capability_change_warnings

    previous = ProviderRecord(
        "ollama", "Ollama", "http://127.0.0.1:11434", "qwen3:8b", needs_tools=True
    )

    assert (
        capability_change_warnings(
            previous, {"tools": True, "vision": False, "context_window": None}
        )
        == []
    )
    assert capability_change_warnings(None, {"tools": False}) == []


def test_persisting_a_weaker_provider_warns_about_the_lost_capabilities(tmp_path):
    memory = _memory(tmp_path)
    printed = []
    previous = ProviderRecord(
        "ollama",
        "Ollama",
        "http://127.0.0.1:11434",
        "qwen3:8b",
        needs_tools=True,
        context_window=32768,
    )

    persist_provider_choice(
        console=SimpleNamespace(
            print=lambda message, **_kwargs: printed.append(message)
        ),
        memory=memory,
        session_id="session-a",
        runtime=_runtime(),
        model="qwen3-8b",
        capabilities={"tools": False, "context_window": 4096},
        previous=previous,
    )

    assert any("tool calling" in message for message in printed)
    assert any("context window shrank" in message for message in printed)
    memory.close()


def _idle_provider(name="Ollama", start_hint="Run `ollama serve`."):
    from providers.adapters import ProviderPreset
    from providers.selection import IdleProvider

    return IdleProvider(
        preset=ProviderPreset(
            name=name,
            kind="ollama",
            default_ports=(11434,),
            install_check=lambda: True,
            start_hint=start_hint,
        ),
        start_hint=start_hint,
    )


def _record(display_name="Ollama", base_url="http://127.0.0.1:11434"):
    return ProviderRecord("ollama", display_name, base_url, "qwen3:8b")


class _Recorder:
    def __init__(self):
        self.messages = []

    def print(self, message, **_kwargs):
        self.messages.append(str(message))


def test_start_offer_is_skipped_when_the_provider_is_not_installed():
    from agent.provider_resume import offer_provider_start

    console = _Recorder()
    runtimes = [object()]

    result, matched = offer_provider_start(
        console=console,
        record=_record(),
        idle_providers=[],
        runtimes=runtimes,
        refresh=lambda: [],
        input_fn=lambda _prompt: "y",
    )

    assert (result, matched) == (runtimes, None)
    assert console.messages == []


def test_start_offer_does_nothing_without_consent():
    from agent.provider_resume import offer_provider_start

    console = _Recorder()
    started = []
    runtimes = [object()]

    def start_provider(name):
        started.append(name)
        return True

    result, matched = offer_provider_start(
        console=console,
        record=_record(),
        idle_providers=[_idle_provider()],
        runtimes=runtimes,
        refresh=lambda: [],
        input_fn=lambda _prompt: "",
        start_provider=start_provider,
    )

    assert (result, matched) == (runtimes, None)
    assert started == []
    assert any("is not running" in message for message in console.messages)
    assert any("ollama serve" in message for message in console.messages)


def test_start_offer_waits_for_the_restarted_provider():
    from agent.provider_resume import offer_provider_start

    console = _Recorder()
    restarted = SimpleNamespace(
        name="Ollama", base_url="http://127.0.0.1:11434", models=("qwen3:8b",)
    )
    polls = []

    def refresh():
        polls.append(True)
        return [restarted] if len(polls) >= 2 else []

    result, matched = offer_provider_start(
        console=console,
        record=_record(),
        idle_providers=[_idle_provider()],
        runtimes=[],
        refresh=refresh,
        input_fn=lambda _prompt: "yes",
        start_provider=lambda _name: True,
        sleep=lambda _seconds: None,
    )

    assert matched is restarted
    assert result == [restarted]
    assert len(polls) == 2
    assert any("running again" in message for message in console.messages)


def test_start_offer_reports_a_failed_launch_and_a_provider_that_stays_down():
    from agent.provider_resume import offer_provider_start

    console = _Recorder()
    result, matched = offer_provider_start(
        console=console,
        record=_record(),
        idle_providers=[_idle_provider()],
        runtimes=[],
        refresh=lambda: [],
        input_fn=lambda _prompt: "y",
        start_provider=lambda _name: False,
    )

    assert (result, matched) == ([], None)
    assert any("Could not start Ollama" in message for message in console.messages)

    console = _Recorder()
    sleeps = []
    result, matched = offer_provider_start(
        console=console,
        record=_record(),
        idle_providers=[_idle_provider()],
        runtimes=[],
        refresh=lambda: [],
        input_fn=lambda _prompt: "y",
        start_provider=lambda _name: True,
        sleep=sleeps.append,
        attempts=3,
        delay=0.5,
    )

    assert (result, matched) == ([], None)
    assert sleeps == [0.5, 0.5]
    assert any("did not answer" in message for message in console.messages)


def test_start_offer_is_skipped_when_not_interactive():
    from agent.provider_resume import offer_provider_start

    console = _Recorder()
    asked = []
    runtimes = [object()]

    result, matched = offer_provider_start(
        console=console,
        record=_record(),
        idle_providers=[_idle_provider()],
        runtimes=runtimes,
        refresh=lambda: [],
        input_fn=lambda prompt: asked.append(prompt) or "y",
        interactive=False,
    )

    assert (result, matched) == (runtimes, None)
    assert asked == []
    assert console.messages == []


def test_initial_phase_offers_installed_but_idle_runtimes():
    from agent.provider_resume import offer_provider_start

    console = _Recorder()
    restarted = SimpleNamespace(
        name="LM Studio", base_url="http://127.0.0.1:1234/v1", models=("m",)
    )
    polls = []

    def refresh():
        polls.append(True)
        return [restarted] if len(polls) >= 2 else []

    idle_provider = _idle_provider("LM Studio", "start LM Studio")
    # Rebuild idle provider with openai-compatible preset for realism.
    from providers.adapters import ProviderPreset
    from providers.selection import IdleProvider

    idle_provider = IdleProvider(
        preset=ProviderPreset(
            name="LM Studio",
            kind="openai-compatible",
            default_ports=(1234,),
            install_check=lambda: True,
            start_hint="start LM Studio",
        ),
        start_hint="start LM Studio",
    )

    result, matched = offer_provider_start(
        console=console,
        record=None,
        idle_providers=[idle_provider],
        runtimes=[],
        refresh=refresh,
        input_fn=lambda _prompt: "yes",
        start_provider=lambda _name: True,
        sleep=lambda _seconds: None,
    )

    assert matched is restarted
    assert result == [restarted]
    assert any("is installed but not running" in message for message in console.messages)


def _openai_idle(name="LM Studio", start_hint="start LM Studio"):
    from providers.adapters import ProviderPreset
    from providers.selection import IdleProvider

    return IdleProvider(
        preset=ProviderPreset(
            name=name,
            kind="openai-compatible",
            default_ports=(1234,),
            install_check=lambda: True,
            start_hint=start_hint,
        ),
        start_hint=start_hint,
    )


def test_initial_phase_declines_without_starting():
    from agent.provider_resume import offer_provider_start

    console = _Recorder()
    started = []
    runtimes = [object()]

    result, matched = offer_provider_start(
        console=console,
        record=None,
        idle_providers=[_openai_idle()],
        runtimes=runtimes,
        refresh=lambda: [],
        input_fn=lambda _prompt: "n",
        start_provider=lambda name: started.append(name) or True,
    )

    assert result == runtimes
    assert matched is None
    assert started == []
    assert any("is installed but not running" in message for message in console.messages)


def test_initial_phase_reports_failed_launch():
    from agent.provider_resume import offer_provider_start

    console = _Recorder()

    result, matched = offer_provider_start(
        console=console,
        record=None,
        idle_providers=[_openai_idle()],
        runtimes=[],
        refresh=lambda: [],
        input_fn=lambda _prompt: "yes",
        start_provider=lambda _name: False,
        sleep=lambda _seconds: None,
    )

    assert result == []
    assert matched is None
    assert any("Could not start LM Studio automatically" in message for message in console.messages)


def test_initial_phase_restarts_after_consent():
    from agent.provider_resume import offer_provider_start

    console = _Recorder()
    restarted = SimpleNamespace(
        name="LM Studio", base_url="http://127.0.0.1:1234/v1", models=("m",)
    )
    polls = []

    def refresh():
        polls.append(True)
        return [restarted] if len(polls) >= 1 else []

    result, matched = offer_provider_start(
        console=console,
        record=None,
        idle_providers=[_openai_idle()],
        runtimes=[],
        refresh=refresh,
        input_fn=lambda _prompt: "y",
        start_provider=lambda _name: True,
        sleep=lambda _seconds: None,
    )

    assert matched is restarted
    assert result == [restarted]
    assert any("is running again" in message for message in console.messages)
