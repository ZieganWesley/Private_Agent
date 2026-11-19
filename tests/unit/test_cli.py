import os
import sys

import pytest

from private_agent import cli


def test_verbose_startup_flag_is_forwarded(monkeypatch):
    from private_agent import run_logging
    from private_agent.agent import runtime

    received = []
    monkeypatch.setattr(sys, "argv", ["private-agent", "--verbose-startup"])
    monkeypatch.setattr(run_logging, "start_debug_logging", lambda: None)
    monkeypatch.setattr(run_logging, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_logging, "stop_debug_logging", lambda: None)
    monkeypatch.setattr(
        runtime,
        "run_agent_cli",
        lambda **kwargs: received.append(kwargs),
    )

    cli.main()

    assert received == [
        {
            "verbose_startup": True,
            "resume_session": None,
            "continue_session": False,
        }
    ]


def test_resume_flags_are_forwarded(monkeypatch):
    from private_agent import run_logging
    from private_agent.agent import runtime

    received = []
    monkeypatch.setattr(sys, "argv", ["private-agent", "--resume", "session_abc"])
    monkeypatch.setattr(run_logging, "start_debug_logging", lambda: None)
    monkeypatch.setattr(run_logging, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_logging, "stop_debug_logging", lambda: None)
    monkeypatch.setattr(
        runtime, "run_agent_cli", lambda **kwargs: received.append(kwargs)
    )

    cli.main()

    assert received[0]["resume_session"] == "session_abc"
    assert received[0]["continue_session"] is False


def test_continue_flag_is_forwarded(monkeypatch):
    from private_agent import run_logging
    from private_agent.agent import runtime

    received = []
    monkeypatch.setattr(sys, "argv", ["private-agent", "--continue"])
    monkeypatch.setattr(run_logging, "start_debug_logging", lambda: None)
    monkeypatch.setattr(run_logging, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_logging, "stop_debug_logging", lambda: None)
    monkeypatch.setattr(
        runtime, "run_agent_cli", lambda **kwargs: received.append(kwargs)
    )

    cli.main()

    assert received[0]["continue_session"] is True
    assert received[0]["resume_session"] is None


def test_resume_and_continue_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["private-agent", "--resume", "session_a", "--continue"]
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2


def _patch_database(monkeypatch, database_path):
    from private_agent.database import PersistentMemory

    monkeypatch.setattr(
        "private_agent.database.PersistentMemory",
        lambda: PersistentMemory(str(database_path)),
    )


def test_list_sessions_prints_saved_sessions(monkeypatch, capsys, tmp_path):
    from private_agent.database import PersistentMemory

    database_path = tmp_path / "sessions.sqlite"
    memory = PersistentMemory(str(database_path))
    memory.save_message("session_listed", "human", "hello")
    memory.close()

    _patch_database(monkeypatch, database_path)
    monkeypatch.setattr(sys, "argv", ["private-agent", "--list-sessions"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "session_listed" in output
    assert "--resume" in output


def test_list_sessions_reports_an_empty_database(monkeypatch, capsys, tmp_path):
    _patch_database(monkeypatch, tmp_path / "empty.sqlite")
    monkeypatch.setattr(sys, "argv", ["private-agent", "--list-sessions"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    assert "No saved sessions yet" in capsys.readouterr().out


def test_list_sessions_does_not_start_the_agent(monkeypatch, tmp_path):
    from private_agent.agent import runtime

    started = []
    _patch_database(monkeypatch, tmp_path / "empty.sqlite")
    monkeypatch.setattr(sys, "argv", ["private-agent", "--list-sessions"])
    monkeypatch.setattr(
        runtime, "run_agent_cli", lambda **_kwargs: started.append(True)
    )

    with pytest.raises(SystemExit):
        cli.main()

    assert started == []


def test_offline_environment_is_set_at_startup_not_import(monkeypatch):
    import importlib

    from private_agent import config

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    importlib.reload(config)
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ

    cli.enforce_offline_environment()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_native_windows_exits_with_wsl_message(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", ["private-agent"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    assert "WSL" in capsys.readouterr().out.replace("\n", " ")


def test_posix_platforms_start_normally(monkeypatch):
    for platform_name in ("linux", "darwin"):
        monkeypatch.setattr(sys, "platform", platform_name)
        cli.ensure_supported_platform()


def test_config_flag_sets_config_path_environment(monkeypatch, tmp_path):
    from private_agent import run_logging
    from private_agent.agent import runtime

    seen = []
    custom = str(tmp_path / "custom.conf")
    monkeypatch.delenv("PRIVATE_AGENT_CONFIG", raising=False)
    monkeypatch.setattr(sys, "argv", ["private-agent", "--config", custom])
    monkeypatch.setattr(run_logging, "start_debug_logging", lambda: None)
    monkeypatch.setattr(run_logging, "log_event", lambda *_a, **_k: None)
    monkeypatch.setattr(run_logging, "stop_debug_logging", lambda: None)
    monkeypatch.setattr(
        runtime,
        "run_agent_cli",
        lambda **_kwargs: seen.append(os.environ.get("PRIVATE_AGENT_CONFIG")),
    )

    cli.main()
    os.environ.pop("PRIVATE_AGENT_CONFIG", None)

    assert seen == [custom]


def test_end_of_input_at_a_startup_prompt_exits_cleanly(monkeypatch, capsys):
    from private_agent import run_logging
    from private_agent.agent import runtime

    def closed_stdin(**_kwargs):
        raise EOFError

    monkeypatch.setattr(sys, "argv", ["private-agent"])
    monkeypatch.setattr(run_logging, "start_debug_logging", lambda: None)
    monkeypatch.setattr(run_logging, "log_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_logging, "stop_debug_logging", lambda: None)
    monkeypatch.setattr(runtime, "run_agent_cli", closed_stdin)

    cli.main()

    output = capsys.readouterr().out
    assert "Input closed; exiting." in output
    assert "Goodbye" in output
