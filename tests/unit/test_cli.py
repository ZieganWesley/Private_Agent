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

    assert received == [{"verbose_startup": True}]


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
