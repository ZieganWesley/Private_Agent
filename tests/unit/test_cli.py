import sys

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
