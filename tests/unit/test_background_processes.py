import contextlib
import os
import signal
import subprocess
import sys
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from sandbox import SandboxManager
from tools import approved_tool_invocation, shell


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    def isolate(command_args, root, **_kwargs):
        return (list(command_args), {})

    monkeypatch.setattr("code_tasks._isolated_command", isolate)


def test_start_and_list_background_process(tmp_path, monkeypatch):
    SandboxManager.set_root(str(tmp_path))
    with (
        approved_tool_invocation(),
        patch("tools.shell.sys.stdin", MagicMock(isatty=lambda: True)),
    ):
        result = shell.start_background_process.invoke({"command": "sleep 5"})
    assert "Started background process" in result
    process_id = result.split()[3]
    listed = shell.list_background_processes.invoke({})
    assert process_id in listed
    assert "running" in listed
    assert shell.background_process_indicator().startswith("background_processes=")
    shell.stop_background_process.invoke({"process_id": process_id})


def test_background_process_completes(tmp_path, monkeypatch):
    SandboxManager.set_root(str(tmp_path))
    with (
        approved_tool_invocation(),
        patch("tools.shell.sys.stdin", MagicMock(isatty=lambda: True)),
    ):
        result = shell.start_background_process.invoke({"command": "echo hello"})
    process_id = result.split()[3]
    time.sleep(0.5)
    status = shell.get_background_process_status.invoke({"process_id": process_id})
    assert "status=completed" in status or "status=failed" in status
    assert "hello" in status


def test_stop_background_process(tmp_path, monkeypatch):
    SandboxManager.set_root(str(tmp_path))
    with (
        approved_tool_invocation(),
        patch("tools.shell.sys.stdin", MagicMock(isatty=lambda: True)),
    ):
        result = shell.start_background_process.invoke({"command": "sleep 30"})
    process_id = result.split()[3]
    stopped = shell.stop_background_process.invoke({"process_id": process_id})
    assert "Stopped background process" in stopped
    status = shell.get_background_process_status.invoke({"process_id": process_id})
    assert "status=stopped" in status


def test_unknown_background_process(tmp_path, monkeypatch):
    SandboxManager.set_root(str(tmp_path))
    assert "Unknown background process" in shell.get_background_process_status.invoke(
        {"process_id": "missing"}
    )
    assert "Unknown background process" in shell.stop_background_process.invoke(
        {"process_id": "missing"}
    )


def test_empty_background_command(tmp_path, monkeypatch):
    SandboxManager.set_root(str(tmp_path))
    with (
        approved_tool_invocation(),
        patch("tools.shell.sys.stdin", MagicMock(isatty=lambda: True)),
    ):
        result = shell.start_background_process.invoke({"command": ""})
    assert "Command must not be empty" in result or "Error" in result


def test_list_empty_background_processes():
    # reset registry
    with shell._BACKGROUND_PROCESS_LOCK:
        shell._BACKGROUND_PROCESSES.clear()
    assert "No background processes" in shell.list_background_processes.invoke({})


def _register_group_process(command, cwd, tmp_path):
    """Register a real background process the way the tool does after Popen."""
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    process_id = uuid.uuid4().hex[:12]
    with shell._BACKGROUND_PROCESS_LOCK:
        shell._BACKGROUND_PROCESSES[process_id] = {
            "process_id": process_id,
            "pid": process.pid,
            "command": " ".join(command),
            "process": process,
            "started_at": time.time(),
            "status": "running",
            "returncode": None,
            "output_path": str(tmp_path / "bg.log"),
            "stdout_handle": None,
            "ended_at": None,
        }
    return process_id


def _wait_for_output(path, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.02)
    return False


def _dispose(process_id):
    with shell._BACKGROUND_PROCESS_LOCK:
        item = shell._BACKGROUND_PROCESSES.pop(process_id, None)
    if item is None:
        return
    process = item["process"]
    with contextlib.suppress(OSError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        process.wait(timeout=5)


def test_stop_background_process_kills_spawned_children(tmp_path):
    ticks = tmp_path / "ticks.log"
    process_id = _register_group_process(
        [
            "bash",
            "-c",
            "( while :; do echo tick >> ticks.log; sleep 0.05; done ) & sleep 60",
        ],
        tmp_path,
        tmp_path,
    )
    try:
        assert _wait_for_output(ticks), "grandchild never started writing"
        shell.stop_background_process.invoke({"process_id": process_id})
        size_after_stop = ticks.stat().st_size
        time.sleep(0.5)
        assert ticks.stat().st_size == size_after_stop, (
            "a spawned child kept running after the background process was stopped"
        )
    finally:
        _dispose(process_id)


def test_stop_background_process_signals_the_whole_group(tmp_path, monkeypatch):
    calls = []
    real_killpg = os.killpg

    def spy(pgid, sig):
        calls.append((pgid, sig))
        return real_killpg(pgid, sig)

    monkeypatch.setattr(shell.os, "killpg", spy)
    process_id = _register_group_process(["sleep", "60"], tmp_path, tmp_path)
    pid = shell._BACKGROUND_PROCESSES[process_id]["pid"]
    try:
        shell.stop_background_process.invoke({"process_id": process_id})
        assert calls[0] == (pid, signal.SIGTERM)
        assert shell._BACKGROUND_PROCESSES[process_id]["status"] == "stopped"
    finally:
        _dispose(process_id)


def test_stop_background_process_escalates_to_sigkill(tmp_path, monkeypatch):
    calls = []
    real_killpg = os.killpg

    def spy(pgid, sig):
        calls.append((pgid, sig))
        return real_killpg(pgid, sig)

    monkeypatch.setattr(shell.os, "killpg", spy)
    monkeypatch.setattr(shell, "_BACKGROUND_STOP_GRACE_SECONDS", 0.5)
    ready = tmp_path / "ready"
    process_id = _register_group_process(
        [
            sys.executable,
            "-c",
            "import pathlib, signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path('ready').write_text('1'); time.sleep(60)",
        ],
        tmp_path,
        tmp_path,
    )
    pid = shell._BACKGROUND_PROCESSES[process_id]["pid"]
    try:
        assert _wait_for_output(ready), "child never installed its SIGTERM handler"
        shell.stop_background_process.invoke({"process_id": process_id})
        assert (pid, signal.SIGTERM) in calls
        assert (pid, signal.SIGKILL) in calls
        item = shell._BACKGROUND_PROCESSES[process_id]
        assert item["returncode"] == -signal.SIGKILL
        assert item["status"] == "stopped"
    finally:
        _dispose(process_id)


def test_stop_background_process_tolerates_vanished_group(tmp_path, monkeypatch):
    def missing_group(_pgid, _sig):
        raise ProcessLookupError

    monkeypatch.setattr(shell.os, "killpg", missing_group)
    monkeypatch.setattr(shell, "_BACKGROUND_STOP_GRACE_SECONDS", 0.1)
    process_id = _register_group_process(["sleep", "60"], tmp_path, tmp_path)
    try:
        result = shell.stop_background_process.invoke({"process_id": process_id})
        assert "Stopped background process" in result
        assert shell._BACKGROUND_PROCESSES[process_id]["status"] == "stopped"
    finally:
        _dispose(process_id)
