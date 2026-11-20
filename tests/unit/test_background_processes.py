import time

from private_agent.sandbox import SandboxManager
from private_agent.tools import shell


def test_start_and_list_background_process(tmp_path, monkeypatch):
    SandboxManager.set_root(str(tmp_path))
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
    result = shell.start_background_process.invoke({"command": "echo hello"})
    process_id = result.split()[3]
    time.sleep(0.5)
    status = shell.get_background_process_status.invoke({"process_id": process_id})
    assert "status=completed" in status or "status=failed" in status
    assert "hello" in status


def test_stop_background_process(tmp_path, monkeypatch):
    SandboxManager.set_root(str(tmp_path))
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
    result = shell.start_background_process.invoke({"command": ""})
    assert "Command must not be empty" in result or "Error" in result


def test_list_empty_background_processes():
    # reset registry
    with shell._BACKGROUND_PROCESS_LOCK:
        shell._BACKGROUND_PROCESSES.clear()
    assert "No background processes" in shell.list_background_processes.invoke({})
