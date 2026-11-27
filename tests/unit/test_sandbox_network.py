import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import code_tasks
import tools.shell as shell
from sandbox import SandboxManager
from tools import approved_tool_invocation, get_tool_context


class _NetworkHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        body = b"sandbox-network-ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def local_http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NetworkHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/health"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.requires_sandbox
def test_isolated_command_network_egress_follows_permission(
    monkeypatch, tmp_path, local_http_server
):
    if not shutil.which("curl"):
        pytest.skip("curl is required for the sandbox network test")
    SandboxManager.set_root(str(tmp_path))

    denied_command, denied_env = code_tasks._isolated_command(
        ["curl", "-sS", "--max-time", "2", local_http_server],
        tmp_path,
        allow_network=False,
    )
    denied = subprocess.run(
        denied_command,
        env=denied_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if code_tasks.detect_platform().is_linux:
        assert "--share-net" not in denied_command
    else:
        # macOS confines networking in the Seatbelt profile instead of argv flags.
        assert "(deny network*)" in " ".join(denied_command)
    assert denied.returncode != 0
    assert "sandbox-network-ok" not in denied.stdout

    allowed_command, allowed_env = code_tasks._isolated_command(
        ["curl", "-sS", "--max-time", "2", local_http_server],
        tmp_path,
        allow_network=True,
    )
    allowed = subprocess.run(
        allowed_command,
        env=allowed_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if code_tasks.detect_platform().is_linux:
        assert "--share-net" in allowed_command
    else:
        assert "(allow network*)" in " ".join(allowed_command)
    assert allowed.returncode == 0
    assert "sandbox-network-ok" in allowed.stdout


@pytest.mark.requires_sandbox
def test_shell_and_background_tools_honor_network_permission(
    monkeypatch, tmp_path, local_http_server
):
    if not shutil.which("curl"):
        pytest.skip("curl is required for the sandbox network test")

    class FakeStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(shell.sys, "stdin", FakeStdin())
    SandboxManager.set_root(str(tmp_path))
    context = get_tool_context()
    monkeypatch.setattr(context, "code_task", None)
    monkeypatch.setattr(context, "db_path", str(tmp_path / "memory.sqlite"))
    shell.reset_shell_cwd()
    command = f"curl -sS --max-time 2 {local_http_server}"
    try:
        SandboxManager.set_network_allowed(False)
        with approved_tool_invocation():
            denied = shell.run_shell_command.invoke({"command": command})
        assert "Exit Code: 7" in denied
        assert "sandbox-network-ok" not in denied

        SandboxManager.set_network_allowed(True)
        with approved_tool_invocation():
            allowed = shell.run_shell_command.invoke({"command": command})
        assert "Exit Code: 0" in allowed
        assert "sandbox-network-ok" in allowed

        SandboxManager.set_network_allowed(False)
        with approved_tool_invocation():
            denied_bg = shell.start_background_process.invoke(
                {"command": command, "timeout": 10}
            )
        denied_match = re.search(r"Started background process ([0-9a-f]+)", denied_bg)
        assert denied_match is not None
        denied_status = shell.get_background_process_status.invoke(
            {"process_id": denied_match.group(1)}
        )
        assert "status=failed" in denied_status
        assert "sandbox-network-ok" not in denied_status

        SandboxManager.set_network_allowed(True)
        with approved_tool_invocation():
            allowed_bg = shell.start_background_process.invoke(
                {"command": command, "timeout": 10}
            )
        allowed_match = re.search(r"Started background process ([0-9a-f]+)", allowed_bg)
        assert allowed_match is not None
        allowed_status = shell.get_background_process_status.invoke(
            {"process_id": allowed_match.group(1)}
        )
        assert "status=completed" in allowed_status
        assert "sandbox-network-ok" in allowed_status
    finally:
        for process_id in list(shell._BACKGROUND_PROCESSES):
            shell.stop_background_process.invoke({"process_id": process_id})
        SandboxManager.disable_network()
