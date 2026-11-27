"""End-to-end PTY coverage for the CLI's Goodbye/shutdown screen."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

if sys.platform != "win32":
    import fcntl
    import pty
    import select
    import struct
    import termios
else:  # pragma: no cover - the test is skipped on Windows
    fcntl = pty = select = struct = termios = None

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="PTY tests require a POSIX terminal"
)

_HARNESS = r"""
import os
import sys

sys.path.insert(0, os.environ["PA_SRC"])

import agent.runtime as runtime
from agent.session_types import ProviderSelection, SessionSetup
from database import PersistentMemory
from sandbox import SandboxManager


class _FakeLLM:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, _messages):
        raise RuntimeError("the shutdown test must not call the model")


async def _prepare_session(**_kwargs):
    workspace = os.environ["PA_WORKSPACE"]
    SandboxManager.set_root(workspace)
    return SessionSetup(
        memory=PersistentMemory(db_path=os.environ["PA_DB"]),
        local_runtimes=[],
        isolation_warning=None,
        vectorstore=None,
        docs_input="",
        is_rag_active=False,
        rag_signature=None,
        rag_last_check=0.0,
        skills_folder_input="",
        loaded_skills={},
        mode_label="pty-test",
        session_id="pty-session",
        network_consent_state={},
        thinking_enabled=False,
        thinking_effort="medium",
        permission_mode="full",
    )


async def _choose_provider(**_kwargs):
    model = _FakeLLM()
    return ProviderSelection(
        provider_type="local",
        selected_model="pty-model",
        capabilities={
            "tools": False,
            "function_calls": False,
            "structured_output": False,
            "thinking": False,
            "vision": False,
            "audio": False,
            "context_window": 4096,
        },
        llm=model,
        llm_with_tools=model,
        tools_list=[],
        selected_local_runtime=None,
        include_private_context=False,
        startup_memory_context="",
        chat_history=[],
        context_summary="",
        session_summary_saved=True,
        available_models=[],
    )


async def _prompt_user_input(*_args, **_kwargs):
    sys.stdout.write("User: ")
    sys.stdout.flush()
    line = sys.stdin.readline()
    return "exit" if not line else line.strip()


runtime._prepare_session = _prepare_session
runtime._choose_provider = _choose_provider
runtime.prompt_user_input = _prompt_user_input
runtime.SHOW_WELCOME_SCREEN = False
runtime.SHOW_TURN_DIVIDER = False
runtime.STATUS_BAR = True

from cli import main

sys.argv = ["private-agent"]
main()
"""

_OSC = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)
_CSI = re.compile(r"\x1b\[([0-9;?]*)([@-~])")


def _plain(raw: str) -> str:
    """Strip ANSI styling while keeping the literal text the terminal showed."""
    return _CSI.sub("", _OSC.sub("", raw))


def _screen(raw: str) -> str:
    """Render the subset of ANSI a Rich Live footer emits into plain lines."""
    lines: list[list[str]] = [[]]
    row = column = 0

    def ensure() -> None:
        while len(lines) <= row:
            lines.append([])
        while len(lines[row]) <= column:
            lines[row].append(" ")

    index = 0
    while index < len(raw):
        char = raw[index]
        if char == "\x1b":
            osc = _OSC.match(raw, index)
            if osc is not None:
                index = osc.end()
                continue
            csi = _CSI.match(raw, index)
            if csi is not None:
                params, final = csi.group(1), csi.group(2)
                numbers = [
                    int(part) if part.isdigit() else 0
                    for part in params.lstrip("?").split(";")
                    if part
                ]
                amount = numbers[0] if numbers else 0
                if final == "A":
                    row = max(0, row - (amount or 1))
                elif final == "B":
                    row += amount or 1
                elif final == "C":
                    column += amount or 1
                elif final == "D":
                    column = max(0, column - (amount or 1))
                elif final == "G":
                    column = max(0, (amount or 1) - 1)
                elif final == "d":
                    row = max(0, (amount or 1) - 1)
                elif final in {"H", "f"}:
                    row = max(0, (numbers[0] if numbers else 1) - 1)
                    column = max(0, (numbers[1] if len(numbers) > 1 else 1) - 1)
                elif final == "K":
                    ensure()
                    if amount == 0:
                        del lines[row][column:]
                    elif amount == 1:
                        lines[row][: column + 1] = [" "] * (column + 1)
                    else:
                        lines[row] = []
                elif final == "J":
                    ensure()
                    if amount == 2:
                        lines = [[] for _ in lines]
                        row = column = 0
                    elif amount == 0:
                        del lines[row][column:]
                        lines[row + 1 :] = [[] for _ in lines[row + 1 :]]
                index = csi.end()
                continue
            index += 2
            continue
        if char == "\r":
            column = 0
        elif char == "\n":
            row += 1
            column = 0
            while len(lines) <= row:
                lines.append([])
        elif char == "\b":
            column = max(0, column - 1)
        elif char >= " ":
            ensure()
            lines[row][column] = char
            column += 1
        index += 1
    return "\n".join("".join(line).rstrip() for line in lines)


def _run_cli_in_pty(tmp_path: Path) -> tuple[str, bytes]:
    repository = Path(__file__).resolve().parents[2]
    harness = tmp_path / "pty_cli_harness.py"
    harness.write_text(_HARNESS, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    environment = os.environ.copy()
    environment.update(
        {
            "PA_SRC": str(repository / "src"),
            "PA_DB": str(tmp_path / "memory.sqlite"),
            "PA_WORKSPACE": str(workspace),
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "PYTHONPATH": str(repository / "src"),
            "TERM": "xterm-256color",
            "COLUMNS": "100",
            "LINES": "30",
        }
    )
    environment.pop("NO_COLOR", None)
    environment.pop("PRIVATE_AGENT_ACCESSIBLE", None)

    master, slave = pty.openpty()
    fcntl.ioctl(
        master,
        termios.TIOCSWINSZ,
        struct.pack("HHHH", 30, 100, 0, 0),
    )
    process = subprocess.Popen(
        [sys.executable, "-u", str(harness)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=repository,
        env=environment,
        start_new_session=True,
    )
    os.close(slave)
    output = bytearray()
    sent_exit = False
    deadline = time.monotonic() + 30
    try:
        while process.poll() is None:
            if time.monotonic() > deadline:
                process.kill()
                raise AssertionError(
                    "the CLI did not exit after 'exit':\n"
                    + output.decode("utf-8", "replace")
                )
            ready, _, _ = select.select([master], [], [], 0.1)
            if not ready:
                continue
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
            if not sent_exit and b"User:" in output:
                os.write(master, b"exit\n")
                sent_exit = True
        while True:
            ready, _, _ = select.select([master], [], [], 0.2)
            if not ready:
                break
            try:
                chunk = os.read(master, 4096)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
    finally:
        os.close(master)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)

    assert sent_exit, "the CLI never reached its user prompt"
    return output.decode("utf-8", "replace"), bytes(output)


def test_cli_pty_erases_the_ribbon_after_goodbye(tmp_path):
    raw, _ = _run_cli_in_pty(tmp_path)
    screen = _screen(raw)
    lines = screen.split("\n")

    assert "Permission: full" in _plain(raw), "the status ribbon never rendered"
    goodbye = [index for index, line in enumerate(lines) if "Goodbye." in line]
    assert goodbye, screen
    last_goodbye = goodbye[-1]

    after = "\n".join(lines[last_goodbye:])
    assert "Model:" not in after
    assert "Context:" not in after
    assert "Permission: full" not in after
    assert all(not line.strip() for line in lines[last_goodbye + 1 :]), screen
    assert last_goodbye + 1 < len(lines)
    assert lines[last_goodbye + 1] == ""
