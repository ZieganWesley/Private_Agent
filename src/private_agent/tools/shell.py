"""Approved shell execution inside the configured OS sandbox."""

import contextlib
import logging
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from langchain_core.tools import tool

from ..config import SHELL_COMMAND_TIMEOUT
from ..sandbox import (
    SandboxManager,
    evaluate_shell_command,
    requires_shell_interpreter,
    shell_command_programs,
)
from . import _tool_invocation_approved, console, get_tool_context
from .schemas import (
    BackgroundProcessIdInput,
    ListBackgroundProcessesInput,
    RunShellInput,
    StartBackgroundProcessInput,
)

logger = logging.getLogger(__name__)

_SANDBOX_SHELL = "/bin/bash"
_SANDBOX_WORKSPACE = "/workspace"


def _sandbox_shell_script(command: str, root: Path) -> str:
    logger.debug(
        "_sandbox_shell_script called with command length=" + str(len(command))
    )
    """Point host workspace paths in a shell script at the sandbox mount."""
    host_root = re.escape(str(root.resolve()))
    return re.sub(rf"{host_root}(?=/|[\s;&|)'\"]|$)", _SANDBOX_WORKSPACE, command)


@tool(
    args_schema=RunShellInput,
    description="Run an approved command inside the configured OS-isolated workspace",
)
def run_shell_command(command: str, timeout: int | None = None) -> str:
    logger.debug("run_shell_command called with command length=" + str(len(command)))
    """Run an approved command inside the configured OS-isolated workspace.

    The command starts in the workspace root (mounted at /workspace), so use
    relative paths. Shell syntax such as `&&`, `|`, `;`, redirects, `cd` and
    `source` is supported.
    """
    try:
        command_args = shlex.split(command)
        if not command_args:
            return "Error: Command must not be empty."
        use_shell = requires_shell_interpreter(command)
        if not sys.stdin.isatty() and not _tool_invocation_approved.get():
            return (
                "Error: Shell execution requires an interactive terminal and explicit "
                "per-command approval; no command was run."
            )
        risk_notice = (
            " This command matches a high-risk pattern."
            if evaluate_shell_command(command)
            else ""
        )
        from .. import code_tasks

        console.print(
            "[yellow]Commands run in a Linux OS sandbox with "
            + (
                "network access enabled by the approved coding access level, "
                if get_tool_context().code_task is not None
                and getattr(get_tool_context().code_task, "allow_network", False)
                else "network disabled, "
            )
            + f"workspace-only writes, and resource limits.{risk_notice}[/yellow]"
        )
        if not _tool_invocation_approved.get():
            return (
                "Error: Command execution requires explicit approval through the "
                "central permission policy; no command was run."
            )
        if get_tool_context().code_task is not None and "git" in (
            shell_command_programs(command)
        ):
            return (
                "Error: Use checkpoint_code_task for local code-task checkpoints; "
                "manual Git commands are disabled."
            )
        sandbox_root = (
            get_tool_context().code_task.root
            if get_tool_context().code_task is not None
            else SandboxManager.get_root()
        )
        if use_shell:
            command_args = [
                _SANDBOX_SHELL,
                "-c",
                _sandbox_shell_script(command, sandbox_root),
            ]
        isolated_command, environment = code_tasks._isolated_command(
            command_args,
            sandbox_root,
            allow_network=(
                get_tool_context().code_task is not None
                and getattr(get_tool_context().code_task, "allow_network", False)
            ),
        )
        effective_timeout = (
            SHELL_COMMAND_TIMEOUT
            if timeout is None
            else min(int(timeout), SHELL_COMMAND_TIMEOUT)
        )
        result = subprocess.run(
            isolated_command,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            env=environment,
        )
        return (
            f"Exit Code: {result.returncode}\nOutput:\n{result.stdout}{result.stderr}"
        )
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {effective_timeout} seconds."
    except RuntimeError as exc:
        return f"Execution blocked: {exc}"
    except (ValueError, OSError) as exc:
        return f"Execution failed: {exc}"


_BACKGROUND_PROCESS_LOCK = threading.RLock()
_BACKGROUND_PROCESSES: dict[str, dict] = {}


def _background_process_count() -> int:
    with _BACKGROUND_PROCESS_LOCK:
        _reap_background_processes()
        return sum(
            1 for item in _BACKGROUND_PROCESSES.values() if item["status"] == "running"
        )


def _reap_background_processes() -> None:
    with _BACKGROUND_PROCESS_LOCK:
        for item in _BACKGROUND_PROCESSES.values():
            process = item["process"]
            if item["status"] == "running" and process.poll() is not None:
                item["status"] = "completed" if process.returncode == 0 else "failed"
                item["returncode"] = process.returncode
                item["ended_at"] = time.time()
                handle = item.get("stdout_handle")
                if handle is not None:
                    with contextlib.suppress(OSError):
                        handle.close()
                    item["stdout_handle"] = None


def _read_background_output(item: dict, tail: int = 4000) -> str:
    try:
        with open(item["output_path"], encoding="utf-8", errors="replace") as handle:
            data = handle.read()
    except OSError as exc:
        return f"Output unavailable: {exc}"
    if len(data) > tail:
        return data[-tail:]
    return data


@tool(args_schema=StartBackgroundProcessInput)
def start_background_process(command: str, timeout: int | None = None) -> str:
    """Start a shell command in the background and return a process id."""
    try:
        command_args = shlex.split(command)
        if not command_args:
            return "Error: Command must not be empty."
        if not sys.stdin.isatty() and not _tool_invocation_approved.get():
            return (
                "Error: Shell execution requires an interactive terminal and explicit "
                "per-command approval; no command was run."
            )
        from .. import code_tasks

        if not _tool_invocation_approved.get():
            return (
                "Error: Command execution requires explicit approval through the "
                "central permission policy; no command was run."
            )
        use_shell = requires_shell_interpreter(command)
        sandbox_root = (
            get_tool_context().code_task.root
            if get_tool_context().code_task is not None
            else SandboxManager.get_root()
        )
        if use_shell:
            command_args = [
                _SANDBOX_SHELL,
                "-c",
                _sandbox_shell_script(command, sandbox_root),
            ]
        isolated_command, environment = code_tasks._isolated_command(
            command_args,
            sandbox_root,
            allow_network=(
                get_tool_context().code_task is not None
                and getattr(get_tool_context().code_task, "allow_network", False)
            ),
        )
        process_id = uuid.uuid4().hex[:12]
        output_handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w+", delete=False, prefix=f"bg-{process_id}-", suffix=".log"
        )
        output_path = output_handle.name
        output_handle.close()
        stdout_handle = open(output_path, "a", encoding="utf-8")  # noqa: SIM115
        process = subprocess.Popen(
            isolated_command,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        started_at = time.time()
        with _BACKGROUND_PROCESS_LOCK:
            _BACKGROUND_PROCESSES[process_id] = {
                "process_id": process_id,
                "pid": process.pid,
                "command": command,
                "process": process,
                "started_at": started_at,
                "status": "running",
                "returncode": None,
                "output_path": output_path,
                "stdout_handle": stdout_handle,
                "ended_at": None,
            }
        if timeout is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=timeout)
            _reap_background_processes()
        running = _background_process_count()
        return (
            f"Started background process {process_id} (pid {process.pid}). "
            f"Running background processes: {running}."
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return f"Error starting background process: {exc}"


@tool(args_schema=BackgroundProcessIdInput)
def get_background_process_status(process_id: str) -> str:
    """Return the status and recent output for a background process."""
    with _BACKGROUND_PROCESS_LOCK:
        _reap_background_processes()
        item = _BACKGROUND_PROCESSES.get(process_id)
        if item is None:
            return f"Error: Unknown background process '{process_id}'."
        status = item["status"]
        returncode = item["returncode"]
        output = _read_background_output(item)
        running = _background_process_count()
    return (
        f"Process {process_id}: status={status}, pid={item['pid']}, "
        f"returncode={returncode}, running_background_processes={running}.\\n"
        f"Recent output:\\n{output}"
    )


@tool(args_schema=BackgroundProcessIdInput)
def stop_background_process(process_id: str) -> str:
    """Stop a running background process."""
    with _BACKGROUND_PROCESS_LOCK:
        _reap_background_processes()
        item = _BACKGROUND_PROCESSES.get(process_id)
        if item is None:
            return f"Error: Unknown background process '{process_id}'."
        process = item["process"]
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        item["status"] = "stopped"
        item["returncode"] = process.returncode
        item["ended_at"] = time.time()
        handle = item.get("stdout_handle")
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()
            item["stdout_handle"] = None
        _reap_background_processes()
        running = _background_process_count()
    return f"Stopped background process {process_id}. Running background processes: {running}."


@tool(args_schema=ListBackgroundProcessesInput)
def list_background_processes() -> str:
    """List tracked background processes and how many are running."""
    with _BACKGROUND_PROCESS_LOCK:
        _reap_background_processes()
        items = list(_BACKGROUND_PROCESSES.values())
        running = sum(1 for item in items if item["status"] == "running")
    if not items:
        return "No background processes are tracked. Running: 0."
    lines = [f"Background processes: {len(items)} total, {running} running."]
    for item in items:
        lines.append(
            f"- {item['process_id']} (pid {item['pid']}): {item['status']} - {item['command']}"
        )
    return "\\n".join(lines)


def background_process_indicator() -> str:
    """Return a compact status indicator for running background processes."""
    return f"background_processes={_background_process_count()}"
