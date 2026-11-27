"""Approved shell execution inside the configured OS sandbox."""

import contextlib
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from langchain_core.tools import tool

from config import SHELL_COMMAND_TIMEOUT
from sandbox import (
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
_SHELL_CWD_LOCK = threading.RLock()
_SHELL_CWD = _SANDBOX_WORKSPACE


def _sandbox_network_allowed() -> bool:
    """Return whether the current sandboxed command may use the network."""
    code_task = None
    try:
        code_task = get_tool_context().code_task
    except Exception:  # noqa: BLE001 - tool context must never crash shell execution
        code_task = None
    if code_task is not None:
        return bool(getattr(code_task, "allow_network", False))
    try:
        return SandboxManager.is_network_allowed()
    except Exception:  # noqa: BLE001 - permission lookup must fail closed
        return False


def _workspace_path(sandbox_root: Path) -> str:
    """Return the directory sandboxed shells see as the workspace root."""
    import code_tasks

    return code_tasks.sandbox_workspace_path(sandbox_root)


def _current_shell_cwd(sandbox_root: Path) -> str:
    """Return the sandbox path used as the starting directory for shell calls."""
    global _SHELL_CWD
    workspace = _workspace_path(sandbox_root)
    with _SHELL_CWD_LOCK:
        candidate = _SHELL_CWD
    if not isinstance(candidate, str) or not candidate.startswith(workspace):
        candidate = workspace
    relative = candidate[len(workspace) :].lstrip("/")
    host_dir = sandbox_root / relative if relative else sandbox_root
    try:
        if not host_dir.is_dir():
            candidate = workspace
    except OSError:
        candidate = workspace
    with _SHELL_CWD_LOCK:
        _SHELL_CWD = candidate
    return candidate


def reset_shell_cwd() -> None:
    """Reset persisted shell state to the workspace root."""
    global _SHELL_CWD
    with _SHELL_CWD_LOCK:
        _SHELL_CWD = _SANDBOX_WORKSPACE


def _sandbox_shell_script(command: str, root: Path) -> str:
    logger.debug(
        "_sandbox_shell_script called with command length=" + str(len(command))
    )
    """Point host workspace paths in a shell script at the sandbox mount."""
    if _workspace_path(root) != _SANDBOX_WORKSPACE:
        return command
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
        import code_tasks

        console.print(
            "[warning]Commands run in a Linux OS sandbox with "
            + (
                "network access enabled by the approved coding access level, "
                if _sandbox_network_allowed()
                else "network disabled, "
            )
            + f"workspace-only writes, and resource limits.{risk_notice}[/warning]"
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
            shell_state_command = (
                f"cd {shlex.quote(_current_shell_cwd(sandbox_root))} && {command}"
            )
            shell_state_command = (
                f"{shell_state_command}; __pa_status=$?; "
                f"printf '\\n__PA_CWD__:%s\\n' \"$(pwd)\"; "
                f"exit $__pa_status"
            )
            command_args = [
                _SANDBOX_SHELL,
                "-c",
                _sandbox_shell_script(shell_state_command, sandbox_root),
            ]
        isolated_command, environment = code_tasks._isolated_command(
            command_args,
            sandbox_root,
            allow_network=_sandbox_network_allowed(),
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
            cwd=str(sandbox_root),
        )
        stdout = result.stdout
        cwd_match = re.search(r"\n__PA_CWD__:(.+?)(?:\n|$)", stdout)
        if cwd_match:
            new_cwd = cwd_match.group(1).strip()
            workspace = _workspace_path(sandbox_root)
            if new_cwd.startswith(workspace):
                relative = new_cwd[len(workspace) :].lstrip("/")
                host_dir = sandbox_root / relative if relative else sandbox_root
                try:
                    if host_dir.is_dir():
                        with _SHELL_CWD_LOCK:
                            _SHELL_CWD = new_cwd
                except OSError:
                    pass
            stdout = re.sub(r"\n__PA_CWD__:[^\n]*", "", stdout, count=1)
        return f"Exit Code: {result.returncode}\nOutput:\n{stdout}{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {effective_timeout} seconds."
    except RuntimeError as exc:
        return f"Execution blocked: {exc}"
    except (ValueError, OSError) as exc:
        return f"Execution failed: {exc}"


_BACKGROUND_PROCESS_LOCK = threading.RLock()
_BACKGROUND_PROCESSES: dict[str, dict] = {}
# Seconds a background group gets to exit after SIGTERM before SIGKILL.
_BACKGROUND_STOP_GRACE_SECONDS = 5.0


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


def _signal_background_group(pgid: int, sig: int) -> bool:
    """Signal a background process group, reporting whether it still exists."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        # The whole group is already gone; nothing left to signal.
        return False
    except PermissionError:
        logger.debug("Cannot signal background process group %s", pgid)
        return False
    return True


def _stop_background_group(process: subprocess.Popen) -> None:
    """Stop a background process and every process it spawned.

    Background commands start in their own session, so the direct child also
    leads a process group. Signalling only that child leaves the grandchildren
    it spawned running (and holding the output file open), so signal the group
    and escalate to SIGKILL when it outlives the grace period.
    """
    pgid = process.pid
    if process.poll() is not None:
        return
    if pgid <= 1 or pgid == os.getpgrp():
        # Defensive: a misconfigured launch must never signal our own group.
        process.terminate()
        try:
            process.wait(timeout=_BACKGROUND_STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_BACKGROUND_STOP_GRACE_SECONDS)
        return
    _signal_background_group(pgid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=_BACKGROUND_STOP_GRACE_SECONDS)
    # The leader may have exited while children that ignored SIGTERM survived,
    # so kill the group rather than assuming the leader's exit was enough.
    _signal_background_group(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=_BACKGROUND_STOP_GRACE_SECONDS)


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
        import code_tasks

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
            command = f"cd {shlex.quote(_current_shell_cwd(sandbox_root))} && {command}"
            command_args = [
                _SANDBOX_SHELL,
                "-c",
                _sandbox_shell_script(command, sandbox_root),
            ]
        isolated_command, environment = code_tasks._isolated_command(
            command_args,
            sandbox_root,
            allow_network=_sandbox_network_allowed(),
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
            cwd=str(sandbox_root),
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
        _stop_background_group(process)
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
