"""Approved shell execution inside the configured OS sandbox."""

import re
import shlex
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool
from rich.markup import escape

from ..config import SHELL_COMMAND_TIMEOUT
from ..sandbox import (
    SandboxManager,
    evaluate_shell_command,
    requires_shell_interpreter,
    shell_command_programs,
)
from . import _tool_invocation_approved, console
from .schemas import RunShellInput

_SANDBOX_SHELL = "/bin/bash"
_SANDBOX_WORKSPACE = "/workspace"


def _sandbox_shell_script(command: str, root: Path) -> str:
    """Point host workspace paths in a shell script at the sandbox mount."""
    host_root = re.escape(str(root.resolve()))
    return re.sub(rf"{host_root}(?=/|[\s;&|)'\"]|$)", _SANDBOX_WORKSPACE, command)


@tool(args_schema=RunShellInput)
def run_shell_command(command: str) -> str:
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
                if code_tasks.ACTIVE_CODE_TASK is not None
                and getattr(code_tasks.ACTIVE_CODE_TASK, "allow_network", False)
                else "network disabled, "
            )
            + f"workspace-only writes, and resource limits.{risk_notice}[/yellow]"
        )
        if not _tool_invocation_approved.get():
            approval = (
                console.input(
                    f"[yellow]Approve command "
                    f"{escape(command if use_shell else shlex.join(command_args))}? "
                    "\\[y/N]: [/yellow]"
                )
                .strip()
                .lower()
            )
            if approval != "y":
                return "Error: Command execution was not approved; no command was run."
        if code_tasks.ACTIVE_CODE_TASK is not None and "git" in (
            shell_command_programs(command)
        ):
            return (
                "Error: Use checkpoint_code_task for local code-task checkpoints; "
                "manual Git commands are disabled."
            )
        sandbox_root = (
            code_tasks.ACTIVE_CODE_TASK.root
            if code_tasks.ACTIVE_CODE_TASK is not None
            else SandboxManager.root_dir
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
                code_tasks.ACTIVE_CODE_TASK is not None
                and getattr(code_tasks.ACTIVE_CODE_TASK, "allow_network", False)
            ),
        )
        result = subprocess.run(
            isolated_command,
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=SHELL_COMMAND_TIMEOUT,
            env=environment,
        )
        return (
            f"Exit Code: {result.returncode}\nOutput:\n{result.stdout}{result.stderr}"
        )
    except RuntimeError as exc:
        return f"Execution blocked: {exc}"
    except Exception as exc:
        return f"Execution failed: {exc}"
