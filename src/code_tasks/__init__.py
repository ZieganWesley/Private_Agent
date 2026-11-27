import os
import platform  # noqa: F401 (used by tests)
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module
    resource = None

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import (
    CODE_TASK_GIT_TIMEOUT,
    CODE_TASK_SANDBOX_ADDRESS_SPACE_BYTES,
    CODE_TASK_SANDBOX_CPU_SECONDS,
    CODE_TASK_SANDBOX_FILE_SIZE_BYTES,
    CODE_TASK_SANDBOX_PROCESS_COUNT,
)
from platform_support import detect_platform
from ui.console import console

from .workspace import CodeTaskWorkspace as CodeTaskWorkspace

_CODE_TASK_PATTERNS = (
    re.compile(
        r"\b(write|create|generate|build|implement|modify|edit|change|fix|"
        r"refactor|scaffold|add|remove|delete)\b.{0,80}\b(code|program|script|"
        r"software|application|app|project|function|class|module|file|bug|"
        r"unit tests?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(code|program|script|software|application|app|project|function|"
        r"class|module|file|bug|unit tests?)\b.{0,50}\b(create|generate|"
        r"modify|edit|change|fix|refactor|implement|add|remove|delete)\b",
        re.IGNORECASE,
    ),
)
_TEST_FILE = re.compile(
    r"(^|/)(tests?/|test_[^/]+\.[^/]+$|[^/]+_test\.[^/]+$|"
    r"[^/]+\.(test|spec)\.[^/]+$)"
)
_SECRET_PATH = re.compile(
    r"(^|/)(\.env($|\.)|id_rsa($|\.)|id_ed25519($|\.)|credentials\.json$)"
    r"|(\.(pem|key|p12|pfx)$)",
    re.IGNORECASE,
)
_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}


def _is_likely_secret(path: str) -> bool:
    if Path(path).name.lower() in _ENV_TEMPLATES:
        return False
    return _SECRET_PATH.search(path) is not None


def is_code_task_request(text: str) -> bool:
    return any(pattern.search(text) for pattern in _CODE_TASK_PATTERNS)


def _run_git(
    root: Path, *args: str, timeout: int = CODE_TASK_GIT_TIMEOUT
) -> subprocess.CompletedProcess:
    command = ["git", "-C", str(root), *args]
    try:
        isolated_command, environment = _isolated_command(command, root)
    except RuntimeError as exc:
        return subprocess.CompletedProcess(command, 126, "", str(exc))
    return subprocess.run(
        isolated_command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )


def isolation_unavailable_reason() -> str | None:
    """Explain why OS-isolated project commands cannot run on this host."""
    info = detect_platform()
    if info.is_macos:
        if not shutil.which("sandbox-exec"):
            return (
                "Install macOS sandbox support (sandbox-exec) for filesystem, "
                "network, and resource isolation. Grant Terminal or the agent "
                "camera and microphone permissions in System Settings if those "
                "features are needed."
            )
        return None
    if not info.is_linux:
        return (
            "OS-level command isolation is currently supported only on Linux and macOS."
        )
    if not shutil.which("bwrap") or not shutil.which("prlimit"):
        return (
            "Install Bubblewrap (bwrap) and util-linux (prlimit) for "
            "filesystem, network, and resource isolation."
        )
    return None


# Read-only system locations a sandboxed command legitimately needs: the dyld
# shared cache, libc/system frameworks, the toolchain, and locale/timezone data.
_SEATBELT_READ_PATHS = (
    "/bin",
    "/sbin",
    "/usr",
    "/System",
    "/Library",
    "/etc",
    "/private/etc",
    "/private/var/db/timezone",
    "/dev",
    "/opt/homebrew",
    "/opt/local",
)

# Scratch space tools expect to write to, plus the character devices every
# process needs. Host data outside these paths and the workspace stays read-only.
_SEATBELT_WRITE_PATHS = (
    "/private/tmp",
    "/private/var/folders",
    "/dev/null",
    "/dev/zero",
    "/dev/stdout",
    "/dev/stderr",
)


def _seatbelt_path(path: Path | str) -> str:
    text = str(path)
    if '"' in text or "\n" in text or "\r" in text:
        raise ValueError("Sandbox paths cannot contain a quote or newline.")
    return text.replace("\\", "\\\\")


def _seatbelt_profile(root: Path, *, allow_network: bool = False) -> str:
    """Return a default-deny Seatbelt profile scoped to the workspace.

    The previous profile started with ``(allow default)`` and only denied or
    re-allowed writes, which left the whole host readable and every binary
    executable. This profile denies everything up front and then allows only
    reads of system/runtime locations, writes inside the workspace and system
    scratch space, and network access when explicitly requested.
    """
    root_text = _seatbelt_path(root.resolve())
    read_paths = [_seatbelt_path(path) for path in _SEATBELT_READ_PATHS]
    for candidate in (sys.prefix, sys.base_prefix):
        resolved = _seatbelt_path(Path(candidate).resolve())
        if resolved not in read_paths:
            read_paths.append(resolved)
    read_paths.append(root_text)
    write_paths = [_seatbelt_path(path) for path in _SEATBELT_WRITE_PATHS]
    write_paths.append(root_text)
    read_rules = " ".join(f'(subpath "{path}")' for path in read_paths)
    write_rules = " ".join(f'(subpath "{path}")' for path in write_paths)
    # Tools call getcwd() and realpath(), which reopen every ancestor of the
    # working directory. Denying those reads leaves a default-deny profile
    # unable to resolve the workspace at all (git and shells fail with
    # "getcwd: cannot access parent directories"). Allow the ancestor
    # directories themselves, never their contents.
    ancestor_rules = " ".join(
        f'(literal "{_seatbelt_path(parent)}")' for parent in root.resolve().parents
    )
    network_rule = "(allow network*)" if allow_network else "(deny network*)"
    return (
        "(version 1)\n"
        "(deny default)\n"
        f"{network_rule}\n"
        "(allow process*)\n"
        "(allow signal (target self))\n"
        "(allow sysctl-read)\n"
        "(allow mach-lookup)\n"
        "(allow ipc-posix-shm)\n"
        "(allow file-read-metadata)\n"
        f'(allow file-read* (literal "/") {ancestor_rules} {read_rules})\n'
        f"(allow file-write* {write_rules})\n"
    )


# Names a sandboxed child may inherit. Anything that looks like a credential is
# dropped, and PRIVATE_AGENT_* settings never cross the boundary.
_SANDBOX_ENVIRONMENT_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LANGUAGE",
        "TZ",
        "TERM",
        "COLORTERM",
        "NO_COLOR",
        "FORCE_COLOR",
        "USER",
        "LOGNAME",
        "SHELL",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "SOURCE_DATE_EPOCH",
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    }
)
_SECRET_ENVIRONMENT_MARKERS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTH",
    "SESSION",
    "COOKIE",
)


def _sandbox_environment_name_allowed(name: str) -> bool:
    """Decide whether one host environment variable may reach the sandbox."""
    if name.startswith("PRIVATE_AGENT"):
        return False
    if name in _SANDBOX_ENVIRONMENT_NAMES:
        return True
    upper = name.upper()
    if any(marker in upper for marker in _SECRET_ENVIRONMENT_MARKERS):
        return False
    return upper.startswith("LC_")


def _sandboxed_environment() -> dict[str, str]:
    """Return only the non-secret environment entries a sandboxed child needs.

    ``dict(os.environ)`` handed every API key, provider token, and
    ``PRIVATE_AGENT_*`` setting to a generated project's commands.
    """
    return {
        name: value
        for name, value in os.environ.items()
        if _sandbox_environment_name_allowed(name)
    }


def _host_resource_limit(resource_name: str) -> int | None:
    """Return this process's hard limit, or ``None`` when it is unlimited."""
    if resource is None:  # pragma: no cover - Windows
        return None
    limit_id = getattr(resource, resource_name, None)
    if limit_id is None:  # pragma: no cover - platform dependent
        return None
    try:
        _soft, hard = resource.getrlimit(limit_id)
    except OSError:  # pragma: no cover - platform dependent
        return None
    except ValueError:  # pragma: no cover - platform dependent
        return None
    if hard is None or hard == resource.RLIM_INFINITY:
        return None
    return int(hard)


def _sandbox_limit(resource_name: str, requested: int) -> int:
    """Fit a sandbox limit inside the hard limit inherited from our parent.

    A parent that already capped a resource (``ulimit -v`` in CI, for example)
    makes ``prlimit`` fail with EPERM when the sandbox asks for more than the
    parent allows, so the tighter of the two limits wins.
    """
    hard = _host_resource_limit(resource_name)
    if hard is None:
        return requested
    return min(requested, hard)


def _isolated_command(
    command: Sequence[str],
    root: Path,
    *,
    allow_network: bool = False,
) -> tuple[list[str], dict[str, str]]:
    unavailable_reason = isolation_unavailable_reason()
    if unavailable_reason:
        raise RuntimeError(f"Command execution is disabled: {unavailable_reason}")
    if detect_platform().is_macos:
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            raise RuntimeError(
                "Command execution is disabled: macOS sandbox support is unavailable."
            )
        profile = _seatbelt_profile(root, allow_network=allow_network)
        environment = _sandboxed_environment()
        environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        return ([sandbox_exec, "-p", profile, *command], environment)
    bwrap = shutil.which("bwrap")
    prlimit = shutil.which("prlimit")
    root = root.resolve()
    rewritten_command = []
    root_prefix = str(root)
    runtime_prefix = Path(sys.prefix).resolve()
    interpreter_base = Path(sys.base_prefix).resolve()
    interpreter_path = Path(sys.executable).resolve()
    interpreter_in_base = interpreter_base != Path(
        "/usr"
    ) and interpreter_path.is_relative_to(interpreter_base)
    dependency_paths = []
    for path in sys.path:
        if not path:
            continue
        resolved_path = Path(path).resolve()
        if (
            resolved_path.is_dir()
            and resolved_path.is_relative_to(Path.home())
            and not resolved_path.is_relative_to(interpreter_base)
            and resolved_path not in dependency_paths
        ):
            dependency_paths.append(resolved_path)
    dependency_mounts = {
        path: Path("/opt/private-agent-dependencies") / str(index)
        for index, path in enumerate(dependency_paths)
    }
    for argument in command:
        try:
            is_active_interpreter = Path(argument).resolve() == interpreter_path
        except OSError, RuntimeError:
            is_active_interpreter = False
        if is_active_interpreter and interpreter_in_base:
            relative_interpreter = interpreter_path.relative_to(interpreter_base)
            argument = str(Path("/opt/private-agent-python") / relative_interpreter)
        elif argument == root_prefix or argument.startswith(root_prefix + os.sep):
            argument = "/workspace" + argument[len(root_prefix) :]
        elif runtime_prefix != Path("/usr") and (
            argument == str(runtime_prefix)
            or argument.startswith(str(runtime_prefix) + os.sep)
        ):
            argument = (
                "/opt/private-agent-runtime" + argument[len(str(runtime_prefix)) :]
            )
        else:
            for dependency_path, sandbox_path in dependency_mounts.items():
                if argument == str(dependency_path) or argument.startswith(
                    str(dependency_path) + os.sep
                ):
                    argument = str(sandbox_path) + argument[len(str(dependency_path)) :]
                    break
        rewritten_command.append(argument)

    args = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--dir",
        "/etc",
    ]
    if allow_network:
        args.append("--share-net")
    system_files = [
        "/etc/ld.so.cache",
        "/etc/passwd",
        "/etc/group",
        "/etc/nsswitch.conf",
        "/etc/os-release",
    ]
    if allow_network:
        system_files.extend(("/etc/resolv.conf", "/etc/hosts"))
    for system_file in system_files:
        if os.path.isfile(system_file):
            args.extend(["--ro-bind", system_file, system_file])
    if os.path.isdir("/etc/ssl/certs"):
        args.extend(
            ["--dir", "/etc/ssl", "--ro-bind", "/etc/ssl/certs", "/etc/ssl/certs"]
        )
    args.extend(
        [
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",  # nosec B108 - container-internal mount point, not a host temp path
            "--tmpfs",
            "/home",
            "--tmpfs",
            "/root",
            "--tmpfs",
            "/run",
            "--tmpfs",
            "/mnt",
            "--tmpfs",
            "/media",
            "--tmpfs",
            "/var",
            "--tmpfs",
            "/opt",
        ]
    )
    if runtime_prefix != Path("/usr"):
        args.extend(
            [
                "--dir",
                "/opt/private-agent-runtime",
                "--ro-bind",
                str(runtime_prefix),
                "/opt/private-agent-runtime",
            ]
        )
    if interpreter_in_base:
        args.extend(
            [
                "--dir",
                "/opt/private-agent-python",
                "--ro-bind",
                str(interpreter_base),
                "/opt/private-agent-python",
            ]
        )
    if dependency_mounts:
        args.extend(["--dir", "/opt/private-agent-dependencies"])
        for source, destination in dependency_mounts.items():
            args.extend(
                [
                    "--dir",
                    str(destination),
                    "--ro-bind",
                    str(source),
                    str(destination),
                ]
            )
    args.extend(
        [
            "--dir",
            "/workspace",
            "--bind",
            str(root),
            "/workspace",
            "--chdir",
            "/workspace",
        ]
    )
    args.extend(
        [
            prlimit,
            f"--cpu={_sandbox_limit('RLIMIT_CPU', CODE_TASK_SANDBOX_CPU_SECONDS)}",
            "--as="
            f"{_sandbox_limit('RLIMIT_AS', CODE_TASK_SANDBOX_ADDRESS_SPACE_BYTES)}",
            "--nproc="
            f"{_sandbox_limit('RLIMIT_NPROC', CODE_TASK_SANDBOX_PROCESS_COUNT)}",
            "--fsize="
            f"{_sandbox_limit('RLIMIT_FSIZE', CODE_TASK_SANDBOX_FILE_SIZE_BYTES)}",
            "--",
            *rewritten_command,
        ]
    )
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",  # nosec B108 - sandbox-internal path
        "TMPDIR": "/tmp",  # nosec B108 - sandbox-internal path
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    try:
        git_config_count = min(20, max(0, int(os.environ.get("GIT_CONFIG_COUNT", "0"))))
    except ValueError:
        git_config_count = 0
    git_identity = []
    for index in range(git_config_count):
        key = os.environ.get(f"GIT_CONFIG_KEY_{index}", "").lower()
        value = os.environ.get(f"GIT_CONFIG_VALUE_{index}", "")
        if key in {"user.name", "user.email"} and len(value) <= 200:
            git_identity.append((key, value))
    if git_identity:
        environment["GIT_CONFIG_COUNT"] = str(len(git_identity))
        for index, (key, value) in enumerate(git_identity):
            environment[f"GIT_CONFIG_KEY_{index}"] = key
            environment[f"GIT_CONFIG_VALUE_{index}"] = value
    if dependency_mounts:
        environment["PYTHONPATH"] = os.pathsep.join(
            str(sandbox_path) for sandbox_path in dependency_mounts.values()
        )
    if interpreter_in_base:
        environment["PYTHONHOME"] = "/opt/private-agent-python"
        # The interpreter's RUNPATH points at its real install path, which is hidden
        # behind the /opt tmpfs here, so the loader would fall back to a system
        # libpython (a different patch release with its own site layout) and the
        # sandboxed interpreter would stop seeing its own site-packages.
        library_dirs = [
            f"/opt/private-agent-python/{name}"
            for name in ("lib", "lib64")
            if (interpreter_base / name).is_dir()
        ]
        if library_dirs:
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs)
    return args, environment


def _ask_without_git(reason: str) -> bool:
    if not sys.stdin.isatty():
        console.print(
            f"[error]Git is unavailable for this code task ({reason}). "
            "A user must explicitly approve continuing without Git.[/error]"
        )
        return False
    console.print(f"[warning]Git checkpointing is unavailable: {reason}[/warning]")
    return (
        console.input(
            "[warning]Proceed with this isolated code task without Git initialization "
            "and checkpoint commits? \\[y/N]: [/warning]"
        )
        .strip()
        .lower()
        == "y"
    )


def _slug(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())[:6]
    return "-".join(words) or "code-task"


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        ".tox",
        ".nox",
        ".next",
        "coverage",
        "dist",
        "build",
        "target",
        ".idea",
        ".vscode",
        ".conf",
        "secrets",
    }
    return {
        name
        for name in names
        if name in ignored
        or (Path(directory) / name).is_symlink()
        or (
            (name == ".env" or name.startswith(".env."))
            and name.lower() not in _ENV_TEMPLATES
        )
        or _is_likely_secret(name)
    }


def _project_files(root: Path):
    excluded_directories = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "dist",
        "build",
        "target",
    }
    for current_root, directories, filenames in os.walk(root):
        directories[:] = [
            name
            for name in directories
            if name not in excluded_directories
            and not (Path(current_root) / name).is_symlink()
        ]
        current = Path(current_root)
        for filename in filenames:
            candidate = current / filename
            if candidate.is_file() and not candidate.is_symlink():
                yield candidate


def _is_git_ignored(root: Path, relative_path: str) -> bool:
    if not (root / ".git").exists():
        return False
    result = _run_git(
        root,
        "check-ignore",
        "-q",
        "--no-index",
        "--",
        relative_path,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError(
        f"Could not determine whether project test '{relative_path}' is ignored."
    )


def _not_ignored_by_project_git(root: Path, relative_path: str) -> bool:
    result = _run_git(
        root,
        "check-ignore",
        "-q",
        "--no-index",
        "--",
        relative_path,
    )
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    raise RuntimeError(
        f"Could not determine whether project file {relative_path!r} is ignored."
    )


class RunTestsInput(BaseModel):
    pass


class CheckpointInput(BaseModel):
    label: str = Field(
        ..., description="Short completed microtask/subtask checkpoint label."
    )
    paths: list[str] = Field(
        ...,
        description="Exact changed project-relative files to include, such as ['src/main.py', 'tests/test_main.py'].",
    )


class FinalizeInput(BaseModel):
    pass


@tool(args_schema=RunTestsInput)
def run_project_unit_tests() -> str:
    """Run the code task's unit tests in its isolated project directory."""
    from tools import get_tool_context

    task = get_tool_context().code_task
    if task is None:
        return "Error: No isolated code task is active."
    return task.run_tests()


@tool(args_schema=CheckpointInput)
def checkpoint_code_task(label: str, paths: list[str]) -> str:
    """Run project tests and create a local-only Git checkpoint for exact changed files."""
    from tools import get_tool_context

    task = get_tool_context().code_task
    if task is None:
        return "Error: No isolated code task is active."
    return task.checkpoint(label, paths)


@tool(args_schema=FinalizeInput)
def finalize_code_task() -> str:
    """Run final project tests and commit a verified local completion checkpoint."""
    from tools import get_tool_context

    task = get_tool_context().code_task
    if task is None:
        return "Error: No isolated code task is active."
    return task.finalize()
