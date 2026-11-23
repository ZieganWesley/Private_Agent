"""Platform detection for Private Agent.

The rest of the codebase should ask this module what kind of host it is
running on instead of calling ``platform.system()`` directly.  Detection is
deliberately side-effect free and easy to fake in tests: every probe accepts
the environment and (where relevant) the ``/proc/version`` path as arguments.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlatformInfo:
    """A small, immutable description of the current host."""

    system: str
    machine: str
    is_linux: bool
    is_macos: bool
    is_windows: bool
    is_wsl: bool
    is_termux: bool
    name: str
    available_features: tuple[str, ...]


def _read_text(path: object) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, TypeError, ValueError):
        return ""


def is_wsl(
    *,
    proc_version_path: object | None = None,
    environ: dict[str, str] | None = None,
) -> bool:
    """Return whether this process is running inside WSL."""
    if environ is None:
        environ = os.environ
    if platform.system() != "Linux":
        return False
    if environ.get("WSL_DISTRO_NAME") or environ.get("WSL_INTEROP"):
        return True
    path = "/proc/version" if proc_version_path is None else proc_version_path
    text = _read_text(path).lower()
    return "microsoft" in text or "wsl" in text


def is_termux(
    *,
    environ: dict[str, str] | None = None,
    path_exists: Callable[[str], bool] | None = None,
) -> bool:
    """Return whether this process is running inside Termux."""
    if environ is None:
        environ = os.environ
    if platform.system() != "Linux":
        return False
    if environ.get("TERMUX_VERSION"):
        return True
    prefix = environ.get("PREFIX", "")
    if "com.termux" in prefix:
        return True
    checker = (
        path_exists if path_exists is not None else lambda path: Path(path).exists()
    )
    try:
        return bool(checker("/data/data/com.termux"))
    except OSError:
        return False


def detect_platform(
    *,
    proc_version_path: object | None = None,
    environ: dict[str, str] | None = None,
) -> PlatformInfo:
    """Detect the current platform and the features it can provide."""
    if environ is None:
        environ = os.environ

    system = platform.system() or sys.platform
    machine = platform.machine()
    is_windows = system == "Windows" or sys.platform == "win32"
    is_linux = system == "Linux" and not is_windows
    is_macos = system == "Darwin"
    running_wsl = is_linux and is_wsl(
        proc_version_path=proc_version_path, environ=environ
    )
    running_termux = is_linux and is_termux(environ=environ)

    if running_termux:
        name = "Termux"
    elif running_wsl:
        name = "WSL"
    elif is_macos:
        name = "macOS"
    elif is_windows:
        name = "Windows"
    elif is_linux:
        name = "Linux"
    else:
        name = system or "Unknown"

    if is_macos:
        features = ("command-isolation", "seatbelt")
    elif running_termux:
        features = ("termux",)
    elif is_linux:
        features = ("command-isolation", "bubblewrap", "prlimit")
    else:
        features = ()

    return PlatformInfo(
        system=system,
        machine=machine,
        is_linux=is_linux,
        is_macos=is_macos,
        is_windows=is_windows,
        is_wsl=running_wsl,
        is_termux=running_termux,
        name=name,
        available_features=features,
    )


def is_windows_mount_path(path: object) -> bool:
    """Return whether a path is under a WSL Windows drive mount."""
    try:
        parts = Path(str(path)).parts
    except (TypeError, ValueError):
        return False
    return (
        len(parts) >= 3
        and parts[0] == "/"
        and parts[1] == "mnt"
        and len(parts[2]) == 1
        and parts[2].isalpha()
    )


def platform_name(info: PlatformInfo | None = None) -> str:
    """Return a human-readable platform name."""
    if info is None:
        info = detect_platform()
    return info.name


@dataclass(frozen=True)
class DistributionInfo:
    """A small, immutable description of the host distribution."""

    distro_id: str
    name: str
    version: str
    package_manager: str


def _parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _read_os_release(path: object | None) -> str:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.extend((Path("/etc/os-release"), Path("/usr/lib/os-release")))
    for candidate in candidates:
        try:
            return candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return ""


def _package_manager_for(
    distro_id: str,
    id_like: str,
    *,
    environ: dict[str, str] | None = None,
) -> str:
    if environ is None:
        environ = os.environ
    if "com.termux" in environ.get("PREFIX", "") or environ.get("TERMUX_VERSION"):
        return "pkg"
    if sys.platform == "darwin":
        return "brew"

    candidates = [distro_id] + id_like.split()
    for candidate in candidates:
        if candidate in {"debian", "ubuntu", "linuxmint", "pop", "raspbian"}:
            return "apt-get"
        if candidate in {"fedora", "rhel", "centos", "rocky", "almalinux", "amzn"}:
            return "dnf" if shutil.which("dnf") else "yum"
        if candidate in {"opensuse", "opensuse-leap", "opensuse-tumbleweed", "sles"}:
            return "zypper"
        if candidate in {"arch", "manjaro", "endeavouros"}:
            return "pacman"
        if candidate == "alpine":
            return "apk"

    for manager in ("dnf", "yum", "apt-get", "apt", "pacman", "zypper", "apk"):
        if shutil.which(manager):
            return manager
    return "unknown"


def detect_distribution(
    *,
    os_release_path: object | None = None,
    environ: dict[str, str] | None = None,
) -> DistributionInfo:
    """Detect the host distribution and its package manager."""
    if environ is None:
        environ = os.environ
    values = _parse_os_release(_read_os_release(os_release_path))
    distro_id = values.get("ID", "unknown").casefold()
    name = values.get("NAME") or distro_id or "Unknown"
    version = values.get("VERSION_ID") or values.get("VERSION") or ""
    package_manager = _package_manager_for(
        distro_id, values.get("ID_LIKE", ""), environ=environ
    )
    return DistributionInfo(
        distro_id=distro_id,
        name=name,
        version=version,
        package_manager=package_manager,
    )


def format_install_command(packages: Iterable[str], package_manager: str) -> str:
    """Return an install command for ``packages`` on ``package_manager``."""
    package_list = " ".join(dict.fromkeys(packages))
    if not package_list:
        return ""
    if package_manager in {"apt", "apt-get"}:
        return f"sudo apt-get update && sudo apt-get install -y {package_list}"
    if package_manager == "dnf":
        return f"sudo dnf install -y {package_list}"
    if package_manager == "yum":
        return f"sudo yum install -y {package_list}"
    if package_manager == "pacman":
        return f"sudo pacman -S --noconfirm {package_list}"
    if package_manager == "zypper":
        return f"sudo zypper install -y {package_list}"
    if package_manager == "apk":
        return f"sudo apk add {package_list}"
    if package_manager == "brew":
        return f"brew install {package_list}"
    if package_manager == "pkg":
        return f"pkg install {package_list}"
    return f"# Install manually with your package manager: {package_list}"


def system_dependency_status(
    info: PlatformInfo | None = None,
) -> list[dict[str, str | bool]]:
    """Return required system tools and whether they are available."""
    if info is None:
        info = detect_platform()
    if info.is_macos:
        tools = [("sandbox-exec", "sandbox-exec")]
    elif info.is_linux:
        tools = [("bwrap", "bubblewrap"), ("prlimit", "util-linux")]
    else:
        tools = []
    return [
        {
            "tool": tool_name,
            "package": package_name,
            "available": shutil.which(tool_name) is not None,
        }
        for tool_name, package_name in tools
    ]


__all__ = [
    "DistributionInfo",
    "PlatformInfo",
    "detect_distribution",
    "detect_platform",
    "format_install_command",
    "is_termux",
    "is_windows_mount_path",
    "is_wsl",
    "platform_name",
    "system_dependency_status",
]
