"""Doctor: dependency, configuration, environment and resource diagnostics."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rich.text import Text

from platform_support import (
    detect_distribution,
    detect_platform,
    format_install_command,
    system_dependency_status,
)
from ui.console import build_console
from ui.status_view import StatusSection, status_panel

console = build_console()

ENDPOINT_PROBE_TIMEOUT_SECONDS = 0.75

OPTIONAL_DEPENDENCIES = {
    "tree_sitter": "Repository intelligence parsing",
    "langchain_chroma": "Local RAG vector store",
    "faster_whisper": "Local speech transcription (media extra)",
    "cv2": "Webcam/image capture (media extra)",
    "sounddevice": "Microphone capture (media extra)",
    "PIL": "Image processing (media extra)",
}

REQUIRED_DEPENDENCIES = {
    "httpx": "HTTP client",
    "rich": "Terminal UI",
    "prompt_toolkit": "Interactive prompts",
    "pydantic": "Settings validation",
    "yaml": "Configuration parsing",
    "sqlite3": "Local database (stdlib)",
}


def check_dependencies() -> dict[str, list[str]]:
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for module, description in REQUIRED_DEPENDENCIES.items():
        try:
            __import__(module)
        except Exception:  # noqa: BLE001 - import probe
            missing_required.append(f"{module} ({description})")
    for module, description in OPTIONAL_DEPENDENCIES.items():
        try:
            __import__(module)
        except Exception:  # noqa: BLE001 - import probe
            missing_optional.append(f"{module} ({description})")
    return {"required": missing_required, "optional": missing_optional}


@dataclass(frozen=True)
class ConfigProblem:
    """One problem found while validating the configuration file."""

    severity: str  # "error" or "warning"
    message: str


def _config_path() -> Path:
    from config import resolve_config_path

    return resolve_config_path()


def _type_label(value: Any) -> str:
    if value is None:
        return "null"
    return type(value).__name__


def _matches_default_type(value: Any, default: Any) -> bool:
    """Return whether a configured value fits the type of its default."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return True
        # configured_bool() also accepts the usual textual spellings.
        return isinstance(value, str) and value.strip().lower() in {
            "true",
            "false",
            "1",
            "0",
            "yes",
            "no",
            "on",
            "off",
        }
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, list):
        return isinstance(value, list)
    if isinstance(default, dict):
        return isinstance(value, dict)
    return value is None or default is None


def check_config_file(path: Path | None = None) -> list[ConfigProblem]:
    """Validate the user's configuration file without loading it into the app.

    Returns an empty list when the file is missing (it is created with defaults on
    first run) or when it is structurally valid.
    """
    from config import DEFAULT_CONFIG

    path = _config_path() if path is None else Path(path)
    problems: list[ConfigProblem] = []
    if not path.exists():
        problems.append(
            ConfigProblem(
                "warning",
                f"{path} does not exist yet; it will be created with defaults.",
            )
        )
        return problems
    if path.is_dir():
        problems.append(ConfigProblem("error", f"{path} is a directory, not a file."))
        return problems
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        problems.append(ConfigProblem("error", f"Cannot read {path}: {exc}"))
        return problems
    if not text.strip():
        problems.append(
            ConfigProblem("warning", f"{path} is empty; defaults are used.")
        )
        return problems
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        problems.append(
            ConfigProblem(
                "error",
                f"{path} is not valid JSON: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno}).",
            )
        )
        return problems
    if not isinstance(config, dict):
        problems.append(
            ConfigProblem(
                "error",
                f"{path} must contain a JSON object, found {_type_label(config)}.",
            )
        )
        return problems

    for section, default_section in DEFAULT_CONFIG.items():
        if section not in config:
            continue
        value = config[section]
        if not isinstance(default_section, dict):
            if not _matches_default_type(value, default_section):
                problems.append(
                    ConfigProblem(
                        "error",
                        f"{section} must be {_type_label(default_section)}, "
                        f"found {_type_label(value)}.",
                    )
                )
            continue
        if not isinstance(value, dict):
            problems.append(
                ConfigProblem(
                    "error",
                    f"{section} must be a JSON object, found {_type_label(value)}.",
                )
            )
            continue
        for key, default_value in default_section.items():
            if key not in value:
                continue
            if not _matches_default_type(value[key], default_value):
                problems.append(
                    ConfigProblem(
                        "error",
                        f"{section}.{key} must be {_type_label(default_value)}, "
                        f"found {_type_label(value[key])}.",
                    )
                )
    return problems


def analyze_machine() -> dict[str, Any]:
    try:
        import psutil

        memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    except Exception:  # noqa: BLE001 - psutil may be unavailable
        memory_gb = None
        cpu_count = os.cpu_count() or 1
    disk = shutil.disk_usage(str(Path.home()))
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": cpu_count,
        "memory_gb": memory_gb,
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "home": str(Path.home()),
    }


def recommend_config(info: dict[str, Any]) -> dict[str, Any]:
    cpu = int(info.get("cpu_count") or 1)
    mem = info.get("memory_gb") or 4.0
    if mem >= 16 and cpu >= 8:
        profile = "high"
    elif mem >= 8 and cpu >= 4:
        profile = "medium"
    else:
        profile = "low"
    profiles = {
        "low": {
            "max_context_tokens": 4096,
            "max_tool_iterations": 8,
            "rag_context_results": 3,
            "max_memory_context_tokens": 600,
        },
        "medium": {
            "max_context_tokens": 8192,
            "max_tool_iterations": 12,
            "rag_context_results": 5,
            "max_memory_context_tokens": 1200,
        },
        "high": {
            "max_context_tokens": 16384,
            "max_tool_iterations": 20,
            "rag_context_results": 8,
            "max_memory_context_tokens": 2400,
        },
    }
    return {"profile": profile, "settings": profiles[profile]}


def probe_endpoint(
    base_url: str,
    *,
    timeout: float = ENDPOINT_PROBE_TIMEOUT_SECONDS,
    client_factory=None,
) -> str:
    """Return a short reachability label for one configured endpoint."""
    factory = client_factory or (lambda **kwargs: httpx.Client(**kwargs))
    client = None
    try:
        client = factory(timeout=timeout)
        response = client.get(f"{base_url.rstrip('/')}/models")
        if response.status_code == 200:
            return "reachable"
        return f"HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001 - doctor must not fail on a probe
        return f"unreachable ({type(exc).__name__})"
    finally:
        if client is not None and hasattr(client, "close"):
            with contextlib.suppress(Exception):
                client.close()


def configured_endpoints() -> list[dict[str, Any]]:
    """Return the endpoints resolved from configuration, with live status."""
    try:
        from config import APP_CONFIG
        from providers.endpoints import EndpointRegistry
    except Exception:  # noqa: BLE001 - doctor must not fail on config errors
        return []
    registry = EndpointRegistry.from_config(APP_CONFIG)
    rows = []
    for endpoint in registry.endpoints:
        rows.append(
            {
                "name": endpoint.name,
                "base_url": endpoint.base_url,
                "roles": ", ".join(endpoint.roles),
                "enabled": endpoint.enabled,
                "models": len(endpoint.models),
                "status": probe_endpoint(endpoint.base_url)
                if endpoint.enabled
                else "disabled",
            }
        )
    return rows


def missing_system_dependencies() -> list[dict[str, str | bool]]:
    """Return required system tools that are not installed."""
    return [
        item
        for item in system_dependency_status(detect_platform())
        if not item["available"]
    ]


def system_dependency_fix_command() -> str:
    """Return an install command for missing system dependencies."""
    missing = missing_system_dependencies()
    if not missing:
        return ""
    distro = detect_distribution()
    return format_install_command(
        [str(item["package"]) for item in missing], distro.package_manager
    )


def _status_label(value: str, level: str) -> Text:
    """Return a status cell; values are literal, only the style is markup."""
    from ui.console import resolve_style

    return Text(value, style=resolve_style(level))


def doctor_sections() -> tuple[StatusSection, ...]:
    """Collect the doctor's report as one section list (shared with /status)."""
    platform_info = detect_platform()
    distro = detect_distribution()
    deps = check_dependencies()
    endpoints = configured_endpoints()
    info = analyze_machine()
    problems = check_config_file()

    dependency_rows: list[tuple[object, object]] = []
    if deps["required"]:
        dependency_rows.extend(
            (_status_label("MISSING (required)", "error"), item)
            for item in deps["required"]
        )
    else:
        dependency_rows.append(
            (_status_label("OK", "success"), "All required dependencies present")
        )
    dependency_rows.extend(
        (_status_label("OPTIONAL", "warning"), item) for item in deps["optional"]
    )

    system_rows = [
        (
            str(item["tool"]),
            str(item["package"]),
            _status_label("available", "success")
            if item["available"]
            else _status_label("missing", "error"),
        )
        for item in system_dependency_status(platform_info)
    ]

    endpoint_rows = [
        (
            str(endpoint["name"]),
            str(endpoint["base_url"]),
            str(endpoint["roles"]),
            str(endpoint["models"]),
            str(endpoint["status"]),
        )
        for endpoint in endpoints
    ] or [
        (
            "none",
            "-",
            "-",
            "-",
            "add models.endpoints or models.local_openai_compatible_endpoints",
        )
    ]

    config_rows: list[tuple[object, object]] = []
    if not problems:
        config_rows.append((_status_label("OK", "success"), "Configuration is valid"))
    config_rows.extend(
        (
            _status_label(
                "ERROR" if problem.severity == "error" else "WARNING",
                "error" if problem.severity == "error" else "warning",
            ),
            problem.message,
        )
        for problem in problems
    )

    return (
        StatusSection(
            "Platform",
            (
                ("Platform", platform_info.name),
                ("Machine", platform_info.machine or "unknown"),
                ("Features", ", ".join(platform_info.available_features) or "none"),
                (
                    "Distribution",
                    f"{distro.name} {distro.version}".strip()
                    + f" (package manager: {distro.package_manager})",
                ),
            ),
            ("Field", "Value"),
        ),
        StatusSection(
            "Dependencies",
            tuple(dependency_rows),
            ("Status", "Item"),
            header=True,
            justify_first="left",
        ),
        StatusSection(
            "System dependencies",
            tuple(system_rows),
            ("Tool", "Package", "Status"),
            header=True,
            justify_first="left",
        ),
        StatusSection(
            "Configured endpoints",
            tuple(endpoint_rows),
            ("Name", "Base URL", "Roles", "Models", "Status"),
            header=True,
            justify_first="left",
        ),
        StatusSection(
            "Machine",
            tuple((str(key), str(value)) for key, value in info.items()),
            ("Field", "Value"),
        ),
        StatusSection(
            "Configuration",
            tuple(config_rows),
            ("Status", "Item"),
            header=True,
            justify_first="left",
        ),
    )


def run_doctor() -> int:
    deps = check_dependencies()
    problems = check_config_file()
    console.print(
        status_panel(
            "Private Agent Doctor",
            doctor_sections(),
            subtitle=f"Configuration file: {_config_path()}",
        )
    )
    fix_command = system_dependency_fix_command()
    if fix_command:
        console.print(
            "[warning]Missing system dependencies. Install them with:[/warning]"
        )
        console.print(f"  {fix_command}", markup=False)
    if deps["required"]:
        console.print("[warning]Install required Python dependencies with:[/warning]")
        console.print('  python -m pip install -e ".[dev]"', markup=False)
    media_missing = [
        item
        for item in deps["optional"]
        if any(name in item for name in ("faster_whisper", "cv2", "sounddevice", "PIL"))
    ]
    if media_missing:
        console.print("[warning]Install optional media dependencies with:[/warning]")
        console.print('  python -m pip install -e ".[media]"', markup=False)

    if deps["required"]:
        console.print(
            "[error]Install missing required dependencies before running.[/error]"
        )
        return 1
    if any(problem.severity == "error" for problem in problems):
        console.print(
            "[error]Fix the configuration errors above; the app is running on "
            "defaults until they are resolved.[/error]"
        )
        return 1
    return 0


def doctor_interactive() -> int:
    run_doctor()
    info = analyze_machine()
    console.print("\n[bold]Detected machine information:[/bold]")
    for key, value in info.items():
        console.print(f"  {key}: {value}")
    try:
        answer = console.input("Is this information correct? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer not in {"y", "yes"}:
        console.print(
            "[warning]Manual config changes cannot be performed from the application. "
            "Please edit the configuration file directly at "
            f"{_config_path()}.[/warning]"
        )
        return 0
    rec = recommend_config(info)
    console.print(f"\n[success]Recommended profile:[/success] {rec['profile']}")
    for key, value in rec["settings"].items():
        console.print(f"  {key}: {value}")
    console.print(
        "[info]Apply these values under the appropriate sections of "
        f"{_config_path()}.[/info]"
    )
    return 0
