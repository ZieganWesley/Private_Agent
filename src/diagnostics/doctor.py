"""Doctor: dependency, environment and machine-resource diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from platform_support import (
    detect_distribution,
    detect_platform,
    format_install_command,
    system_dependency_status,
)

console = Console()

OPTIONAL_DEPENDENCIES = {
    "tree_sitter": "Repository intelligence parsing",
    "langchain_chroma": "Local RAG vector store",
    "faster_whisper": "Local speech transcription (media extra)",
    "cv2": "Webcam/image capture (media extra)",
    "sounddevice": "Microphone capture (media extra)",
    "PIL": "Image processing (media extra)",
    "ollama": "Ollama Python client (ollama extra)",
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


def ollama_configuration_summary() -> tuple[list[str], bool]:
    """Return configured Ollama endpoints and remote-host permission."""
    try:
        from config import APP_CONFIG, OLLAMA_BASE_URLS
    except Exception:  # noqa: BLE001 - doctor must not fail on config errors
        return [], False
    endpoints = [
        str(url).strip() for url in (OLLAMA_BASE_URLS or []) if str(url).strip()
    ]
    allow_remote = False
    if isinstance(APP_CONFIG, dict):
        model_config = APP_CONFIG.get("models", {})
        if isinstance(model_config, dict):
            raw = model_config.get("allow_remote_hosts", False)
            allow_remote = (
                raw
                if isinstance(raw, bool)
                else str(raw).strip().lower() in {"true", "1", "yes", "on"}
            )
    return endpoints, allow_remote


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


def run_doctor() -> int:
    console.print("[bold info]Private Agent Doctor[/bold info]")
    platform_info = detect_platform()
    distro = detect_distribution()
    platform_table = Table(title="Platform")
    platform_table.add_column("Field")
    platform_table.add_column("Value")
    platform_table.add_row("Platform", platform_info.name)
    platform_table.add_row("Machine", platform_info.machine or "unknown")
    platform_table.add_row(
        "Features", ", ".join(platform_info.available_features) or "none"
    )
    console.print(platform_table)
    console.print(
        "[info]Distribution:[/info] "
        + f"{distro.name} {distro.version}".strip()
        + f" (package manager: {distro.package_manager})"
    )

    deps = check_dependencies()
    table = Table(title="Dependencies")
    table.add_column("Status")
    table.add_column("Item")
    if deps["required"]:
        for item in deps["required"]:
            table.add_row("[error]MISSING (required)[/error]", item)
    else:
        table.add_row("[success]OK[/success]", "All required dependencies present")
    for item in deps["optional"]:
        table.add_row("[warning]OPTIONAL[/warning]", item)
    console.print(table)

    system_table = Table(title="System dependencies")
    system_table.add_column("Tool")
    system_table.add_column("Package")
    system_table.add_column("Status")
    for item in system_dependency_status(platform_info):
        system_table.add_row(
            str(item["tool"]),
            str(item["package"]),
            "[success]available[/success]"
            if item["available"]
            else "[error]missing[/error]",
        )
    console.print(system_table)
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
    if any("ollama" in item for item in deps["optional"]):
        console.print("[warning]Install optional Ollama dependencies with:[/warning]")
        console.print('  python -m pip install -e ".[ollama]"', markup=False)

    endpoints, allow_remote = ollama_configuration_summary()
    ollama_table = Table(title="Ollama")
    ollama_table.add_column("Field")
    ollama_table.add_column("Value")
    ollama_table.add_row(
        "Configured endpoints",
        ", ".join(endpoints) if endpoints else "none (default localhost:11434)",
    )
    ollama_table.add_row("Remote hosts allowed", "yes" if allow_remote else "no")
    console.print(ollama_table)
    if detect_platform().is_wsl:
        console.print(
            "[info][WSL][/info] Ollama inside WSL: http://localhost:11434. "
            "Windows-side Ollama: use the WSL host gateway (commonly "
            "172.x.x.1:11434) and set models.allow_remote_hosts to true."
        )

    info = analyze_machine()
    mtable = Table(title="Machine")
    mtable.add_column("Field")
    mtable.add_column("Value")
    for key, value in info.items():
        mtable.add_row(str(key), str(value))
    console.print(mtable)

    console.print(
        "[info]Configuration file path:[/info] "
        f"{Path.home() / '.config' / 'private_agent' / 'conf'}"
    )
    if deps["required"]:
        console.print(
            "[error]Install missing required dependencies before running.[/error]"
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
            f"{Path.home() / '.config' / 'private_agent' / 'conf'}.[/warning]"
        )
        return 0
    rec = recommend_config(info)
    console.print(f"\n[success]Recommended profile:[/success] {rec['profile']}")
    for key, value in rec["settings"].items():
        console.print(f"  {key}: {value}")
    console.print(
        "[info]Apply these values under the appropriate sections of "
        f"{Path.home() / '.config' / 'private_agent' / 'conf'}.[/info]"
    )
    return 0
