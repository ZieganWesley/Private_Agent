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


def run_doctor() -> int:
    console.print("[bold cyan]Private Agent Doctor[/bold cyan]")
    deps = check_dependencies()
    table = Table(title="Dependencies")
    table.add_column("Status")
    table.add_column("Item")
    if deps["required"]:
        for item in deps["required"]:
            table.add_row("[red]MISSING (required)[/red]", item)
    else:
        table.add_row("[green]OK[/green]", "All required dependencies present")
    for item in deps["optional"]:
        table.add_row("[yellow]OPTIONAL[/yellow]", item)
    console.print(table)

    info = analyze_machine()
    mtable = Table(title="Machine")
    mtable.add_column("Field")
    mtable.add_column("Value")
    for key, value in info.items():
        mtable.add_row(str(key), str(value))
    console.print(mtable)

    console.print(
        "[cyan]Configuration file path:[/cyan] "
        f"{Path.home() / '.config' / 'private_agent' / 'conf'}"
    )
    if deps["required"]:
        console.print(
            "[red]Install missing required dependencies before running.[/red]"
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
            "[yellow]Manual config changes cannot be performed from the application. "
            "Please edit the configuration file directly at "
            f"{Path.home() / '.config' / 'private_agent' / 'conf'}.[/yellow]"
        )
        return 0
    rec = recommend_config(info)
    console.print(f"\n[green]Recommended profile:[/green] {rec['profile']}")
    for key, value in rec["settings"].items():
        console.print(f"  {key}: {value}")
    console.print(
        "[cyan]Apply these values under the appropriate sections of "
        f"{Path.home() / '.config' / 'private_agent' / 'conf'}.[/cyan]"
    )
    return 0
