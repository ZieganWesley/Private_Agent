"""Production-ready terminal CLI for Private Agent using Rich and pyfiglet."""

from __future__ import annotations

import contextlib
import os
import signal
import stat
import sys
import termios
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pyfiglet
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console()

CONFIG_STEPS: tuple[str, ...] = (
    "1. Workspace",
    "2. RAG & Runtime",
    "3. Session & Permissions",
    "4. Provider & Model",
)

DEFAULT_RECENT_SESSIONS: tuple[dict[str, str], ...] = (
    {
        "session_id": "session_dfa567cbee7f4ab88227ee99907aa181",
        "timestamp": "2026-11-27 20:22:16",
        "model": "LFM3-Omni-2B-Q4_K_M",
        "msgs": "12",
        "tokens": "4,820",
        "focus": "4B Llama Q4_K_M models for tooling, vision & coding",
    },
    {
        "session_id": "session_44bf05c0dc5046c59bd265a2db95466e",
        "timestamp": "2026-11-27 15:20:02",
        "model": "Qwen2.5-Coder-7B-Instruct-Q8_0",
        "msgs": "8",
        "tokens": "3,190",
        "focus": "Technical takeaways, code fixes & user preferences",
    },
    {
        "session_id": "session_91c08e4a2d7b41f6a39d0e11b82f5504",
        "timestamp": "2026-11-26 22:05:40",
        "model": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M",
        "msgs": "16",
        "tokens": "7,640",
        "focus": "llama.cpp router concurrency & OpenVINO NPU fallback",
    },
)

_CAPABILITY_ICONS: dict[str, tuple[str, str, str]] = {
    "tools": ("🔧", "🛠️", "🚫"),
    "function_calls": ("⚡", "❓", "➖"),
    "structured_output": ("📐", "❔", "➖"),
    "thinking": ("🧠", "💭", "➖"),
    "vision": ("👁️", "🙈", "🙈"),
    "audio": ("🎙️", "🔇", "🔇"),
}


# ---------------------------------------------------------------------------
# Terminal & Resize Utilities
# ---------------------------------------------------------------------------


def _sync_console_size() -> tuple[int, int]:
    """Return current (width, height) while refreshing on resize signals."""
    try:
        cols, rows = os.get_terminal_size()
        if cols > 0 and rows > 0:
            console._width = cols
            console._height = rows
            return cols, rows
    except OSError:
        pass
    return console.size.width, console.size.height


def _install_sigwinch_handler() -> None:
    """Refresh console dimensions on SIGWINCH."""
    if hasattr(signal, "SIGWINCH"):
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGWINCH, lambda _signum, _frame: _sync_console_size())


_install_sigwinch_handler()


def _panel_width(min_width: int = 32) -> int:
    """Compute a safe full-width panel width for the current terminal."""
    width, _ = _sync_console_size()
    return max(min_width, width)


def _make_panel(
    renderable: RenderableType,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    border_style: str = "blue",
    padding: tuple[int, int] = (0, 1),
    height: int | None = None,
    style: str = "",
) -> Panel:
    """Create a responsive full-width Rich Panel with consistent defaults."""
    return Panel(
        renderable,
        title=title,
        subtitle=subtitle,
        border_style=border_style,
        padding=padding,
        height=height,
        style=style,
        width=_panel_width(),
    )


@contextlib.contextmanager
def _raw_terminal_input() -> Iterator[int | None]:
    """Put stdin into cbreak mode when attached to a TTY and restore on exit."""
    if not sys.stdin.isatty():
        yield None
        return
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _read_ansi_escape(fd: int) -> str:
    """Read a complete CSI/SS3 escape sequence after ESC (0x1b)."""
    first = os.read(fd, 1).decode("utf-8", errors="ignore")
    if first != "[":
        return first
    seq = ["["]
    while len(seq) < 12:
        nxt = os.read(fd, 1).decode("utf-8", errors="ignore")
        if not nxt:
            break
        seq.append(nxt)
        if nxt == "~" or nxt.isalpha():
            break
    return "".join(seq)


def _read_key_event() -> str:
    """Read a single key or ANSI escape sequence from stdin."""
    with _raw_terminal_input() as fd:
        if fd is None:
            line = sys.stdin.readline()
            if not line:
                raise EOFError
            return "\r"
        ch = os.read(fd, 1).decode("utf-8", errors="ignore")
        if ch == "\x1b":
            return "\x1b" + _read_ansi_escape(fd)
        return ch


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


StreamFn = Callable[[str, str], Iterable[str]]


def _default_model_stream(prompt: str, model: str) -> Iterable[str]:
    """Default token stream used when no external inference backend is injected."""
    if prompt.strip().lower() == "hi":
        text = (
            "Hi! I can see two recent episodic memories stored:\n\n"
            "1. **Episode 78** (`session_dfa567cbee7f4ab88227ee99907aa181`, 2026-11-27 20:22:16) — "
            "Focused on 4B Llama Q4_K_M models supporting tooling, vision, and coding.\n"
            "2. **Episode 35** (`session_44bf05c0dc5046c59bd265a2db95466e`, 2026-11-27 15:20:02) — "
            "Focused on technical takeaways, code solutions, and user preferences.\n\n"
            "Would you like me to retrieve the full content of either episode?"
        )
    else:
        text = f"I'll handle `{prompt}` on **{model}** using a plan/act/verify step and report back."

    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


@dataclass(frozen=True)
class FolderPermissionReport:
    path: Path
    exists: bool
    is_dir: bool
    readable: bool
    writable: bool
    executable: bool
    mode_octal: str
    error: str | None = None

    @property
    def accessible(self) -> bool:
        return (
            self.exists
            and self.is_dir
            and self.readable
            and self.writable
            and self.executable
        )


@dataclass
class AgentConfig:
    workspace: Path = Path(
        "/home/zieganwesley/Documents/PROGRAM_FILES/PYTHON/Private_Agent/workspace"
    )
    rag_dir: Path = Path("/home/zieganwesley/.config/private_agent/rag")
    runtime: str = "llama.cpp"
    endpoint: str = "http://127.0.0.1:9931"
    embedding_model: str = "embeddinggemma-300M-Q8_0"
    latest_session: str | None = "session_dfa567cbee7f4ab88227ee99907aa181"
    recent_sessions: tuple[dict[str, str], ...] = DEFAULT_RECENT_SESSIONS
    local_models: list[str] = field(
        default_factory=lambda: [
            "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M",
            "LFM3-Omni-2B-Q4_K_M",
            "Phi-4-mini-instruct-Q4_K_M",
            "Qwen2.5-Coder-7B-Instruct-Q8_0",
            "bge-reranker-v2-m3-Q4_K_M",
            "qwen3-5-2B-Q4_K_M",
            "senter-lora-500",
        ]
    )
    default_model: str = "LFM3-Omni-2B-Q4_K_M"
    isolation: str = "Available — bubblewrap + prlimit"
    skills_count: int = 0
    banner_font: str = "slant"
    ctx_max: int = 8192
    model_stream_fn: StreamFn = _default_model_stream


@dataclass
class SessionState:
    workspace: Path | None
    rag_dir: Path | None
    workspace_enabled: bool = True
    rag_enabled: bool = True
    provider: str = "Local (llama.cpp)"
    model: str = "LFM3-Omni-2B-Q4_K_M"
    permission: str = "Full"
    session_id: str | None = None
    rag_active: bool = False
    rag_detail: str = "disabled"
    isolation: str = "Available — bubblewrap + prlimit"
    skills_count: int = 0
    thinking_enabled: bool = False
    thinking_effort: str = "med"
    mode: str = "agentic"
    bg_tasks: int = 0
    tools_used: int = 0
    ctx_used: int = 1280
    ctx_max: int = 8192
    internet_online: bool = True
    internet_allowed: bool = True
    processing: bool = False
    capabilities: dict[str, str] = field(
        default_factory=lambda: {
            "tools": "unknown",
            "function_calls": "unknown",
            "structured_output": "unknown",
            "thinking": "unknown",
            "vision": "no",
            "audio": "no",
        }
    )
    started_at: float = field(default_factory=time.perf_counter)
    time_in_model_calls: float = 0.0
    prompts_count: int = 0
    model_calls: int = 0
    model_constructions: int = 1
    suspected_cold_loads: int = 1
    input_tokens: int = 1280
    output_tokens: int = 0
    tool_breakdown: dict[str, int] = field(default_factory=dict)
    model_call_counts: dict[str, int] = field(default_factory=dict)
    summary_tokens: int = 0
    summary_time: float = 0.0
    summary_stored_sqlite: bool = False

    @property
    def active_workspace_label(self) -> str:
        return (
            str(self.workspace)
            if (self.workspace_enabled and self.workspace)
            else "disabled"
        )

    @property
    def active_rag_label(self) -> str:
        return str(self.rag_dir) if (self.rag_enabled and self.rag_dir) else "disabled"

    def record_model_call(
        self, model_name: str, in_tok: int, out_tok: int, elapsed: float
    ) -> None:
        self.prompts_count += 1
        self.model_calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.time_in_model_calls += elapsed
        self.model_call_counts[model_name] = (
            self.model_call_counts.get(model_name, 0) + 1
        )

    def record_tool_use(self, tool_name: str = "agent_loop") -> None:
        self.tools_used += 1
        self.tool_breakdown[tool_name] = self.tool_breakdown.get(tool_name, 0) + 1


# ---------------------------------------------------------------------------
# Reusable UI Builders & Formatters
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m {secs}s"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def format_capabilities_emojis(caps: dict[str, str]) -> str:
    """Render capability flags as compact emoji indicators."""
    out: list[str] = []
    for key, (yes_icon, unk_icon, no_icon) in _CAPABILITY_ICONS.items():
        val = caps.get(key, "no")
        if val == "yes":
            out.append(yes_icon)
        elif val == "unknown":
            out.append(unk_icon)
        else:
            out.append(no_icon)
    return "".join(out)


def _build_kv_grid(
    rows: Iterable[tuple[str, str]],
    *,
    key_width: int = 18,
    padding: tuple[int, int] = (0, 2),
) -> Table:
    """Build a clean two-column key/value grid."""
    grid = Table.grid(padding=padding, expand=True)
    grid.add_column(style="bold dim", width=key_width, no_wrap=True)
    grid.add_column(style="bright_white")
    for k, v in rows:
        grid.add_row(k, v)
    return grid


def build_pyfiglet_banner(
    title: str = "Private Agent",
    font: str = "slant",
    subtitle: str | None = None,
) -> Panel:
    """Build a responsive pyfiglet ASCII banner inside a full-width panel."""
    width = _panel_width()
    raw_art = pyfiglet.figlet_format(title, font=font).rstrip("\n")
    max_art_len = max((len(line) for line in raw_art.splitlines()), default=0)

    art_text = (
        Text(title.upper(), style="bold cyan")
        if max_art_len > max(20, width - 6)
        else Text(raw_art, style="bold cyan", no_wrap=True)
    )
    body = (
        Align.center(
            Text.assemble(art_text, "\n", Text(subtitle, style="dim italic"))
        )
        if subtitle
        else Align.center(art_text)
    )
    return _make_panel(body, border_style="bright_blue", padding=(0, 2))


def build_ready_header(state: SessionState, font: str = "slant") -> Group:
    """Build the Ready banner + Model & Runtime Info panel."""
    banner = build_pyfiglet_banner(
        title="Private Agent",
        font=font,
        subtitle="READY · Local-first AI assistant with repository intelligence, memory and tools",
    )

    if not state.rag_enabled:
        rag_status = "[dim]disabled[/dim]"
    elif state.rag_active:
        rag_status = "[green]active[/green]"
    else:
        rag_status = f"[yellow]inactive[/yellow] [dim]({state.rag_detail})[/dim]"

    caps_str = ", ".join(f"{k}={v}" for k, v in state.capabilities.items())

    left = _build_kv_grid(
        [
            ("Model", f"[bold bright_white]{state.model}[/bold bright_white]"),
            ("Provider", f"[cyan]{state.provider}[/cyan]"),
            ("Permission", f"[yellow]{state.permission}[/yellow]"),
            ("Isolation", f"[green]{state.isolation}[/green]"),
        ],
        key_width=12,
        padding=(0, 1),
    )
    right = _build_kv_grid(
        [
            ("Workspace", f"[dim]{state.active_workspace_label}[/dim]"),
            ("RAG Dir", f"[dim]{state.active_rag_label}[/dim]"),
            ("Local RAG", rag_status),
            ("Skills / Caps", f"{state.skills_count} · [dim]{caps_str}[/dim]"),
        ],
        key_width=13,
        padding=(0, 1),
    )

    info_panel = _make_panel(
        Columns([left, right], padding=(0, 2), expand=True),
        title="[bold bright_green]Model & Runtime Info[/bold bright_green]",
        subtitle="[dim]Commands: /help · /mode · /think · /net · /up · /down · switch · exit[/dim]",
        border_style="bright_green",
        padding=(0, 1),
    )
    return Group(banner, info_panel)


def render_grouped_session_statistics(state: SessionState) -> Panel:
    """Render session statistics grouped into logical sections."""
    elapsed_total = time.perf_counter() - state.started_at
    total_tokens = state.input_tokens + state.output_tokens

    tools_summary = (
        f"{state.tools_used} ({', '.join(f'{k} x{v}' for k, v in state.tool_breakdown.items())})"
        if state.tool_breakdown
        else str(state.tools_used)
    )
    models_summary = (
        ", ".join(f"{m} ({c} calls)" for m, c in state.model_call_counts.items())
        if state.model_call_counts
        else f"{state.model} (0 calls)"
    )

    table = Table(
        show_header=True, header_style="bold cyan", box=None, padding=(0, 2), expand=True
    )
    table.add_column("Metric", style="white", min_width=22)
    table.add_column("Value", style="bright_white")

    sections: list[tuple[str, list[tuple[str, str]]]] = [
        (
            "Session & Environment",
            [
                ("Session ID", state.session_id or "new_ephemeral_session"),
                ("Session mode", state.mode.capitalize()),
                ("Permission mode", state.permission.lower()),
                ("Workspace", state.active_workspace_label),
                (
                    "Local RAG",
                    f"on ({state.rag_dir})"
                    if (state.rag_enabled and state.rag_dir)
                    else "off",
                ),
                ("Skills loaded", str(state.skills_count)),
                ("Resumed history messages", "4" if state.session_id else "0"),
            ],
        ),
        (
            "Timing & Tokens",
            [
                ("Total session time", _format_duration(elapsed_total)),
                ("Time in model calls", _format_duration(state.time_in_model_calls)),
                ("Prompts this session", str(state.prompts_count)),
                ("Input tokens", f"{state.input_tokens:,}"),
                ("Output tokens", f"{state.output_tokens:,}"),
                ("Total tokens", f"{total_tokens:,}"),
                (
                    "Token accuracy",
                    f"{state.model_calls} of {state.model_calls} call(s) estimated (local router)",
                ),
            ],
        ),
        (
            "Model, Tools & Memory",
            [
                ("Model provider", state.provider),
                ("Active model", state.model),
                (
                    "Thinking",
                    f"{'on' if state.thinking_enabled else 'off'} (effort: {state.thinking_effort})",
                ),
                ("Model calls", str(state.model_calls)),
                ("Model constructions", str(state.model_constructions)),
                ("Suspected cold loads", str(state.suspected_cold_loads)),
                ("Tool uses", tools_summary),
                ("Models used", models_summary),
                ("Summary tokens generated", str(state.summary_tokens)),
                ("Summary generation time", f"{state.summary_time:.2f}s"),
                (
                    "Summary stored in SQLite",
                    "yes" if state.summary_stored_sqlite else "no",
                ),
            ],
        ),
    ]

    for idx, (group_title, rows) in enumerate(sections):
        if idx > 0:
            table.add_row("", "")
        table.add_row(f"[bold magenta]── {group_title} ──[/bold magenta]", "")
        for metric, val in rows:
            table.add_row(metric, val)

    return _make_panel(
        table,
        title="[bold bright_cyan]Session Statistics[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Folder Permissions & Interactive Directory Picker
# ---------------------------------------------------------------------------


def check_folder_permissions(target: Path) -> FolderPermissionReport:
    """Inspect directory existence and POSIX r/w/x access flags."""
    resolved = target.expanduser()
    try:
        st = resolved.stat()
        exists = True
        is_dir = stat.S_ISDIR(st.st_mode)
        mode_octal = oct(stat.S_IMODE(st.st_mode))
    except OSError as exc:
        return FolderPermissionReport(
            path=resolved,
            exists=False,
            is_dir=False,
            readable=False,
            writable=False,
            executable=False,
            mode_octal="----",
            error=str(exc),
        )

    return FolderPermissionReport(
        path=resolved,
        exists=exists,
        is_dir=is_dir,
        readable=os.access(resolved, os.R_OK),
        writable=os.access(resolved, os.W_OK),
        executable=os.access(resolved, os.X_OK),
        mode_octal=mode_octal,
        error=None if is_dir else "Target exists but is not a directory",
    )


def render_permission_badge_table(
    report: FolderPermissionReport, label: str = "Folder"
) -> Panel:
    """Render a permission matrix panel for a target path."""
    table = Table(
        show_header=True, header_style="bold cyan", box=None, padding=(0, 2), expand=True
    )
    table.add_column("Check", style="bold dim")
    table.add_column("Status")

    checks: list[tuple[str, bool, str, str]] = [
        ("Path exists", report.exists, "EXISTS", "MISSING"),
        ("Directory type", report.is_dir, "DIR", "NOT DIR"),
        ("Read entries", report.readable, "READ (r)", "NO READ"),
        ("Create/modify files", report.writable, "WRITE (w)", "NO WRITE"),
        ("Enter/traverse folder", report.executable, "EXEC/TRAVERSE (x)", "NO EXEC"),
    ]
    for desc, ok, yes_txt, no_txt in checks:
        badge = (
            f"[bold green]✓ {yes_txt}[/bold green]"
            if ok
            else f"[bold red]✗ {no_txt}[/bold red]"
        )
        table.add_row(desc, badge)

    table.add_row("POSIX Mode", f"[bright_white]{report.mode_octal}[/bright_white]")
    if report.error:
        table.add_row("Note", f"[yellow]{report.error}[/yellow]")

    overall = (
        "[bold bright_green]ACCESSIBLE[/bold bright_green]"
        if report.accessible
        else "[bold bright_red]RESTRICTED / INACCESSIBLE[/bold bright_red]"
    )
    return _make_panel(
        table,
        title=f"[bold]{label} Permissions[/bold] · {overall}",
        subtitle=f"[dim]{report.path}[/dim]",
        border_style="green" if report.accessible else "red",
        padding=(0, 1),
    )


def _list_subdirs(current: Path) -> list[Path]:
    """Return sorted visible subdirectories under current, skipping unreadable entries."""
    entries: list[Path] = []
    with contextlib.suppress(OSError):
        for child in sorted(current.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            with contextlib.suppress(OSError):
                if child.is_dir():
                    entries.append(child)
    return entries


def _build_nav_entries(current: Path, max_subdirs: int) -> list[tuple[str, Path, str]]:
    """Build parent + visible subdirectory entries with rwx permission tags."""
    items: list[tuple[str, Path, str]] = [(".. (parent directory)", current.parent, "up")]
    for d in _list_subdirs(current)[:max_subdirs]:
        rep = check_folder_permissions(d)
        tag = (
            f"[green]{'r' if rep.readable else '-'}"
            f"{'w' if rep.writable else '-'}"
            f"{'x' if rep.executable else '-'}[/green]"
        )
        items.append((f"{d.name}/", d, tag))
    return items


def pick_directory_path(
    title: str,
    initial_path: Path,
    presets: list[tuple[str, Path]] | None = None,
) -> Path:
    """Interactive directory browser with keyboard navigation, manual entry, and rwx validation."""
    start_dir = initial_path.expanduser()
    current = (
        start_dir
        if start_dir.is_dir()
        else (start_dir.parent if start_dir.parent.is_dir() else Path.home())
    )
    cursor = 0

    def _render_picker(cur_dir: Path, idx: int) -> Panel:
        _, height = _sync_console_size()
        nav_items = _build_nav_entries(cur_dir, max_subdirs=max(4, height - 13))
        clamped = max(0, min(idx, len(nav_items) - 1))

        rows = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        rows.add_column("Ptr", width=2, no_wrap=True)
        rows.add_column("Entry")
        rows.add_column("Access", justify="right", no_wrap=True)

        for i, (label, _, tag) in enumerate(nav_items):
            ptr = "[bold cyan]›[/bold cyan]" if i == clamped else " "
            style = "bold bright_white" if i == clamped else "white"
            rows.add_row(ptr, f"[{style}]{label}[/{style}]", tag)

        cur_rep = check_folder_permissions(cur_dir)
        status_str = (
            "[green]rwx ok[/green]"
            if cur_rep.accessible
            else f"[red]r:{int(cur_rep.readable)} w:{int(cur_rep.writable)} x:{int(cur_rep.executable)}[/red]"
        )

        parts: list[RenderableType] = [
            Text.from_markup(
                f"Current folder: [bold cyan]{cur_dir}[/bold cyan]  ({status_str})"
            ),
            rows,
        ]
        if presets:
            parts.append(
                Text(
                    "Presets: " + " | ".join(f"{name}={p}" for name, p in presets),
                    style="dim",
                )
            )

        return _make_panel(
            Group(*parts),
            title=f"[bold bright_white]{title}[/bold bright_white]",
            subtitle="[dim]↑/↓ move · Enter open dir · c use dir · t type path · q keep default[/dim]",
            border_style="blue",
            padding=(0, 1),
        )

    console.clear()
    with Live(
        _render_picker(current, cursor),
        console=console,
        refresh_per_second=12,
        transient=True,
    ) as live:
        while True:
            _, height = _sync_console_size()
            nav_items = _build_nav_entries(current, max_subdirs=max(4, height - 13))
            cursor = max(0, min(cursor, len(nav_items) - 1))
            live.update(_render_picker(current, cursor))

            key = _read_key_event()
            if key in ("\x1b[A", "k"):
                cursor = (cursor - 1) % len(nav_items)
            elif key in ("\x1b[B", "j"):
                cursor = (cursor + 1) % len(nav_items)
            elif key in ("\r", "\n"):
                chosen_target = nav_items[cursor][1]
                if chosen_target.is_dir():
                    current = chosen_target.resolve()
                    cursor = 0
            elif key.lower() == "c":
                break
            elif key.lower() == "q":
                current = start_dir
                break
            elif key.lower() == "t":
                live.stop()
                typed = Prompt.ask(
                    "[bold green]Enter directory path[/bold green]", default=str(current)
                )
                current = Path(typed.strip() or str(current)).expanduser()
                cursor = 0
                console.clear()
                live.start()
            elif key == "\x03":
                raise KeyboardInterrupt

    console.clear()
    if not current.exists() and Confirm.ask(
        f"Directory [cyan]{current}[/cyan] does not exist. Create it?",
        default=True,
    ):
        try:
            current.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            console.print(f"[bold red]Failed to create directory:[/bold red] {exc}")

    report = check_folder_permissions(current)
    console.print(render_permission_badge_table(report, label=title))
    if not report.accessible and Confirm.ask(
        "Permissions look restricted. Enter a different path?",
        default=False,
    ):
        return pick_directory_path(title, initial_path=Path.home(), presets=presets)

    return current


# ---------------------------------------------------------------------------
# Configuration Steps & Gates
# ---------------------------------------------------------------------------


def _build_config_process_banner(step_idx: int, detail: str = "") -> Panel:
    """Banner shown across configuration screens announcing active configuration."""
    badges: list[str] = []
    for i, label in enumerate(CONFIG_STEPS):
        if i < step_idx:
            badges.append(f"[dim green]✓ {label}[/dim green]")
        elif i == step_idx:
            badges.append(f"[bold bright_cyan]▸ {label}[/bold bright_cyan]")
        else:
            badges.append(f"[dim]{label}[/dim]")

    parts: list[RenderableType] = [
        Text.from_markup(
            "[bold bright_yellow]⚙  Configuration under Process[/bold bright_yellow]"
        ),
        Text.from_markup("  ·  ".join(badges)),
    ]
    if detail:
        parts.append(Text(detail, style="dim"))

    return _make_panel(
        Group(*parts),
        title="[bold blue]Private Agent Setup[/bold blue]",
        border_style="bright_blue",
        padding=(0, 1),
    )


def _isolate_current_config_view(step_idx: int, status_note: str = "") -> None:
    """Clear screen and render only the active step banner."""
    _sync_console_size()
    console.clear()
    console.print(_build_config_process_banner(step_idx, status_note))


def select_menu_option(
    title: str,
    options: list[tuple[str, str]],
    default_index: int = 0,
) -> int:
    """Display a full-width numbered option list and return the chosen 0-based index."""
    table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    table.add_column("Key", style="bold cyan", width=5, justify="right", no_wrap=True)
    table.add_column("Option", style="white")
    table.add_column("Details", style="dim")

    for idx, (label, detail) in enumerate(options):
        row_style = "bold bright_white" if idx == default_index else "white"
        table.add_row(f"[{idx + 1}]", f"[{row_style}]{label}[/{row_style}]", detail)

    console.print()
    console.print(
        _make_panel(
            table, title=f"[bold bright_white]{title}[/bold bright_white]", border_style="blue"
        )
    )

    valid = [str(i) for i in range(1, len(options) + 1)]
    raw = IntPrompt.ask(
        "[bold green]Select option[/bold green]",
        choices=valid,
        default=default_index + 1,
        show_choices=False,
    )
    return raw - 1


def _render_recent_sessions_table(sessions: list[dict[str, str]]) -> Panel:
    """Render numbered recent sessions across current width."""
    table = Table(
        show_header=True, header_style="bold cyan", box=None, padding=(0, 1), expand=True
    )
    table.add_column("#", style="bold cyan", width=5, justify="right", no_wrap=True)
    table.add_column("Session ID", style="cyan")
    table.add_column("When", style="dim")
    table.add_column("Model", style="white")
    table.add_column("Msgs", justify="right", style="bright_white", no_wrap=True)
    table.add_column("Focus", style="dim")

    for idx, item in enumerate(sessions, start=1):
        tag = "[1] latest" if idx == 1 else f"[{idx}]"
        short_id = item["session_id"][:18] + "…"
        table.add_row(
            tag,
            short_id,
            item["timestamp"],
            item["model"],
            item["msgs"],
            item["focus"],
        )

    return _make_panel(
        table,
        title="[bold bright_white]Recent Sessions[/bold bright_white]",
        subtitle="[dim]y or 1 = resume latest · 2..N = resume older session · n = start new[/dim]",
        border_style="dim blue",
        padding=(0, 1),
    )


def _prompt_session_selection(cfg: AgentConfig) -> str | None:
    """Display recent sessions and accept 'y' (latest), a session number, or 'n' (new)."""
    sessions = list(cfg.recent_sessions)
    if cfg.latest_session and (
        not sessions or sessions[0]["session_id"] != cfg.latest_session
    ):
        sessions.insert(
            0,
            {
                "session_id": cfg.latest_session,
                "timestamp": "latest",
                "model": cfg.default_model,
                "msgs": "—",
                "tokens": "—",
                "focus": "Latest recorded session",
            },
        )

    if not sessions:
        return None

    console.print(_render_recent_sessions_table(sessions))
    valid_nums = {str(i) for i in range(1, len(sessions) + 1)}

    while True:
        raw = (
            Prompt.ask(
                f"[bold green]Resume latest (y), pick session # (1-{len(sessions)}), or start new (n)?[/bold green]",
                default="n",
            )
            .strip()
            .lower()
        )
        if raw in {"y", "yes"}:
            chosen = sessions[0]["session_id"]
            console.print(f"[cyan]Resuming latest session:[/cyan] {chosen}")
            return chosen
        if raw in {"n", "no"}:
            console.print("[dim]Starting a new session.[/dim]")
            return None
        if raw in valid_nums:
            chosen = sessions[int(raw) - 1]["session_id"]
            console.print(f"[cyan]Resuming session #{raw}:[/cyan] {chosen}")
            return chosen
        console.print(
            f"[yellow]Enter 'y', 'n', or a number from 1 to {len(sessions)}.[/yellow]"
        )


def configure_workspace_step(cfg: AgentConfig) -> tuple[bool, Path | None]:
    """Step 1: Workspace directory enablement and path selection."""
    _isolate_current_config_view(0, "Configure project workspace folder.")
    if not Confirm.ask(
        "Enable [bold cyan]Workspace[/bold cyan] directory?", default=True
    ):
        console.print("[dim]Workspace disabled for this session.[/dim]")
        return False, None

    chosen = pick_directory_path(
        title="Select Workspace Directory",
        initial_path=cfg.workspace,
        presets=[("Default WS", cfg.workspace), ("Home", Path.home())],
    )
    return True, chosen


def configure_rag_and_runtime_step(
    cfg: AgentConfig,
) -> tuple[bool, Path | None, bool, str]:
    """Step 2: Local RAG directory and runtime probe."""
    _isolate_current_config_view(1, "Configure local RAG knowledge base and runtime.")
    if not Confirm.ask("Enable [bold cyan]Local RAG[/bold cyan]?", default=True):
        with console.status("[cyan]Checking local runtime...[/cyan]", spinner="dots"):
            time.sleep(0.2)
        console.print("[dim]Local RAG disabled; continuing without RAG.[/dim]")
        return False, None, False, "disabled"

    rag_dir = pick_directory_path(
        title="Select RAG Documents Directory",
        initial_path=cfg.rag_dir,
        presets=[("Default RAG", cfg.rag_dir), ("Workspace", cfg.workspace)],
    )

    with Live(
        Group(
            Text(f"  • RAG Dir:  {rag_dir}", style="dim"),
            Text(f"  • Runtime:  {cfg.runtime}", style="dim"),
        ),
        console=console,
        refresh_per_second=10,
        transient=True,
    ):
        time.sleep(0.25)

    rag_ok = False
    rag_detail = f"{cfg.embedding_model} @ {cfg.endpoint} -> InternalServerError"
    console.print(
        f"[yellow]RAG fallback:[/yellow] Ollama unavailable; using OpenAI-compatible endpoint "
        f"([bold]{cfg.embedding_model}[/bold] @ {cfg.endpoint})."
    )
    console.print(
        "[yellow]RAG warning:[/yellow] Embedding runtime unavailable (InternalServerError); continuing without RAG."
    )
    return True, rag_dir, rag_ok, rag_detail


def resolve_session_and_permissions(cfg: AgentConfig) -> tuple[str | None, str]:
    """Step 3: Session selection and tool permission policy."""
    _isolate_current_config_view(2, "Select session resume and tool permission policy.")
    session_id = _prompt_session_selection(cfg)

    perm_options = [
        ("Manual", "ask before every tool action"),
        ("Auto", "read/search freely; ask before edits and command execution"),
        ("Full", "approve all tool calls for this session"),
    ]
    permission = perm_options[
        select_menu_option("Permission Mode", perm_options, default_index=2)
    ][0]

    if permission == "Full":
        console.print(
            _make_panel(
                "Full mode approves every tool call for this session: file writes, shell commands, "
                "Git operations, and stored-data changes run without another prompt. Internet access "
                "still needs a one-time approval when first requested.",
                title="[bold yellow]Permission Mode: Full[/bold yellow]",
                border_style="yellow",
            )
        )
        confirmation = Prompt.ask(
            "Type [bold]yes[/bold] to run without further tool approvals",
            default="yes",
        )
        if confirmation.strip().lower() != "yes":
            permission = "Manual"
            console.print("[dim]Falling back to Manual permission mode.[/dim]")

    return session_id, permission


def select_provider_and_model(cfg: AgentConfig, step_idx: int = 3) -> tuple[str, str]:
    """Step 4: Choose provider and target model."""
    _isolate_current_config_view(step_idx, "Choose runtime provider and target model.")

    provider_options = [
        (f"Local ({cfg.runtime})", f"{len(cfg.local_models)} model(s) (last used)"),
        ("Online", "enter details manually"),
        ("Exit", "quit Private Agent"),
    ]
    p_idx = select_menu_option("Model Provider", provider_options, default_index=0)

    if p_idx == 2:
        return "Exit", ""
    if p_idx == 1:
        custom_model = Prompt.ask("[bold green]Enter online model name[/bold green]")
        return "Online", custom_model.strip()

    model_options = [(name, "GGUF local") for name in cfg.local_models]
    default_idx = (
        cfg.local_models.index(cfg.default_model)
        if cfg.default_model in cfg.local_models
        else 0
    )
    m_idx = select_menu_option("Select Model", model_options, default_index=default_idx)
    return f"Local ({cfg.runtime})", cfg.local_models[m_idx]


def initialize_agent(cfg: AgentConfig) -> SessionState | None:
    """Execute setup steps and return a ready SessionState, or None if exited."""
    ws_enabled, workspace = configure_workspace_step(cfg)
    rag_enabled, rag_dir, rag_active, rag_detail = configure_rag_and_runtime_step(cfg)
    session_id, permission = resolve_session_and_permissions(cfg)
    provider, model = select_provider_and_model(cfg, step_idx=3)

    if provider == "Exit":
        return None

    _isolate_current_config_view(3, f"Initializing client for {model}...")
    with console.status(f"[cyan]Loading {model}...[/cyan]", spinner="dots"):
        time.sleep(0.25)

    console.clear()
    return SessionState(
        workspace=workspace,
        rag_dir=rag_dir,
        workspace_enabled=ws_enabled,
        rag_enabled=rag_enabled,
        provider=provider,
        model=model,
        permission=permission,
        session_id=session_id,
        rag_active=rag_active,
        rag_detail=rag_detail,
        isolation=cfg.isolation,
        skills_count=cfg.skills_count,
        ctx_max=cfg.ctx_max,
    )


def _show_bye_and_exit(font: str) -> None:
    """Display a BYE banner for 3 seconds before exiting."""
    _sync_console_size()
    console.clear()
    console.print(
        build_pyfiglet_banner(
            title="BYE",
            font=font,
            subtitle="No active local or online provider confirmed — exiting in 3s...",
        )
    )
    time.sleep(3)


def run_provider_readiness_gate(cfg: AgentConfig) -> bool:
    """Confirm once whether a local or online provider instance is ready."""
    _sync_console_size()
    console.clear()
    console.print(
        build_pyfiglet_banner(
            title="Private Agent",
            font=cfg.banner_font,
            subtitle="Local-first AI assistant · Provider readiness check",
        )
    )
    console.print(
        _make_panel(
            "Confirm if a local provider instance is running and active to use, "
            "or an online provider endpoint/key is ready to consume.",
            title="[bold bright_cyan]Provider Readiness[/bold bright_cyan]",
            border_style="bright_cyan",
        )
    )

    ready = Confirm.ask(
        "[bold green]Is a local provider instance running/active or an online provider ready to use?[/bold green]",
        default=True,
    )
    if not ready:
        _show_bye_and_exit(cfg.banner_font)
        return False
    return True


def confirm_enter_chat(state: SessionState) -> str:
    """Gate after configuration before entering ChatScreenUI. Returns 'enter' or 'reconfigure'."""
    _sync_console_size()
    console.clear()
    summary = _build_kv_grid(
        [
            ("Provider / Model", f"{state.provider} · {state.model}"),
            ("Permission Mode", state.permission),
            ("Workspace", state.active_workspace_label),
            ("Local RAG", state.active_rag_label),
            ("Session", state.session_id or "new session"),
        ]
    )
    console.print(
        _make_panel(
            summary,
            title="[bold bright_green]Configuration Complete — Ready for Chat[/bold bright_green]",
            subtitle="[dim]Enter / y: open chat window · r: re-run config steps[/dim]",
            border_style="bright_green",
        )
    )

    action = (
        Prompt.ask(
            "[bold green]Enter chat window now?[/bold green] ([bold]y[/bold]=enter, [bold]r[/bold]=reconfigure)",
            choices=["y", "yes", "r", "reconfigure", ""],
            default="y",
            show_choices=False,
        )
        .strip()
        .lower()
    )
    return "reconfigure" if action in {"r", "reconfigure"} else "enter"


# ---------------------------------------------------------------------------
# Chat Screen UI & Turn Execution
# ---------------------------------------------------------------------------


class ChatScreenUI:
    """Full-screen layout with scrollable chat history, input box, and bottom ribbon."""

    def __init__(self, state: SessionState, banner_font: str = "slant") -> None:
        self.state = state
        self.banner_font = banner_font
        self.history: deque[RenderableType] = deque(maxlen=250)
        self.streaming_renderable: RenderableType | None = None
        self.input_buffer: str = ""
        self.prompt_active: bool = False
        self.scroll_offset: int = 0
        self._last_max_offset: int = 0
        self._height_cache: dict[tuple[int, int], int] = {}
        self._cached_width: int = 0

        self.history.append(build_ready_header(state, font=banner_font))

        self.layout = Layout()
        self.layout.split_column(
            Layout(name="chat", ratio=1),
            Layout(name="input", size=3),
            Layout(name="ribbon", size=5),
        )
        self.live = Live(
            self._render(),
            console=console,
            screen=True,
            refresh_per_second=15,
        )

    def _chat_pane_height(self) -> int:
        _, height = _sync_console_size()
        return max(height - 8, 6)

    def _estimate_lines(self, item: RenderableType, eff_w: int) -> int:
        key = (id(item), eff_w)
        cached = self._height_cache.get(key)
        if cached is not None:
            return cached

        if isinstance(item, Text):
            lines = item.plain.splitlines() or [""]
            est = sum(max(1, (len(line) + eff_w - 1) // eff_w) for line in lines)
        elif isinstance(item, Rule):
            est = 1
        elif isinstance(item, Tree):
            est = 5
        elif isinstance(item, Group):
            est = sum(self._estimate_lines(child, eff_w) for child in item.renderables)
        elif isinstance(item, Markdown):
            raw = getattr(item, "markup", "")
            lines = raw.splitlines() or [" "]
            est = max(3, sum(max(1, (len(line) + eff_w - 1) // eff_w) for line in lines) + 1)
        else:
            est = 4

        self._height_cache[key] = est
        return est

    def scroll_by(self, delta: int) -> None:
        self.scroll_offset = max(
            0, min(self._last_max_offset, self.scroll_offset + delta)
        )
        self.refresh()

    def scroll_to_top(self) -> None:
        self.scroll_offset = self._last_max_offset
        self.refresh()

    def scroll_to_bottom(self) -> None:
        self.scroll_offset = 0
        self.refresh()

    def _compute_visible_items(
        self, items: list[RenderableType], pane_h: int, pane_w: int
    ) -> tuple[list[RenderableType], int, int]:
        total = len(items)
        if total == 0:
            return [], 0, 0

        eff_w = max(pane_w - 4, 20)
        if eff_w != self._cached_width:
            self._height_cache.clear()
            self._cached_width = eff_w

        avail_lines = max(4, pane_h - 2)
        max_offset = max(0, total - 1)
        self._last_max_offset = max_offset
        if self.scroll_offset > max_offset:
            self.scroll_offset = max_offset

        end_idx = total - self.scroll_offset
        used_lines = 0
        start_idx = end_idx
        while start_idx > 0:
            h = self._estimate_lines(items[start_idx - 1], eff_w)
            if used_lines + h > avail_lines and start_idx < end_idx:
                break
            used_lines += h
            start_idx -= 1

        return items[start_idx:end_idx], start_idx, end_idx

    def _render_chat_pane(self) -> Panel:
        width, _ = _sync_console_size()
        pane_h = self._chat_pane_height()
        pane_w = max(width - 2, 24)

        items: list[RenderableType] = list(self.history)
        if self.streaming_renderable is not None and self.scroll_offset == 0:
            items.append(self.streaming_renderable)

        visible, start_idx, end_idx = self._compute_visible_items(items, pane_h, pane_w)
        total = len(items)

        if self.scroll_offset > 0:
            subtitle = (
                f"[bold yellow]Scrolled +{self.scroll_offset} block(s) "
                f"({start_idx + 1}-{end_idx}/{total}) · ↓/PgDn or /bottom for live tail[/bold yellow]"
            )
            border = "yellow"
        else:
            subtitle = (
                f"[dim]Auto-scroll active · History: {total} block(s) · ↑/↓/PgUp/PgDn or /up /down[/dim]"
            )
            border = "dim blue"

        return Panel(
            Group(*visible),
            title="[bold]Conversation[/bold]",
            subtitle=subtitle,
            border_style=border,
            padding=(0, 1),
            height=pane_h,
        )

    def _render_input_pane(self) -> Panel:
        if self.prompt_active:
            content = Text.assemble(
                ("You › ", "bold cyan"),
                (self.input_buffer, "bold bright_white"),
                ("█", "bold bright_cyan"),
            )
            border = "cyan"
        else:
            content = Text("Model is responding...", style="dim italic")
            border = "dim"
        return Panel(content, border_style=border, padding=(0, 1))

    def _render_ribbon(self) -> Panel:
        s = self.state
        busy = s.processing

        bg = "#1b2432" if busy else "#173753"
        border = "dim bright_blue" if busy else "bright_cyan"
        val_style = "dim bright_white" if busy else "bold bright_white"
        accent = "dim cyan" if busy else "bold cyan"
        group_label = "dim magenta" if busy else "bold magenta"

        think_state = "ON" if s.thinking_enabled else "OFF"
        caps_emojis = format_capabilities_emojis(s.capabilities)
        net_icon = "🌐" if s.internet_online else "📴"
        net_perm = "allow" if s.internet_allowed else "block"
        ctx_pct = int((s.ctx_used / max(s.ctx_max, 1)) * 100)
        run_badge = "⏳ busy" if busy else "● ready"
        ws_flag = "on" if s.workspace_enabled else "off"
        rag_flag = "on" if s.rag_enabled else "off"

        row_top = (
            f"[{group_label}][Model & Mode][/{group_label}] "
            f"[{accent}]{s.provider}/{s.model}[/{accent}] · "
            f"Mode:[{val_style}]{s.mode}[/{val_style}] · "
            f"🧠Think:[{val_style}]{think_state}({s.thinking_effort})[/{val_style}]"
            f"   │   "
            f"[{group_label}][Safety & Execution][/{group_label}] "
            f"🔐Perm:[{val_style}]{s.permission}[/{val_style}] · "
            f"🧰Tools:[{val_style}]{s.tools_used}[/{val_style}] · "
            f"⚙️BG:[{val_style}]{s.bg_tasks}[/{val_style}] · "
            f"{net_icon}Net:[{val_style}]{net_perm}[/{val_style}]"
        )
        row_bottom = (
            f"[{group_label}][Context & Caps][/{group_label}] "
            f"📊Ctx:[{val_style}]{s.ctx_used}/{s.ctx_max} ({ctx_pct}%)[/{val_style}] · "
            f"📁WS:[{val_style}]{ws_flag}[/{val_style}] · "
            f"🔎RAG:[{val_style}]{rag_flag}[/{val_style}] · "
            f"📚Skills:[{val_style}]{s.skills_count}[/{val_style}] · "
            f"Caps:{caps_emojis} · "
            f"[{accent}][{run_badge}][/{accent}]"
        )

        return Panel(
            Text.from_markup(f"{row_top}\n{row_bottom}"),
            title="[dim]Live Status Ribbon[/dim]",
            style=f"on {bg}",
            border_style=border,
            padding=(0, 1),
        )

    def _render(self) -> Layout:
        _sync_console_size()
        self.layout["chat"].update(self._render_chat_pane())
        self.layout["input"].update(self._render_input_pane())
        self.layout["ribbon"].update(self._render_ribbon())
        return self.layout

    def refresh(self) -> None:
        self.live.update(self._render())

    def append(self, item: RenderableType) -> None:
        self.history.append(item)
        self.refresh()

    def set_streaming(self, item: RenderableType | None) -> None:
        self.streaming_renderable = item
        self.refresh()

    def set_processing(self, active: bool) -> None:
        self.state.processing = active
        self.refresh()

    def reset_with_banner(self) -> None:
        self.history.clear()
        self._height_cache.clear()
        self.scroll_offset = 0
        self.history.append(build_ready_header(self.state, font=self.banner_font))
        self.refresh()

    def read_line(self) -> str:
        """Read input live inside the bottom input box while handling scroll keys."""
        self.input_buffer = ""
        self.prompt_active = True
        self.refresh()

        with _raw_terminal_input() as fd:
            if fd is None:
                line = sys.stdin.readline()
                self.prompt_active = False
                self.refresh()
                if not line:
                    raise EOFError
                return line.strip()

            while True:
                ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                if ch in ("\r", "\n"):
                    result = self.input_buffer.strip()
                    self.input_buffer = ""
                    self.prompt_active = False
                    self.scroll_offset = 0
                    self.refresh()
                    return result
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch == "\x04":
                    raise EOFError
                if ch in ("\x7f", "\b"):
                    if self.input_buffer:
                        self.input_buffer = self.input_buffer[:-1]
                        self.refresh()
                elif ch == "\x1b":
                    seq = _read_ansi_escape(fd)
                    scroll_map = {"[A": 1, "[B": -1, "[5~": 4, "[6~": -4}
                    if seq in scroll_map:
                        self.scroll_by(scroll_map[seq])
                elif ch.isprintable():
                    self.input_buffer += ch
                    self.refresh()

    def __enter__(self) -> "ChatScreenUI":
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.live.stop()


def _stream_assistant_reply(
    ui: ChatScreenUI, model: str, chunks: Iterable[str]
) -> tuple[str, int]:
    buffer = ""
    token_est = 0

    for chunk in chunks:
        buffer += chunk
        token_est += 1
        ui.state.ctx_used = min(ui.state.ctx_max, ui.state.ctx_used + 1)
        ui.set_streaming(
            Group(
                Text.from_markup(f"[bold green]AI[/bold green] [dim]({model})[/dim]"),
                Markdown(buffer),
            )
        )

    ui.set_streaming(None)
    ui.append(
        Group(
            Text.from_markup(f"[bold green]AI[/bold green] [dim]({model})[/dim]"),
            Markdown(buffer),
        )
    )
    return buffer, token_est


def _execute_agent_loop_step(ui: ChatScreenUI, step_num: int, goal: str) -> None:
    ui.state.record_tool_use("agent_loop")
    tree = Tree(f"[bold cyan]Agent Loop · Step {step_num}[/bold cyan] — {goal}")
    tree.add("[dim]Plan:[/dim] Determine required action and parameters")
    tree.add("[dim]Act:[/dim] Dispatch tool or retrieval handler")
    tree.add("[dim]Verify:[/dim] Validate output against constraints")
    ui.append(tree)


def run_chat_turn(
    ui: ChatScreenUI,
    state: SessionState,
    user_input: str,
    stream_fn: StreamFn = _default_model_stream,
) -> None:
    """Execute a single chat turn with live streaming and telemetry recording."""
    ui.append(
        Text.from_markup(f"[bold bright_white]User:[/bold bright_white] {user_input}")
    )
    ui.set_processing(True)
    start = time.perf_counter()

    try:
        if user_input.strip().lower() != "hi":
            _execute_agent_loop_step(ui, step_num=1, goal=user_input.strip())

        reply, token_count = _stream_assistant_reply(
            ui, state.model, stream_fn(user_input, state.model)
        )
        elapsed = max(time.perf_counter() - start, 0.01)
        actual_tokens = max(len(reply.split()), token_count)
        in_tokens = max(1, len(user_input.split()) * 2)
        state.record_model_call(
            model_name=state.model,
            in_tok=in_tokens,
            out_tok=actual_tokens,
            elapsed=elapsed,
        )
        tok_s = actual_tokens / elapsed
        ui.append(
            Text(
                f"↳ {actual_tokens} tokens · {elapsed:.2f}s · {tok_s:.1f} tok/s",
                style="dim",
            )
        )
        ui.append(Rule(style="dim", characters="─"))
    finally:
        ui.set_processing(False)


def _handle_slash_command(cmd: str, state: SessionState, ui: ChatScreenUI) -> bool:
    """Handle runtime toggles and scroll commands. Returns True if handled."""
    parts = cmd.strip().split()
    if not parts:
        return False
    head = parts[0].lower()

    if head == "/help":
        ui.append(
            Text(
                "Commands: /help · /mode <chat|research|agentic> · "
                "/think <on|off> [low|med|high] · /net <allow|block> · "
                "/up [n] · /down [n] · /top · /bottom · switch · exit",
                style="dim",
            )
        )
        return True

    scroll_cmds: dict[str, Callable[[int], None]] = {
        "/up": lambda n: ui.scroll_by(n),
        "/down": lambda n: ui.scroll_by(-n),
    }
    if head in scroll_cmds:
        step = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 4
        scroll_cmds[head](step)
        return True

    if head == "/top":
        ui.scroll_to_top()
        return True
    if head == "/bottom":
        ui.scroll_to_bottom()
        return True

    if (
        head == "/mode"
        and len(parts) >= 2
        and parts[1].lower() in {"chat", "research", "agentic"}
    ):
        state.mode = parts[1].lower()
        ui.append(Text(f"Mode set to {state.mode}", style="cyan"))
        return True

    if head == "/think" and len(parts) >= 2 and parts[1].lower() in {"on", "off"}:
        state.thinking_enabled = parts[1].lower() == "on"
        if len(parts) >= 3 and parts[2].lower() in {"low", "med", "high"}:
            state.thinking_effort = parts[2].lower()
        ui.append(
            Text(
                f"Thinking {'ON' if state.thinking_enabled else 'OFF'} ({state.thinking_effort})",
                style="cyan",
            )
        )
        return True

    if head == "/net" and len(parts) >= 2 and parts[1].lower() in {"allow", "block"}:
        state.internet_allowed = parts[1].lower() == "allow"
        ui.append(
            Text(
                f"Internet permission set to {'allow' if state.internet_allowed else 'block'}",
                style="cyan",
            )
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def run_cli(cfg: AgentConfig | None = None) -> None:
    """Run provider gate, interactive configuration, chat session, and exit telemetry."""
    config = cfg or AgentConfig()

    if not run_provider_readiness_gate(config):
        return

    while True:
        state = initialize_agent(config)
        if state is None:
            console.print("[dim]Exited before starting session.[/dim]")
            return
        if confirm_enter_chat(state) == "enter":
            console.clear()
            break

    exit_reason = "Session ended."
    try:
        with ChatScreenUI(state, banner_font=config.banner_font) as ui:
            while True:
                user_input = ui.read_line()
                if not user_input:
                    continue

                lowered = user_input.lower()
                if lowered == "exit":
                    exit_reason = "Goodbye."
                    break
                if lowered == "switch":
                    ui.live.stop()
                    _, new_model = select_provider_and_model(config, step_idx=3)
                    if new_model and new_model != state.model:
                        state.model = new_model
                        state.model_constructions += 1
                        state.suspected_cold_loads += 1
                    console.clear()
                    ui.live.start()
                    ui.reset_with_banner()
                    continue
                if _handle_slash_command(user_input, state, ui):
                    continue

                run_chat_turn(ui, state, user_input, stream_fn=config.model_stream_fn)
    except KeyboardInterrupt:
        exit_reason = "Interrupted with Ctrl+C."
    except EOFError:
        exit_reason = "EOF received; session ended."

    console.print(f"\n[dim]{exit_reason}[/dim]")
    console.print(render_grouped_session_statistics(state))


if __name__ == "__main__":
    run_cli()
