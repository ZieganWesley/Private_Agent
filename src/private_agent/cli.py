import argparse
import logging
import os
import sys
import time

from rich.console import Console

console = Console()


def enforce_offline_environment() -> None:
    """Keep Hugging Face libraries from making network requests."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


WINDOWS_UNSUPPORTED_MESSAGE = (
    "Private Agent supports Linux, macOS and WSL2 only; native Windows is not "
    "supported. Install WSL (https://learn.microsoft.com/windows/wsl/install) "
    "and run private-agent inside it."
)


def ensure_supported_platform() -> None:
    """Exit with a pointer to WSL when started on native Windows."""
    if sys.platform == "win32":
        console.print(
            f"[red][Unsupported platform] {WINDOWS_UNSUPPORTED_MESSAGE}[/red]"
        )
        raise SystemExit(1)


def list_saved_sessions(limit: int = 20) -> int:
    """Print saved session ids with activity, newest first."""
    import sqlite3

    from rich.table import Table

    memory = None
    try:
        from .database import PersistentMemory

        memory = PersistentMemory()
        sessions = memory.list_sessions_with_metadata(limit=limit)
    except (ValueError, sqlite3.Error, OSError) as exc:
        console.print(f"[red][Session list] Could not read saved sessions: {exc}[/red]")
        return 1
    finally:
        if memory is not None:
            memory.close()
    if not sessions:
        console.print("[cyan][Session list][/cyan] No saved sessions yet.")
        return 0
    table = Table(title="Saved sessions")
    table.add_column("Session ID", overflow="fold")
    table.add_column("Messages", justify="right")
    table.add_column("Last activity")
    for session in sessions:
        table.add_row(
            str(session["session_id"]),
            str(session["messages"]),
            str(session["last_at"] or ""),
        )
    console.print(table)
    console.print(
        "[cyan][Session list][/cyan] Resume one with "
        "`private-agent --resume <SESSION ID>`."
    )
    return 0


def main():
    ensure_supported_platform()
    enforce_offline_environment()
    parser = argparse.ArgumentParser(
        prog="private-agent",
        description="Run the local-first Private Agent CLI.",
    )
    parser.add_argument(
        "--verbose-startup",
        action="store_true",
        help="Show detailed startup diagnostics and configured paths.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="Use this configuration file instead of ~/.private_agent.conf "
        "(also settable with PRIVATE_AGENT_CONFIG).",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Resume the most recently saved session.",
    )
    resume_group.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume a specific saved session (see --list-sessions).",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List saved sessions with their message counts and exit.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        metavar="PROMPT",
        help="Run a single prompt non-interactively and exit (headless mode).",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format for non-interactive mode (default: text).",
    )
    parser.add_argument(
        "--permission-mode",
        choices=("manual", "auto", "full", "monitored"),
        default=None,
        help="Permission mode for this run (default: configured).",
    )
    parser.add_argument(
        "--provider",
        metavar="NAME",
        default=None,
        help="Preferred local provider name (e.g. Ollama, LM Studio).",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Use a light-terminal colour theme.",
    )
    args = parser.parse_args()
    if args.config:
        os.environ["PRIVATE_AGENT_CONFIG"] = args.config
    if args.light:
        os.environ["PRIVATE_AGENT_LIGHT"] = "1"
    if args.provider:
        os.environ["PRIVATE_AGENT_PROVIDER"] = args.provider
    if args.list_sessions:
        raise SystemExit(list_saved_sessions())
    originals = []
    exit_code = 0
    from .config import DEBUG_LOG_ENABLED
    from .run_logging import (
        RUN_LOGGER,
        LoggingConsoleAdapter,
        install_logging_console_adapter,
        log_event,
        start_debug_logging,
        stop_debug_logging,
    )

    try:
        startup_started = time.monotonic()
        log_path = start_debug_logging()
        if DEBUG_LOG_ENABLED and log_path is None:
            console.print(
                "[yellow][Debug log] Logging is enabled but no file was created.[/yellow]"
            )
        elif log_path is not None:
            console.print(f"[cyan][Debug log][/cyan] Writing run trace to {log_path}")
        from .agent import run_agent_cli

        if log_path is not None:
            originals = install_logging_console_adapter(LoggingConsoleAdapter())
        log_event(RUN_LOGGER, "application.started")
        log_event(
            RUN_LOGGER,
            "application.startup_completed",
            elapsed_seconds=round(time.monotonic() - startup_started, 3),
        )
        non_interactive = bool(args.prompt)
        run_agent_cli(
            verbose_startup=args.verbose_startup,
            resume_session=args.resume,
            continue_session=args.continue_session,
            prompt=args.prompt,
            output=args.output,
            permission_mode=args.permission_mode,
            non_interactive=non_interactive,
            interactive=not non_interactive,
            input_provider=(lambda *a, **k: "") if non_interactive else None,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow][Info] Interrupted by user.[/yellow]")
    except EOFError:
        # stdin was closed at an interactive prompt (Ctrl+D), which is an
        # orderly exit rather than an unexpected crash.
        console.print("\n[yellow][Info] Input closed; exiting.[/yellow]")
    except RuntimeError as exc:
        log_event(
            RUN_LOGGER,
            "application.startup_failed",
            level=logging.ERROR,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        console.print(f"[red][Startup error] {exc}[/red]")
        exit_code = 1
    except (ValueError, OSError):
        log_event(
            RUN_LOGGER,
            "application.terminated_unexpectedly",
            level=logging.ERROR,
            exc_info=True,
        )
        raise
    finally:
        log_event(RUN_LOGGER, "application.exiting")
        if originals:
            from .run_logging import restore_logging_consoles

            restore_logging_consoles(originals)
        stop_debug_logging()
        console.print("[cyan][Info] Goodbye.[/cyan]")
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
