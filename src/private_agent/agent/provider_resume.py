"""Provider record persistence and resume fallback helpers (Part I, I1p)."""

import logging
import sqlite3

from ..run_logging import RUN_LOGGER, log_event

__all__ = [
    "always_ask_provider",
    "provider_loss_guidance",
    "auto_select_single_provider",
    "capability_change_warnings",
    "warn_capability_changes",
    "provider_record_for_resume",
    "match_runtime_by_address",
    "persist_provider_choice",
]


def provider_record_for_resume(memory, session_id):
    """Return the provider record saved for a session being resumed.

    A task can outlive the session that created it, so when the session has no
    record of its own fall back to the provider the active task was run with.
    """
    if memory is None or not session_id:
        return None
    try:
        record = memory.get_session_provider(session_id)
        if record is not None:
            return record
        active_task = memory.get_session_active_task(session_id)
        if not active_task:
            return None
        return memory.get_task_provider(active_task.get("task_id"))
    except (ValueError, sqlite3.Error) as exc:
        log_event(
            RUN_LOGGER,
            "providers.resume_record_unreadable",
            level=logging.WARNING,
            error_type=type(exc).__name__,
        )
        return None


def capability_change_warnings(previous, capabilities):
    """Describe what a resumed model lost compared with its provider record."""
    if previous is None:
        return []
    current = capabilities or {}
    warnings = []
    if previous.needs_tools and not current.get("tools"):
        warnings.append("it was recorded with tool calling, which it no longer reports")
    if previous.needs_vision and not current.get("vision"):
        warnings.append("it was recorded with vision, which it no longer reports")
    recorded_window = previous.context_window
    current_window = current.get("context_window")
    if recorded_window and current_window and current_window < recorded_window:
        warnings.append(
            f"its context window shrank from {recorded_window} to "
            f"{current_window} tokens"
        )
    return warnings


def warn_capability_changes(console, previous, capabilities, model):
    """Print one warning per capability a resumed model no longer reports."""
    for warning in capability_change_warnings(previous, capabilities):
        console.print(
            f"[yellow][Resume][/yellow] {model}: {warning}; "
            "continuing, but expect a weaker result."
        )


def match_runtime_by_address(base_url, runtimes):
    """Match a recorded endpoint to a reachable runtime when names differ."""
    wanted = str(base_url or "").rstrip("/").lower()
    if not wanted:
        return None
    for runtime in runtimes:
        if str(getattr(runtime, "base_url", "")).rstrip("/").lower() == wanted:
            return runtime
    return None


def persist_provider_choice(
    *,
    console,
    memory,
    session_id,
    runtime,
    model,
    capabilities,
    previous,
):
    """Record the session's provider, remember it, and journal a switch."""
    from ..database.provider_records import provider_record_payload
    from ..providers.adapters import ProviderRecord
    from ..providers.selection import save_last_provider

    record = ProviderRecord(
        adapter_type=str(getattr(runtime, "kind", "openai-compatible")),
        display_name=str(runtime.name),
        base_url=str(runtime.base_url),
        model=str(model),
        needs_tools=bool((capabilities or {}).get("tools")),
        needs_vision=bool((capabilities or {}).get("vision")),
        context_window=(capabilities or {}).get("context_window"),
    )
    active_task = None
    if memory is not None and session_id:
        try:
            memory.record_session_provider(session_id, record)
            active_task = memory.get_session_active_task(session_id)
            if active_task:
                memory.record_task_provider(active_task["task_id"], record)
        except (ValueError, sqlite3.Error) as exc:
            log_event(
                RUN_LOGGER,
                "providers.record_write_failed",
                level=logging.WARNING,
                error_type=type(exc).__name__,
            )
    warn_capability_changes(console, previous, capabilities, record.model)
    if previous is not None and (
        previous.display_name != record.display_name
        or previous.base_url != record.base_url
        or previous.model != record.model
    ):
        if active_task:
            try:
                memory._record_task_event(
                    active_task["task_id"],
                    int(active_task.get("current_revision") or 1),
                    "provider_changed",
                    {
                        "previous": provider_record_payload(previous),
                        "current": provider_record_payload(record),
                    },
                )
            except (ValueError, sqlite3.Error) as exc:
                log_event(
                    RUN_LOGGER,
                    "providers.task_event_failed",
                    level=logging.WARNING,
                    error_type=type(exc).__name__,
                )
        console.print(
            f"[yellow][Provider changed][/yellow] This session now runs on "
            f"{record.display_name} · {record.model}."
        )
    save_last_provider(
        adapter_type=record.adapter_type,
        display_name=record.display_name,
        base_url=record.base_url,
        model=record.model,
    )
    return record


def _config_flag(models_config, key: str) -> bool:
    """Read one boolean from the models section, tolerating strings."""
    raw = models_config.get(key, False) if isinstance(models_config, dict) else False
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"true", "1", "yes", "on"}


def auto_select_single_provider(app_config) -> bool:
    """Return models.auto_select_single_provider (default: off)."""
    return _config_flag(app_config.get("models", {}), "auto_select_single_provider")


def always_ask_provider(app_config) -> bool:
    """Return models.always_ask (default: off)."""
    return _config_flag(app_config.get("models", {}), "always_ask")


def provider_loss_guidance(*, runtime_name, session_id, discover):
    """Explain what to do when the active local provider stops answering.

    ``discover`` returns the reachable runtimes (a callable so this stays
    cheap when there is no failure). Returns an empty string when nothing
    useful can be said.
    """
    try:
        reachable = [
            runtime
            for runtime in discover()
            if str(getattr(runtime, "name", "")) != str(runtime_name)
        ]
    except (OSError, RuntimeError):
        reachable = []
    lines = [
        f"[yellow][Provider lost][/yellow] {runtime_name} stopped answering; "
        "the turn was not completed."
    ]
    if reachable:
        lines.append(
            "Still reachable: "
            + ", ".join(
                f"{getattr(runtime, 'name', '?')} "
                f"({len(getattr(runtime, 'models', ()) or ())} model(s))"
                for runtime in reachable
            )
        )
        lines.append(
            "Start it again, or continue this session on another provider with "
            f"`private-agent --resume {session_id} --provider "
            f"'{getattr(reachable[0], 'name', '')}'`."
        )
    else:
        lines.append(
            "No other local provider is reachable. Start one (for example "
            "`ollama serve`) and resume this session, or continue online."
        )
    return "\n".join(lines)
