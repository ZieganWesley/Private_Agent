"""Provider record persistence and resume fallback helpers (Part I, I1p)."""

import logging
import sqlite3

from ..run_logging import RUN_LOGGER, log_event

__all__ = [
    "always_ask_provider",
    "auto_select_single_provider",
    "provider_record_for_resume",
    "match_runtime_by_address",
    "persist_provider_choice",
]


def provider_record_for_resume(memory, session_id):
    """Return the provider record saved for a session being resumed."""
    if memory is None or not session_id:
        return None
    try:
        return memory.get_session_provider(session_id)
    except (ValueError, sqlite3.Error) as exc:
        log_event(
            RUN_LOGGER,
            "providers.resume_record_unreadable",
            level=logging.WARNING,
            error_type=type(exc).__name__,
        )
        return None


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
    if memory is not None and session_id:
        try:
            memory.record_session_provider(session_id, record)
        except (ValueError, sqlite3.Error) as exc:
            log_event(
                RUN_LOGGER,
                "providers.record_write_failed",
                level=logging.WARNING,
                error_type=type(exc).__name__,
            )
    if previous is not None and (
        previous.display_name != record.display_name
        or previous.base_url != record.base_url
        or previous.model != record.model
    ):
        active_task = None
        try:
            if memory is not None and session_id:
                active_task = memory.get_session_active_task(session_id)
        except (ValueError, sqlite3.Error):
            active_task = None
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
