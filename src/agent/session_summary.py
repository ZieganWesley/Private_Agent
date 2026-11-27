"""Session summary and end-of-turn reporting helpers.

These were extracted from the runtime module so that it stays below the size
that the module-layout guard enforces; the runtime re-exports them unchanged.
"""

import asyncio
import json
import logging
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import MAX_SUMMARY_CHARS, SUMMARY_PROMPT_SENTENCES
from database import PersistentMemory
from ui.status_bar import get_active_footer
from utils.run_logging import RUN_LOGGER, log_event

from .console import console
from .prompts import token_count as _token_count
from .session_state import SESSION_INFO as _SESSION_INFO
from .session_state import SESSION_STATS as _SESSION_STATS
from .session_stats import build_stats_table, format_duration
from .status import (
    SUMMARIZING,
    StatusIndicator,
    StatusState,
    status_ribbon,
    summarizing_message,
)


def _print_context_state(
    stats: dict,
    output_tokens: int,
    model_name: str | None = None,
    permission_mode: str = "auto",
) -> tuple[int, int]:
    """Print the consolidated status line and return ``(used, maximum)``."""
    values = stats if isinstance(stats, dict) else {}
    context_tokens = values.get("tokens", 0)
    context_maximum = values.get("max", 0)
    try:
        from tools.shell import background_process_indicator

        process_status = background_process_indicator()
    except ImportError, OSError, RuntimeError, ValueError:  # pragma: no cover
        process_status = ""
    if process_status.startswith("background_processes="):
        try:
            background_processes = int(process_status.split("=", 1)[1])
        except TypeError, ValueError:
            background_processes = 0
    else:
        background_processes = 0
    state = StatusState(
        model=model_name,
        context_tokens=context_tokens,
        context_maximum=context_maximum,
        permission_mode=permission_mode,
        output_tokens=output_tokens,
        background_processes=background_processes,
    )
    # The status footer owns the ribbon: Mode A paints it live and Mode B
    # prints it as the turn's closing line. Printing it here as well left a
    # second, never-updated ribbon in the transcript every turn, so it is only
    # emitted when no footer is presenting one (status bar disabled or off).
    footer = get_active_footer(console)
    if footer is None or not footer.presents_ribbon:
        console.print(f"[dim]{status_ribbon(state)}[/dim]")
    used = context_tokens
    try:
        used = int(context_tokens)
    except TypeError, ValueError:
        used = 0
    maximum = context_maximum
    try:
        maximum = int(context_maximum)
    except TypeError, ValueError:
        maximum = 0
    return used, maximum


def _history_with_summary(summary: str, history: list) -> list:
    if not summary:
        return history
    return [AIMessage(content=f"[Earlier in this session] {summary}")] + history


async def _summarize_and_save_session(
    console_obj,
    memory: PersistentMemory,
    llm,
    session_id: str,
    chat_history: list,
    provider_type: str,
    reason: str = "before exit",
    model_name: str = "model",
) -> bool:
    """Summarize the session into SQLite with concise stage output.

    A further Ctrl+C while this runs skips the summary instead of exiting
    with a traceback.
    """
    if not chat_history:
        console_obj.print(
            "[info][Memory][/info] No chat history was available "
            "to summarize; slash commands such as /maintenance "
            "are not stored as episodic memories."
        )
        return False
    console_obj.print(
        f"\n[info][Info] Summarizing session {reason} "
        "(press Ctrl+C again to skip)...[/info]"
    )
    summary_started = time.monotonic()
    try:
        console_obj.print("[info][Summary 1/3][/info] Preparing session context...")
        summary_schema = memory.get_table_schema("chat_history")
        summary_limit = MAX_SUMMARY_CHARS
        summary_prompt = (
            "Summarize the key technical takeaways, code solutions, "
            "and user preferences from this session. Do not include "
            "secrets, credentials, unnecessary private details, or "
            "raw tool output. The destination is SQLite table "
            "chat_history, column content for role=summary. Use the "
            "provided table schema and application content budget; "
            f"the summary must be at most {summary_limit} characters. "
            f"Prefer approximately {SUMMARY_PROMPT_SENTENCES} "
            "sentences when that fits. Preserve key meaning rather "
            "than truncating text."
        )
        summary_messages = (
            [
                SystemMessage(
                    content=(
                        "Prepare one meaning-preserving episodic "
                        "summary value that fits the SQLite destination. "
                        "Schema metadata and limits:\n"
                        + json.dumps(
                            {
                                "schema": summary_schema,
                                "application_limit_characters": summary_limit,
                            },
                            ensure_ascii=False,
                        )
                    )
                )
            ]
            + chat_history
            + [HumanMessage(content=summary_prompt)]
        )
        console_obj.print(
            "[info][Summary 2/3][/info] Model is summarizing the session..."
        )
        with StatusIndicator(
            summarizing_message(model_name, "the session"),
            phase=SUMMARIZING,
            console=console_obj,
        ):
            from providers.utility import complete_utility

            raw_summary = await asyncio.to_thread(
                complete_utility, summary_messages, model=model_name, llm=llm
            )
            if raw_summary is not None:
                summary_text = raw_summary.text.strip()
            else:
                call_started = time.monotonic()
                summary_res = await llm.ainvoke(summary_messages)
                _SESSION_STATS.record_model_call(
                    summary_messages, summary_res, time.monotonic() - call_started
                )
                summary_text = str(summary_res.content).strip()
            if len(summary_text) > summary_limit:
                shorten_messages = [
                    SystemMessage(
                        content=(
                            "Semantically summarize the provided "
                            "candidate for the exact SQLite schema "
                            "budget below. Do not slice or merely "
                            "remove its ending. Return only the "
                            "revised summary value.\n"
                            + json.dumps(
                                {
                                    "schema": summary_schema,
                                    "maximum_characters": summary_limit,
                                },
                                ensure_ascii=False,
                            )
                        )
                    ),
                    HumanMessage(content=summary_text),
                ]
                call_started = time.monotonic()
                summary_res = await llm.ainvoke(shorten_messages)
                _SESSION_STATS.record_model_call(
                    shorten_messages,
                    summary_res,
                    time.monotonic() - call_started,
                )
                summary_text = str(summary_res.content).strip()
        console_obj.print("[info][Summary 3/3][/info] Saving summary to SQLite...")
        memory.save_summary(session_id, summary_text)
        elapsed = time.monotonic() - summary_started
        _SESSION_STATS.summary_seconds += elapsed
        _SESSION_STATS.summary_tokens += _token_count(summary_text)
        _SESSION_STATS.summary_saved = True
        console_obj.print(
            f"[success][Summary][/success] Saved to SQLite memory "
            f"({len(summary_text)} chars, ~{_token_count(summary_text)} tokens, "
            f"{format_duration(elapsed)})."
        )
        return True
    except KeyboardInterrupt, asyncio.CancelledError:
        console_obj.print("[warning][Summary] Skipped; no summary was saved.[/warning]")
    except Exception as exc:  # noqa: BLE001 - summary mixes model/transport/SQLite failures
        log_event(
            RUN_LOGGER,
            "session.summary_failed",
            level=logging.ERROR,
            session_id=session_id,
            provider=provider_type,
            error_type=type(exc).__name__,
            exc_info=True,
        )
        error_detail = type(exc).__name__ if provider_type == "online" else str(exc)
        console_obj.print(
            f"[warning][Memory warning] Could not summarize session: "
            f"{error_detail}[/warning]"
        )
    _SESSION_STATS.summary_seconds += time.monotonic() - summary_started
    return False


def _print_session_statistics() -> None:
    console.print(build_stats_table(_SESSION_STATS, _SESSION_INFO))


def _acknowledge_interrupt() -> None:
    task = asyncio.current_task()
    if task is not None:
        while task.cancelling():
            task.uncancel()
    console.print("\n[warning][Info] Ctrl+C received; ending session.[/warning]")
