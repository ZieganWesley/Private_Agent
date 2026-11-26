"""Per-turn context wiring kept out of the session loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import (
    APP_CONFIG,
    CONTEXT_OUTPUT_RESERVE_RATIO,
    CONTEXT_SAFETY_MARGIN_TOKENS,
)

from . import runtime_f_integration as f_integration
from .context_engine import (
    ContextAllocation,
    ContextEngine,
    assemble_owned_prompt,
    start_turn_allocation,
)
from .context_engine import is_enabled as context_engine_is_enabled
from .model_selection import request_limits_for_provider
from .prompts import build_bounded_user_input, trim_history_to_context_budget


def start_turn_context(
    provider_type: str,
    selected_local_runtime: Any,
    selected_model: str,
    capabilities: Any,
    user_input: str,
) -> tuple[int, int, ContextEngine | None, ContextAllocation | None]:
    """Resolve limits and allocate every turn source once before retrieval."""
    limits = request_limits_for_provider(
        provider_type, selected_local_runtime, selected_model, capabilities
    )
    effective_context = limits.context
    output_ceiling = limits.output_tokens
    output_reserve = min(
        output_ceiling,
        max(1, int(effective_context * CONTEXT_OUTPUT_RESERVE_RATIO)),
    )
    engine = None
    allocation = None
    if context_engine_is_enabled():
        try:
            engine = ContextEngine.from_config(APP_CONFIG)
            allocation = start_turn_allocation(
                engine,
                effective_context,
                user_input,
                output_reserve=output_reserve,
                safety_margin=CONTEXT_SAFETY_MARGIN_TOKENS,
            )
        except Exception:  # noqa: BLE001 - optional best-effort context feature
            engine = None
            allocation = None
    return effective_context, output_ceiling, engine, allocation


@dataclass
class TurnPrompt:
    system_text: str | None
    enriched_input: str
    prompt_history: list[Any]


def build_turn_prompt(
    engine: ContextEngine | None,
    allocation: ContextAllocation | None,
    *,
    context_label: str,
    system_prompts: list[Any],
    tools_list: Any,
    active_skill: Any,
    active_task_instruction: str,
    attached_file_context: str,
    context_summary: str,
    episodic_context: str,
    context_text: str,
    user_input: str,
    chat_history: list[Any],
    prompt_budget: int,
) -> TurnPrompt:
    """Build the prompt once, from the allocation when the engine is on."""
    if engine is not None and allocation is not None:
        system_text, enriched_input, prompt_history = assemble_owned_prompt(
            engine,
            allocation,
            context_label=context_label,
            system_prompts=system_prompts,
            tools_list=tools_list,
            active_skill=active_skill,
            active_task_instruction=active_task_instruction,
            attached_file_context=attached_file_context,
            context_summary=context_summary,
            episodic_context=episodic_context,
            context_text=context_text,
            user_input=user_input,
            chat_history=chat_history,
        )
        return TurnPrompt(system_text, enriched_input, prompt_history)
    enriched_input = build_bounded_user_input(
        context_label,
        episodic_context,
        context_text,
        user_input,
        budget=prompt_budget,
    )
    enriched_input = f_integration.trim_context(enriched_input, prompt_budget)
    prompt_history = trim_history_to_context_budget(
        chat_history,
        enriched_input,
        max_context_tokens=prompt_budget,
    )
    return TurnPrompt(None, enriched_input, prompt_history)


def rag_budget_for(allocation: ContextAllocation | None) -> int | None:
    """Return the RAG token budget from the turn allocation."""
    if allocation is None:
        return None
    return allocation.budgets.get("rag")


def assemble_turn_prompt(
    engine: ContextEngine,
    allocation: ContextAllocation,
    **kwargs: Any,
) -> tuple[str, str, list[Any]]:
    """Render the flag-on prompt from the turn's allocation."""
    return assemble_owned_prompt(engine, allocation, **kwargs)


__all__ = [
    "TurnPrompt",
    "assemble_turn_prompt",
    "build_turn_prompt",
    "rag_budget_for",
    "start_turn_context",
]
