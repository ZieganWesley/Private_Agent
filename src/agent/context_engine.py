"""Context engine coordination layer (Phase 3).

Wraps the budget model, source adapters, and profiles into a single
per-turn allocator.  It is only used when ``context_engine_enabled`` is True;
otherwise callers keep their existing fallback paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import CONTEXT_ENGINE_ENABLED, CONTEXT_ENGINE_PROFILE

from .compaction import (
    messages_tokens,
    needs_history_compaction,
    trim_history_for_budget,
)
from .context_manager import ContextBudget, ContextSource
from .context_profiles import (
    apply_profile_to_budget,
    load_profile_overrides,
    load_share_overrides,
    select_profile,
)
from .context_sources import ADAPTERS, adapt_source
from .context_telemetry import build_context_telemetry
from .prompts import token_count, trim_history_messages


@dataclass(frozen=True)
class CompactionDecision:
    """Why the next turn does or does not need summarization."""

    reason: str
    history: list[Any]
    post_trim_tokens: int


@dataclass
class ContextAllocation:
    profile: str
    budgets: dict[str, int]
    sources: dict[str, ContextSource]
    total_tokens: int
    enabled: bool
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextEngine:
    enabled: bool = False
    profile: str = "auto"
    config: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)

    def needs_compaction(
        self,
        history: Any,
        current_input_tokens: int,
        prompt_budget: int,
        threshold: float,
        max_messages: int,
    ) -> bool:
        """Compaction trigger moved into the engine (Phase 7)."""
        try:
            return needs_history_compaction(
                history,
                current_input_tokens,
                prompt_budget,
                threshold,
                max_messages,
            )
        except Exception:  # noqa: BLE001 - optional best-effort context feature
            return False

    def should_compact(
        self,
        history: Any,
        current_input_tokens: int,
        prompt_budget: int,
        threshold: float,
        max_messages: int,
        *,
        keep_recent: int = 2,
    ) -> CompactionDecision:
        """Trim first, then decide whether summarization is still required."""
        messages = list(history or [])
        if threshold <= 0 or not messages:
            return CompactionDecision("below_threshold", messages, 0)
        if not needs_history_compaction(
            messages,
            current_input_tokens,
            prompt_budget,
            threshold,
            max_messages,
        ):
            return CompactionDecision(
                "below_threshold",
                messages,
                messages_tokens(messages) + current_input_tokens,
            )
        trimmed, post_trim = trim_history_for_budget(
            messages,
            current_input_tokens,
            prompt_budget,
            threshold,
            max_messages,
            keep_recent=keep_recent,
        )
        if post_trim <= threshold * prompt_budget and len(trimmed) <= max_messages:
            return CompactionDecision("trim_sufficient", trimmed, post_trim)
        return CompactionDecision("compact", trimmed, post_trim)

    def resolve_profile(self, request_text: str = "") -> str:
        if self.profile and self.profile != "auto":
            return self.profile
        # config override for explicit profile
        configured = None
        try:
            configured = self.config.get("context_engine_profile") if isinstance(self.config, dict) else None
        except Exception:  # noqa: BLE001 - optional best-effort context feature
            configured = None
        if configured and configured != "auto":
            return str(configured)
        return select_profile(request_text)

    def _build_source(self, name: str, text: Any, budget: int) -> ContextSource:
        adapter = ADAPTERS.get(name)
        if adapter is not None:
            try:
                return adapter(text=text, budget_tokens=budget)
            except Exception:  # noqa: BLE001 - optional best-effort context feature
                pass
        return adapt_source(name, text=str(text or ""), budget_tokens=budget)

    def build_sources(
        self,
        source_texts: dict[str, Any] | None = None,
        budgets: dict[str, int] | None = None,
    ) -> dict[str, ContextSource]:
        source_texts = source_texts or {}
        budgets = budgets or {}
        return {
            name: self._build_source(name, text, int(budgets.get(name, 0)))
            for name, text in source_texts.items()
        }

    def _raw_sources(
        self, source_texts: dict[str, Any]
    ) -> dict[str, ContextSource]:
        """Build unbounded sources so adapters can contribute constraints."""
        raw: dict[str, ContextSource] = {}
        for name, text in source_texts.items():
            full_text = text if isinstance(text, str) else str(text or "")
            required = token_count(full_text) if full_text else 0
            source = self._build_source(name, text, required)
            # Keep the full text when the estimator cannot fit the exact count.
            source.text = full_text
            raw[name] = source
        return raw

    def fill_sources(
        self,
        allocation: ContextAllocation,
        source_texts: dict[str, Any],
    ) -> ContextAllocation:
        """Update an allocation's source text without changing its budgets."""
        if not allocation.enabled:
            return allocation
        requested = {
            name: token_count(str(text or ""))
            for name, text in source_texts.items()
            if name in allocation.budgets
        }
        for name, text in source_texts.items():
            if name not in allocation.budgets:
                continue
            budget = int(allocation.budgets[name])
            adapter = ADAPTERS.get(name)
            if isinstance(text, (list, tuple)) and adapter is not None:
                try:
                    source = adapter(messages=text, budget_tokens=budget)
                except Exception:  # noqa: BLE001 - optional best-effort context feature
                    source = self._build_source(name, str(text), budget)
            else:
                source = self._build_source(name, text, budget)
            allocation.sources[name] = source
        previous = allocation.telemetry.get("requested", {}) if allocation.telemetry else {}
        allocation.telemetry = build_context_telemetry(
            profile=allocation.profile,
            total_tokens=allocation.total_tokens,
            budgets=allocation.budgets,
            sources=allocation.sources,
            requested={**previous, **requested},
        ).to_dict()
        return allocation

    def allocate(
        self,
        total_tokens: int,
        source_texts: dict[str, Any] | None = None,
        request_text: str = "",
        profile: str | None = None,
        overrides: dict[str, Any] | None = None,
        *,
        output_reserve: int = 0,
        safety_margin: int = 0,
    ) -> ContextAllocation:
        if not self.enabled:
            # Fallback: empty allocation; callers use existing paths.
            return ContextAllocation(
                profile="chat",
                budgets={},
                sources={},
                total_tokens=total_tokens,
                enabled=False,
            )
        prof = profile if profile is not None else self.resolve_profile(request_text)
        if prof == "auto":
            prof = select_profile(request_text)
        cfg_overrides = self.overrides or load_profile_overrides(self.config or {})
        budget = ContextBudget(
            total_tokens=total_tokens,
            output_reserve=max(0, int(output_reserve or 0)),
            safety_margin=max(0, int(safety_margin or 0)),
        )
        apply_profile_to_budget(
            budget,
            prof,
            request_text=request_text,
            overrides=cfg_overrides if cfg_overrides else (overrides or {}),
        )
        share_overrides = load_share_overrides(self.config or {})
        if share_overrides:
            merged = {**budget.shares, **share_overrides}
            total_share = sum(merged.values())
            if total_share > 0:
                merged = {
                    name: value / total_share for name, value in merged.items()
                }
            budget.shares = merged
        raw_sources = self._raw_sources(source_texts or {})
        budgets = budget.allocate(
            list(raw_sources.values()) if raw_sources else None
        )
        # Ensure all source texts have budgets.
        for name in (source_texts or {}):
            if name not in budgets:
                budgets[name] = 0
        sources = self.build_sources(source_texts, budgets)
        requested = {
            name: token_count(str(text or ""))
            for name, text in (source_texts or {}).items()
        }
        telemetry = build_context_telemetry(
            profile=prof,
            total_tokens=total_tokens,
            budgets=budgets,
            sources=sources,
            requested=requested,
        ).to_dict()
        return ContextAllocation(
            profile=prof,
            budgets=budgets,
            sources=sources,
            total_tokens=total_tokens,
            enabled=True,
            telemetry=telemetry,
        )

    @classmethod
    def from_config(
        cls, config: dict[str, Any] | None = None, enabled: bool | None = None
    ) -> ContextEngine:
        cfg = config or {}
        eng_enabled = CONTEXT_ENGINE_ENABLED if enabled is None else bool(enabled)
        # allow config override
        try:
            agent = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
            if isinstance(agent, dict) and "context_engine_enabled" in agent:
                eng_enabled = bool(agent.get("context_engine_enabled"))
        except Exception:  # noqa: BLE001 - optional best-effort context feature
            pass
        profile = CONTEXT_ENGINE_PROFILE
        try:
            agent = cfg.get("agent", {}) if isinstance(cfg, dict) else {}
            if isinstance(agent, dict) and agent.get("context_engine_profile"):
                profile = str(agent.get("context_engine_profile"))
        except Exception:  # noqa: BLE001 - optional best-effort context feature
            pass
        return cls(
            enabled=eng_enabled,
            profile=profile,
            config=cfg,
            overrides=load_profile_overrides(cfg),
        )


TURN_SOURCE_NAMES: tuple[str, ...] = (
    "system",
    "tool_definitions",
    "skills",
    "plan",
    "files",
    "summary",
    "episodic",
    "history",
    "rag",
    "tool_results",
    "user",
)


def plan_turn_compaction(
    engine: ContextEngine,
    history: Any,
    current_input_tokens: int,
    prompt_budget: int,
    threshold: float,
    max_messages: int,
    *,
    keep_recent: int = 2,
) -> tuple[list[Any], bool]:
    """Return the possibly trimmed history and whether to summarize."""
    decision = engine.should_compact(
        history,
        current_input_tokens,
        prompt_budget,
        threshold,
        max_messages,
        keep_recent=keep_recent,
    )
    if decision.reason == "trim_sufficient":
        return decision.history, False
    return list(history or []), decision.reason == "compact"


def start_turn_allocation(
    engine: ContextEngine,
    total_tokens: int,
    user_input: str,
    *,
    output_reserve: int = 0,
    safety_margin: int = 0,
) -> ContextAllocation:
    """Allocate every turn source once, before retrieval and prompt assembly."""
    source_texts = {name: "" for name in TURN_SOURCE_NAMES}
    source_texts["user"] = user_input
    return engine.allocate(
        total_tokens,
        source_texts=source_texts,
        request_text=user_input,
        output_reserve=output_reserve,
        safety_margin=safety_margin,
    )


def assemble_owned_prompt(
    engine: ContextEngine,
    allocation: ContextAllocation,
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
) -> tuple[str, str, list[Any]]:
    """Fill the turn allocation and render the flag-on prompt from it."""
    engine.fill_sources(
        allocation,
        {
            "system": "\n".join(
                str(getattr(message, "content", "")) for message in system_prompts
            ),
            "tool_definitions": "\n".join(
                f"{getattr(tool, 'name', '')}: "
                f"{getattr(tool, 'description', '')}"
                for tool in (tools_list or [])
            ),
            "skills": active_skill.system_prompt if active_skill else "",
            "plan": active_task_instruction,
            "files": attached_file_context,
            "summary": context_summary,
            "episodic": episodic_context,
            "rag": context_text,
            "user": user_input,
        },
    )
    sources = allocation.sources
    input_parts = [
        part
        for part in (
            sources["episodic"].text,
            sources["rag"].text,
            sources["files"].text,
        )
        if part
    ]
    enriched_input = (
        f"{context_label}:\n"
        + "\n".join(input_parts)
        + f"\n\n[User Query]: {sources['user'].text}"
    )
    prompt_history = trim_history_messages(
        chat_history, allocation.budgets.get("history", 0)
    )
    engine.fill_sources(allocation, {"history": prompt_history})
    return sources["system"].text, enriched_input, prompt_history


def is_enabled(config: dict[str, Any] | None = None) -> bool:
    if config is None:
        return bool(CONTEXT_ENGINE_ENABLED)
    try:
        agent = config.get("agent", {}) if isinstance(config, dict) else {}
        if isinstance(agent, dict) and "context_engine_enabled" in agent:
            return bool(agent.get("context_engine_enabled"))
    except Exception:  # noqa: BLE001 - optional best-effort context feature
        pass
    return bool(CONTEXT_ENGINE_ENABLED)


__all__ = [
    "ContextEngine",
    "ContextAllocation",
    "TURN_SOURCE_NAMES",
    "assemble_owned_prompt",
    "is_enabled",
    "start_turn_allocation",
]
