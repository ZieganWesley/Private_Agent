"""Context engine coordination layer (Phase 3).

Wraps the budget model, source adapters, and profiles into a single
per-turn allocator.  It is only used when ``context_engine_enabled`` is True;
otherwise callers keep their existing fallback paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import CONTEXT_ENGINE_ENABLED, CONTEXT_ENGINE_PROFILE

from .compaction import needs_history_compaction
from .context_manager import ContextBudget, ContextSource
from .context_profiles import apply_profile_to_budget, load_profile_overrides, select_profile
from .context_sources import ADAPTERS
from .context_telemetry import build_context_telemetry


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

    def build_sources(
        self,
        source_texts: dict[str, Any] | None = None,
        budgets: dict[str, int] | None = None,
    ) -> dict[str, ContextSource]:
        source_texts = source_texts or {}
        budgets = budgets or {}
        result: dict[str, ContextSource] = {}
        for name, text in source_texts.items():
            adapter = ADAPTERS.get(name)
            budget = int(budgets.get(name, 0))
            if adapter is not None:
                try:
                    src = adapter(text=text if not isinstance(text, (list, tuple, dict)) else text, budget_tokens=budget)
                except Exception:  # noqa: BLE001 - optional best-effort context feature
                    from .context_sources import adapt_source

                    src = adapt_source(name, text=str(text), budget_tokens=budget)
                result[name] = src
            else:
                from .context_sources import adapt_source

                result[name] = adapt_source(name, text=str(text), budget_tokens=budget)
        return result

    def allocate(
        self,
        total_tokens: int,
        source_texts: dict[str, Any] | None = None,
        request_text: str = "",
        profile: str | None = None,
        overrides: dict[str, Any] | None = None,
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
        budget = ContextBudget(total_tokens=total_tokens)
        apply_profile_to_budget(budget, prof, request_text=request_text, overrides=cfg_overrides if cfg_overrides else (overrides or {}))
        budgets = budget.allocate(list((source_texts or {}).keys()) if source_texts else None)
        # Ensure all source texts have budgets.
        for name in (source_texts or {}):
            if name not in budgets:
                budgets[name] = 0
        sources = self.build_sources(source_texts, budgets)
        telemetry = build_context_telemetry(
            profile=prof,
            total_tokens=total_tokens,
            budgets=budgets,
            sources=sources,
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
    def from_config(cls, config: dict[str, Any] | None = None, enabled: bool | None = None) -> "ContextEngine":
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


__all__ = ["ContextEngine", "ContextAllocation", "is_enabled"]
