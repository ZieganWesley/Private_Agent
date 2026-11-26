"""Context telemetry and inspection helpers (Phase 6).

Structured per-source usage reporting for one turn.  No source content is
included in the report; only names, token counts, budgets and overflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .prompts import token_count as _token_count


@dataclass
class ContextTelemetry:
    profile: str = "chat"
    total_tokens: int = 0
    budgets: dict[str, int] = field(default_factory=dict)
    used: dict[str, int] = field(default_factory=dict)
    overflow: dict[str, int] = field(default_factory=dict)
    truncated: list[str] = field(default_factory=list)
    cacheable_prefix_tokens: int = 0
    invalidations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "total_tokens": int(self.total_tokens),
            "budgets": dict(self.budgets),
            "used": dict(self.used),
            "overflow": dict(self.overflow),
            "truncated": list(self.truncated),
            "cacheable_prefix_tokens": int(self.cacheable_prefix_tokens),
            "invalidations": int(self.invalidations),
            "sum_budgets": int(sum(self.budgets.values())),
            "sum_used": int(sum(self.used.values())),
            "usage_percent": (
                (sum(self.used.values()) / self.total_tokens * 100.0)
                if self.total_tokens > 0
                else 0.0
            ),
        }


def _source_text(source: Any) -> str:
    if source is None:
        return ""
    if hasattr(source, "text"):
        try:
            return str(source.text or "")
        except Exception:  # noqa: BLE001 - optional best-effort context feature
            return ""
    if isinstance(source, str):
        return source
    return str(source)


def build_context_telemetry(
    profile: str = "chat",
    total_tokens: int = 0,
    budgets: dict[str, int] | None = None,
    sources: Any = None,
    cacheable_prefix_tokens: int = 0,
    invalidations: int = 0,
) -> ContextTelemetry:
    budgets = dict(budgets or {})
    used: dict[str, int] = {}
    overflow: dict[str, int] = {}
    truncated: list[str] = []
    src_map = sources or {}
    # sources may be dict name->ContextSource or name->text
    for name, budget in budgets.items():
        src = src_map.get(name) if isinstance(src_map, dict) else None
        text = _source_text(src) if src is not None else ""
        # If not found in sources, maybe src_map maps name->text directly.
        if src is None and isinstance(src_map, dict) and name in src_map:
            val = src_map[name]
            if not hasattr(val, "text"):
                text = str(val or "")
        u = _token_count(text)
        used[name] = u
        ov = max(0, u - int(budget or 0))
        overflow[name] = ov
        if ov > 0:
            truncated.append(name)
    # ensure all sources present in used
    if isinstance(src_map, dict):
        for name in src_map:
            if name not in used:
                text = _source_text(src_map[name])
                if not hasattr(src_map[name], "text"):
                    text = str(src_map[name] or "")
                used[name] = _token_count(text)
                budgets.setdefault(name, 0)
                overflow[name] = max(0, used[name] - budgets.get(name, 0))
                if overflow[name] > 0:
                    truncated.append(name)
    return ContextTelemetry(
        profile=profile,
        total_tokens=int(total_tokens or 0),
        budgets=budgets,
        used=used,
        overflow=overflow,
        truncated=truncated,
        cacheable_prefix_tokens=int(cacheable_prefix_tokens or 0),
        invalidations=int(invalidations or 0),
    )


def format_context_report(telemetry: Any) -> str:
    data = telemetry.to_dict() if hasattr(telemetry, "to_dict") else (telemetry or {})
    lines = [f"Context usage ({data.get('profile', 'chat')}):"]
    budgets = data.get("budgets", {}) or {}
    used = data.get("used", {}) or {}
    for name in sorted(set(budgets) | set(used)):
        b = int(budgets.get(name, 0))
        u = int(used.get(name, 0))
        lines.append(f"  {name}: used={u} budget={b}")
    lines.append(
        f"  total: used={data.get('sum_used', 0)} / {data.get('total_tokens', 0)} "
        f"({data.get('usage_percent', 0.0):.1f}%)"
    )
    if data.get("truncated"):
        lines.append(f"  overflow: {', '.join(data['truncated'])}")
    cache = data.get("cache") if isinstance(data.get("cache"), dict) else {}
    if cache:
        lines.append(
            "  cache prefix: "
            f"tokens={cache.get('prefix_tokens', 0)} "
            f"qualifies={bool(cache.get('qualifies'))} "
            f"invalidated={bool(cache.get('invalidated'))}"
        )
    return "\n".join(lines)


def build_runtime_context_report(
    effective_context: Any,
    prompt_history: Any,
    system_prompts: Any,
    enriched_input: Any,
    user_input: Any,
    allocation: Any = None,
) -> dict[str, Any] | None:
    """Build a structured per-turn context report (Phase 6).

    When the turn's ``ContextAllocation`` is supplied it is the single source
    of truth, including overflow measured before trimming.  Returns ``None``
    only when no report can be constructed.
    """
    if allocation is not None and getattr(allocation, "telemetry", None):
        return dict(allocation.telemetry)
    report = None
    try:
        from .prompts import estimate_context_window

        report = {
            "profile": "chat",
            "total_tokens": int(effective_context),
            "budgets": {},
            "used": {
                "history": int(
                    estimate_context_window(
                        prompt_history,
                        "",
                        max_context=effective_context,
                        system_prompts=[],
                    ).get("tokens", 0)
                ),
            },
        }
    except Exception:  # noqa: BLE001 - best-effort reporting
        return None
    try:
        from .context_engine import ContextEngine
        from .context_engine import is_enabled as _is_enabled

        if _is_enabled():
            from config import APP_CONFIG

            eng = ContextEngine.from_config(APP_CONFIG)
            alloc = eng.allocate(
                effective_context,
                source_texts={
                    "system": "\n".join(
                        str(getattr(m, "content", "")) for m in (system_prompts or [])
                    ),
                    "history": "\n".join(
                        str(getattr(m, "content", "")) for m in (prompt_history or [])
                    ),
                    "user": enriched_input,
                },
                request_text=user_input,
            )
            report = alloc.telemetry or report
    except Exception:  # noqa: BLE001 - best-effort reporting
        pass
    return report




__all__ = ["ContextTelemetry", "build_context_telemetry", "format_context_report", "build_runtime_context_report"]
