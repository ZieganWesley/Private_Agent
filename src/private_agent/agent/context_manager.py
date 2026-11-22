"""Token-budgeted context manager (Part F5)."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SHARES = {
    "system": 0.15,
    "recent": 0.35,
    "plan": 0.10,
    "rag": 0.15,
    "memory": 0.10,
    "files": 0.15,
}


@dataclass
class ContextBudget:
    total_tokens: int
    shares: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SHARES))

    def allocate(self) -> dict[str, int]:
        result: dict[str, int] = {}
        used = 0
        for name, share in self.shares.items():
            amt = int(self.total_tokens * share)
            result[name] = amt
            used += amt
        # give remainder to recent
        result["recent"] += self.total_tokens - used
        return result

    def overflow(self, used: dict[str, int]) -> dict[str, int]:
        alloc = self.allocate()
        return {k: max(0, used.get(k, 0) - alloc.get(k, 0)) for k in alloc}


def build_context(sources: dict[str, str], total_tokens: int) -> dict[str, str]:
    """Trim each source to its budget (char proxy: 4 chars/token)."""
    budget = ContextBudget(total_tokens=total_tokens)
    alloc = budget.allocate()
    out: dict[str, str] = {}
    for name, text in sources.items():
        limit = alloc.get(name, 0) * 4
        out[name] = text[:limit] if limit > 0 else ""
    return out
