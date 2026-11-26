"""Token-budgeted context manager (Part F5).

Phase 1 of the context window management plan: a production-ready
``ContextBudget`` allocator with per-source priorities, minimum/maximum
constraints, deterministic overflow redistribution, and a ``ContextSource``
dataclass.  All existing behaviour is preserved when the engine is disabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SHARES = {
    "system": 0.15,
    "tool_definitions": 0.04,
    "skills": 0.02,
    "plan": 0.09,
    "files": 0.10,
    "summary": 0.05,
    "episodic": 0.05,
    "history": 0.27,
    "rag": 0.15,
    "tool_results": 0.05,
    "user": 0.03,
}

# Older profile files and callers used coarse names. Map them onto the real
# source names so RAG no longer absorbs the unused history/memory share (C2).
LEGACY_SHARE_ALIASES: dict[str, tuple[str, ...]] = {
    "recent": ("history",),
    "memory": ("episodic", "summary"),
}


def resolve_shares(shares: Mapping[str, float] | None) -> dict[str, float]:
    """Return canonical per-source shares with legacy names expanded."""
    resolved: dict[str, float] = dict(shares or {})
    for legacy, targets in LEGACY_SHARE_ALIASES.items():
        if legacy not in resolved:
            continue
        value = float(resolved.pop(legacy) or 0.0)
        remaining = [name for name in targets if name not in resolved]
        if not remaining:
            continue
        per_target = value / len(remaining)
        for name in remaining:
            resolved[name] = per_target
    return {name: float(value) for name, value in resolved.items()}

# Sources considered stable/cacheable for KV-cache prefix purposes.
CACHEABLE_SOURCES = {"system", "tool_definitions", "skills", "plan", "files"}


@dataclass
class ContextSource:
    """One allocatable context source.

    Attributes:
        name: Stable source identifier.
        text: Source text (may be empty for budget-only planning).
        priority: Higher values are preserved first during redistribution.
        min_tokens: Never trim below this unless the total is too small.
        max_tokens: Hard cap; ``None`` means no cap.
        compressible: Whether unused budget may be redistributed here.
        trim_strategy: One of ``truncate``, ``summarize``, ``drop``, ``compact``.
        cacheable: Whether the source belongs to the cacheable prefix.
    """

    name: str
    text: str = ""
    priority: int = 0
    min_tokens: int = 0
    max_tokens: int | None = None
    compressible: bool = True
    trim_strategy: str = "truncate"
    cacheable: bool = False


@dataclass
class ContextBudget:
    total_tokens: int
    shares: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SHARES))
    priorities: dict[str, int] = field(default_factory=dict)
    min_tokens: dict[str, int] = field(default_factory=dict)
    max_tokens: dict[str, int | None] = field(default_factory=dict)
    compressible: dict[str, bool] = field(default_factory=dict)
    output_reserve: int = 0
    safety_margin: int = 0

    def _effective_total(self) -> int:
        total = int(self.total_tokens) - int(self.output_reserve) - int(self.safety_margin)
        return total if total > 0 else 0

    def _resolve(
        self, sources: Any = None
    ) -> tuple[list[str], dict[str, float], dict[str, int], dict[str, int | None], dict[str, int]]:
        """Resolve source names and per-source constraints."""
        names: list[str] = []
        src_min: dict[str, int] = {}
        src_max: dict[str, int | None] = {}
        src_priority: dict[str, int] = {}
        src_shares: dict[str, float] = {}

        if sources is None:
            names = list(self.shares.keys())
        elif isinstance(sources, (list, tuple)):
            for s in sources:
                if isinstance(s, ContextSource):
                    names.append(s.name)
                    src_min[s.name] = max(0, int(s.min_tokens))
                    src_max[s.name] = s.max_tokens
                    src_priority[s.name] = int(s.priority)
                    self.compressible.setdefault(s.name, bool(s.compressible))
                    src_shares[s.name] = self.shares.get(s.name, 0.0)
                    if s.name not in self.shares:
                        src_shares[s.name] = 0.0
                elif isinstance(s, str):
                    names.append(s)

        elif isinstance(sources, dict):
            for key, value in sources.items():
                if isinstance(value, ContextSource):
                    names.append(key)
                    src_min[key] = max(0, int(value.min_tokens))
                    src_max[key] = value.max_tokens
                    src_priority[key] = int(value.priority)
                    self.compressible.setdefault(key, bool(value.compressible))
                    src_shares[key] = self.shares.get(key, 0.0)
                elif isinstance(value, str):
                    names.append(key)
                    src_shares[key] = self.shares.get(key, 0.0)
                else:
                    names.append(key)
                    src_shares[key] = self.shares.get(key, 0.0)
        else:
            names = list(self.shares.keys())

        # de-duplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        names = unique

        if not names:
            names = list(self.shares.keys())

        for n in names:
            if n not in src_shares or src_shares.get(n, 0.0) == 0.0 and n in self.shares:
                src_shares[n] = self.shares.get(n, 0.0)
            if n not in src_shares:
                src_shares[n] = 0.0
            if n not in src_min:
                src_min[n] = max(0, int(self.min_tokens.get(n, 0)))
            else:
                src_min[n] = max(src_min[n], int(self.min_tokens.get(n, 0)))
            if n not in src_max:
                src_max[n] = self.max_tokens.get(n, None)
            else:
                if self.max_tokens.get(n, None) is not None:
                    cur = src_max[n]
                    src_max[n] = (
                        min(cur, self.max_tokens[n])
                        if cur is not None
                        else self.max_tokens[n]
                    )
            if n not in src_priority:
                src_priority[n] = int(self.priorities.get(n, 0))

        # default shares for unknown sources: equal small share
        if all(src_shares.get(n, 0.0) == 0.0 for n in names):
            for n in names:
                src_shares[n] = 1.0

        return names, src_shares, src_min, src_max, src_priority

    def allocate(self, sources: Any = None, **kwargs: Any) -> dict[str, int]:
        # Accept sources via keyword too.
        if sources is None and "sources" in kwargs:
            sources = kwargs.pop("sources", None)
        if kwargs:
            # apply overrides
            if "min_tokens" in kwargs and isinstance(kwargs["min_tokens"], dict):
                self.min_tokens = {**self.min_tokens, **kwargs["min_tokens"]}
            if "max_tokens" in kwargs and isinstance(kwargs["max_tokens"], dict):
                self.max_tokens = {**self.max_tokens, **kwargs["max_tokens"]}
            if "priorities" in kwargs and isinstance(kwargs["priorities"], dict):
                self.priorities = {**self.priorities, **kwargs["priorities"]}
            if "shares" in kwargs and isinstance(kwargs["shares"], dict):
                self.shares = {**self.shares, **kwargs["shares"]}

        self.shares = resolve_shares(self.shares)
        names, shares, mins, maxs, prios = self._resolve(sources)
        T = self._effective_total()

        # normalize maxs
        norm_max: dict[str, int] = {}
        for n in names:
            m = maxs.get(n, None)
            norm_max[n] = int(m) if m is not None else T

        # clamp mins to maxs
        min_clamped: dict[str, int] = {}
        for n in names:
            min_clamped[n] = min(mins.get(n, 0), norm_max[n])
            if min_clamped[n] < 0:
                min_clamped[n] = 0

        alloc: dict[str, int] = {n: 0 for n in names}

        # If mins infeasible, allocate by priority.
        if sum(min_clamped.values()) > T:
            order = sorted(names, key=lambda x: (-prios.get(x, 0), x))
            left = T
            for n in order:
                give = min(min_clamped[n], left)
                alloc[n] = give
                left -= give
                if left <= 0:
                    break
            # ensure sum == T (pad if needed, though unlikely)
            self._pad_to_total(alloc, names, norm_max, T)
            return alloc

        # start with mins
        for n in names:
            alloc[n] = min_clamped[n]
        remaining = T - sum(alloc.values())

        # distribute remaining by priority then share, respecting maxs.
        # Use iterative largest-remainder style with priority tiers.
        # Build an order of candidate sources sorted by priority desc, share desc, name.
        order = sorted(
            names,
            key=lambda x: (-prios.get(x, 0), -shares.get(x, 0.0), x),
        )

        # First, allocate proportional to shares (floor) capped by headroom.
        headroom = {n: max(0, norm_max[n] - alloc[n]) for n in names}
        total_headroom = sum(headroom.values())
        if remaining > 0 and total_headroom > 0:
            # proportional floors
            weight_sum = sum(shares.get(n, 0.0) for n in names if headroom[n] > 0)
            if weight_sum <= 0:
                weight_sum = float(sum(1 for n in names if headroom[n] > 0))
                weights = {n: (1.0 if headroom[n] > 0 else 0.0) for n in names}
            else:
                weights = {n: (shares.get(n, 0.0) if headroom[n] > 0 else 0.0) for n in names}

            # floor allocation
            extra: dict[str, int] = {n: 0 for n in names}
            allocated_extra = 0
            # compute desired
            desired: dict[str, float] = {}
            for n in names:
                if headroom[n] <= 0:
                    desired[n] = 0.0
                else:
                    desired[n] = remaining * (weights[n] / weight_sum) if weight_sum > 0 else 0.0
            for n in names:
                extra[n] = min(headroom[n], int(desired[n] // 1))
                allocated_extra += extra[n]
            # apply
            for n in names:
                alloc[n] += extra[n]
            remaining -= allocated_extra
            headroom = {n: max(0, norm_max[n] - alloc[n]) for n in names}

            # redistribute leftover by priority order (one at a time) to respect maxs.
            # Sort by priority desc, share desc, name.
            leftover_order = sorted(
                names,
                key=lambda x: (-prios.get(x, 0), -shares.get(x, 0.0), x),
            )
            guard = 0
            max_guard = T + 5
            while remaining > 0 and guard < max_guard:
                progressed = False
                for n in leftover_order:
                    if remaining <= 0:
                        break
                    if alloc[n] < norm_max[n]:
                        alloc[n] += 1
                        remaining -= 1
                        progressed = True
                if not progressed:
                    # Infeasible max constraints: overflow into highest priority.
                    order = sorted(names, key=lambda x: (-prios.get(x, 0), x))
                    i = 0
                    guard2 = 0
                    while remaining > 0 and guard2 < T + 5:
                        n = order[i % len(order)] if order else None
                        if n is None:
                            break
                        alloc[n] = alloc.get(n, 0) + 1
                        remaining -= 1
                        i += 1
                        guard2 += 1
                    break
                guard += 1

        # Final pad (safety) to ensure exact total.
        self._pad_to_total(alloc, names, norm_max, T)

        # Reorder result to a stable key order: names order then any missing shares.
        result: dict[str, int] = {}
        for n in names:
            result[n] = int(alloc.get(n, 0))
        # pad again across all keys if needed
        self._pad_to_total(
            result, list(result.keys()), {k: norm_max.get(k, T) for k in result}, T
        )
        return result

    def _pad_to_total(
        self, alloc: dict[str, int], names: list[str], maxs: dict[str, int], T: int
    ) -> None:
        cur = sum(alloc.get(n, 0) for n in alloc)
        # remove excess if over
        if cur > T:
            order = sorted(names, key=lambda x: (prios_sort_key(self, x)))
            # reduce from lowest priority
            reduce_order = sorted(names, key=lambda x: (self.priorities.get(x, 0), x))
            idx = 0
            while cur > T and idx < len(reduce_order) * (T + 1):
                n = reduce_order[idx % len(reduce_order)] if reduce_order else None
                if n is None:
                    break
                if alloc.get(n, 0) > 0:
                    alloc[n] -= 1
                    cur -= 1
                idx += 1
        # add deficit
        cur = sum(alloc.get(n, 0) for n in alloc)
        if cur < T:
            order = sorted(names, key=lambda x: (-self.priorities.get(x, 0), x))
            guard = 0
            while cur < T and guard < T + 5:
                progressed = False
                for n in order:
                    if cur >= T:
                        break
                    cap = maxs.get(n, T)
                    if alloc.get(n, 0) < cap:
                        alloc[n] = alloc.get(n, 0) + 1
                        cur += 1
                        progressed = True
                if not progressed:
                    # Last resort: ignore caps to reach total.
                    order = sorted(names, key=lambda x: (-self.priorities.get(x, 0), x))
                    if not order:
                        order = list(names)
                    i = 0
                    g = 0
                    while cur < T and g < T + 5:
                        n = order[i % len(order)]
                        alloc[n] = alloc.get(n, 0) + 1
                        cur += 1
                        i += 1
                        g += 1
                    break
                guard += 1

    def overflow(self, used: dict[str, int]) -> dict[str, int]:
        alloc = self.allocate()
        return {k: max(0, used.get(k, 0) - alloc.get(k, 0)) for k in alloc}

    def redistribute_unused(
        self, used: dict[str, int], total: int | None = None
    ) -> dict[str, int]:
        """Return allocations after redistributing unused budget in priority order."""
        T = self._effective_total() if total is None else int(total)
        alloc = self.allocate()
        # compute unused per source
        unused = 0
        for n in alloc:
            u = used.get(n, 0)
            if u < alloc[n]:
                unused += alloc[n] - u
        # redistribute unused to compressible sources by priority
        if unused <= 0:
            return alloc
        # build candidate list
        candidates = sorted(
            [n for n in alloc if self._is_compressible(n)],
            key=lambda x: (-self.priorities.get(x, 0), x),
        )
        # add one at a time
        guard = 0
        while unused > 0 and guard < T + 5:
            progressed = False
            for n in candidates:
                if unused <= 0:
                    break
                cap = self.max_tokens.get(n, None)
                cap = T if cap is None else int(cap)
                if alloc[n] < cap:
                    alloc[n] += 1
                    unused -= 1
                    progressed = True
            if not progressed:
                break
            guard += 1
        # Keep the total at T without taking budget back from sources that are
        # not compressible (system prompts, the user request).
        current = sum(alloc.values())
        if current > T:
            excess = current - T
            reduce_order = sorted(
                candidates, key=lambda x: (self.priorities.get(x, 0), x)
            )
            idx = 0
            while excess > 0 and reduce_order:
                name = reduce_order[idx % len(reduce_order)]
                removable = max(0, alloc[name] - self.min_tokens.get(name, 0))
                give_back = min(removable, excess)
                alloc[name] -= give_back
                excess -= give_back
                idx += 1
                if idx > len(reduce_order) * (T + 5):
                    break
        elif current < T:
            deficit = T - current
            for name in candidates:
                if deficit <= 0:
                    break
                cap = self.max_tokens.get(name, None)
                cap = T if cap is None else int(cap)
                add = min(max(0, cap - alloc[name]), deficit)
                alloc[name] += add
                deficit -= add
        return alloc

    def _is_compressible(self, name: str) -> bool:
        return bool(self.compressible.get(name, True))



def build_context(sources: dict[str, str], total_tokens: int) -> dict[str, str]:
    """Trim each source to its budget (char proxy: 4 chars/token)."""
    # Ensure every provided source has a share so it can receive budget.
    shares = None
    if isinstance(sources, dict):
        missing = [n for n in sources if n not in DEFAULT_SHARES]
        if missing:
            shares = dict(DEFAULT_SHARES)
            for n in missing:
                shares[n] = 1.0 / max(1, len(sources))
    if shares is not None:
        budget = ContextBudget(total_tokens=total_tokens, shares=shares)
    else:
        budget = ContextBudget(total_tokens=total_tokens)
    alloc = budget.allocate(sources if isinstance(sources, dict) else None)
    # If sources keys differ from shares, allocate() may not include them;
    # ensure all source keys have a budget.
    for name in sources:
        if name not in alloc:
            alloc[name] = 0
    # redistribute if needed to cover all keys? keep simple: equal split of 0
    out: dict[str, str] = {}
    for name, text in sources.items():
        limit = alloc.get(name, 0) * 4
        out[name] = text[:limit] if limit > 0 else ""
    return out


def prios_sort_key(budget: ContextBudget, name: str):
    return (budget.priorities.get(name, 0), name)
