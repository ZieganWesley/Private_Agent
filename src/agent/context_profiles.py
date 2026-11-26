"""Request-aware context profiles (Phase 4).

Profiles adjust per-source budget shares based on the request type.  Selection
is deterministic and falls back to ``chat`` for unknown inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .context_manager import DEFAULT_SHARES, resolve_shares

PROFILE_NAMES: tuple[str, ...] = ("chat", "coding", "research", "memory")

# Share overrides per profile (only keys that differ from the base).
PROFILE_SHARE_OVERRIDES: dict[str, dict[str, float]] = {
    "chat": {},
    "coding": {
        "files": 0.24,
        "plan": 0.14,
        "history": 0.24,
    },
    "research": {
        "rag": 0.25,
        "episodic": 0.09,
        "summary": 0.06,
        "history": 0.22,
    },
    "memory": {
        "episodic": 0.18,
        "summary": 0.12,
        "history": 0.22,
    },
}

# Heuristic keyword sets.
_CODING_KEYWORDS = {
    "code", "coding", "implement", "refactor", "debug", "test", "function",
    "class", "module", "file", "repository", "git", "build", "compile",
    "python", "javascript", "typescript", "bug", "error", "stacktrace",
}
_RESEARCH_KEYWORDS = {
    "research", "search", "find", "lookup", "documentation", "docs", "paper",
    "article", "source", "reference", "investigate", "explore", "web",
}
_MEMORY_KEYWORDS = {
    "remember", "recall", "memory", "previous", "earlier", "history",
    "episode", "past", "summary", "context",
}


@dataclass
class ContextProfile:
    name: str
    shares: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SHARES))
    description: str = ""

    def normalized_shares(self) -> dict[str, float]:
        # Ensure all default keys present and sum to ~1.
        result = dict(DEFAULT_SHARES)
        result.update(self.shares)
        total = sum(result.values())
        if total <= 0:
            return dict(DEFAULT_SHARES)
        # normalize to sum 1.0
        return {k: v / total for k, v in result.items()}


def _keyword_score(text: str, keywords: set[str]) -> int:
    lowered = (text or "").lower()
    tokens = set(re_split(lowered))
    return len(tokens & keywords)


def re_split(text: str) -> list[str]:
    import re

    return re.findall(r"[a-zA-Z0-9_]+", text)


def select_profile(request_text: str = "", explicit: str | None = None) -> str:
    """Deterministically select a profile name."""
    if explicit:
        cand = str(explicit).strip().lower()
        if cand in PROFILE_NAMES:
            return cand
        if cand == "auto":
            pass
        else:
            return "chat"
    text = request_text or ""
    scores = {
        "coding": _keyword_score(text, _CODING_KEYWORDS),
        "research": _keyword_score(text, _RESEARCH_KEYWORDS),
        "memory": _keyword_score(text, _MEMORY_KEYWORDS),
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "chat"
    # tie-break: coding > research > memory
    max_score = scores[best]
    candidates = [k for k in ("coding", "research", "memory") if scores[k] == max_score]
    if len(candidates) > 1:
        for preferred in ("coding", "research", "memory"):
            if preferred in candidates:
                return preferred
    return best


def profile_shares(
    profile: str, base: dict[str, float] | None = None, overrides: dict[str, Any] | None = None
) -> dict[str, float]:
    """Return shares for a profile, applying optional config overrides."""
    if base is None:
        base = dict(DEFAULT_SHARES)
    result = resolve_shares(base)
    name = profile if profile in PROFILE_NAMES else "chat"
    result.update(resolve_shares(PROFILE_SHARE_OVERRIDES.get(name, {})))
    # config overrides: overrides may be {profile: {source: share}} or {source: share}
    if overrides:
        if isinstance(overrides, dict):
            if name in overrides and isinstance(overrides[name], dict):
                result.update(resolve_shares(overrides[name]))
            else:
                # treat as direct source overrides
                result.update(
                    resolve_shares(
                        {k: v for k, v in overrides.items() if isinstance(v, (int, float))}
                    )
                )
    # normalize
    total = sum(result.values())
    if total <= 0:
        return dict(base)
    return {k: v / total for k, v in result.items()}


def build_profile(profile: str = "chat", **kwargs: Any) -> ContextProfile:
    shares = profile_shares(profile, overrides=kwargs.get("overrides"))
    return ContextProfile(name=profile if profile in PROFILE_NAMES else "chat", shares=shares)


def apply_profile_to_budget(budget: Any, profile: str, request_text: str = "", overrides: dict[str, Any] | None = None) -> Any:
    """Mutate/return a ContextBudget with profile shares applied."""
    if profile == "auto":
        profile = select_profile(request_text)
    shares = profile_shares(profile, overrides=overrides)
    try:
        budget.shares = shares
    except Exception:  # noqa: BLE001 - optional best-effort context feature
        pass
    return budget


# Config loading helper: read profile overrides from a config mapping.
def load_profile_overrides(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    # support agent.context_profiles
    agent = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
    cp = agent.get("context_profiles", {}) if isinstance(agent.get("context_profiles"), dict) else {}
    # also top-level
    if not cp and isinstance(config.get("context_profiles"), dict):
        cp = config["context_profiles"]
    return cp if isinstance(cp, dict) else {}


def load_share_overrides(config: dict[str, Any] | None) -> dict[str, float]:
    """Read per-source share overrides from ``[agent.context_shares]``."""
    if not isinstance(config, dict):
        return {}
    agent = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    shares = agent.get("context_shares", {}) if isinstance(agent, dict) else {}
    if not shares and isinstance(config.get("context_shares"), dict):
        shares = config["context_shares"]
    if not isinstance(shares, dict):
        return {}
    cleaned = {
        str(key): float(value)
        for key, value in shares.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
    }
    return resolve_shares(cleaned)


__all__ = [
    "PROFILE_NAMES",
    "PROFILE_SHARE_OVERRIDES",
    "ContextProfile",
    "select_profile",
    "profile_shares",
    "build_profile",
    "apply_profile_to_budget",
    "load_profile_overrides",
    "load_share_overrides",
]
