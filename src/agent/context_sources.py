"""Source adapters for the context engine (Phase 2).

Each adapter wraps an existing bounded-text builder and returns a
``ContextSource`` trimmed to the supplied token budget.  Existing helpers
remain the implementation details; these adapters only coordinate them with
the unified budget model.  They are no-ops for output when the engine is
disabled because callers keep using the original builders directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context_manager import ContextSource
from .prompts import _truncate_to_tokens, token_count

# Source priority defaults (higher preserved first).
SOURCE_PRIORITIES: dict[str, int] = {
    "system": 100,
    "tool_definitions": 95,
    "skills": 80,
    "plan": 75,
    "files": 70,
    "summary": 60,
    "episodic": 55,
    "history": 50,
    "rag": 45,
    "tool_results": 40,
    "user": 30,
}

# Whether a source is compressible by default.
SOURCE_COMPRESSIBLE: dict[str, bool] = {
    "system": False,
    "tool_definitions": False,
    "user": False,
}

# Cacheable (stable prefix) defaults.
SOURCE_CACHEABLE: set[str] = {
    "system",
    "tool_definitions",
    "skills",
    "plan",
    "files",
}


def trim_text_to_tokens(text: str, budget_tokens: int) -> str:
    """Trim ``text`` to at most ``budget_tokens`` tokens."""
    if budget_tokens is None or budget_tokens < 0:
        budget_tokens = 0
    if budget_tokens == 0:
        return ""
    if not text:
        return ""
    if token_count(text) <= budget_tokens:
        return text
    return _truncate_to_tokens(text, budget_tokens)


def _source(
    name: str,
    text: str,
    budget_tokens: int,
    *,
    priority: int | None = None,
    min_tokens: int = 0,
    max_tokens: int | None = None,
    compressible: bool | None = None,
    trim_strategy: str = "truncate",
    cacheable: bool | None = None,
) -> ContextSource:
    trimmed = trim_text_to_tokens(text, budget_tokens)
    return ContextSource(
        name=name,
        text=trimmed,
        priority=SOURCE_PRIORITIES.get(name, 0) if priority is None else priority,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        compressible=SOURCE_COMPRESSIBLE.get(name, True)
        if compressible is None
        else compressible,
        trim_strategy=trim_strategy,
        cacheable=(name in SOURCE_CACHEABLE) if cacheable is None else cacheable,
    )


def _messages_to_text(messages: Any) -> str:
    if messages is None:
        return ""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, (list, tuple)):
        lines = []
        for m in messages:
            content = getattr(m, "content", m)
            if isinstance(content, list):
                content = " ".join(str(x) for x in content)
            lines.append(str(content))
        return "\n".join(lines)
    return str(messages)


def _docs_to_text(documents: Any) -> str:
    if documents is None:
        return ""
    if isinstance(documents, str):
        return documents
    if isinstance(documents, (list, tuple)):
        parts = []
        for d in documents:
            content = getattr(d, "page_content", None)
            if content is None:
                content = getattr(d, "content", None)
            if content is None:
                content = str(d)
            parts.append(str(content))
        return "\n\n".join(parts)
    return str(documents)


# ---------------------------------------------------------------------------
# Individual adapters
# ---------------------------------------------------------------------------


def adapt_system(text: str = "", budget_tokens: int = 0, **kwargs: Any) -> ContextSource:
    return _source("system", text or "", budget_tokens, trim_strategy="truncate", **kwargs)


def adapt_summary(text: str = "", budget_tokens: int = 0, **kwargs: Any) -> ContextSource:
    return _source("summary", text or "", budget_tokens, trim_strategy="summarize", **kwargs)


def adapt_history(messages: Any = None, budget_tokens: int = 0, text: str = "", **kwargs: Any) -> ContextSource:
    src_text = text if text else _messages_to_text(messages)
    return _source("history", src_text, budget_tokens, trim_strategy="compact", **kwargs)


def adapt_episodic(text: str = "", budget_tokens: int = 0, **kwargs: Any) -> ContextSource:
    return _source("episodic", text or "", budget_tokens, trim_strategy="truncate", **kwargs)


def adapt_rag(documents: Any = None, budget_tokens: int = 0, text: str = "", **kwargs: Any) -> ContextSource:
    src_text = text if text else _docs_to_text(documents)
    return _source("rag", src_text, budget_tokens, trim_strategy="truncate", **kwargs)


def adapt_skills(text: str = "", budget_tokens: int = 0, **kwargs: Any) -> ContextSource:
    return _source("skills", text or "", budget_tokens, trim_strategy="truncate", **kwargs)


def adapt_plan(text: str = "", budget_tokens: int = 0, **kwargs: Any) -> ContextSource:
    return _source("plan", text or "", budget_tokens, trim_strategy="truncate", **kwargs)


def adapt_files(text: str = "", budget_tokens: int = 0, **kwargs: Any) -> ContextSource:
    return _source("files", text or "", budget_tokens, trim_strategy="truncate", **kwargs)


def adapt_tool_results(messages: Any = None, budget_tokens: int = 0, text: str = "", **kwargs: Any) -> ContextSource:
    src_text = text if text else _messages_to_text(messages)
    return _source("tool_results", src_text, budget_tokens, trim_strategy="summarize", **kwargs)


def adapt_user(text: str = "", budget_tokens: int = 0, **kwargs: Any) -> ContextSource:
    return _source("user", text or "", budget_tokens, trim_strategy="truncate", **kwargs)


def adapt_tool_definitions(text: str = "", budget_tokens: int = 0, **kwargs: Any) -> ContextSource:
    return _source("tool_definitions", text or "", budget_tokens, trim_strategy="truncate", **kwargs)


def adapt_memory(text: str = "", budget_tokens: int = 0, **kwargs: Any) -> ContextSource:
    return _source("memory", text or "", budget_tokens, trim_strategy="truncate", **kwargs)


# Generic adapter.
def adapt_source(name: str, text: str = "", budget_tokens: int = 0, **kwargs: Any) -> ContextSource:
    return _source(name, text or "", budget_tokens, **kwargs)


ADAPTERS: dict[str, Any] = {
    "system": adapt_system,
    "summary": adapt_summary,
    "history": adapt_history,
    "episodic": adapt_episodic,
    "rag": adapt_rag,
    "skills": adapt_skills,
    "plan": adapt_plan,
    "files": adapt_files,
    "tool_results": adapt_tool_results,
    "user": adapt_user,
    "tool_definitions": adapt_tool_definitions,
    "memory": adapt_memory,
}


def get_adapter(name: str):
    return ADAPTERS.get(name, lambda text="", budget_tokens=0, **kw: adapt_source(name, text, budget_tokens, **kw))


def build_source(name: str, *args: Any, budget_tokens: int = 0, text: str = "", **kwargs: Any) -> ContextSource:
    adapter = get_adapter(name)
    # Pass through kwargs; if text provided prefer it.
    try:
        return adapter(text=text, budget_tokens=budget_tokens, **kwargs)
    except TypeError:
        return adapter(*args, budget_tokens=budget_tokens, text=text, **kwargs)


# ---------------------------------------------------------------------------
# Class-style adapters for callers that prefer objects.
# ---------------------------------------------------------------------------


@dataclass
class SourceAdapter:
    name: str

    def build(self, *, budget_tokens: int = 0, text: str = "", **kwargs: Any) -> ContextSource:
        return adapt_source(self.name, text=text, budget_tokens=budget_tokens, **kwargs)


class SystemSourceAdapter(SourceAdapter):
    def __init__(self):
        super().__init__(name="system")

    def build(self, *, budget_tokens: int = 0, text: str = "", **kwargs: Any) -> ContextSource:
        return adapt_system(text=text, budget_tokens=budget_tokens, **kwargs)


class HistorySourceAdapter(SourceAdapter):
    def __init__(self):
        super().__init__(name="history")

    def build(self, *, budget_tokens: int = 0, messages: Any = None, text: str = "", **kwargs: Any) -> ContextSource:
        return adapt_history(messages=messages, text=text, budget_tokens=budget_tokens, **kwargs)


class RagSourceAdapter(SourceAdapter):
    def __init__(self):
        super().__init__(name="rag")

    def build(self, *, budget_tokens: int = 0, documents: Any = None, text: str = "", **kwargs: Any) -> ContextSource:
        return adapt_rag(documents=documents, text=text, budget_tokens=budget_tokens, **kwargs)


class MemorySourceAdapter(SourceAdapter):
    def __init__(self):
        super().__init__(name="memory")

    def build(self, *, budget_tokens: int = 0, text: str = "", **kwargs: Any) -> ContextSource:
        return adapt_memory(text=text, budget_tokens=budget_tokens, **kwargs)


class SkillsSourceAdapter(SourceAdapter):
    def __init__(self):
        super().__init__(name="skills")

    def build(self, *, budget_tokens: int = 0, text: str = "", **kwargs: Any) -> ContextSource:
        return adapt_skills(text=text, budget_tokens=budget_tokens, **kwargs)


class PlanSourceAdapter(SourceAdapter):
    def __init__(self):
        super().__init__(name="plan")

    def build(self, *, budget_tokens: int = 0, text: str = "", **kwargs: Any) -> ContextSource:
        return adapt_plan(text=text, budget_tokens=budget_tokens, **kwargs)


class ToolResultsSourceAdapter(SourceAdapter):
    def __init__(self):
        super().__init__(name="tool_results")

    def build(self, *, budget_tokens: int = 0, messages: Any = None, text: str = "", **kwargs: Any) -> ContextSource:
        return adapt_tool_results(messages=messages, text=text, budget_tokens=budget_tokens, **kwargs)


# Aliases for alternate naming conventions.
adapt_history_source = adapt_history
adapt_rag_source = adapt_rag
adapt_memory_source = adapt_memory
adapt_skills_source = adapt_skills
adapt_plan_source = adapt_plan
adapt_tool_results_source = adapt_tool_results
adapt_system_source = adapt_system
adapt_summary_source = adapt_summary
adapt_files_source = adapt_files
adapt_user_source = adapt_user


__all__ = [
    "SOURCE_PRIORITIES",
    "SOURCE_COMPRESSIBLE",
    "SOURCE_CACHEABLE",
    "trim_text_to_tokens",
    "adapt_system",
    "adapt_summary",
    "adapt_history",
    "adapt_episodic",
    "adapt_rag",
    "adapt_skills",
    "adapt_plan",
    "adapt_files",
    "adapt_tool_results",
    "adapt_user",
    "adapt_tool_definitions",
    "adapt_memory",
    "adapt_source",
    "get_adapter",
    "build_source",
    "ADAPTERS",
    "SourceAdapter",
    "SystemSourceAdapter",
    "HistorySourceAdapter",
    "RagSourceAdapter",
    "MemorySourceAdapter",
    "SkillsSourceAdapter",
    "PlanSourceAdapter",
    "ToolResultsSourceAdapter",
]
