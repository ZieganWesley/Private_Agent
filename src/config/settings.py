"""Typed, validated settings derived from the per-user configuration."""

from __future__ import annotations

import os
import pathlib
import posixpath
import re
import urllib.parse
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hardware import normalize_acceleration_mode

from .defaults import (
    DEFAULT_CONFIG,
    DEFAULT_PATHS,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT_PATH,  # noqa: F401  (re-exported for compatibility)
)


def default_for(section: str, key: str) -> Any:
    """Return the generated default for one settings key; defaults live once."""
    return DEFAULT_CONFIG[section][key]


def config_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one config section, tolerating missing or malformed values."""
    value = config.get(name, {})
    return value if isinstance(value, Mapping) else {}


def configured_str(
    config: Mapping[str, Any],
    section: str,
    key: str,
    default: str,
) -> str:
    """Return a stripped string setting, tolerating missing or malformed values."""
    raw = config_section(config, section).get(key, default)
    return str(raw).strip()


def configured_choice(
    config: Mapping[str, Any],
    section: str,
    key: str,
    default: str,
    choices: tuple[str, ...],
) -> str:
    """Return a lower-cased setting, falling back to `default` when unknown."""
    value = configured_str(config, section, key, default).lower()
    return value if value in choices else default


def configured_path(value: Any, default: str) -> str:
    """Expand config paths and anchor relative paths to the user's home."""
    path = pathlib.Path(str(value or default)).expanduser()
    if not path.is_absolute():
        path = pathlib.Path.home() / path
    return str(path.resolve())


def positive_int(config: Mapping[str, Any], key: str, default: int) -> int:
    try:
        value = int(config.get(key, default))
        return value if value > 0 else default
    except TypeError, ValueError:
        return default


def positive_float(config: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(config.get(key, default))
        return value if value > 0 else default
    except TypeError, ValueError:
        return default


def configured_int(
    config: Mapping[str, Any], section: str, key: str, default: int
) -> int:
    try:
        value = int(config_section(config, section).get(key, default))
        return value if value > 0 else default
    except AttributeError, TypeError, ValueError:
        return default


def configured_nonnegative_int(
    config: Mapping[str, Any], section: str, key: str, default: int
) -> int:
    try:
        value = int(config_section(config, section).get(key, default))
        return value if value >= 0 else default
    except AttributeError, TypeError, ValueError:
        return default


def configured_int_range(
    config: Mapping[str, Any],
    section: str,
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = configured_int(config, section, key, default)
    return value if minimum <= value <= maximum else default


def configured_bool(
    config: Mapping[str, Any], section: str, key: str, default: bool
) -> bool:
    value = config_section(config, section).get(key, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        return default
    return value if isinstance(value, bool) else default


def configured_float(
    config: Mapping[str, Any], section: str, key: str, default: float
) -> float:
    try:
        value = float(config_section(config, section).get(key, default))
        return value if value > 0 else default
    except AttributeError, TypeError, ValueError:
        return default


# Flat legacy keys that used to be read directly from the config root. The
# categorized default config stores the same values under a section, so a flat key
# is only consulted when that section does not set the canonical key.
_LEGACY_FLAT_KEYS: dict[str, tuple[str, str]] = {
    "default_db_path": ("paths", "database"),
    "workspace_root": ("paths", "workspace"),
    "code_output_root": ("paths", "code_output"),
    "embedding_model": ("models", "embedding_model"),
    "ollama_base_url": ("models", "ollama_base_url"),
    "ollama_base_urls": ("models", "ollama_base_urls"),
    "hardware_acceleration": ("models", "hardware_acceleration"),
    "preferred_model": ("models", "preferred_model"),
    "max_tool_iterations": ("agent", "max_tool_iterations"),
    "max_tool_calls": ("agent", "max_tool_calls"),
    "max_task_seconds": ("agent", "max_task_seconds"),
    "max_tool_output_chars": ("agent", "max_tool_output_chars"),
    "max_history_messages": ("agent", "max_history_messages"),
    "max_context_tokens": ("agent", "max_context_tokens"),
    "conversation_retention_days": ("memory", "conversation_retention_days"),
    "episode_retention_days": ("memory", "episode_retention_days"),
    "registered_skill_retention_days": ("memory", "registered_skill_retention_days"),
    "task_stale_after_days": ("memory", "task_stale_after_days"),
    "max_summary_chars": ("memory", "max_summary_chars"),
    "max_network_concurrency": ("network", "max_concurrent_requests"),
    "network_request_timeout": ("network", "request_timeout_seconds"),
    "rag_max_file_bytes": ("rag", "max_file_bytes"),
    "rag_max_corpus_bytes": ("rag", "max_corpus_bytes"),
    "rag_max_pdf_pages": ("rag", "max_pdf_pages"),
    "rag_max_documents": ("rag", "max_documents"),
    "enable_web_research": ("network", "web_research_enabled"),
    "web_research_consent": ("network", "web_research_consent"),
    "internet_check_host": ("network", "internet_check_host"),
    "internet_check_port": ("network", "internet_check_port"),
    "internet_check_timeout": ("network", "internet_check_timeout_seconds"),
    "thinking_toggle_default": ("models", "thinking_enabled_by_default"),
    "thinking_effort_default": ("models", "thinking_effort"),
    "agent_permission_mode": ("agent", "permission_mode"),
    "DEBUG_LOG_ENABLED": ("logging", "debug_enabled"),
    "mcpServers": ("mcp", "servers"),
    "mcp_auto_approve_tools": ("mcp", "auto_approve_tools"),
}


def materialize_legacy_keys(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy flat legacy keys into their section when the section key is unset."""
    materialized = dict(config)
    for legacy_key, (section, key) in _LEGACY_FLAT_KEYS.items():
        if legacy_key not in config:
            continue
        values = dict(config_section(materialized, section))
        values.setdefault(key, config[legacy_key])
        materialized[section] = values
    return materialized


def first_configured_bool(
    config: Mapping[str, Any],
    candidates: tuple[tuple[str, str], ...],
    default: bool,
) -> bool:
    """Return the first explicitly configured alias, so false can defeat defaults."""
    for section, key in candidates:
        if key in config_section(config, section):
            return configured_bool(config, section, key, default)
    return default


def resolve_config_values(
    config: Mapping[str, Any],
    *,
    default_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve one merged config mapping into validated, runtime-ready values."""
    default_paths = DEFAULT_PATHS if default_paths is None else default_paths
    config = materialize_legacy_keys(config)

    DEFAULT_DB_PATH = os.path.expanduser(
        str(config_section(config, "paths").get("database", default_paths["database"]))
    )
    WORKSPACE_ROOT_DEFAULT = str(
        config_section(config, "paths").get("workspace", default_paths["workspace"])
    )
    CODE_OUTPUT_ROOT = str(
        config_section(config, "paths").get("code_output", default_paths["code_output"])
    )
    SKILLS_FOLDER_DEFAULT = configured_path(
        config_section(config, "paths").get("skills"),
        default_paths["skills"],
    )
    RAG_DOCS_DEFAULT = configured_path(
        config_section(config, "paths").get("rag_documents"),
        default_paths["rag_documents"],
    )

    MODEL_TEMPERATURE = configured_float(
        config, "models", "temperature", default_for("models", "temperature")
    )

    ONLINE_PERMISSION_OVERRIDE = configured_bool(
        config,
        "models",
        "online_permission_override",
        default_for("models", "online_permission_override"),
    )
    EMBEDDING_MODEL = configured_str(
        config, "models", "embedding_model", default_for("models", "embedding_model")
    )
    EMBEDDING_PROVIDER = configured_choice(
        config,
        "rag",
        "embedding_provider",
        default_for("rag", "embedding_provider"),
        ("auto", "none"),
    )
    EMBEDDING_BASE_URL = str(
        config_section(config, "rag").get(
            "embedding_base_url", "http://127.0.0.1:1234/v1"
        )
    )
    RAG_INDEX_PATH = configured_path(
        config_section(config, "paths").get("rag_index"),
        default_paths["rag_index"],
    )
    _models_section = config_section(config, "models")
    _raw_ollama_urls = _models_section.get(
        "ollama_base_urls",
        _models_section.get("ollama_base_url", "http://localhost:11434"),
    )
    if isinstance(_raw_ollama_urls, str):
        OLLAMA_BASE_URLS = (
            [_raw_ollama_urls.strip()] if _raw_ollama_urls.strip() else []
        )
    elif isinstance(_raw_ollama_urls, list):
        OLLAMA_BASE_URLS = [
            str(item).strip()
            for item in _raw_ollama_urls
            if isinstance(item, str) and str(item).strip()
        ]
    else:
        OLLAMA_BASE_URLS = []
    if not OLLAMA_BASE_URLS:
        import shutil

        if shutil.which("ollama"):
            OLLAMA_BASE_URLS = ["http://localhost:11434"]
    OLLAMA_BASE_URL = OLLAMA_BASE_URLS[0] if OLLAMA_BASE_URLS else ""
    HARDWARE_ACCELERATION_MODE = normalize_acceleration_mode(
        _models_section.get("hardware_acceleration", "auto")
    )
    PREFERRED_MODEL = _models_section.get("preferred_model")

    _agent_section = config_section(config, "agent")
    MAX_TOOL_ITERATIONS = positive_int(_agent_section, "max_tool_iterations", 15)
    MAX_TOOL_CALLS = positive_int(_agent_section, "max_tool_calls", 60)
    MAX_TASK_SECONDS = positive_int(_agent_section, "max_task_seconds", 600)
    MAX_TOOL_OUTPUT_CHARS = positive_int(_agent_section, "max_tool_output_chars", 12000)
    MAX_HISTORY_MESSAGES = positive_int(_agent_section, "max_history_messages", 20)
    MAX_CONTEXT_TOKENS = positive_int(_agent_section, "max_context_tokens", 12000)
    CONTEXT_COMPACTION_SOURCE_TOKENS = configured_nonnegative_int(
        config,
        "agent",
        "compaction_source_tokens",
        default_for("agent", "compaction_source_tokens"),
    )
    if not CONTEXT_COMPACTION_SOURCE_TOKENS:
        # Derive the transcript cap from the context window; a fixed cap wastes
        # large windows and overflows small ones.
        CONTEXT_COMPACTION_SOURCE_TOKENS = max(1000, MAX_CONTEXT_TOKENS // 2)
    TOOL_RESULT_PREVIEW_CHARS = configured_int_range(
        config,
        "agent",
        "tool_result_preview_chars",
        default_for("agent", "tool_result_preview_chars"),
        100,
        20000,
    )
    TOOL_ARGUMENT_PREVIEW_CHARS = configured_int_range(
        config,
        "agent",
        "tool_argument_preview_chars",
        default_for("agent", "tool_argument_preview_chars"),
        50,
        2000,
    )
    SEARCH_MAX_FILES = configured_int_range(
        config,
        "tools",
        "search_max_files",
        default_for("tools", "search_max_files"),
        100,
        100000,
    )
    SEARCH_MAX_BYTES = configured_int_range(
        config,
        "tools",
        "search_max_bytes",
        default_for("tools", "search_max_bytes"),
        1024**2,
        256 * 1024**2,
    )
    SEARCH_OUTPUT_CHARS = configured_int_range(
        config,
        "tools",
        "search_output_chars",
        default_for("tools", "search_output_chars"),
        1000,
        100000,
    )
    TASK_CITATION_BUDGET_CHARS = configured_int_range(
        config,
        "tools",
        "task_citation_budget_chars",
        default_for("tools", "task_citation_budget_chars"),
        1000,
        50000,
    )
    MAX_OUTPUT_TOKENS = configured_int_range(
        config,
        "agent",
        "max_output_tokens",
        default_for("agent", "max_output_tokens"),
        1,
        32768,
    )
    STREAMING_OUTPUT = configured_bool(
        config, "agent", "streaming_output", default_for("agent", "streaming_output")
    )
    VISIBLE_REASONING = configured_bool(
        config, "agent", "visible_reasoning", default_for("agent", "visible_reasoning")
    )
    VERBOSE_TIMINGS = configured_bool(
        config, "agent", "verbose_timings", default_for("agent", "verbose_timings")
    )
    STATUS_BAR = configured_bool(
        config, "agent", "status_bar", default_for("agent", "status_bar")
    )
    SHOW_WELCOME_SCREEN = first_configured_bool(
        config,
        (
            ("ui", "show_welcome_screen"),
            ("agent", "show_welcome_screen"),
            ("ui", "welcome_screen"),
            ("agent", "welcome_screen"),
        ),
        True,
    )
    INFORMATIVE_PROMPT = first_configured_bool(
        config,
        (("ui", "informative_prompt"), ("agent", "informative_prompt")),
        True,
    )
    SHOW_TURN_DIVIDER = first_configured_bool(
        config,
        (
            ("ui", "show_turn_divider"),
            ("agent", "show_turn_divider"),
            ("ui", "turn_divider"),
            ("agent", "turn_divider"),
        ),
        True,
    )
    SHOW_TURN_STATS = first_configured_bool(
        config,
        (
            ("ui", "show_turn_stats"),
            ("agent", "show_turn_stats"),
            ("ui", "turn_stats"),
            ("agent", "turn_stats"),
            ("ui", "per_turn_tokens"),
            ("agent", "per_turn_tokens"),
        ),
        True,
    )
    GROUPED_HELP = first_configured_bool(
        config,
        (("ui", "grouped_help"), ("agent", "grouped_help")),
        True,
    )
    _configured_system_prompt = config_section(config, "agent").get("system_prompt")
    PROJECT_INSTRUCTIONS_FILE = config_section(config, "agent").get(
        "project_instructions_file", "AGENTS.md"
    )
    PERMISSION_ALLOW_RULES = list(
        config_section(config, "permissions").get("allow", []) or []
    )
    PERMISSION_DENY_RULES = list(
        config_section(config, "permissions").get("deny", []) or []
    )
    APPROVED_EXTRA_FOLDERS = list(
        config_section(config, "permissions").get("approved_extra_folders", []) or []
    )
    SANDBOX_NETWORK_DEFAULT = first_configured_bool(
        config,
        (("permissions", "sandbox_network"), ("sandbox", "network")),
        False,
    )
    SYSTEM_PROMPT = (
        _configured_system_prompt.strip()
        if isinstance(_configured_system_prompt, str)
        and _configured_system_prompt.strip()
        else DEFAULT_SYSTEM_PROMPT
    )
    CONVERSATION_RETENTION_DAYS = configured_nonnegative_int(
        config,
        "memory",
        "conversation_retention_days",
        default_for("memory", "conversation_retention_days"),
    )
    EPISODE_RETENTION_DAYS = configured_nonnegative_int(
        config,
        "memory",
        "episode_retention_days",
        default_for("memory", "episode_retention_days"),
    )
    REGISTERED_SKILL_RETENTION_DAYS = configured_nonnegative_int(
        config,
        "memory",
        "registered_skill_retention_days",
        default_for("memory", "registered_skill_retention_days"),
    )
    TASK_STALE_AFTER_DAYS = configured_nonnegative_int(
        config,
        "memory",
        "task_stale_after_days",
        default_for("memory", "task_stale_after_days"),
    )
    MAX_SUMMARY_CHARS = configured_int(
        config,
        "memory",
        "max_summary_chars",
        default_for("memory", "max_summary_chars"),
    )
    MAX_LEARNED_ITEM_CHARS = configured_int_range(
        config,
        "memory",
        "max_learned_item_chars",
        default_for("memory", "max_learned_item_chars"),
        1,
        10000,
    )
    MAX_RESUME_STATE_CHARS = configured_int_range(
        config,
        "memory",
        "max_resume_state_chars",
        default_for("memory", "max_resume_state_chars"),
        256,
        100000,
    )
    MAX_MEMORY_CONTEXT_TOKENS = configured_int_range(
        config,
        "memory",
        "max_memory_context_tokens",
        default_for("memory", "max_memory_context_tokens"),
        128,
        12000,
    )
    MAX_RELEVANT_EPISODES = configured_int_range(
        config,
        "memory",
        "max_relevant_episodes",
        default_for("memory", "max_relevant_episodes"),
        1,
        20,
    )
    LEARNED_ITEM_EXPIRY_DAYS = configured_int_range(
        config,
        "memory",
        "learned_item_expiry_days",
        default_for("memory", "learned_item_expiry_days"),
        1,
        3650,
    )
    LEARNED_ITEM_REVIEW_DAYS = configured_int_range(
        config,
        "memory",
        "learned_item_review_days",
        default_for("memory", "learned_item_review_days"),
        1,
        3650,
    )
    DEFAULT_HISTORY_MESSAGES = configured_int(
        config,
        "memory",
        "default_history_messages",
        default_for("memory", "default_history_messages"),
    )
    DEFAULT_EPISODIC_SUMMARIES = configured_int(
        config,
        "memory",
        "default_episodic_summaries",
        default_for("memory", "default_episodic_summaries"),
    )
    MAX_NETWORK_CONCURRENCY = configured_int(
        config,
        "network",
        "max_concurrent_requests",
        default_for("network", "max_concurrent_requests"),
    )
    NETWORK_REQUEST_TIMEOUT = configured_int(
        config,
        "network",
        "request_timeout_seconds",
        default_for("network", "request_timeout_seconds"),
    )
    RAG_MAX_FILE_BYTES = configured_int(
        config, "rag", "max_file_bytes", 5 * 1024 * 1024
    )
    RAG_MAX_CORPUS_BYTES = configured_int(
        config, "rag", "max_corpus_bytes", 25 * 1024 * 1024
    )
    RAG_MAX_PDF_PAGES = configured_int(
        config, "rag", "max_pdf_pages", default_for("rag", "max_pdf_pages")
    )
    RAG_MAX_DOCUMENTS = configured_int(
        config, "rag", "max_documents", default_for("rag", "max_documents")
    )
    RAG_CHUNK_SIZE_CHARS = configured_int(
        config, "rag", "chunk_size_chars", default_for("rag", "chunk_size_chars")
    )
    RAG_CHUNK_OVERLAP_CHARS = configured_int(
        config, "rag", "chunk_overlap_chars", default_for("rag", "chunk_overlap_chars")
    )
    CHUNKING_STRATEGY = (
        str(config_section(config, "chunking").get("strategy", "character"))
        .strip()
        .lower()
    )
    CHUNKING_UNIT = (
        str(config_section(config, "chunking").get("unit", "chars")).strip().lower()
    )
    # Unset chunk_size/chunk_overlap fall back to the legacy rag.chunk_* keys.
    CHUNKING_SIZE = configured_int(
        config, "chunking", "chunk_size", RAG_CHUNK_SIZE_CHARS
    )
    CHUNKING_OVERLAP = configured_nonnegative_int(
        config, "chunking", "chunk_overlap", RAG_CHUNK_OVERLAP_CHARS
    )
    CHUNKING_MIN_CHUNK_CHARS = configured_nonnegative_int(
        config,
        "chunking",
        "min_chunk_chars",
        default_for("chunking", "min_chunk_chars"),
    )
    _raw_separators = config_section(config, "chunking").get("separators")
    CHUNKING_SEPARATORS = (
        tuple(_raw_separators)
        if isinstance(_raw_separators, list)
        and _raw_separators
        and all(isinstance(item, str) for item in _raw_separators)
        else ("\n\n", "\n", ". ", " ")
    )
    CHUNKING_SEMANTIC_PERCENTILE = min(
        100.0,
        configured_float(
            config,
            "chunking",
            "semantic_breakpoint_percentile",
            default_for("chunking", "semantic_breakpoint_percentile"),
        ),
    )
    CHUNKING_SEMANTIC_BUFFER = configured_nonnegative_int(
        config,
        "chunking",
        "semantic_buffer_sentences",
        default_for("chunking", "semantic_buffer_sentences"),
    )
    CHUNKING_SEMANTIC_MAX_SENTENCES = configured_int(
        config,
        "chunking",
        "semantic_max_sentences",
        default_for("chunking", "semantic_max_sentences"),
    )
    RAG_SIMILARITY_RESULTS = configured_int(
        config, "rag", "similarity_results", default_for("rag", "similarity_results")
    )
    _raw_rag_file_types = config_section(config, "rag").get("file_types", [])
    RAG_FILE_TYPES = []
    if isinstance(_raw_rag_file_types, list):
        for _suffix in _raw_rag_file_types[:50]:
            if not isinstance(_suffix, str):
                continue
            _normalized = _suffix.strip().lower()
            if _normalized and not _normalized.startswith("."):
                _normalized = f".{_normalized}"
            if 2 <= len(_normalized) <= 16 and all(
                character.isalnum() or character == "." for character in _normalized
            ):
                RAG_FILE_TYPES.append(_normalized)
    if not RAG_FILE_TYPES:
        RAG_FILE_TYPES = list(default_for("rag", "file_types"))
    RAG_SQLITE_ROWS_PER_CHUNK = configured_int_range(
        config,
        "rag",
        "sqlite_rows_per_chunk",
        default_for("rag", "sqlite_rows_per_chunk"),
        1,
        1000,
    )
    RAG_SQLITE_MAX_CELL_CHARS = configured_int_range(
        config,
        "rag",
        "sqlite_max_cell_chars",
        default_for("rag", "sqlite_max_cell_chars"),
        20,
        10000,
    )
    RAG_SQLITE_MAX_ROWS_PER_TABLE = configured_int(
        config,
        "rag",
        "sqlite_max_rows_per_table",
        default_for("rag", "sqlite_max_rows_per_table"),
    )
    RAG_AUTO_REFRESH_SECONDS = configured_nonnegative_int(
        config,
        "rag",
        "auto_refresh_seconds",
        default_for("rag", "auto_refresh_seconds"),
    )
    _network_section = config_section(config, "network")
    _raw_web_research = _network_section.get("web_research_enabled", True)
    if isinstance(_raw_web_research, str):
        ENABLE_WEB_RESEARCH = _raw_web_research.lower() in ("true", "1", "yes", "on")
    else:
        ENABLE_WEB_RESEARCH = bool(_raw_web_research)
    WEB_RESEARCH_CONSENT = str(
        _network_section.get("web_research_consent", "ask")
    ).lower()
    if WEB_RESEARCH_CONSENT not in {"ask", "session", "never"}:
        WEB_RESEARCH_CONSENT = "ask"
    INTERNET_CHECK_HOST = str(_network_section.get("internet_check_host", "1.1.1.1"))
    INTERNET_CHECK_PORT = positive_int(_network_section, "internet_check_port", 443)
    INTERNET_CHECK_TIMEOUT = positive_float(
        _network_section, "internet_check_timeout_seconds", 3.0
    )

    _sandbox_section = config_section(config, "sandbox")
    SANDBOX_EXTRA_RISKY_PROGRAMS = []
    _raw_extra_programs = _sandbox_section.get("extra_risky_programs", [])
    if isinstance(_raw_extra_programs, list):
        for _name in _raw_extra_programs[:100]:
            if not isinstance(_name, str):
                continue
            _candidate = _name.strip().lower()
            if re.fullmatch(r"[a-z0-9._+-]{1,64}", _candidate):
                SANDBOX_EXTRA_RISKY_PROGRAMS.append(_candidate)
    SANDBOX_EXTRA_SYSTEM_PATHS = []
    _raw_extra_paths = _sandbox_section.get("extra_system_paths", [])
    if isinstance(_raw_extra_paths, list):
        for _path in _raw_extra_paths[:50]:
            if not isinstance(_path, str):
                continue
            _candidate = posixpath.normpath(_path.strip())
            if _candidate.startswith("/") and ".." not in _candidate:
                SANDBOX_EXTRA_SYSTEM_PATHS.append(_candidate)
    RAG_EXCLUDED_FILE_TYPES = []
    _raw_excluded_types = config_section(config, "rag").get("excluded_file_types", [])
    if isinstance(_raw_excluded_types, list):
        for _suffix in _raw_excluded_types[:50]:
            if not isinstance(_suffix, str):
                continue
            _normalized = _suffix.strip().lower()
            if _normalized and not _normalized.startswith("."):
                _normalized = f".{_normalized}"
            if 2 <= len(_normalized) <= 16:
                RAG_EXCLUDED_FILE_TYPES.append(_normalized)
    _raw_override_tool_list = config_section(config, "tools").get(
        "override_tool_list", []
    )
    if isinstance(_raw_override_tool_list, list):
        OVERRIDE_TOOL_LIST = frozenset(
            name.strip()
            for name in _raw_override_tool_list
            if isinstance(name, str) and name.strip()
        )
    else:
        OVERRIDE_TOOL_LIST = frozenset()

    _raw_thinking = _models_section.get("thinking_enabled_by_default", False)
    if isinstance(_raw_thinking, str):
        THINKING_TOGGLE_DEFAULT = _raw_thinking.lower() in ("true", "1", "yes", "on")
    else:
        THINKING_TOGGLE_DEFAULT = bool(_raw_thinking)

    THINKING_EFFORT_DEFAULT = str(
        _models_section.get("thinking_effort", "medium")
    ).lower()
    if THINKING_EFFORT_DEFAULT not in {"low", "medium", "high"}:
        THINKING_EFFORT_DEFAULT = "medium"

    AGENT_PERMISSION_MODE = str(_agent_section.get("permission_mode", "auto")).lower()
    if AGENT_PERMISSION_MODE not in {"manual", "auto", "full"}:
        AGENT_PERMISSION_MODE = "auto"

    _env_debug = os.environ.get("PRIVATE_AGENT_DEBUG", "")
    if _env_debug.strip().lower() in {"1", "true", "yes", "on"}:
        DEBUG_LOG_ENABLED = True
    else:
        _debug_log_enabled = config_section(config, "logging").get("debug_enabled", 0)
        if isinstance(_debug_log_enabled, str):
            DEBUG_LOG_ENABLED = _debug_log_enabled.strip() == "1"
        else:
            DEBUG_LOG_ENABLED = _debug_log_enabled == 1 or _debug_log_enabled is True

    DEBUG_LOG_LEVEL = (
        str(config_section(config, "logging").get("level", "INFO")).strip().upper()
    )
    if DEBUG_LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        DEBUG_LOG_LEVEL = "INFO"

    _mcp_section = config_section(config, "mcp")
    _raw_mcp = _mcp_section.get("servers", {})
    MCP_SERVERS = _raw_mcp if isinstance(_raw_mcp, dict) else {}
    _raw_mcp_approvals = _mcp_section.get("auto_approve_tools", [])
    MCP_AUTO_APPROVE_TOOLS = (
        {str(name) for name in _raw_mcp_approvals}
        if isinstance(_raw_mcp_approvals, list)
        else set()
    )

    ONLINE_REQUEST_TIMEOUT = configured_int(
        config,
        "models",
        "online_request_timeout_seconds",
        default_for("models", "online_request_timeout_seconds"),
    )
    ONLINE_MODEL_LIST_TIMEOUT = configured_int(
        config,
        "models",
        "online_model_list_timeout_seconds",
        default_for("models", "online_model_list_timeout_seconds"),
    )
    ONLINE_MAX_RETRIES = configured_nonnegative_int(
        config,
        "models",
        "online_max_retries",
        default_for("models", "online_max_retries"),
    )
    ONLINE_MODEL_LIST_LIMIT = configured_int(
        config,
        "models",
        "online_model_list_limit",
        default_for("models", "online_model_list_limit"),
    )
    PROVIDER_API_MODE = configured_choice(
        config,
        "models",
        "provider_api_mode",
        default_for("models", "provider_api_mode"),
        ("chat", "completions", "responses"),
    )
    RAW_COMPLETIONS_ENABLED = configured_bool(
        config,
        "models",
        "raw_completions_enabled",
        default_for("models", "raw_completions_enabled"),
    )
    MAX_MODEL_CAPABILITY_CACHE_ENTRIES = configured_int(
        config,
        "agent",
        "max_model_capability_cache_entries",
        default_for("agent", "max_model_capability_cache_entries"),
    )
    MAX_WEBPAGE_BYTES = configured_int(
        config, "network", "max_webpage_bytes", 2 * 1024 * 1024
    )
    MAX_DOWNLOAD_BYTES = configured_int(
        config, "network", "max_download_bytes", 25 * 1024 * 1024
    )
    MAX_HTTP_REDIRECTS = configured_nonnegative_int(
        config,
        "network",
        "max_http_redirects",
        default_for("network", "max_http_redirects"),
    )
    MAX_SEARCH_QUERY_CHARS = configured_int(
        config,
        "network",
        "max_search_query_chars",
        default_for("network", "max_search_query_chars"),
    )
    MAX_SEARCH_RESULTS = configured_int(
        config,
        "network",
        "max_search_results",
        default_for("network", "max_search_results"),
    )
    SEARCH_PROVIDER = configured_choice(
        config,
        "network",
        "search_provider",
        default_for("network", "search_provider"),
        ("auto", "searxng", "duckduckgo"),
    )
    SEARXNG_URL = configured_str(
        config, "network", "searxng_url", default_for("network", "searxng_url")
    )

    def _https_url(value: object, default: str) -> str:
        url = str(value or "").strip()
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return url
        return default

    _network_values = config_section(config, "network")
    GEOCODING_URL = _https_url(
        _network_values.get("geocoding_url"),
        default_for("network", "geocoding_url"),
    )
    FORECAST_URL = _https_url(
        _network_values.get("forecast_url"),
        default_for("network", "forecast_url"),
    )
    ARCHIVE_URL = _https_url(
        _network_values.get("archive_url"),
        default_for("network", "archive_url"),
    )
    _raw_ip_services = _network_values.get("ip_geolocation_urls", [])
    IP_GEOLOCATION_URLS = []
    if isinstance(_raw_ip_services, list):
        for _candidate in _raw_ip_services[:5]:
            _url = _https_url(_candidate, "")
            if _url:
                IP_GEOLOCATION_URLS.append(_url)
    if not IP_GEOLOCATION_URLS:
        IP_GEOLOCATION_URLS = list(default_for("network", "ip_geolocation_urls"))
    LOCATION_MAX_RESPONSE_BYTES = configured_int_range(
        config,
        "network",
        "location_max_response_bytes",
        default_for("network", "location_max_response_bytes"),
        65536,
        16 * 1024 * 1024,
    )
    WEATHER_MAX_DAYS = configured_int_range(
        config,
        "network",
        "weather_max_days",
        default_for("network", "weather_max_days"),
        1,
        16,
    )
    WEATHER_PAST_DAYS = configured_int_range(
        config,
        "network",
        "weather_past_days",
        default_for("network", "weather_past_days"),
        0,
        366,
    )
    TOKENIZER_ENCODING = configured_str(
        config,
        "models",
        "tokenizer_encoding",
        default_for("models", "tokenizer_encoding"),
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", TOKENIZER_ENCODING):
        TOKENIZER_ENCODING = default_for("models", "tokenizer_encoding")
    SEARCH_FALLBACK = configured_bool(
        config, "network", "search_fallback", default_for("network", "search_fallback")
    )
    MAX_FETCHED_WEBPAGE_CHARS = configured_int(
        config,
        "network",
        "max_fetched_webpage_chars",
        default_for("network", "max_fetched_webpage_chars"),
    )
    MAX_READ_FILE_BYTES = configured_int(
        config, "tools", "max_read_file_bytes", 1024 * 1024
    )
    FILE_READ_CHUNK_BYTES = configured_int(
        config,
        "tools",
        "file_read_chunk_bytes",
        default_for("tools", "file_read_chunk_bytes"),
    )
    BINARY_PROBE_BYTES = configured_int(
        config,
        "tools",
        "binary_probe_bytes",
        default_for("tools", "binary_probe_bytes"),
    )
    SHELL_COMMAND_TIMEOUT = configured_int(
        config,
        "tools",
        "shell_command_timeout_seconds",
        default_for("tools", "shell_command_timeout_seconds"),
    )
    DEFAULT_HISTORY_READ_LIMIT = configured_int(
        config,
        "tools",
        "default_history_read_limit",
        default_for("tools", "default_history_read_limit"),
    )
    MAX_SKILL_NAME_CHARS = configured_int(
        config, "skills", "max_name_chars", default_for("skills", "max_name_chars")
    )
    MAX_STREAM_TIMEOUT = configured_int(
        config,
        "network",
        "max_stream_timeout_seconds",
        default_for("network", "max_stream_timeout_seconds"),
    )
    MAX_CAPTURED_IMAGE_BYTES = configured_int(
        config, "media", "max_captured_image_bytes", 2 * 1024 * 1024
    )
    MAX_DECODED_IMAGE_PIXELS = configured_int(
        config,
        "media",
        "max_decoded_image_pixels",
        default_for("media", "max_decoded_image_pixels"),
    )
    MAX_MEDIA_FILE_BYTES = configured_int(
        config, "media", "max_media_file_bytes", 25 * 1024 * 1024
    )
    MAX_IMAGES_PER_TURN = configured_int(
        config,
        "media",
        "max_images_per_turn",
        default_for("media", "max_images_per_turn"),
    )
    MAX_MICROPHONE_SECONDS = configured_int(
        config,
        "media",
        "max_microphone_seconds",
        default_for("media", "max_microphone_seconds"),
    )
    MAX_VIDEO_SECONDS = configured_int(
        config, "media", "max_video_seconds", default_for("media", "max_video_seconds")
    )
    DEFAULT_MICROPHONE_SECONDS = configured_float(
        config,
        "media",
        "default_microphone_seconds",
        default_for("media", "default_microphone_seconds"),
    )
    MICROPHONE_SAMPLE_RATE = configured_int(
        config,
        "media",
        "microphone_sample_rate",
        default_for("media", "microphone_sample_rate"),
    )
    DEFAULT_CAMERA_DEVICE_INDEX = configured_nonnegative_int(
        config,
        "media",
        "default_camera_device_index",
        default_for("media", "default_camera_device_index"),
    )
    MAX_CAMERA_DEVICE_INDEX = configured_int(
        config,
        "media",
        "max_camera_device_index",
        default_for("media", "max_camera_device_index"),
    )
    MAX_MICROPHONE_DEVICE_INDEX = configured_int(
        config,
        "media",
        "max_microphone_device_index",
        default_for("media", "max_microphone_device_index"),
    )
    MAX_IMAGE_WIDTH = configured_int(
        config, "media", "max_image_width", default_for("media", "max_image_width")
    )
    MAX_IMAGE_HEIGHT = configured_int(
        config, "media", "max_image_height", default_for("media", "max_image_height")
    )
    JPEG_QUALITY = configured_int_range(
        config, "media", "jpeg_quality", default_for("media", "jpeg_quality"), 1, 100
    )
    MAX_SKILL_DESCRIPTION_CHARS = configured_int(
        config,
        "skills",
        "max_description_chars",
        default_for("skills", "max_description_chars"),
    )
    MAX_SKILL_INSTRUCTION_CHARS = configured_int(
        config,
        "skills",
        "max_instruction_chars",
        default_for("skills", "max_instruction_chars"),
    )
    MAX_SKILL_PREVIEW_CHARS = configured_int_range(
        config,
        "skills",
        "max_preview_chars",
        default_for("skills", "max_preview_chars"),
        512,
        100000,
    )
    SKILL_MAX_ITERATIONS = configured_int(
        config, "skills", "max_iterations", default_for("skills", "max_iterations")
    )
    SKILL_DESCRIPTION_PREVIEW_CHARS = configured_int(
        config,
        "skills",
        "description_preview_chars",
        default_for("skills", "description_preview_chars"),
    )
    SKILL_RELEVANCE_THRESHOLD = configured_float(
        config,
        "skills",
        "relevance_threshold",
        default_for("skills", "relevance_threshold"),
    )
    SKILL_RELEVANCE_EXACT_MATCH_SCORE = configured_float(
        config,
        "skills",
        "relevance_exact_match_score",
        default_for("skills", "relevance_exact_match_score"),
    )
    SKILL_RELEVANCE_PREFIX_MATCH_SCORE = configured_float(
        config,
        "skills",
        "relevance_prefix_match_score",
        default_for("skills", "relevance_prefix_match_score"),
    )
    SKILL_RELEVANCE_WORD_MATCH_SCORE = configured_float(
        config,
        "skills",
        "relevance_word_match_score",
        default_for("skills", "relevance_word_match_score"),
    )
    SKILL_RELEVANCE_EMBEDDING_THRESHOLD = configured_float(
        config,
        "skills",
        "relevance_embedding_threshold",
        default_for("skills", "relevance_embedding_threshold"),
    )
    RAG_CONTEXT_RESULTS = configured_int(
        config,
        "agent",
        "rag_context_results",
        default_for("agent", "rag_context_results"),
    )
    RAG_DOCUMENTS_ONLY = configured_bool(
        config,
        "agent",
        "rag_documents_only",
        default_for("agent", "rag_documents_only"),
    )
    RAG_RELEVANCE_THRESHOLD = configured_float(
        config, "rag", "relevance_threshold", default_for("rag", "relevance_threshold")
    )
    RERANK_MODE = configured_choice(
        config,
        "rag",
        "rerank_mode",
        default_for("rag", "rerank_mode"),
        ("api", "local", "auto"),
    )
    RERANK_MODEL = configured_str(
        config, "rag", "rerank_model", default_for("rag", "rerank_model")
    )
    RERANK_TIMEOUT_SECONDS = configured_float(
        config,
        "rag",
        "rerank_timeout_seconds",
        default_for("rag", "rerank_timeout_seconds"),
    )
    RERANK_TOP_N = configured_int(
        config, "rag", "rerank_top_n", default_for("rag", "rerank_top_n")
    )
    RERANK_MAX_CANDIDATES = configured_int(
        config,
        "rag",
        "rerank_max_candidates",
        default_for("rag", "rerank_max_candidates"),
    )
    RERANK_LOCAL_THRESHOLD = configured_nonnegative_int(
        config,
        "rag",
        "rerank_local_threshold",
        default_for("rag", "rerank_local_threshold"),
    )
    RERANK_CACHE_ENTRIES = configured_int(
        config,
        "rag",
        "rerank_cache_entries",
        default_for("rag", "rerank_cache_entries"),
    )
    RERANK_BREAKER_FAILURES = configured_int(
        config,
        "rag",
        "rerank_breaker_failures",
        default_for("rag", "rerank_breaker_failures"),
    )
    RERANK_BREAKER_COOLDOWN_SECONDS = configured_float(
        config,
        "rag",
        "rerank_breaker_cooldown_seconds",
        default_for("rag", "rerank_breaker_cooldown_seconds"),
    )
    RAG_INCLUDE_CODING_GUIDANCE = configured_bool(
        config,
        "rag",
        "include_coding_guidance",
        default_for("rag", "include_coding_guidance"),
    )
    SUMMARY_PROMPT_SENTENCES = configured_int(
        config,
        "agent",
        "summary_prompt_sentences",
        default_for("agent", "summary_prompt_sentences"),
    )
    try:
        CONTEXT_COMPACTION_THRESHOLD = float(
            config_section(config, "agent").get("context_compaction_threshold", 0.8)
        )
    except AttributeError, TypeError, ValueError:
        CONTEXT_COMPACTION_THRESHOLD = 0.8
    if not 0 <= CONTEXT_COMPACTION_THRESHOLD <= 1:
        CONTEXT_COMPACTION_THRESHOLD = 0.8
    CONTEXT_KEEP_RECENT_MESSAGES = configured_int(
        config,
        "agent",
        "context_keep_recent_messages",
        default_for("agent", "context_keep_recent_messages"),
    )
    CONTEXT_SUMMARY_CHARS = configured_int(
        config,
        "agent",
        "context_summary_chars",
        default_for("agent", "context_summary_chars"),
    )
    CONTEXT_ENGINE_ENABLED = configured_bool(
        config,
        "agent",
        "context_engine_enabled",
        default_for("agent", "context_engine_enabled"),
    )
    CONTEXT_ENGINE_PROFILE = configured_choice(
        config,
        "agent",
        "context_engine_profile",
        default_for("agent", "context_engine_profile"),
        ("auto", "chat", "coding", "research", "memory"),
    )
    CONTEXT_SAFETY_MARGIN_TOKENS = configured_nonnegative_int(
        config,
        "agent",
        "context_safety_margin_tokens",
        default_for("agent", "context_safety_margin_tokens"),
    )
    CONTEXT_OUTPUT_RESERVE_RATIO = configured_float(
        config,
        "agent",
        "context_output_reserve_ratio",
        default_for("agent", "context_output_reserve_ratio"),
    )

    # KV cache settings may live under agent, agent.context_cache, or a
    # top-level context_cache section.
    def _kv_raw(key, default):
        agent_section = config_section(config, "agent")
        if key in agent_section:
            return agent_section.get(key, default)
        nested = agent_section.get("context_cache", {})
        if isinstance(nested, Mapping) and key in nested:
            return nested.get(key, default)
        top = config_section(config, "context_cache")
        if key in top:
            return top.get(key, default)
        return default

    _kv_enabled_raw = _kv_raw("kv_cache_enabled", False)
    _kv_hints_raw = _kv_raw("kv_cache_provider_hints", "auto")
    _kv_min_prefix_raw = _kv_raw("kv_cache_min_prefix_tokens", 1024)
    CONTEXT_KV_CACHE_ENABLED = configured_bool(
        {"_": {"v": _kv_enabled_raw}}, "_", "v", False
    )
    CONTEXT_KV_CACHE_PROVIDER_HINTS = configured_choice(
        {"_": {"v": _kv_hints_raw}}, "_", "v", "auto", ("auto", "on", "off")
    )
    CONTEXT_KV_CACHE_MIN_PREFIX_TOKENS = configured_int(
        {"_": {"v": _kv_min_prefix_raw}}, "_", "v", 1024
    )
    CODE_TASK_GIT_TIMEOUT = configured_int(
        config,
        "code_tasks",
        "git_command_timeout_seconds",
        default_for("code_tasks", "git_command_timeout_seconds"),
    )
    CODE_TASK_TEST_TIMEOUT = configured_int(
        config,
        "code_tasks",
        "test_command_timeout_seconds",
        default_for("code_tasks", "test_command_timeout_seconds"),
    )
    CODE_TASK_SANDBOX_CPU_SECONDS = configured_int_range(
        config,
        "code_tasks",
        "sandbox_cpu_seconds",
        default_for("code_tasks", "sandbox_cpu_seconds"),
        1,
        86400,
    )
    CODE_TASK_SANDBOX_ADDRESS_SPACE_BYTES = configured_int_range(
        config,
        "code_tasks",
        "sandbox_address_space_bytes",
        4 * 1024**3,
        1024**3,
        32 * 1024**3,
    )
    CODE_TASK_SANDBOX_PROCESS_COUNT = configured_int_range(
        config,
        "code_tasks",
        "sandbox_process_count",
        default_for("code_tasks", "sandbox_process_count"),
        8,
        4096,
    )
    CODE_TASK_SANDBOX_FILE_SIZE_BYTES = configured_int_range(
        config,
        "code_tasks",
        "sandbox_file_size_bytes",
        512 * 1024**2,
        1024**2,
        10 * 1024**3,
    )
    _raw_code_task_test_commands = config_section(config, "code_tasks").get(
        "test_commands", []
    )
    CODE_TASK_TEST_COMMANDS = []
    if isinstance(_raw_code_task_test_commands, list):
        if len(_raw_code_task_test_commands) > 16:
            warnings.warn(
                "Only the first 16 code_tasks.test_commands entries will be used.",
                RuntimeWarning,
                stacklevel=2,
            )
        for _test_command in _raw_code_task_test_commands[:16]:
            if (
                isinstance(_test_command, list)
                and 1 <= len(_test_command) <= 64
                and all(
                    isinstance(argument, str) and 0 < len(argument) <= 2048
                    for argument in _test_command
                )
            ):
                CODE_TASK_TEST_COMMANDS.append(_test_command)
            else:
                warnings.warn(
                    "Ignoring invalid code_tasks.test_commands entry; use a non-empty "
                    "array of at most 64 non-empty argument strings.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    elif _raw_code_task_test_commands:
        warnings.warn(
            "Ignoring invalid code_tasks.test_commands; expected an array of argument arrays.",
            RuntimeWarning,
            stacklevel=2,
        )
    CODE_TASK_SUCCESS_OUTPUT_CHARS = configured_int(
        config,
        "code_tasks",
        "successful_test_output_chars",
        default_for("code_tasks", "successful_test_output_chars"),
    )
    CODE_TASK_FAILURE_OUTPUT_CHARS = configured_int(
        config,
        "code_tasks",
        "failed_test_output_chars",
        default_for("code_tasks", "failed_test_output_chars"),
    )

    return {name.lower(): value for name, value in locals().items() if name.isupper()}


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings, one field per runtime constant."""

    default_db_path: str
    workspace_root_default: str
    code_output_root: str
    skills_folder_default: str
    rag_docs_default: str
    model_temperature: float
    online_permission_override: bool
    embedding_model: str
    embedding_provider: str
    embedding_base_url: str
    rag_index_path: str
    ollama_base_url: str
    ollama_base_urls: list[str]
    hardware_acceleration_mode: str
    preferred_model: str | None
    max_tool_iterations: int
    max_tool_calls: int
    max_task_seconds: int
    max_tool_output_chars: int
    max_history_messages: int
    max_context_tokens: int
    max_output_tokens: int
    streaming_output: bool
    visible_reasoning: bool
    verbose_timings: bool
    status_bar: bool
    show_welcome_screen: bool
    informative_prompt: bool
    show_turn_divider: bool
    show_turn_stats: bool
    grouped_help: bool
    system_prompt: str
    permission_allow_rules: list
    permission_deny_rules: list
    approved_extra_folders: list
    sandbox_network_default: bool
    project_instructions_file: str | None
    conversation_retention_days: Any
    episode_retention_days: Any
    registered_skill_retention_days: Any
    task_stale_after_days: Any
    max_summary_chars: int
    max_learned_item_chars: int
    max_resume_state_chars: int
    max_memory_context_tokens: int
    max_relevant_episodes: int
    learned_item_expiry_days: int
    learned_item_review_days: int
    default_history_messages: int
    default_episodic_summaries: int
    max_network_concurrency: int
    network_request_timeout: int
    rag_max_file_bytes: int
    rag_max_corpus_bytes: int
    rag_max_pdf_pages: int
    rag_max_documents: int
    rag_chunk_size_chars: int
    rag_chunk_overlap_chars: int
    chunking_strategy: str
    chunking_unit: str
    chunking_size: int
    chunking_overlap: int
    chunking_min_chunk_chars: int
    chunking_separators: tuple[str, ...]
    chunking_semantic_percentile: float
    chunking_semantic_buffer: int
    chunking_semantic_max_sentences: int
    rag_similarity_results: int
    rag_sqlite_max_rows_per_table: int
    rag_file_types: list[str]
    rag_sqlite_rows_per_chunk: int
    rag_sqlite_max_cell_chars: int
    rag_auto_refresh_seconds: int
    enable_web_research: bool
    web_research_consent: str
    internet_check_host: str
    internet_check_port: int
    internet_check_timeout: float
    override_tool_list: frozenset[str]
    sandbox_extra_risky_programs: list[str]
    sandbox_extra_system_paths: list[str]
    rag_excluded_file_types: list[str]
    thinking_toggle_default: bool
    thinking_effort_default: str
    agent_permission_mode: str
    debug_log_enabled: bool
    debug_log_level: str
    mcp_servers: dict[str, Any]
    mcp_auto_approve_tools: set[str]
    online_request_timeout: int
    online_model_list_timeout: int
    online_max_retries: int
    online_model_list_limit: int
    provider_api_mode: str
    raw_completions_enabled: bool
    max_model_capability_cache_entries: int
    max_webpage_bytes: int
    max_download_bytes: int
    max_http_redirects: int
    max_search_query_chars: int
    max_search_results: int
    search_provider: str
    searxng_url: str
    geocoding_url: str
    forecast_url: str
    archive_url: str
    ip_geolocation_urls: list[str]
    location_max_response_bytes: int
    weather_max_days: int
    weather_past_days: int
    tokenizer_encoding: str
    search_fallback: bool
    max_fetched_webpage_chars: int
    max_read_file_bytes: int
    file_read_chunk_bytes: int
    binary_probe_bytes: int
    shell_command_timeout: int
    default_history_read_limit: int
    max_skill_name_chars: int
    max_stream_timeout: int
    max_captured_image_bytes: int
    max_decoded_image_pixels: int
    max_media_file_bytes: int
    max_images_per_turn: int
    max_microphone_seconds: int
    max_video_seconds: int
    default_microphone_seconds: float
    microphone_sample_rate: int
    default_camera_device_index: int
    max_camera_device_index: int
    max_microphone_device_index: int
    max_image_width: int
    max_image_height: int
    jpeg_quality: int
    max_skill_description_chars: int
    max_skill_instruction_chars: int
    max_skill_preview_chars: int
    skill_max_iterations: int
    skill_description_preview_chars: int
    skill_relevance_threshold: float
    skill_relevance_exact_match_score: float
    skill_relevance_prefix_match_score: float
    skill_relevance_word_match_score: float
    skill_relevance_embedding_threshold: float
    rag_context_results: int
    rag_documents_only: bool
    rag_relevance_threshold: float
    rag_include_coding_guidance: bool
    rerank_mode: str
    rerank_model: str
    rerank_timeout_seconds: float
    rerank_top_n: int
    rerank_max_candidates: int
    rerank_local_threshold: int
    rerank_cache_entries: int
    rerank_breaker_failures: int
    rerank_breaker_cooldown_seconds: float
    summary_prompt_sentences: int
    context_compaction_threshold: float
    context_keep_recent_messages: int
    context_summary_chars: int
    context_engine_enabled: bool
    context_engine_profile: str
    context_safety_margin_tokens: int
    context_output_reserve_ratio: float
    context_kv_cache_enabled: bool
    context_kv_cache_provider_hints: str
    context_kv_cache_min_prefix_tokens: int
    code_task_git_timeout: int
    code_task_test_timeout: int
    code_task_sandbox_cpu_seconds: int
    code_task_sandbox_address_space_bytes: int
    code_task_sandbox_process_count: int
    code_task_sandbox_file_size_bytes: int
    code_task_test_commands: list[list[str]]
    code_task_success_output_chars: int
    code_task_failure_output_chars: int
    context_compaction_source_tokens: int
    tool_result_preview_chars: int
    tool_argument_preview_chars: int
    search_max_files: int
    search_max_bytes: int
    search_output_chars: int
    task_citation_budget_chars: int

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        default_paths: Mapping[str, str] | None = None,
    ) -> Settings:
        """Build typed settings from a merged configuration mapping."""
        return cls(**resolve_config_values(config, default_paths=default_paths))
