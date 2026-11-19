"""Typed, validated settings derived from the per-user configuration."""

from __future__ import annotations

import os
import pathlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .hardware import normalize_acceleration_mode

DEFAULT_SYSTEM_PROMPT_PATH = (
    pathlib.Path(__file__).parent / "resources" / "default_system_prompt.md"
)
try:
    DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_PATH.read_text(
        encoding="utf-8"
    ).strip()
except OSError as exc:
    raise RuntimeError(
        f"Could not load packaged default system prompt at "
        f"{DEFAULT_SYSTEM_PROMPT_PATH}: {exc}"
    ) from exc


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
    except (TypeError, ValueError):
        return default


def positive_float(config: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(config.get(key, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def configured_int(
    config: Mapping[str, Any], section: str, key: str, default: int
) -> int:
    try:
        value = int(config_section(config, section).get(key, default))
        return value if value > 0 else default
    except (AttributeError, TypeError, ValueError):
        return default


def configured_nonnegative_int(
    config: Mapping[str, Any], section: str, key: str, default: int
) -> int:
    try:
        value = int(config_section(config, section).get(key, default))
        return value if value >= 0 else default
    except (AttributeError, TypeError, ValueError):
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
    except (AttributeError, TypeError, ValueError):
        return default


DEFAULT_PATHS: dict[str, str] = {
    "database": "~/.local_ai_memory.db",
    "workspace": ".",
    "code_output": "~/.private_agent/projects",
    "skills": "~/.private_agent/resources/skills",
    "rag_documents": "~/.private_agent/rag/",
    "rag_index": "~/.private_agent/rag_index",
}


def resolve_config_values(
    config: Mapping[str, Any],
    *,
    default_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve one merged config mapping into validated, runtime-ready values."""
    default_paths = DEFAULT_PATHS if default_paths is None else default_paths

    DEFAULT_DB_PATH = os.path.expanduser(
        str(config.get("default_db_path", "~/.local_ai_memory.db"))
    )
    WORKSPACE_ROOT_DEFAULT = str(config.get("workspace_root", "."))
    CODE_OUTPUT_ROOT = str(config.get("code_output_root", "~/.private_agent/projects"))
    SKILLS_FOLDER_DEFAULT = configured_path(
        config_section(config, "paths").get("skills"),
        default_paths["skills"],
    )
    RAG_DOCS_DEFAULT = configured_path(
        config_section(config, "paths").get("rag_documents"),
        default_paths["rag_documents"],
    )

    MODEL_TEMPERATURE = configured_float(config, "models", "temperature", 0.1)

    ONLINE_PERMISSION_OVERRIDE = configured_bool(
        config, "models", "online_permission_override", False
    )
    EMBEDDING_MODEL = str(config.get("embedding_model", "nomic-embed-text"))
    RAG_INDEX_PATH = configured_path(
        config_section(config, "paths").get("rag_index"),
        default_paths["rag_index"],
    )
    OLLAMA_BASE_URL = str(config.get("ollama_base_url", "http://localhost:11434"))
    HARDWARE_ACCELERATION_MODE = normalize_acceleration_mode(
        config.get("hardware_acceleration", "auto")
    )
    PREFERRED_MODEL = config.get("preferred_model")

    MAX_TOOL_ITERATIONS = positive_int(config, "max_tool_iterations", 15)
    MAX_TOOL_CALLS = positive_int(config, "max_tool_calls", 60)
    MAX_TASK_SECONDS = positive_int(config, "max_task_seconds", 600)
    MAX_TOOL_OUTPUT_CHARS = positive_int(config, "max_tool_output_chars", 12000)
    MAX_HISTORY_MESSAGES = positive_int(config, "max_history_messages", 20)
    MAX_CONTEXT_TOKENS = positive_int(config, "max_context_tokens", 12000)
    MAX_OUTPUT_TOKENS = configured_int_range(
        config, "agent", "max_output_tokens", 2048, 1, 32768
    )
    STREAMING_OUTPUT = configured_bool(config, "agent", "streaming_output", True)
    VISIBLE_REASONING = configured_bool(config, "agent", "visible_reasoning", False)
    _configured_system_prompt = config_section(config, "agent").get("system_prompt")
    SYSTEM_PROMPT = (
        _configured_system_prompt.strip()
        if isinstance(_configured_system_prompt, str)
        and _configured_system_prompt.strip()
        else DEFAULT_SYSTEM_PROMPT
    )
    CONVERSATION_RETENTION_DAYS = (
        max(0, positive_int(config, "conversation_retention_days", 0))
        if config.get("conversation_retention_days", 0)
        else 0
    )
    EPISODE_RETENTION_DAYS = (
        max(0, positive_int(config, "episode_retention_days", 0))
        if config.get("episode_retention_days", 0)
        else 0
    )
    REGISTERED_SKILL_RETENTION_DAYS = (
        max(0, positive_int(config, "registered_skill_retention_days", 0))
        if config.get("registered_skill_retention_days", 0)
        else 0
    )
    TASK_STALE_AFTER_DAYS = (
        max(0, positive_int(config, "task_stale_after_days", 0))
        if config.get("task_stale_after_days", 0)
        else 0
    )
    MAX_SUMMARY_CHARS = positive_int(config, "max_summary_chars", 2000)
    MAX_LEARNED_ITEM_CHARS = configured_int_range(
        config, "memory", "max_learned_item_chars", 1000, 1, 10000
    )
    MAX_RESUME_STATE_CHARS = configured_int_range(
        config, "memory", "max_resume_state_chars", 12000, 256, 100000
    )
    MAX_MEMORY_CONTEXT_TOKENS = configured_int_range(
        config, "memory", "max_memory_context_tokens", 1200, 128, 12000
    )
    MAX_RELEVANT_EPISODES = configured_int_range(
        config, "memory", "max_relevant_episodes", 3, 1, 20
    )
    LEARNED_ITEM_EXPIRY_DAYS = configured_int_range(
        config, "memory", "learned_item_expiry_days", 365, 1, 3650
    )
    LEARNED_ITEM_REVIEW_DAYS = configured_int_range(
        config, "memory", "learned_item_review_days", 180, 1, 3650
    )
    DEFAULT_HISTORY_MESSAGES = configured_int(
        config, "memory", "default_history_messages", 20
    )
    DEFAULT_EPISODIC_SUMMARIES = configured_int(
        config, "memory", "default_episodic_summaries", 5
    )
    MAX_NETWORK_CONCURRENCY = positive_int(config, "max_network_concurrency", 4)
    NETWORK_REQUEST_TIMEOUT = positive_int(config, "network_request_timeout", 30)
    RAG_MAX_FILE_BYTES = positive_int(config, "rag_max_file_bytes", 5 * 1024 * 1024)
    RAG_MAX_CORPUS_BYTES = positive_int(
        config, "rag_max_corpus_bytes", 25 * 1024 * 1024
    )
    RAG_MAX_PDF_PAGES = positive_int(config, "rag_max_pdf_pages", 250)
    RAG_MAX_DOCUMENTS = positive_int(config, "rag_max_documents", 20000)
    RAG_CHUNK_SIZE_CHARS = configured_int(config, "rag", "chunk_size_chars", 1200)
    RAG_CHUNK_OVERLAP_CHARS = configured_int(config, "rag", "chunk_overlap_chars", 200)
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
        config, "chunking", "min_chunk_chars", 0
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
        configured_float(config, "chunking", "semantic_breakpoint_percentile", 90.0),
    )
    CHUNKING_SEMANTIC_BUFFER = configured_nonnegative_int(
        config, "chunking", "semantic_buffer_sentences", 1
    )
    CHUNKING_SEMANTIC_MAX_SENTENCES = configured_int(
        config, "chunking", "semantic_max_sentences", 2000
    )
    RAG_SIMILARITY_RESULTS = configured_int(config, "rag", "similarity_results", 4)
    RAG_SQLITE_MAX_ROWS_PER_TABLE = configured_int(
        config, "rag", "sqlite_max_rows_per_table", 5000
    )
    RAG_AUTO_REFRESH_SECONDS = configured_nonnegative_int(
        config, "rag", "auto_refresh_seconds", 30
    )
    _raw_web_research = config.get("enable_web_research", True)
    if isinstance(_raw_web_research, str):
        ENABLE_WEB_RESEARCH = _raw_web_research.lower() in ("true", "1", "yes", "on")
    else:
        ENABLE_WEB_RESEARCH = bool(_raw_web_research)
    WEB_RESEARCH_CONSENT = str(config.get("web_research_consent", "ask")).lower()
    if WEB_RESEARCH_CONSENT not in {"ask", "session", "never"}:
        WEB_RESEARCH_CONSENT = "ask"
    INTERNET_CHECK_HOST = str(config.get("internet_check_host", "1.1.1.1"))
    INTERNET_CHECK_PORT = positive_int(config, "internet_check_port", 443)
    INTERNET_CHECK_TIMEOUT = positive_float(config, "internet_check_timeout", 3.0)

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
        warnings.warn(
            "tools.override_tool_list must be a list of tool names; ignoring it.",
            RuntimeWarning,
            stacklevel=2,
        )
        OVERRIDE_TOOL_LIST = frozenset()

    _raw_thinking = config.get("thinking_toggle_default", False)
    if isinstance(_raw_thinking, str):
        THINKING_TOGGLE_DEFAULT = _raw_thinking.lower() in ("true", "1", "yes", "on")
    else:
        THINKING_TOGGLE_DEFAULT = bool(_raw_thinking)

    THINKING_EFFORT_DEFAULT = str(
        config.get("thinking_effort_default", "medium")
    ).lower()
    if THINKING_EFFORT_DEFAULT not in {"low", "medium", "high"}:
        THINKING_EFFORT_DEFAULT = "medium"

    AGENT_PERMISSION_MODE = str(config.get("agent_permission_mode", "auto")).lower()
    if AGENT_PERMISSION_MODE not in {"manual", "auto", "full"}:
        AGENT_PERMISSION_MODE = "auto"

    _debug_log_enabled = config.get("DEBUG_LOG_ENABLED", 0)
    if isinstance(_debug_log_enabled, str):
        DEBUG_LOG_ENABLED = _debug_log_enabled.strip() == "1"
    else:
        DEBUG_LOG_ENABLED = _debug_log_enabled == 1 or _debug_log_enabled is True

    DEBUG_LOG_LEVEL = (
        str(config_section(config, "logging").get("level", "INFO")).strip().upper()
    )
    if DEBUG_LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        DEBUG_LOG_LEVEL = "INFO"

    _raw_mcp = config.get("mcpServers", {})
    MCP_SERVERS = _raw_mcp if isinstance(_raw_mcp, dict) else {}
    _raw_mcp_approvals = config.get("mcp_auto_approve_tools", [])
    MCP_AUTO_APPROVE_TOOLS = (
        {str(name) for name in _raw_mcp_approvals}
        if isinstance(_raw_mcp_approvals, list)
        else set()
    )

    ONLINE_REQUEST_TIMEOUT = configured_int(
        config, "models", "online_request_timeout_seconds", 60
    )
    ONLINE_MODEL_LIST_TIMEOUT = configured_int(
        config, "models", "online_model_list_timeout_seconds", 10
    )
    ONLINE_MAX_RETRIES = configured_nonnegative_int(
        config, "models", "online_max_retries", 1
    )
    ONLINE_MODEL_LIST_LIMIT = configured_int(
        config, "models", "online_model_list_limit", 30
    )
    MAX_MODEL_CAPABILITY_CACHE_ENTRIES = configured_int(
        config, "agent", "max_model_capability_cache_entries", 128
    )
    MAX_WEBPAGE_BYTES = configured_int(
        config, "network", "max_webpage_bytes", 2 * 1024 * 1024
    )
    MAX_DOWNLOAD_BYTES = configured_int(
        config, "network", "max_download_bytes", 25 * 1024 * 1024
    )
    MAX_HTTP_REDIRECTS = configured_nonnegative_int(
        config, "network", "max_http_redirects", 5
    )
    MAX_SEARCH_QUERY_CHARS = configured_int(
        config, "network", "max_search_query_chars", 2000
    )
    MAX_SEARCH_RESULTS = configured_int(config, "network", "max_search_results", 3)
    SEARCH_PROVIDER = configured_choice(
        config,
        "network",
        "search_provider",
        "auto",
        ("auto", "searxng", "duckduckgo"),
    )
    SEARXNG_URL = configured_str(config, "network", "searxng_url", "")
    SEARCH_FALLBACK = configured_bool(config, "network", "search_fallback", True)
    MAX_FETCHED_WEBPAGE_CHARS = configured_int(
        config, "network", "max_fetched_webpage_chars", 12000
    )
    MAX_READ_FILE_BYTES = configured_int(
        config, "tools", "max_read_file_bytes", 1024 * 1024
    )
    FILE_READ_CHUNK_BYTES = configured_int(
        config, "tools", "file_read_chunk_bytes", 65536
    )
    BINARY_PROBE_BYTES = configured_int(config, "tools", "binary_probe_bytes", 2048)
    SHELL_COMMAND_TIMEOUT = configured_int(
        config, "tools", "shell_command_timeout_seconds", 30
    )
    DEFAULT_HISTORY_READ_LIMIT = configured_int(
        config, "tools", "default_history_read_limit", 20
    )
    MAX_SKILL_NAME_CHARS = configured_int(config, "skills", "max_name_chars", 64)
    MAX_STREAM_TIMEOUT = configured_int(
        config, "network", "max_stream_timeout_seconds", 600
    )
    MAX_CAPTURED_IMAGE_BYTES = configured_int(
        config, "media", "max_captured_image_bytes", 2 * 1024 * 1024
    )
    MAX_DECODED_IMAGE_PIXELS = configured_int(
        config, "media", "max_decoded_image_pixels", 20_000_000
    )
    MAX_MEDIA_FILE_BYTES = configured_int(
        config, "media", "max_media_file_bytes", 25 * 1024 * 1024
    )
    MAX_IMAGES_PER_TURN = configured_int(config, "media", "max_images_per_turn", 3)
    MAX_MICROPHONE_SECONDS = configured_int(
        config, "media", "max_microphone_seconds", 30
    )
    MAX_VIDEO_SECONDS = configured_int(config, "media", "max_video_seconds", 120)
    DEFAULT_MICROPHONE_SECONDS = configured_float(
        config, "media", "default_microphone_seconds", 5.0
    )
    MICROPHONE_SAMPLE_RATE = configured_int(
        config, "media", "microphone_sample_rate", 16000
    )
    DEFAULT_CAMERA_DEVICE_INDEX = configured_nonnegative_int(
        config, "media", "default_camera_device_index", 0
    )
    MAX_CAMERA_DEVICE_INDEX = configured_int(
        config, "media", "max_camera_device_index", 32
    )
    MAX_MICROPHONE_DEVICE_INDEX = configured_int(
        config, "media", "max_microphone_device_index", 128
    )
    MAX_IMAGE_WIDTH = configured_int(config, "media", "max_image_width", 1280)
    MAX_IMAGE_HEIGHT = configured_int(config, "media", "max_image_height", 720)
    JPEG_QUALITY = configured_int_range(config, "media", "jpeg_quality", 85, 1, 100)
    MAX_SKILL_DESCRIPTION_CHARS = configured_int(
        config, "skills", "max_description_chars", 500
    )
    MAX_SKILL_INSTRUCTION_CHARS = configured_int(
        config, "skills", "max_instruction_chars", 20000
    )
    MAX_SKILL_PREVIEW_CHARS = configured_int_range(
        config, "skills", "max_preview_chars", 12000, 512, 100000
    )
    SKILL_MAX_ITERATIONS = configured_int(config, "skills", "max_iterations", 15)
    SKILL_DESCRIPTION_PREVIEW_CHARS = configured_int(
        config, "skills", "description_preview_chars", 100
    )
    SKILL_RELEVANCE_THRESHOLD = configured_float(
        config, "skills", "relevance_threshold", 1.5
    )
    SKILL_RELEVANCE_EXACT_MATCH_SCORE = configured_float(
        config, "skills", "relevance_exact_match_score", 3.0
    )
    SKILL_RELEVANCE_PREFIX_MATCH_SCORE = configured_float(
        config, "skills", "relevance_prefix_match_score", 2.0
    )
    SKILL_RELEVANCE_WORD_MATCH_SCORE = configured_float(
        config, "skills", "relevance_word_match_score", 1.0
    )
    SKILL_RELEVANCE_EMBEDDING_THRESHOLD = configured_float(
        config, "skills", "relevance_embedding_threshold", 0.55
    )
    RAG_CONTEXT_RESULTS = configured_int(config, "agent", "rag_context_results", 2)
    SUMMARY_PROMPT_SENTENCES = configured_int(
        config, "agent", "summary_prompt_sentences", 2
    )
    try:
        CONTEXT_COMPACTION_THRESHOLD = float(
            config_section(config, "agent").get("context_compaction_threshold", 0.8)
        )
    except (AttributeError, TypeError, ValueError):
        CONTEXT_COMPACTION_THRESHOLD = 0.8
    if not 0 <= CONTEXT_COMPACTION_THRESHOLD <= 1:
        CONTEXT_COMPACTION_THRESHOLD = 0.8
    CONTEXT_KEEP_RECENT_MESSAGES = configured_int(
        config, "agent", "context_keep_recent_messages", 6
    )
    CONTEXT_SUMMARY_CHARS = configured_int(
        config, "agent", "context_summary_chars", 1500
    )
    CODE_TASK_GIT_TIMEOUT = configured_int(
        config, "code_tasks", "git_command_timeout_seconds", 30
    )
    CODE_TASK_TEST_TIMEOUT = configured_int(
        config, "code_tasks", "test_command_timeout_seconds", 180
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
        config, "code_tasks", "successful_test_output_chars", 6000
    )
    CODE_TASK_FAILURE_OUTPUT_CHARS = configured_int(
        config, "code_tasks", "failed_test_output_chars", 3000
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
    rag_index_path: str
    ollama_base_url: str
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
    system_prompt: str
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
    rag_auto_refresh_seconds: int
    enable_web_research: bool
    web_research_consent: str
    internet_check_host: str
    internet_check_port: int
    internet_check_timeout: float
    override_tool_list: frozenset[str]
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
    max_model_capability_cache_entries: int
    max_webpage_bytes: int
    max_download_bytes: int
    max_http_redirects: int
    max_search_query_chars: int
    max_search_results: int
    search_provider: str
    searxng_url: str
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
    summary_prompt_sentences: int
    context_compaction_threshold: float
    context_keep_recent_messages: int
    context_summary_chars: int
    code_task_git_timeout: int
    code_task_test_timeout: int
    code_task_test_commands: list[list[str]]
    code_task_success_output_chars: int
    code_task_failure_output_chars: int

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        default_paths: Mapping[str, str] | None = None,
    ) -> Settings:
        """Build typed settings from a merged configuration mapping."""
        return cls(**resolve_config_values(config, default_paths=default_paths))
