import contextlib
import json
import os
import pathlib
import tempfile
import warnings

from .settings import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT_PATH,  # noqa: F401  (re-exported for compatibility)
    Settings,
    configured_bool,
    configured_float,
    configured_int,
    configured_int_range,
    configured_nonnegative_int,
    configured_path,
    positive_float,
    positive_int,
)

CONFIG_PATH_ENV_VAR = "PRIVATE_AGENT_CONFIG"


def resolve_config_path(environ=None) -> pathlib.Path:
    """Return the config file path: $PRIVATE_AGENT_CONFIG or ~/.private_agent.conf."""
    override = (environ if environ is not None else os.environ).get(
        CONFIG_PATH_ENV_VAR, ""
    )
    if override.strip():
        return pathlib.Path(override.strip()).expanduser().absolute()
    return pathlib.Path.home() / ".private_agent.conf"


CONFIG_FILE_PATH = resolve_config_path()

DEFAULT_CONFIG = {
    "paths": {
        "database": "~/.local_ai_memory.db",
        "workspace": ".",
        "code_output": "~/.private_agent/projects",
        "skills": "~/.private_agent/resources/skills",
        "rag_documents": "~/.private_agent/rag/",
        "rag_index": "~/.private_agent/rag_index",
    },
    "permissions": {
        "allow": [],
        "deny": [],
    },
    "models": {
        "ollama_base_url": "http://localhost:11434",
        "local_openai_compatible_endpoints": [
            {
                "name": "LM Studio",
                "base_url": "http://127.0.0.1:1234/v1",
            },
            {
                "name": "llama.cpp",
                "base_url": "http://127.0.0.1:8080/v1",
            },
        ],
        "hardware_acceleration": "auto",
        "preferred_model": None,
        "online_base_url": "https://api.openai.com/v1",
        "online_model": None,
        "online_api_key": None,
        "online_permission_override": False,
        "temperature": 0.1,
        "embedding_model": "nomic-embed-text",
        "thinking_enabled_by_default": False,
        "thinking_effort": "medium",
        "online_request_timeout_seconds": 60,
        "online_model_list_timeout_seconds": 10,
        "online_max_retries": 1,
        "online_model_list_limit": 30,
    },
    "agent": {
        "permission_mode": "auto",
        "max_tool_iterations": 15,
        "max_tool_calls": 60,
        "max_task_seconds": 600,
        "max_tool_output_chars": 12000,
        "max_history_messages": 20,
        "max_context_tokens": 12000,
        "max_output_tokens": 2048,
        "max_model_capability_cache_entries": 128,
        "summary_prompt_sentences": 2,
        "context_compaction_threshold": 0.8,
        "context_keep_recent_messages": 6,
        "context_summary_chars": 1500,
        "rag_context_results": 2,
        "streaming_output": True,
        "visible_reasoning": False,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "project_instructions_file": "AGENTS.md",
    },
    "memory": {
        "conversation_retention_days": 0,
        "episode_retention_days": 0,
        "registered_skill_retention_days": 0,
        "task_stale_after_days": 0,
        "max_summary_chars": 2000,
        "max_learned_item_chars": 1000,
        "learned_item_expiry_days": 365,
        "learned_item_review_days": 180,
        "max_resume_state_chars": 12000,
        "max_memory_context_tokens": 1200,
        "max_relevant_episodes": 3,
        "default_history_messages": 20,
        "default_episodic_summaries": 5,
    },
    "rag": {
        "max_file_bytes": 5242880,
        "max_corpus_bytes": 26214400,
        "max_pdf_pages": 250,
        "max_documents": 20000,
        "chunk_size_chars": 1200,
        "chunk_overlap_chars": 200,
        "similarity_results": 4,
        "sqlite_max_rows_per_table": 5000,
        "auto_refresh_seconds": 30,
    },
    "chunking": {
        "strategy": "character",
        "chunk_size": None,
        "chunk_overlap": None,
        "unit": "chars",
        "min_chunk_chars": 0,
        "separators": ["\n\n", "\n", ". ", " "],
        "semantic_breakpoint_percentile": 90,
        "semantic_buffer_sentences": 1,
        "semantic_max_sentences": 2000,
    },
    "network": {
        "web_research_enabled": True,
        "web_research_consent": "ask",
        "internet_check_host": "1.1.1.1",
        "internet_check_port": 443,
        "internet_check_timeout_seconds": 3.0,
        "max_concurrent_requests": 4,
        "request_timeout_seconds": 30,
        "max_stream_timeout_seconds": 600,
        "max_webpage_bytes": 2097152,
        "max_download_bytes": 26214400,
        "max_http_redirects": 5,
        "max_search_query_chars": 2000,
        "max_search_results": 3,
        "search_provider": "auto",
        "searxng_url": "",
        "search_fallback": True,
        "max_fetched_webpage_chars": 12000,
    },
    "tools": {
        "max_read_file_bytes": 1048576,
        "file_read_chunk_bytes": 65536,
        "binary_probe_bytes": 2048,
        "shell_command_timeout_seconds": 30,
        "default_history_read_limit": 20,
        "override_tool_list": [],
    },
    "media": {
        "max_captured_image_bytes": 2097152,
        "max_images_per_turn": 3,
        "max_microphone_seconds": 30,
        "default_microphone_seconds": 5.0,
        "microphone_sample_rate": 16000,
        "default_camera_device_index": 0,
        "max_camera_device_index": 32,
        "max_microphone_device_index": 128,
        "max_image_width": 1280,
        "max_image_height": 720,
        "jpeg_quality": 85,
        "max_decoded_image_pixels": 20000000,
        "max_media_file_bytes": 26214400,
        "max_video_seconds": 120,
    },
    "skills": {
        "max_name_chars": 64,
        "max_description_chars": 500,
        "max_instruction_chars": 20000,
        "max_preview_chars": 12000,
        "max_iterations": 15,
        "description_preview_chars": 100,
        "relevance_threshold": 1.5,
        "relevance_exact_match_score": 3.0,
        "relevance_prefix_match_score": 2.0,
        "relevance_word_match_score": 1.0,
    },
    "code_tasks": {
        "git_command_timeout_seconds": 30,
        "test_command_timeout_seconds": 180,
        "test_commands": [],
        "successful_test_output_chars": 6000,
        "failed_test_output_chars": 3000,
    },
    "logging": {
        "debug_enabled": 0,
        "level": "INFO",
    },
    "mcp": {
        "auto_approve_tools": [],
        "servers": {},
    },
}

_LEGACY_CONFIG_KEYS = {
    "default_db_path": ("paths", "database"),
    "workspace_root": ("paths", "workspace"),
    "code_output_root": ("paths", "code_output"),
    "skills_folder": ("paths", "skills"),
    "rag_docs_path": ("paths", "rag_documents"),
    "rag_index_path": ("paths", "rag_index"),
    "ollama_base_url": ("models", "ollama_base_url"),
    "hardware_acceleration": ("models", "hardware_acceleration"),
    "preferred_model": ("models", "preferred_model"),
    "online_base_url": ("models", "online_base_url"),
    "online_model": ("models", "online_model"),
    "online_api_key": ("models", "online_api_key"),
    "default_model_temperature": ("models", "temperature"),
    "embedding_model": ("models", "embedding_model"),
    "thinking_toggle_default": ("models", "thinking_enabled_by_default"),
    "thinking_effort_default": ("models", "thinking_effort"),
    "agent_permission_mode": ("agent", "permission_mode"),
    "max_tool_iterations": ("agent", "max_tool_iterations"),
    "max_tool_calls": ("agent", "max_tool_calls"),
    "max_task_seconds": ("agent", "max_task_seconds"),
    "max_tool_output_chars": ("agent", "max_tool_output_chars"),
    "max_history_messages": ("agent", "max_history_messages"),
    "max_context_tokens": ("agent", "max_context_tokens"),
    "max_output_tokens": ("agent", "max_output_tokens"),
    "STREAMING_OUTPUT": ("agent", "streaming_output"),
    "streaming_output": ("agent", "streaming_output"),
    "VISIBILE_REASONING": ("agent", "visible_reasoning"),
    "visible_reasoning": ("agent", "visible_reasoning"),
    "max_model_capability_cache_entries": (
        "agent",
        "max_model_capability_cache_entries",
    ),
    "conversation_retention_days": ("memory", "conversation_retention_days"),
    "episode_retention_days": ("memory", "episode_retention_days"),
    "registered_skill_retention_days": ("memory", "registered_skill_retention_days"),
    "task_stale_after_days": ("memory", "task_stale_after_days"),
    "max_summary_chars": ("memory", "max_summary_chars"),
    "rag_max_file_bytes": ("rag", "max_file_bytes"),
    "rag_max_corpus_bytes": ("rag", "max_corpus_bytes"),
    "rag_max_pdf_pages": ("rag", "max_pdf_pages"),
    "rag_max_documents": ("rag", "max_documents"),
    "enable_web_research": ("network", "web_research_enabled"),
    "web_research_consent": ("network", "web_research_consent"),
    "internet_check_host": ("network", "internet_check_host"),
    "internet_check_port": ("network", "internet_check_port"),
    "internet_check_timeout": ("network", "internet_check_timeout_seconds"),
    "max_network_concurrency": ("network", "max_concurrent_requests"),
    "network_request_timeout": ("network", "request_timeout_seconds"),
    "max_webpage_bytes": ("network", "max_webpage_bytes"),
    "max_download_bytes": ("network", "max_download_bytes"),
    "max_http_redirects": ("network", "max_http_redirects"),
    "max_search_query_chars": ("network", "max_search_query_chars"),
    "max_search_results": ("network", "max_search_results"),
    "max_fetched_webpage_chars": ("network", "max_fetched_webpage_chars"),
    "max_read_file_bytes": ("tools", "max_read_file_bytes"),
    "file_read_chunk_bytes": ("tools", "file_read_chunk_bytes"),
    "binary_probe_bytes": ("tools", "binary_probe_bytes"),
    "shell_command_timeout_seconds": ("tools", "shell_command_timeout_seconds"),
    "default_history_read_limit": ("tools", "default_history_read_limit"),
    "rag_context_results": ("agent", "rag_context_results"),
    "summary_prompt_sentences": ("agent", "summary_prompt_sentences"),
    "DEBUG_LOG_ENABLED": ("logging", "debug_enabled"),
    "mcp_auto_approve_tools": ("mcp", "auto_approve_tools"),
    "mcpServers": ("mcp", "servers"),
}


def _merge_config(config: dict) -> dict:
    """Merge categorized settings and legacy flat settings into runtime aliases."""
    merged = {section: values.copy() for section, values in DEFAULT_CONFIG.items()}
    for section, values in config.items():
        if section in merged and isinstance(values, dict):
            merged[section].update(values)
    for old_key, (section, key) in _LEGACY_CONFIG_KEYS.items():
        if old_key in config and not (
            isinstance(config.get(section), dict) and key in config[section]
        ):
            merged[section][key] = config[old_key]
    for section, value in config.items():
        if section not in merged and section not in _LEGACY_CONFIG_KEYS:
            merged[section] = value
    if merged["paths"].get("rag_documents") is None:
        merged["paths"]["rag_documents"] = DEFAULT_CONFIG["paths"]["rag_documents"]
    if merged["paths"].get("skills") == "resources/skills":
        merged["paths"]["skills"] = DEFAULT_CONFIG["paths"]["skills"]
    if merged["paths"].get("rag_index") == "./.local_ai_chroma_db":
        merged["paths"]["rag_index"] = DEFAULT_CONFIG["paths"]["rag_index"]
    if not isinstance(merged["agent"].get("system_prompt"), str) or not (
        merged["agent"]["system_prompt"].strip()
    ):
        merged["agent"]["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    for old_key, (section, key) in _LEGACY_CONFIG_KEYS.items():
        merged[old_key] = merged[section][key]
    return merged


def _categorized_config_for_file(config: dict, merged: dict) -> dict:
    """Return normalized sectioned settings while preserving unknown extensions."""
    categorized = {section: merged[section] for section in DEFAULT_CONFIG}
    categorized.update(
        {
            key: value
            for key, value in config.items()
            if key not in DEFAULT_CONFIG and key not in _LEGACY_CONFIG_KEYS
        }
    )
    return categorized


def _persist_categorized_config(config: dict) -> None:
    """Atomically migrate old flat settings and materialize new categorized defaults."""
    if CONFIG_FILE_PATH.is_symlink():
        warnings.warn(
            f"Not migrating configuration through symbolic link {CONFIG_FILE_PATH}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    normalized = json.dumps(config, indent=4)
    try:
        if CONFIG_FILE_PATH.read_text(encoding="utf-8") == normalized:
            return
    except OSError:
        pass

    temporary_path = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{CONFIG_FILE_PATH.name}.",
            suffix=".tmp",
            dir=CONFIG_FILE_PATH.parent,
        )
        temporary_path = pathlib.Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
            config_file.write(normalized)
            config_file.flush()
            os.fsync(config_file.fileno())
        os.replace(temporary_path, CONFIG_FILE_PATH)
        os.chmod(CONFIG_FILE_PATH, 0o600)
    except OSError as exc:
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink(missing_ok=True)
        warnings.warn(
            f"Could not update categorized configuration at {CONFIG_FILE_PATH}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _deep_merge_config(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = {key: value for key, value in (base or {}).items()}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_config(result[key], value)
        else:
            result[key] = value
    return result


def discover_project_config_paths(workspace_root=None) -> list[pathlib.Path]:
    """Return candidate project-level config files for a workspace."""
    root = (
        pathlib.Path(workspace_root)
        if workspace_root is not None
        else pathlib.Path.cwd()
    )
    base = root / ".private_agent"
    return [base / "settings.json", base / "settings.local.json"]


def load_project_config(workspace_root=None) -> dict:
    """Load and merge project-level config files for a workspace."""
    merged: dict = {}
    for path in discover_project_config_paths(workspace_root):
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            data = json.loads(content)
            if isinstance(data, dict):
                merged = _deep_merge_config(merged, data)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            warnings.warn(
                f"Malformed project configuration in {path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return merged


def merge_project_config(config: dict, workspace_root=None) -> dict:
    """Merge project-level config over an already merged config."""
    return _deep_merge_config(config or {}, load_project_config(workspace_root))


def load_or_create_config() -> dict:
    """Load the per-user config, creating it with defaults when missing."""
    if not CONFIG_FILE_PATH.exists():
        try:
            descriptor = os.open(
                CONFIG_FILE_PATH,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as config_file:
                config_file.write(json.dumps(DEFAULT_CONFIG, indent=4))
        except FileExistsError:
            pass
        except (ValueError, OSError) as e:
            warnings.warn(
                f"Failed to create default configuration file at {CONFIG_FILE_PATH}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            merged = _merge_config(DEFAULT_CONFIG)
            return _deep_merge_config(merged, load_project_config())

    try:
        if CONFIG_FILE_PATH.is_dir():
            raise IsADirectoryError(
                f"Config path {CONFIG_FILE_PATH} is a directory, not a file."
            )

        content = CONFIG_FILE_PATH.read_text(encoding="utf-8").strip()
        if not content:
            merged = _merge_config(DEFAULT_CONFIG)
            return _deep_merge_config(merged, load_project_config())

        config = json.loads(content)
        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration root must be a JSON object (dict), got {type(config).__name__}"
            )

        merged = _merge_config(config)
        merged = _deep_merge_config(merged, load_project_config())
        _persist_categorized_config(_categorized_config_for_file(config, merged))
        return merged
    except (
        json.JSONDecodeError,
        ValueError,
        IsADirectoryError,
        PermissionError,
        OSError,
    ) as jde:
        warnings.warn(
            f"Malformed or invalid configuration encountered in {CONFIG_FILE_PATH}: {jde}. Falling back to default configuration.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _merge_config(DEFAULT_CONFIG)


def load_config() -> dict:
    """Alias for load_or_create_config for compatibility with tool/agent modules."""
    return load_or_create_config()


# Load global configuration immediately with safe parsing/type conversions
APP_CONFIG = load_or_create_config()


SETTINGS = Settings.from_config(APP_CONFIG, default_paths=DEFAULT_CONFIG["paths"])


def _configured_path(value, default: str) -> str:
    """Expand config paths and anchor relative paths to the user's home."""
    return configured_path(value, default)


def _positive_int(key: str, default: int) -> int:
    """Read a positive integer from the flat runtime aliases."""
    return positive_int(APP_CONFIG, key, default)


def _positive_float(key: str, default: float) -> float:
    """Read a positive float from the flat runtime aliases."""
    return positive_float(APP_CONFIG, key, default)


def _configured_int(section: str, key: str, default: int) -> int:
    """Read a positive integer from a config section."""
    return configured_int(APP_CONFIG, section, key, default)


def _configured_nonnegative_int(section: str, key: str, default: int) -> int:
    """Read a non-negative integer from a config section."""
    return configured_nonnegative_int(APP_CONFIG, section, key, default)


def _configured_int_range(
    section: str,
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Read an integer constrained to an inclusive range."""
    return configured_int_range(APP_CONFIG, section, key, default, minimum, maximum)


def _configured_bool(section: str, key: str, default: bool) -> bool:
    """Read a boolean that also accepts the usual textual spellings."""
    return configured_bool(APP_CONFIG, section, key, default)


def _configured_float(section: str, key: str, default: float) -> float:
    """Read a positive float from a config section."""
    return configured_float(APP_CONFIG, section, key, default)


DEFAULT_DB_PATH = SETTINGS.default_db_path
WORKSPACE_ROOT_DEFAULT = SETTINGS.workspace_root_default
CODE_OUTPUT_ROOT = SETTINGS.code_output_root
SKILLS_FOLDER_DEFAULT = SETTINGS.skills_folder_default
RAG_DOCS_DEFAULT = SETTINGS.rag_docs_default
MODEL_TEMPERATURE = SETTINGS.model_temperature
ONLINE_PERMISSION_OVERRIDE = SETTINGS.online_permission_override
EMBEDDING_MODEL = SETTINGS.embedding_model
RAG_INDEX_PATH = SETTINGS.rag_index_path
OLLAMA_BASE_URL = SETTINGS.ollama_base_url
HARDWARE_ACCELERATION_MODE = SETTINGS.hardware_acceleration_mode
PREFERRED_MODEL = SETTINGS.preferred_model
MAX_TOOL_ITERATIONS = SETTINGS.max_tool_iterations
MAX_TOOL_CALLS = SETTINGS.max_tool_calls
MAX_TASK_SECONDS = SETTINGS.max_task_seconds
MAX_TOOL_OUTPUT_CHARS = SETTINGS.max_tool_output_chars
MAX_HISTORY_MESSAGES = SETTINGS.max_history_messages
MAX_CONTEXT_TOKENS = SETTINGS.max_context_tokens
MAX_OUTPUT_TOKENS = SETTINGS.max_output_tokens
STREAMING_OUTPUT = SETTINGS.streaming_output
VISIBLE_REASONING = SETTINGS.visible_reasoning
SYSTEM_PROMPT = SETTINGS.system_prompt
PERMISSION_ALLOW_RULES = SETTINGS.permission_allow_rules
PERMISSION_DENY_RULES = SETTINGS.permission_deny_rules
PROJECT_INSTRUCTIONS_FILE = SETTINGS.project_instructions_file
CONVERSATION_RETENTION_DAYS = SETTINGS.conversation_retention_days
EPISODE_RETENTION_DAYS = SETTINGS.episode_retention_days
REGISTERED_SKILL_RETENTION_DAYS = SETTINGS.registered_skill_retention_days
TASK_STALE_AFTER_DAYS = SETTINGS.task_stale_after_days
MAX_SUMMARY_CHARS = SETTINGS.max_summary_chars
MAX_LEARNED_ITEM_CHARS = SETTINGS.max_learned_item_chars
MAX_RESUME_STATE_CHARS = SETTINGS.max_resume_state_chars
MAX_MEMORY_CONTEXT_TOKENS = SETTINGS.max_memory_context_tokens
MAX_RELEVANT_EPISODES = SETTINGS.max_relevant_episodes
LEARNED_ITEM_EXPIRY_DAYS = SETTINGS.learned_item_expiry_days
LEARNED_ITEM_REVIEW_DAYS = SETTINGS.learned_item_review_days
DEFAULT_HISTORY_MESSAGES = SETTINGS.default_history_messages
DEFAULT_EPISODIC_SUMMARIES = SETTINGS.default_episodic_summaries
MAX_NETWORK_CONCURRENCY = SETTINGS.max_network_concurrency
NETWORK_REQUEST_TIMEOUT = SETTINGS.network_request_timeout
RAG_MAX_FILE_BYTES = SETTINGS.rag_max_file_bytes
RAG_MAX_CORPUS_BYTES = SETTINGS.rag_max_corpus_bytes
RAG_MAX_PDF_PAGES = SETTINGS.rag_max_pdf_pages
RAG_MAX_DOCUMENTS = SETTINGS.rag_max_documents
RAG_CHUNK_SIZE_CHARS = SETTINGS.rag_chunk_size_chars
RAG_CHUNK_OVERLAP_CHARS = SETTINGS.rag_chunk_overlap_chars
CHUNKING_STRATEGY = SETTINGS.chunking_strategy
CHUNKING_UNIT = SETTINGS.chunking_unit
CHUNKING_SIZE = SETTINGS.chunking_size
CHUNKING_OVERLAP = SETTINGS.chunking_overlap
CHUNKING_MIN_CHUNK_CHARS = SETTINGS.chunking_min_chunk_chars
CHUNKING_SEPARATORS = SETTINGS.chunking_separators
CHUNKING_SEMANTIC_PERCENTILE = SETTINGS.chunking_semantic_percentile
CHUNKING_SEMANTIC_BUFFER = SETTINGS.chunking_semantic_buffer
CHUNKING_SEMANTIC_MAX_SENTENCES = SETTINGS.chunking_semantic_max_sentences
RAG_SIMILARITY_RESULTS = SETTINGS.rag_similarity_results
RAG_SQLITE_MAX_ROWS_PER_TABLE = SETTINGS.rag_sqlite_max_rows_per_table
RAG_AUTO_REFRESH_SECONDS = SETTINGS.rag_auto_refresh_seconds
ENABLE_WEB_RESEARCH = SETTINGS.enable_web_research
WEB_RESEARCH_CONSENT = SETTINGS.web_research_consent
INTERNET_CHECK_HOST = SETTINGS.internet_check_host
INTERNET_CHECK_PORT = SETTINGS.internet_check_port
INTERNET_CHECK_TIMEOUT = SETTINGS.internet_check_timeout
OVERRIDE_TOOL_LIST = SETTINGS.override_tool_list
THINKING_TOGGLE_DEFAULT = SETTINGS.thinking_toggle_default
THINKING_EFFORT_DEFAULT = SETTINGS.thinking_effort_default
AGENT_PERMISSION_MODE = SETTINGS.agent_permission_mode
DEBUG_LOG_ENABLED = SETTINGS.debug_log_enabled
DEBUG_LOG_LEVEL = SETTINGS.debug_log_level
MCP_SERVERS = SETTINGS.mcp_servers
MCP_AUTO_APPROVE_TOOLS = SETTINGS.mcp_auto_approve_tools
ONLINE_REQUEST_TIMEOUT = SETTINGS.online_request_timeout
ONLINE_MODEL_LIST_TIMEOUT = SETTINGS.online_model_list_timeout
ONLINE_MAX_RETRIES = SETTINGS.online_max_retries
ONLINE_MODEL_LIST_LIMIT = SETTINGS.online_model_list_limit
MAX_MODEL_CAPABILITY_CACHE_ENTRIES = SETTINGS.max_model_capability_cache_entries
MAX_WEBPAGE_BYTES = SETTINGS.max_webpage_bytes
MAX_DOWNLOAD_BYTES = SETTINGS.max_download_bytes
MAX_HTTP_REDIRECTS = SETTINGS.max_http_redirects
MAX_SEARCH_QUERY_CHARS = SETTINGS.max_search_query_chars
MAX_SEARCH_RESULTS = SETTINGS.max_search_results
SEARCH_PROVIDER = SETTINGS.search_provider
SEARXNG_URL = SETTINGS.searxng_url
SEARCH_FALLBACK = SETTINGS.search_fallback
MAX_FETCHED_WEBPAGE_CHARS = SETTINGS.max_fetched_webpage_chars
MAX_READ_FILE_BYTES = SETTINGS.max_read_file_bytes
FILE_READ_CHUNK_BYTES = SETTINGS.file_read_chunk_bytes
BINARY_PROBE_BYTES = SETTINGS.binary_probe_bytes
SHELL_COMMAND_TIMEOUT = SETTINGS.shell_command_timeout
DEFAULT_HISTORY_READ_LIMIT = SETTINGS.default_history_read_limit
MAX_SKILL_NAME_CHARS = SETTINGS.max_skill_name_chars
MAX_STREAM_TIMEOUT = SETTINGS.max_stream_timeout
MAX_CAPTURED_IMAGE_BYTES = SETTINGS.max_captured_image_bytes
MAX_DECODED_IMAGE_PIXELS = SETTINGS.max_decoded_image_pixels
MAX_MEDIA_FILE_BYTES = SETTINGS.max_media_file_bytes
MAX_IMAGES_PER_TURN = SETTINGS.max_images_per_turn
MAX_MICROPHONE_SECONDS = SETTINGS.max_microphone_seconds
MAX_VIDEO_SECONDS = SETTINGS.max_video_seconds
DEFAULT_MICROPHONE_SECONDS = SETTINGS.default_microphone_seconds
MICROPHONE_SAMPLE_RATE = SETTINGS.microphone_sample_rate
DEFAULT_CAMERA_DEVICE_INDEX = SETTINGS.default_camera_device_index
MAX_CAMERA_DEVICE_INDEX = SETTINGS.max_camera_device_index
MAX_MICROPHONE_DEVICE_INDEX = SETTINGS.max_microphone_device_index
MAX_IMAGE_WIDTH = SETTINGS.max_image_width
MAX_IMAGE_HEIGHT = SETTINGS.max_image_height
JPEG_QUALITY = SETTINGS.jpeg_quality
MAX_SKILL_DESCRIPTION_CHARS = SETTINGS.max_skill_description_chars
MAX_SKILL_INSTRUCTION_CHARS = SETTINGS.max_skill_instruction_chars
MAX_SKILL_PREVIEW_CHARS = SETTINGS.max_skill_preview_chars
SKILL_MAX_ITERATIONS = SETTINGS.skill_max_iterations
SKILL_DESCRIPTION_PREVIEW_CHARS = SETTINGS.skill_description_preview_chars
SKILL_RELEVANCE_THRESHOLD = SETTINGS.skill_relevance_threshold
SKILL_RELEVANCE_EXACT_MATCH_SCORE = SETTINGS.skill_relevance_exact_match_score
SKILL_RELEVANCE_PREFIX_MATCH_SCORE = SETTINGS.skill_relevance_prefix_match_score
SKILL_RELEVANCE_WORD_MATCH_SCORE = SETTINGS.skill_relevance_word_match_score
SKILL_RELEVANCE_EMBEDDING_THRESHOLD = SETTINGS.skill_relevance_embedding_threshold
RAG_CONTEXT_RESULTS = SETTINGS.rag_context_results
SUMMARY_PROMPT_SENTENCES = SETTINGS.summary_prompt_sentences
CONTEXT_COMPACTION_THRESHOLD = SETTINGS.context_compaction_threshold
CONTEXT_KEEP_RECENT_MESSAGES = SETTINGS.context_keep_recent_messages
CONTEXT_SUMMARY_CHARS = SETTINGS.context_summary_chars
CODE_TASK_GIT_TIMEOUT = SETTINGS.code_task_git_timeout
CODE_TASK_TEST_TIMEOUT = SETTINGS.code_task_test_timeout
CODE_TASK_TEST_COMMANDS = SETTINGS.code_task_test_commands
CODE_TASK_SUCCESS_OUTPUT_CHARS = SETTINGS.code_task_success_output_chars
CODE_TASK_FAILURE_OUTPUT_CHARS = SETTINGS.code_task_failure_output_chars
