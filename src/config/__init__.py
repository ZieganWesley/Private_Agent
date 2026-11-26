import contextlib
import inspect
import json
import os
import pathlib
import tempfile
import warnings

# Compatibility shim for Python 3.14+: chromadb still calls the deprecated
# asyncio.iscoroutinefunction. Route it to the recommended inspect helper before
# optional RAG dependencies are imported.
try:
    import asyncio

    asyncio.iscoroutinefunction = inspect.iscoroutinefunction
except Exception:  # noqa: BLE001 - defensive compatibility shim
    pass

from .defaults import (
    DEFAULT_CONFIG,
    DEFAULT_PATHS,  # noqa: F401  (re-exported for compatibility)
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT_PATH,  # noqa: F401  (re-exported for compatibility)
)
from .settings import (
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
    """Return the config file path: $PRIVATE_AGENT_CONFIG or ~/.config/private_agent/private_config.json."""
    override = (environ if environ is not None else os.environ).get(
        CONFIG_PATH_ENV_VAR, ""
    )
    if override.strip():
        return pathlib.Path(override.strip()).expanduser().absolute()
    return pathlib.Path.home() / ".config" / "private_agent" / "private_config.json"


CONFIG_FILE_PATH = resolve_config_path()


def _merge_config(config: dict) -> dict:
    """Merge categorized settings into a normalized configuration."""
    merged = {section: values.copy() for section, values in DEFAULT_CONFIG.items()}
    for section, values in (config or {}).items():
        if section in merged and isinstance(values, dict):
            merged[section].update(values)
        elif section not in merged and isinstance(values, dict):
            merged[section] = values
        # Flat legacy keys and unknown scalar settings are ignored.
    if merged["paths"].get("rag_documents") is None:
        merged["paths"]["rag_documents"] = DEFAULT_CONFIG["paths"]["rag_documents"]
    if merged["paths"].get("database") in {None, "", "~/.local_ai_memory.db"}:
        merged["paths"]["database"] = DEFAULT_CONFIG["paths"]["database"]
    if merged["paths"].get("skills") in {
        None,
        "",
        "resources/skills",
        "~/.config/private_agent/resources/skills",
    }:
        merged["paths"]["skills"] = DEFAULT_CONFIG["paths"]["skills"]
    if merged["paths"].get("rag_index") in {
        None,
        "",
        "./.local_ai_chroma_db",
        "~/.config/private_agent/rag_index",
    }:
        merged["paths"]["rag_index"] = DEFAULT_CONFIG["paths"]["rag_index"]
    if not isinstance(merged["agent"].get("system_prompt"), str) or not (
        merged["agent"]["system_prompt"].strip()
    ):
        merged["agent"]["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    # The local endpoint list is the single description of locally hosted
    # servers; mirror a legacy ollama_base_url into it instead of keeping a
    # second representation alive.
    from endpoint_legacy import fold_legacy_ollama_endpoints

    merged["models"] = fold_legacy_ollama_endpoints(merged["models"])
    # The agent section is the canonical home for context-cache settings; fold an
    # explicitly configured legacy context_cache section into it so both spellings
    # take effect instead of one silently shadowing the other.
    merged["agent"] = _fold_legacy_context_cache(config, merged["agent"])
    return merged


def _fold_legacy_context_cache(config: dict, agent: dict) -> dict:
    """Mirror a legacy ``context_cache`` section into ``agent`` without overriding it."""
    legacy = (config or {}).get("context_cache")
    if not isinstance(legacy, dict):
        return agent
    explicit_agent = (config or {}).get("agent")
    explicit_agent = explicit_agent if isinstance(explicit_agent, dict) else {}
    folded = dict(agent)
    for key, value in legacy.items():
        if key in explicit_agent:
            continue
        folded[key] = value
    return folded


def _categorized_config_for_file(config: dict, merged: dict) -> dict:
    """Return normalized sectioned settings while preserving unknown extensions."""
    categorized = {section: merged[section] for section in DEFAULT_CONFIG}
    categorized.update(
        {
            key: value
            for key, value in (config or {}).items()
            if key not in DEFAULT_CONFIG and isinstance(value, dict)
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

    CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
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
            CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
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
EMBEDDING_PROVIDER = SETTINGS.embedding_provider
PROVIDER_API_MODE = SETTINGS.provider_api_mode
RAW_COMPLETIONS_ENABLED = SETTINGS.raw_completions_enabled
EMBEDDING_BASE_URL = SETTINGS.embedding_base_url
RAG_INDEX_PATH = SETTINGS.rag_index_path
OLLAMA_BASE_URL = SETTINGS.ollama_base_url
OLLAMA_BASE_URLS = SETTINGS.ollama_base_urls
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
VERBOSE_TIMINGS = SETTINGS.verbose_timings
STATUS_BAR = SETTINGS.status_bar
SHOW_WELCOME_SCREEN = SETTINGS.show_welcome_screen
INFORMATIVE_PROMPT = SETTINGS.informative_prompt
SHOW_TURN_DIVIDER = SETTINGS.show_turn_divider
SHOW_TURN_STATS = SETTINGS.show_turn_stats
GROUPED_HELP = SETTINGS.grouped_help
SYSTEM_PROMPT = SETTINGS.system_prompt
PERMISSION_ALLOW_RULES = SETTINGS.permission_allow_rules
PERMISSION_DENY_RULES = SETTINGS.permission_deny_rules
APPROVED_EXTRA_FOLDERS = SETTINGS.approved_extra_folders
SANDBOX_NETWORK_DEFAULT = SETTINGS.sandbox_network_default
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
RAG_FILE_TYPES = SETTINGS.rag_file_types
RAG_SQLITE_ROWS_PER_CHUNK = SETTINGS.rag_sqlite_rows_per_chunk
RAG_SQLITE_MAX_CELL_CHARS = SETTINGS.rag_sqlite_max_cell_chars
RAG_AUTO_REFRESH_SECONDS = SETTINGS.rag_auto_refresh_seconds
ENABLE_WEB_RESEARCH = SETTINGS.enable_web_research
WEB_RESEARCH_CONSENT = SETTINGS.web_research_consent
INTERNET_CHECK_HOST = SETTINGS.internet_check_host
INTERNET_CHECK_PORT = SETTINGS.internet_check_port
INTERNET_CHECK_TIMEOUT = SETTINGS.internet_check_timeout
OVERRIDE_TOOL_LIST = SETTINGS.override_tool_list
SANDBOX_EXTRA_RISKY_PROGRAMS = SETTINGS.sandbox_extra_risky_programs
SANDBOX_EXTRA_SYSTEM_PATHS = SETTINGS.sandbox_extra_system_paths
RAG_EXCLUDED_FILE_TYPES = SETTINGS.rag_excluded_file_types
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
GEOCODING_URL = SETTINGS.geocoding_url
FORECAST_URL = SETTINGS.forecast_url
ARCHIVE_URL = SETTINGS.archive_url
IP_GEOLOCATION_URLS = SETTINGS.ip_geolocation_urls
LOCATION_MAX_RESPONSE_BYTES = SETTINGS.location_max_response_bytes
WEATHER_MAX_DAYS = SETTINGS.weather_max_days
WEATHER_PAST_DAYS = SETTINGS.weather_past_days
TOKENIZER_ENCODING = SETTINGS.tokenizer_encoding
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
RAG_DOCUMENTS_ONLY = SETTINGS.rag_documents_only
RAG_RELEVANCE_THRESHOLD = SETTINGS.rag_relevance_threshold
RAG_INCLUDE_CODING_GUIDANCE = SETTINGS.rag_include_coding_guidance
RERANK_MODE = SETTINGS.rerank_mode
RERANK_MODEL = SETTINGS.rerank_model
RERANK_TIMEOUT_SECONDS = SETTINGS.rerank_timeout_seconds
RERANK_TOP_N = SETTINGS.rerank_top_n
RERANK_MAX_CANDIDATES = SETTINGS.rerank_max_candidates
RERANK_LOCAL_THRESHOLD = SETTINGS.rerank_local_threshold
RERANK_CACHE_ENTRIES = SETTINGS.rerank_cache_entries
RERANK_BREAKER_FAILURES = SETTINGS.rerank_breaker_failures
RERANK_BREAKER_COOLDOWN_SECONDS = SETTINGS.rerank_breaker_cooldown_seconds
SUMMARY_PROMPT_SENTENCES = SETTINGS.summary_prompt_sentences
CONTEXT_COMPACTION_THRESHOLD = SETTINGS.context_compaction_threshold
CONTEXT_KEEP_RECENT_MESSAGES = SETTINGS.context_keep_recent_messages
CONTEXT_SUMMARY_CHARS = SETTINGS.context_summary_chars
CONTEXT_ENGINE_ENABLED = SETTINGS.context_engine_enabled
CONTEXT_ENGINE_PROFILE = SETTINGS.context_engine_profile
CONTEXT_SAFETY_MARGIN_TOKENS = SETTINGS.context_safety_margin_tokens
CONTEXT_OUTPUT_RESERVE_RATIO = SETTINGS.context_output_reserve_ratio
CONTEXT_KV_CACHE_ENABLED = SETTINGS.context_kv_cache_enabled
CONTEXT_KV_CACHE_PROVIDER_HINTS = SETTINGS.context_kv_cache_provider_hints
CONTEXT_KV_CACHE_MIN_PREFIX_TOKENS = SETTINGS.context_kv_cache_min_prefix_tokens
CODE_TASK_GIT_TIMEOUT = SETTINGS.code_task_git_timeout
CODE_TASK_TEST_TIMEOUT = SETTINGS.code_task_test_timeout
CODE_TASK_TEST_COMMANDS = SETTINGS.code_task_test_commands
CODE_TASK_SUCCESS_OUTPUT_CHARS = SETTINGS.code_task_success_output_chars
CODE_TASK_FAILURE_OUTPUT_CHARS = SETTINGS.code_task_failure_output_chars
CODE_TASK_SANDBOX_CPU_SECONDS = SETTINGS.code_task_sandbox_cpu_seconds
CODE_TASK_SANDBOX_ADDRESS_SPACE_BYTES = SETTINGS.code_task_sandbox_address_space_bytes
CODE_TASK_SANDBOX_PROCESS_COUNT = SETTINGS.code_task_sandbox_process_count
CODE_TASK_SANDBOX_FILE_SIZE_BYTES = SETTINGS.code_task_sandbox_file_size_bytes
CONTEXT_COMPACTION_SOURCE_TOKENS = SETTINGS.context_compaction_source_tokens
TOOL_RESULT_PREVIEW_CHARS = SETTINGS.tool_result_preview_chars
TOOL_ARGUMENT_PREVIEW_CHARS = SETTINGS.tool_argument_preview_chars
SEARCH_MAX_FILES = SETTINGS.search_max_files
SEARCH_MAX_BYTES = SETTINGS.search_max_bytes
SEARCH_OUTPUT_CHARS = SETTINGS.search_output_chars
TASK_CITATION_BUDGET_CHARS = SETTINGS.task_citation_budget_chars
