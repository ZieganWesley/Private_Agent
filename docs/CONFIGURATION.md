# Configuration

Private Agent reads one JSON configuration file and merges a project-local file
over it. Everything is validated when it loads: an unknown choice, a value of
the wrong type, or an unreadable file falls back to the documented default and
the app keeps running. Run `private-agent --doctor` to validate the file before
a session.

## File locations

| Path | Purpose |
|---|---|
| `~/.config/private_agent/private_config.json` | User configuration (created with defaults on first run, mode `0600`). |
| `$PRIVATE_AGENT_CONFIG` | Overrides the user configuration path when set to a file path. |
| `<workspace>/.private_agent/settings.json` | Project configuration, deep-merged over the user file. |
| `<workspace>/.private_agent/settings.local.json` | Machine-local project overrides, merged last. |
| `private-agent --config <path>` | One-off override for a session; sets `PRIVATE_AGENT_CONFIG`. |

Unknown top-level sections are preserved when the file is rewritten. Categorized
sections are merged key by key, so a partial file is valid.

## Minimal example

```json
{
    "models": {
        "endpoints": [
            {
                "name": "workstation",
                "base_url": "http://127.0.0.1:8080/v1",
                "roles": ["chat", "embedding", "rerank"],
                "models": ["qwen3-8b"],
                "max_context_tokens": 16384,
                "request_timeout_seconds": 60
            }
        ],
        "online_endpoints": [
            {
                "name": "primary",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "OPENAI_API_KEY",
                "models": ["gpt-4o-mini"]
            },
            {
                "name": "gateway",
                "base_url": "https://gateway.example.com/v1",
                "api_key": "sk-...",
                "models": ["llama-3.3-70b"]
            }
        ],
        "preferred_online_endpoint": "primary",
        "provider_api_mode": "chat",
        "online_model": "gpt-4o-mini"
    },
    "rag": {
        "embedding_provider": "openai-compatible",
        "embedding_base_url": "http://127.0.0.1:8080/v1",
        "rerank_mode": "auto",
        "rerank_local_threshold": 20
    }
}
```

## Endpoint lists

### `models.endpoints`

The preferred way to describe every server. Each entry accepts:

| Field | Values | Meaning |
|---|---|---|
| `name` | string, up to 64 chars | Label shown in selection and diagnostics. Defaults to `endpoint-N`. |
| `base_url` | `http(s)://host[:port][/path]` | OpenAI-compatible root. HTTPS is required for remote hosts; HTTP is allowed for loopback addresses. |
| `roles` | array of `chat`, `embedding`, `rerank`, `native` | Which work this server may do. Defaults to `["chat"]`. A server can serve several roles. |
| `models` | array of strings | Model identifiers to suggest; discovery still lists `/models` at runtime. |
| `enabled` | `true` / `false` | A disabled endpoint is never selected. |
| `allow_remote` | `true` / `false` | Permit a non-loopback host for this endpoint. |
| `max_context_tokens` | positive integer | Overrides the detected context window. |
| `max_output_tokens` | positive integer | Caps generated tokens for this endpoint. |
| `request_timeout_seconds` | positive number | Per-request timeout; defaults to 30. |
| `auth_header_env` | environment variable name | Sends `Authorization: <value>` read from the environment. The value is never stored in config. |
| `tls_verify` | `true` / `false` | Verify TLS certificates; keep `true` outside local testing. |

### `models.local_openai_compatible_endpoints`

Legacy local-server list, still fully supported. Entries default to all four
roles and accept the same fields as `models.endpoints`. Use this list for
locally hosted servers that predate `models.endpoints`.

### `models.ollama_base_url` / `models.ollama_base_urls`

Legacy Ollama URLs. They are mirrored into
`models.local_openai_compatible_endpoints` as an `Ollama` entry with the `/v1`
suffix and every role, so one list describes all local servers. An existing
entry for the same URL is never duplicated. `ollama_base_urls` accepts an array
and numbers additional entries `Ollama 2`, `Ollama 3`, and so on.

The default file contains both representations:

```json
"ollama_base_url": "http://localhost:11434",
"local_openai_compatible_endpoints": [
    { "name": "LM Studio", "base_url": "http://127.0.0.1:1234/v1" },
    { "name": "llama.cpp", "base_url": "http://127.0.0.1:8080/v1" },
    { "name": "Ollama", "base_url": "http://localhost:11434/v1" }
]
```

### `models.online_endpoints`

Remote OpenAI-compatible endpoints, one entry per account or gateway. The
session chooses `models.preferred_online_endpoint` by name; when it is unset the
first enabled entry is used, and with several configured endpoints the
interactive selector offers a numbered choice.

| Field | Values | Meaning |
|---|---|---|
| `name` | string | Name used by `preferred_online_endpoint`. Defaults to `online-N`. |
| `base_url` | `https://host/v1` | Remote base URL. HTTPS is required; HTTP is accepted only for loopback. `/v1` is appended when missing. |
| `api_key` | string | Key stored in the config file. Prefer `api_key_env`. |
| `api_key_env` | environment variable name | Reads the key from the environment at startup. |
| `models` | array of strings | Suggested model identifiers; the first is the default when listing fails. |
| `enabled` | `true` / `false` | Skip this endpoint without deleting it. |

### `models.online_base_url` / `models.online_api_key`

The legacy single-endpoint pair. It becomes the `OpenAI` online endpoint unless
`models.online_endpoints` already lists the same URL, so existing files keep
working. `models.preferred_online_endpoint` is empty by default.

## Models and providers
| Key | Default | Accepted values |
|---|---|---|
| `embedding_model` | `"nomic-embed-text"` | string |
| `endpoint_probe_ttl_seconds` | `300` | How long a reachable-endpoint result is cached. |
| `endpoints` | `[]` | array |
| `hardware_acceleration` | `"auto"` | `auto`, `cpu` or `accelerator`; recorded policy for local runtimes. |
| `local_openai_compatible_endpoints` | `[{"name": "LM Studio", "base_url": "http://127.0.0.1:1234...` | array |
| `ollama_base_url` | `"http://localhost:11434"` | string |
| `online_api_key` | `null` | Legacy single online key; prefer `online_endpoints[].api_key_env`. |
| `online_base_url` | `"https://api.openai.com/v1"` | Legacy single online URL, mirrored into `online_endpoints`. |
| `online_endpoints` | `[]` | Array of remote endpoint objects; see the endpoint tables above. |
| `online_max_retries` | `1` | Retries after a failed online request; `0` disables retries. |
| `online_model` | `null` | Pre-selected online model identifier or `null`. |
| `online_model_list_limit` | `30` | How many models the online picker lists. |
| `online_model_list_timeout_seconds` | `10` | Timeout for `GET /models` during online setup. |
| `online_permission_override` | `false` | `true` allows online providers without the per-session confirmation prompt. |
| `online_request_timeout_seconds` | `60` | Timeout for online chat requests. |
| `preferred_model` | `null` | Model to preselect when a provider starts, or `null`. |
| `preferred_online_endpoint` | `""` | Name of the online endpoint to use; empty means the first enabled entry. |
| `provider_api_mode` | `"chat"` | `chat`, `completions` or `responses`; `chat` is the default chat-completions API. |
| `raw_completions_enabled` | `false` | `true` enables raw `/v1/completions` for titles and short internal summaries. |
| `temperature` | `0.1` | Sampling temperature, typically `0.0` to `1.5`. |
| `thinking_effort` | `"medium"` | string |
| `thinking_enabled_by_default` | `false` | `true` or `false` |

## Retrieval and reranking
| Key | Default | Accepted values |
|---|---|---|
| `auto_refresh_seconds` | `30` | Re-index interval; `0` disables automatic refresh. |
| `chunk_overlap_chars` | `200` | integer greater than 0 |
| `chunk_size_chars` | `1200` | integer greater than 0 |
| `include_coding_guidance` | `true` | `true` adds repository-intelligence guidance to RAG context. |
| `max_corpus_bytes` | `26214400` | Largest total corpus indexed, in bytes (a size cap, not memory). |
| `max_documents` | `20000` | Hard limit on indexed documents. |
| `max_file_bytes` | `5242880` | Largest single file indexed, in bytes. |
| `max_pdf_pages` | `250` | integer greater than 0 |
| `relevance_threshold` | `0.25` | Minimum similarity score, `0` to `1`. |
| `rerank_breaker_cooldown_seconds` | `60` | Seconds before a failed rerank endpoint is tried again. |
| `rerank_breaker_failures` | `3` | Consecutive failures before the circuit breaker opens. |
| `rerank_cache_entries` | `128` | Bounded rerank result cache size. |
| `rerank_local_threshold` | `20` | Candidate count at or below which `auto` mode skips the API. |
| `rerank_max_candidates` | `50` | Upper bound on documents sent to the rerank endpoint. |
| `rerank_mode` | `"auto"` | `api` always calls the rerank endpoint, `local` never does, `auto` calls it only when there are more than `rerank_local_threshold` candidates. |
| `rerank_model` | `""` | Rerank model name; empty uses the rerank endpoint's first advertised model. |
| `rerank_timeout_seconds` | `2.5` | Per-request rerank timeout; a failure falls back to local cosine ordering. |
| `rerank_top_n` | `20` | How many documents the rerank response may keep. |
| `similarity_results` | `6` | Documents retrieved per query. |
| `sqlite_max_rows_per_table` | `5000` | integer greater than 0 |

## Agent loop and context
| Key | Default | Accepted values |
|---|---|---|
| `context_compaction_threshold` | `0.8` | Fraction `0`-`1` of the context budget that triggers compaction. |
| `context_engine_enabled` | `false` | `true` enables the profile-aware context engine. |
| `context_engine_profile` | `"auto"` | `auto`, `chat`, `coding`, `research` or `memory`. |
| `context_keep_recent_messages` | `6` | Messages always kept verbatim during compaction. |
| `context_output_reserve_ratio` | `0.33` | Share of the window reserved for the response, `0`-`1`. |
| `context_safety_margin_tokens` | `0` | Extra tokens kept free at the end of the prompt. |
| `context_summary_chars` | `1500` | Maximum characters of a compaction summary. |
| `kv_cache_enabled` | `false` | `true` records KV-cache hints for local servers. |
| `kv_cache_min_prefix_tokens` | `1024` | integer greater than 0 |
| `kv_cache_provider_hints` | `"auto"` | `auto`, `on` or `off`. |
| `max_context_tokens` | `12000` | Prompt budget in tokens. |
| `max_history_messages` | `20` | Hard cap on retained chat messages. |
| `max_model_capability_cache_entries` | `128` | integer greater than 0 |
| `max_output_tokens` | `2048` | Generation cap, `1` to `32768`. |
| `max_task_seconds` | `600` | Wall-clock budget for one request. |
| `max_tool_calls` | `60` | Total tool calls per turn. |
| `max_tool_iterations` | `15` | Tool round-trips per user turn. |
| `max_tool_output_chars` | `12000` | integer greater than 0 |
| `permission_mode` | `"auto"` | `manual`, `auto` or `full`. |
| `project_instructions_file` | `"AGENTS.md"` | Project instruction file name, e.g. `AGENTS.md`. |
| `rag_context_results` | `5` | RAG passages added to each turn. |
| `rag_documents_only` | `false` | `true` answers only from indexed documents. |
| `streaming_output` | `true` | `true` streams tokens as they arrive. |
| `summary_prompt_sentences` | `2` | integer greater than 0 |
| `system_prompt` | `"You are Private Agent, a privacy-first AI assistant.\n\n...` | System prompt text; empty means the packaged default. |
| `visible_reasoning` | `false` | `true` shows the model's reasoning when the provider sends it. |

## Memory and retention
| Key | Default | Accepted values |
|---|---|---|
| `conversation_retention_days` | `0` | integer of 0 or more |
| `default_episodic_summaries` | `5` | integer greater than 0 |
| `default_history_messages` | `20` | integer greater than 0 |
| `episode_retention_days` | `0` | integer of 0 or more |
| `learned_item_expiry_days` | `365` | integer greater than 0 |
| `learned_item_review_days` | `180` | integer greater than 0 |
| `max_learned_item_chars` | `1000` | integer greater than 0 |
| `max_memory_context_tokens` | `1200` | integer greater than 0 |
| `max_relevant_episodes` | `3` | integer greater than 0 |
| `max_resume_state_chars` | `12000` | integer greater than 0 |
| `max_summary_chars` | `2000` | integer greater than 0 |
| `registered_skill_retention_days` | `0` | integer of 0 or more |
| `task_stale_after_days` | `0` | integer of 0 or more |

## Paths
| Key | Default | Accepted values |
|---|---|---|
| `code_output` | `"~/.config/private_agent/projects"` | string |
| `database` | `"~/.config/private_agent/memory/private_agent.db"` | SQLite file for memory and sessions. |
| `rag_documents` | `"~/.config/private_agent/rag/"` | string |
| `rag_index` | `"~/.config/private_agent/rag/index"` | string |
| `skills` | `"~/.config/private_agent/skills"` | string |
| `workspace` | `"."` | Default workspace root, `.` for the current directory. |

## Permissions
| Key | Default | Accepted values |
|---|---|---|
| `allow` | `[]` | Array of permission rules, e.g. `shell:npm test`. |
| `approved_extra_folders` | `[]` | array |
| `deny` | `[]` | Array of rules evaluated before `allow`. |

## Network
| Key | Default | Accepted values |
|---|---|---|
| `internet_check_host` | `"1.1.1.1"` | string |
| `internet_check_port` | `443` | integer greater than 0 |
| `internet_check_timeout_seconds` | `3.0` | positive number |
| `max_concurrent_requests` | `4` | integer greater than 0 |
| `max_download_bytes` | `26214400` | integer greater than 0 |
| `max_fetched_webpage_chars` | `12000` | integer greater than 0 |
| `max_http_redirects` | `5` | integer greater than 0 |
| `max_search_query_chars` | `2000` | integer greater than 0 |
| `max_search_results` | `8` | integer greater than 0 |
| `max_stream_timeout_seconds` | `600` | integer greater than 0 |
| `max_webpage_bytes` | `2097152` | integer greater than 0 |
| `request_timeout_seconds` | `30` | integer greater than 0 |
| `search_fallback` | `true` | `true` or `false` |
| `search_provider` | `"auto"` | `auto`, `searxng` or `duckduckgo`. |
| `searxng_url` | `""` | SearxNG base URL; empty disables the SearxNG provider. |
| `web_research_consent` | `"ask"` | `ask`, `session` or `never`. |
| `web_research_enabled` | `true` | `true` or `false` |

## Tools
| Key | Default | Accepted values |
|---|---|---|
| `binary_probe_bytes` | `2048` | integer greater than 0 |
| `default_history_read_limit` | `20` | integer greater than 0 |
| `file_read_chunk_bytes` | `65536` | integer greater than 0 |
| `max_read_file_bytes` | `1048576` | Largest file the read tool returns. |
| `override_tool_list` | `[]` | Array of tool names that replaces the default tool set; empty means all. |
| `shell_command_timeout_seconds` | `30` | integer greater than 0 |

## Media capture
| Key | Default | Accepted values |
|---|---|---|
| `default_camera_device_index` | `0` | Camera index used when none is given. |
| `default_microphone_seconds` | `5.0` | positive number |
| `jpeg_quality` | `85` | JPEG quality, `1`-`100`. |
| `max_camera_device_index` | `32` | integer greater than 0 |
| `max_captured_image_bytes` | `2097152` | integer greater than 0 |
| `max_decoded_image_pixels` | `20000000` | integer greater than 0 |
| `max_image_height` | `720` | integer greater than 0 |
| `max_image_width` | `1280` | integer greater than 0 |
| `max_images_per_turn` | `3` | integer greater than 0 |
| `max_media_file_bytes` | `26214400` | integer greater than 0 |
| `max_microphone_device_index` | `128` | integer greater than 0 |
| `max_microphone_seconds` | `30` | integer greater than 0 |
| `max_video_seconds` | `120` | integer greater than 0 |
| `microphone_sample_rate` | `16000` | integer greater than 0 |

## Chunking
| Key | Default | Accepted values |
|---|---|---|
| `chunk_overlap` | `null` | string or `null` |
| `chunk_size` | `null` | string or `null` |
| `min_chunk_chars` | `0` | integer of 0 or more |
| `semantic_breakpoint_percentile` | `90` | integer greater than 0 |
| `semantic_buffer_sentences` | `1` | integer greater than 0 |
| `semantic_max_sentences` | `2000` | integer greater than 0 |
| `separators` | `["\n\n", "\n", ". ", " "]` | Array of split strings, most significant first. |
| `strategy` | `"character"` | `character`, `recursive`, `sentence`, `paragraph`, `token`, `markdown` or `semantic`, depending on available extras. |
| `unit` | `"chars"` | `chars` or `tokens`. |

## Skills
| Key | Default | Accepted values |
|---|---|---|
| `description_preview_chars` | `100` | integer greater than 0 |
| `max_description_chars` | `500` | integer greater than 0 |
| `max_instruction_chars` | `20000` | integer greater than 0 |
| `max_iterations` | `15` | integer greater than 0 |
| `max_name_chars` | `64` | integer greater than 0 |
| `max_preview_chars` | `12000` | integer greater than 0 |
| `relevance_exact_match_score` | `3.0` | positive number |
| `relevance_prefix_match_score` | `2.0` | positive number |
| `relevance_threshold` | `1.5` | positive number |
| `relevance_word_match_score` | `1.0` | number between 0 and 1 |

## Code tasks
| Key | Default | Accepted values |
|---|---|---|
| `failed_test_output_chars` | `3000` | integer greater than 0 |
| `git_command_timeout_seconds` | `30` | integer greater than 0 |
| `successful_test_output_chars` | `6000` | integer greater than 0 |
| `test_command_timeout_seconds` | `180` | integer greater than 0 |
| `test_commands` | `[]` | Array of argument arrays, e.g. `[["pytest", "-q"]]`. |

## Logging
| Key | Default | Accepted values |
|---|---|---|
| `debug_enabled` | `0` | `1` or `0`. |
| `level` | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |

## Hooks

`hooks` is an object keyed by hook name. Each hook accepts `command` (shell
string) and an optional timeout; the default is `{}` because no hook runs unless
configured.

## MCP servers
| Key | Default | Accepted values |
|---|---|---|
| `auto_approve_tools` | `[]` | Array of MCP tool names that skip approval. |
| `servers` | `{}` | Object of MCP server definitions keyed by name. |

## KV cache hints
| Key | Default | Accepted values |
|---|---|---|
| `kv_cache_enabled` | `false` | `true` or `false` |
| `kv_cache_min_prefix_tokens` | `1024` | integer greater than 0 |
| `kv_cache_provider_hints` | `"auto"` | `auto`, `on` or `off`; mirrors `agent.kv_cache_provider_hints`. |

## Environment variables

| Variable | Effect |
|---|---|
| `PRIVATE_AGENT_CONFIG` | Configuration file path. |
| `PRIVATE_AGENT_DEBUG` | `1`/`true` enables debug logging for one session. |
| `PRIVATE_AGENT_ACCESSIBLE` | `1` switches the UI to an accessible transcript. |
| `NO_COLOR` | Disables colour output. |
| `OPENAI_API_KEY` | Usable as `online_endpoints[].api_key_env` for an OpenAI endpoint. |

Anything named by `api_key_env`, `auth_header_env` or `models.*.api_key_env` is
read from the environment; keys are never written to logs.

## Validation rules

- The file must contain a single JSON object. A syntax error is reported with its
  line and column by `private-agent --doctor`, and the app then runs on defaults.
- Each documented section must be an object; each documented key must have the
  type shown above. `true`/`false`, `1`/`0`, `yes`/`no` and `on`/`off` are all
  accepted for booleans.
- Out-of-range numbers and unknown choices are replaced by the default instead of
  failing the session.
- URLs must be `http(s)` with a host and no embedded credentials, query or
  fragment. Remote hosts require HTTPS; loopback hosts may use HTTP.
