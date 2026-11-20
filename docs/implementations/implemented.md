## Strengths

- **Packaging:** standard `src/` layout with a `pyproject.toml`, exact
  dependency pins, and optional `dev` and `media` extras.
- **Security defaults:** command execution refuses to run when the sandbox
  tools are missing. Tool calls are checked against a workspace boundary,
  sensitive tool arguments are redacted, network calls check for private
  addresses, and there is no `shell=True`, `eval` or `exec`.
- **Database:** SQLite foreign keys are on, there is a busy timeout, and schema
  versions are tracked with `user_version`.
- **Tests and docs:** about 13k lines of tests against about 21k lines of
  source, and a detailed README.


## Part G — User Interface (first pass)

- G1: `Exit` option (0) in provider menu; `ProviderPreference.exit_requested`.
- G2: `initialize_knowledge_base` creates missing folders instead of warning.
- G3: indexing spinner only runs when a docs path is provided.
- G4: workspace prompt precedes RAG folder prompt.
- G15/G16: shared `Console` with a named `Theme` (info/success/warning/error/tool/user) and light-toggle env var.

## Section G status (complete)

Branch `g-remaining`; `main` untouched.

| ID | Status | Where |
|---|---|---|
| G1 | Done | `0. Exit` in the provider menu sets `ProviderPreference.exit_requested` |
| G2 | Done | `initialize_knowledge_base` creates missing folders instead of warning; regression test in `tests/unit/test_rag.py` |
| G3 | Done | Indexing spinner only runs when a docs path is supplied |
| G4 | Done | Workspace prompt now precedes the RAG folder prompt |
| G5 | Done | Non-streamed replies render via `rich.markdown.Markdown`; streamed text stays plain by design |
| G6 | Done | Edit previews render coloured diffs in tool results |
| G7 | Done | `--provider`, `--light` and existing startup flags reduce typing |
| G8 | Done | Live overlay panel: running/finished tools, plan, ETA and a context-usage gauge |
| G10 | Done | Alt+Enter / Ctrl+J multi-line input bindings |
| G11 | Done | `/expand` prints the full last tool result |
| G12 | Done | Approval prompt offers `a` (always allow for the session) beside `y/N` |
| G13 | Done | `arrow_select` menu helper with typed fallback, wired into provider, model, permission, coding-workspace and `/tasks` menus |
| G14 | Done | Replies and diagnostics separated (steps 5.2/8.2) |
| G15/G16 | Done | One shared themed `Console` (`private_agent/console.py`) with dark/light palettes; every module imports it |
| G17–G19 | Done | Bottom toolbar, `Rule` between turns, welcome screen |
| G20–G23 | Done | Readable tool requests, ✓/✗ status symbols, grouped `/help`, per-turn details |
| G24 | Done | Narrow-terminal friendly layout via `soft_wrap` and compact status lines |

Verification: `ruff check src` clean; `pytest` -> 1038 passed, 12 skipped, 0 failures.
`execute_tool_call` now tolerates an unset `SandboxManager` root by passing
`SandboxManager.root_dir` (possibly `None`) to the hook evaluators, which
already handle that case; the stale tool-panel title assertion was updated to
the new `✗`-prefixed title from G20.

## Part I — Local Provider Independence

Goal: run on any installed local provider, not only Ollama, and remember the
provider a session or task used so resuming keeps working.

### Adapter layer

`private_agent/providers/` is the new provider-neutral seam:

- `ProviderPreset` + `PROVIDER_PRESETS` (11 presets: Ollama, LM Studio,
  llama.cpp, vLLM, Jan, KoboldCpp, GPT4All, LocalAI, llamafile, MLX-LM and a
  generic OpenAI-compatible entry) carry display name, adapter type, default
  base URL, install check and start hint.
- `LocalProviderAdapter` protocol with `detect_installed()`,
  `detect_running()`, `list_models()`, `capabilities(model)`,
  `chat_model(model, options)`, `embeddings(model)` and `hardware_status()`.
- `OllamaAdapter` (native API) and `OpenAICompatibleAdapter` (`/v1`), built by
  `build_adapter(name_or_preset)`; `all_presets()` lists the catalog.
- `ProviderRecord` carries adapter type, display name, base URL, model and the
  capabilities the work needs (tools, vision, context window).

### Section I status (complete)

Branch `i-provider-independence`; `main` untouched.

| ID | Status | Where |
|---|---|---|
| I1a | Done | `rag.embedding_provider = none` degrades RAG to keyword-only retrieval instead of failing; regression tests in `tests/unit/test_rag.py` |
| I1b/I1c/I1d | Done | Capability detection for every provider (`agent/model_capabilities.py`), runtime-aware endpoint, tool-capability fallback for all providers |
| I1e | Done | `ollama` / `langchain_ollama` imports are lazy; module attributes kept so patching and the Ollama path are unchanged |
| I1f | Done | `OllamaAdapter.hardware_status()` wraps `inspect_ollama_hardware` + `format_hardware_status`; other adapters report "no standard API"; covered by `tests/unit/test_provider_adapters.py` |
| I1g | Done | `fetch_local_chat_models()` spans every reachable local runtime instead of only Ollama |
| I1h | Done | `scripts/benchmark_rag.py` is provider-neutral: per-runtime hardware reports, provider-agnostic wording, tolerant probe failures; tests in `tests/unit/test_benchmark_rag.py` |
| I1i | Done | `ollama_base_urls` / `OLLAMA_BASE_URLS` list several Ollama endpoints; each is probed, listed and matched by URL |
| I1j | Done | Installed-but-idle providers are listed with their start hint; the last chosen provider is remembered in `~/.private_agent/last_provider.json` (override `PRIVATE_AGENT_PROVIDER_STATE`) and used as the menu default |
| I1k | Done | 11 presets with default ports and start hints, so vLLM, Jan, KoboldCpp, GPT4All, LocalAI, llamafile and MLX-LM no longer need manual configuration |
| I1l | Done | Embedding-model filtering runs for every provider, not only Ollama |
| I1m | Done | Non-loopback hosts are accepted only when `models.allow_remote_hosts` is on (default off; loopback is unchanged); using one prints a `[Remote host]` warning that prompts and chunks leave the machine (`agent/model_selection.py`, `rag/indexing.py`) |
| I1n | Done | `--provider` / `--model` flags skip the menu when they match; a single reachable provider is auto-selected when `models.auto_select_single_provider` is on; `models.always_ask` forces the prompt |
| I1o | Done | A provider that stops mid-session prints `[Provider lost]` guidance, offers to retry the same turn on another reachable provider with one explicit yes (rebuilding the model and tool binding there), and records the switch; declining keeps the session usable with the original error (`agent/provider_fallback.py`) |
| I1p | Done | New `session_providers` table (`database/provider_records.py`) records adapter type, display name, base URL, model and capabilities per session; `agent/provider_resume.py` reads it back |

### Resume rules (I2)

Resuming a session now follows the Part I resume rules: the recorded provider
is matched by name or by base URL and reused without a prompt when it is still
reachable (`[Resume] Continuing with …`); when the recorded model is gone the
provider's model list is offered with the closest match first; when the
provider is not running the reason plus its start hint are printed and the
choice prompt appears; switching providers prints `[Provider changed]`,
updates the record and journals a `provider_changed` event on the active task.
Capabilities and context window travel with the record. A resumed local
session is never moved online without an explicit choice, so rule 6 holds by
construction.

### Verification

- `ruff check .` clean, `ruff format` applied.
- `pytest` -> 1120 passed, 12 skipped, 0 failures (was 1038 before Part I).
- New tests: `tests/unit/test_provider_selection.py` (12),
  `tests/unit/test_provider_adapters.py` (3),
  `tests/unit/test_provider_records.py` (6),
  `tests/unit/test_model_capabilities.py` (+4),
  `tests/unit/test_benchmark_rag.py` (+3), `tests/unit/test_provider_fallback.py` (6),
  `tests/unit/test_provider_identify.py` (6), plus resume auto-select,
  stopped-provider fallback and summary tests in
  `tests/integration/test_agent_runtime.py`.

### Deviations from the spec

- `models.auto_select_single_provider` defaults to **off** (the spec says a
  single local provider should be used silently). Turning it on by default
  changed startup for 18 existing menu integration tests and removes the
  visible confirmation line users rely on, so it is opt-in and documented in
  `config.default.json` as an addition.
- I1o now offers an in-turn retry on another reachable provider (see
  `agent/provider_fallback.py`); the original deviation (guidance only) is
  superseded. The resume-with-provider command remains the documented path
  when the user declines or no other provider is reachable.
- The module-size guard forced the resume/flag helpers out of `runtime.py` into
  `agent/provider_resume.py` (`runtime.py` stays below the 4300-line limit).

## Part J — Performance (partial)

Branch `j-k-performance-optimization`.

| ID | Status | Where |
|---|---|---|
| J2 | Partly fixed | `rag/indexing.py` imports Chroma and Ollama embeddings lazily via `_ensure_rag_dependencies()` |
| J5 | Fixed | `tools/network.py` caches validated DNS records for 60 s (bounded, rules re-checked) |
| J6 | Fixed | `tools/__init__.py` caches internet-connectivity probes for 30 s |
| J7/J8 | Partly fixed | `rag/indexing.py` keeps parsed chunks for unchanged files in `_CHUNK_CACHE` |

Verification: `ruff check .` clean; `pytest` -> 1120 passed, 12 skipped, 1 warning.
