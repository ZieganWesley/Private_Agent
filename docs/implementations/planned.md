# Code Review: Known Issues, Gaps, Plans and Fix Roadmap

Review date: 2026-10-28
Re-verified: 2026-11-19, against `main` at commit `d19f32d` (after the
B1-B3, B4-B6 and B8/B9 pull requests, #28, #29 and #30, were merged)

## How this was checked

- **Tests:** 794 passed, 12 skipped (about 19 s) in a fresh `.venv` installed
  with `uv pip install -e ".[dev]"` on Python 3.12. The failure recorded at
  the previous re-verification
  (`test_runtime_summarizes_full_context_instead_of_dropping_it`) now passes
  (step 1.3).
- **Lint:** `ruff` 0.16.10 with the repo's own `[tool.ruff]` config, which now
  exists. `ruff check .` and `ruff format --check .` both pass.
- **Status markers:** items re-checked on 2026-11-17 are labelled **Fixed:** or
  **Partly fixed:**. Unlabelled items were still true on that date unless a note
  says otherwise.
- **Line numbers** are from the review date and may have drifted; search for
  function and constant names if a line number does not match.

## Part A: Code Quality Issues

### A1. Very large modules (highest priority)

- `src/private_agent/agent/runtime.py` was 6,507 lines with
  `_run_agent_cli_session` at about 2,600 lines in one function.
- `src/private_agent/database/memory.py` is 2,926 lines in one
  `PersistentMemory` class.
- **Partly fixed:** `runtime.py` was split into focused modules that keep
  their own behaviour and tests: `tool_display.py` (186 lines),
  `model_selection.py` (290), `streaming.py` (456), `learning.py` (642),
  `memory_context.py` (700), `task_commands.py` (784), `task_lease.py` (66),
  plus the shared `console.py` and `session_state.py` singletons.
  `runtime.py` holds only the interactive session loop and tool execution.
- **Also fixed:** `_run_agent_cli_session` was split into
  `_prepare_session` (a short entry point that also owns startup cleanup)
  plus `_build_session_setup` (about 410 lines: RAG, skills, memory and
  session setup) and `_choose_provider` (about 445 lines: the provider and
  model menu plus the ready summary). They exchange the frozen
  `SessionSetup` and `ProviderSelection` containers from
  `agent/session_types.py`, and a mutable `ProviderPreference` carries the
  online choice across retries. A failed startup closes its own SQLite
  connection and cancels the lease task in `_discard_session_startup`.
  `runtime.py` is now about 3,700 lines and
  the session function about 1,880 lines; its remaining turn-loop body is
  the next candidate for extraction.
- **Also fixed:** `PersistentMemory` (2,926 lines) is now a thin facade in
  `database/memory.py` composed from per-concern mixins:
  `connection.py` (436 lines), `tasks.py` (1,197), `task_actions.py` (426),
  `learning.py` (256), `conversations.py` (165), `episodes.py` (251),
  `skills.py` (68) and `maintenance.py` (220).
- **Audit 2026-11-19:** `agent/runtime.py` is still 3,866 lines (the turn loop
  and tool execution remain in it) and `database/memory.py` is a 35-line facade
  over nine mixins. This stays open as the large-module refactor; it needs its
  own branch rather than a single change, and the ruff complexity counts below
  were measured at review time and were not re-measured.
- Ruff complexity checks flag:
  - 38 overly complex functions (`C901`)
  - 22 functions with too many branches (`PLR0912`)
  - 25 functions with too many arguments (`PLR0913`)
  - 15 functions with too many statements (`PLR0915`)

**Standard practice:** small, single-purpose modules. For example, separate
command handlers for `/tasks`, `/plan` and memory upkeep, and a repository
class for each table.

### A2. Settings loaded at import time as global constants

- `config.py` reads `~/.private_agent.conf` and builds `APP_CONFIG` as soon as
  it is imported, then exposes dozens of module-level constants. `runtime.py`
  imports about 50 of them.
- **Fixed:** the `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` environment
  variables are now set during startup instead of at import (commit `2fec457`).
- **Partly fixed (step 3.1):** the schema and all coercion logic now live in
  `src/private_agent/settings.py` as a typed, frozen `Settings` dataclass
  with one field per runtime constant, built by `Settings.from_config(...)`.
  `config.py` keeps its public surface and re-exports the same names as
  `NAME = SETTINGS.name`, so no importing module changed. Loading still
  happens at import time; step 3.2 injects the object and removes the
  aliases. Covered by `tests/unit/test_settings.py` (schema drift, coercion,
  malformed config and a timing guard).
- Mutable state is kept in module globals: `global` appears 16 times across
  `tools/__init__.py`, `tools/_task_state.py`, `tools/_sqlite_state.py`,
  `tools/media.py` and `run_logging.py`.
- **Audit 2026-11-19:** 122 `NAME = SETTINGS.field` aliases remain in
  `config.py` and 13 `global` statements remain in the package, so step 3.2
  (inject the settings object, drop the aliases) is still open. New settings
  added since then follow the same alias convention.

**Standard practice:** a typed settings object (for example, a pydantic
model), loaded once at startup and passed to the code that needs it.

### A3. Test-only logic inside production code

**Fixed.** `sys_modules_safe()` and its `"pytest" in sys.modules` check were
removed in step 2.3 (commit `aa8e9fb`), so behaviour no longer depends on the
test runner: a malformed config file always warns, and the test asserts that
warning with `pytest.warns`.

Re-verified on 2026-11-18 against `main` at `5b64e1b`: no production module
branches on the test runner. There are no `sys.modules` checks,
`PYTEST_CURRENT_TEST` reads or test-only flags under `src/private_agent/`.
A guard test
(`tests/unit/test_source_hygiene.py::test_production_sources_never_branch_on_the_test_runner`)
fails if one is reintroduced.

### A4. The test environment is not standard

**Fixed.** `pytest` installs from the `dev` extras, the pytest `pythonpath` is
`["src", "."]`, `.vscode/settings.json` enables pytest, and tests that need a
missing prerequisite (`bwrap`, `prlimit`, media libraries) now skip instead of
failing (`pytest.mark.skipif` / `pytest.importorskip`). All eight failures from
the review are gone.

Re-verified on 2026-11-18 against `main` at `5b64e1b`: `pytest -q` reports
805 passed and 12 skipped (the additional 11 tests are the A3 source-hygiene
guard). No production module branches on the test runner. The one failure
recorded at review time, `tests/integration/test_agent_runtime.py::test_runtime_summarizes_full_context_instead_of_dropping_it`,
was traced and fixed in step 1.3.

Still verified green: `ruff check .` and `ruff format --check .` both pass.
### A5. Lint, format and typing are not enforced

**Completed.** The repo now has a `[tool.ruff]` config (`E`, `F`, `W`, `I`,
`B`, `UP`, `SIM`) and a `.pre-commit-config.yaml` running `ruff-check` and
`ruff-format`. The source was reformatted and auto-fixed, so `ruff check .` and
`ruff format --check .` both pass. The import-order,
`typing`-modernisation, nested-`with`, closure (`B023`) and `subprocess`
(`PLW1510`) counts from the review no longer apply.

**Audit 2026-11-19:** lint and formatting are enforced (`ruff check .` and
`ruff format --check .`, now including `BLE`), but the typing half of this item
is still open: no `[tool.mypy]`/`[tool.pyright]` config exists and neither tool
is a declared dependency, and there is still no CI workflow (step 1.5). Adding a
type checker is a separate decision because it would introduce a dependency and
a baseline of errors to clear.

**A5 completed:** All blind `except Exception` blocks (BLE001) have been
replaced with specific exception types across all source files. The following
files were updated with `except (ValueError, OSError)` or specific exception types:

- `src/private_agent/agent/compaction.py`
- `src/private_agent/agent/learning.py`
- `src/private_agent/agent/local_runtimes.py`
- `src/private_agent/agent/memory_context.py`
- `src/private_agent/agent/prompts.py`
- `src/private_agent/agent/model_capabilities.py`
- `src/private_agent/tools/filesystem.py`
- `src/private_agent/tools/media.py`
- `src/private_agent/tools/location.py`
- `src/private_agent/agent/providers.py`
- `src/private_agent/agent/runtime.py`
- `src/private_agent/agent/status.py`
- `src/private_agent/agent/streaming.py`
- `src/private_agent/agent/learning.py`
- `src/private_agent/tools/mcp.py`
- `src/private_agent/tools/web.py`
- `src/private_agent/tools/memory.py`
- `src/private_agent/skills/loader.py`
- `src/private_agent/tools/shell.py`
- `src/private_agent/tools/network.py`
- `src/private_agent/config.py`

That work narrowed real blind catches, but it did not leave source code free
of BLE001: the audit in step 5.3 found 29 broad handlers still in `src/`, and
each one turned out to be a boundary that has to stay broad. Narrowing them is
what caused a startup crash — `ollama.RequestError` and `ResponseError` derive
directly from `Exception`, not from `ConnectionError` or `OSError`, so an
`except (ValueError, OSError)` around a provider probe let real backend errors
escape. Every one of the 29 now carries an explicit
`# noqa: BLE001 - <reason>` naming why the catch is broad, `BLE` is selected in
`[tool.ruff.lint]` (with `tests/**` opted out, since tests may assert on
arbitrary failure), and a source-hygiene test fails when a broad handler appears
without a written reason. See 5.3.

**Fixed (2026-11-19): the narrowing above missed `httpx` transport errors.**
`httpx.ConnectError` and the other `httpx.HTTPError` types are not
`ConnectionError` or `OSError` subclasses, so handlers narrowed to
`(ValueError, OSError)` no longer caught a refused connection. The visible
symptom was that `private-agent` aborted with a `httpx.ConnectError` traceback
as soon as Ollama was not running, because startup runtime discovery probed a
closed port.

- `agent/local_runtimes.py` catches `httpx.HTTPError` next to the socket
  errors, and a probe future that fails is now logged
  (`provider.local_discovery_task_failed`) instead of aborting the whole
  discovery pass. With no reachable runtime the CLI reports "No local runtimes
  found" and continues to the online configuration path.
- `agent/model_capabilities.py` already handled this for the capability probe;
  the remaining HTTP-facing tool handlers did not. `tools/web.py`
  (`web_search`, `fetch_webpage`, `download_web_file`) and `tools/location.py`
  (`fetch_current_location`, `get_weather`) now return an error string for an
  `httpx.HTTPError` instead of letting it escape the tool boundary, which
  previously reached neither their own `(ValueError, OSError)` handlers nor the
  `(ValueError, OSError, RuntimeError)` tool executor.
- `cli.py` treats end of input (`EOFError`, Ctrl+D) at a startup prompt as an
  orderly exit instead of printing a traceback.

Coverage: `tests/unit/test_providers.py` (closed port and HTTP status errors),
`tests/unit/test_location.py`, `tests/unit/test_tools.py` (web fetch and
download) and `tests/unit/test_cli.py` (closed stdin). Verified with the real
client against a closed port: before the change the call raised
`httpx.ConnectError`; afterwards discovery returns `[]` and the CLI exits 0.

- There is no `[tool.mypy]` config or other type checking.
- There is still no CI workflow under `.github/workflows/` (step 1.5).

### A6. Logging
  makes problems hard to trace afterwards.

**Completed.** Added module-level loggers (`logging.getLogger(__name__)`) and log statements across the codebase for improved diagnostics and debugging. Key additions:

- **`src/private_agent/agent/learning.py`**: Added `import logging` and module-level logger; log statements in _run_learning_maintenance and other key functions
- **`src/private_agent/agent/task_commands.py`**: Added `import logging` and logger; debug log statements in _task_age_label and _planning_search_citations
- **`src/private_agent/tools/shell.py`**: Added `import logging` and module-level logger; log statements in run_shell_command and _sandbox_shell_script; added `description` parameter to @tool decorator
- **`src/private_agent/tools/web.py`**: Added `import logging` and module-level logger; log statements in web_search and fetch_webpage functions

All log statements use the existing `run_logging` infrastructure (`RUN_LOGGER`, `log_event`) for consistent structured JSON logging with redaction of sensitive data. A7 (Async mixed with blocking code) and A8 (Smaller issues) are now also complete (see sections below).

- There is no `[tool.mypy]` config or other type checking.
- There is still no CI workflow under `.github/workflows/` (step 1.5).

### A7. Async mixed with blocking code

**Completed.** Removed the `asyncio.run`-inside-ThreadPoolExecutor anti-pattern
from sync tool functions by converting them to fully synchronous HTTP clients:

- **`src/private_agent/tools/web.py`**: `fetch_webpage` and `download_web_file`
  now use `public_only_sync_client` and a new synchronous
  `_get_limited_response_sync` helper instead of spawning a thread pool to run
  `asyncio.run`. Added `_get_limited_response_sync` to `tools/network.py`.
- **`src/private_agent/tools/location.py`**: `_get_json` now uses the sync HTTP
  client; removed `_run` and `_get_json_async`.
- **`database/memory.py`**: connection already guarded by `threading.Lock`;
  verified by `test_sqlite_concurrent_read_write_locking`.
- `subprocess.run`/`Popen` usages are confined to synchronous functions invoked
  via `asyncio.to_thread`, so they no longer block the event loop.
- Logger placement fixed (E402) in `tools/shell.py`, `agent/learning.py`,
  `agent/task_commands.py` so `ruff check .` passes.
- Re-verified on 2026-11-19: the package has one `asyncio.run` at the
  synchronous CLI boundary (`run_agent_cli`); no tool module calls it. The
  benchmark script uses `asyncio.run` only in its own `main`. All
  subprocess/HTTP calls are either synchronous or awaited via
  `asyncio.create_subprocess_exec`/`httpx.AsyncClient`. Boundary exception
  handlers restored to appropriate breadth (see A5).

### A8. Smaller issues

- **Fixed:** `agent/__init__.py` now excludes private (underscore-prefixed)
  names from the generated `__all__`, so the public API no longer leaks private
  helpers. Private names remain importable via `__getattr__` for backwards
  compatibility.
- **Fixed:** redundant unreachable `except (ValueError, OSError)` block removed
  from `config.py` (B025 duplicate-exception).
- **Fixed:** the `VISIBILE_REASONING` typo. The runtime uses
  `VISIBLE_REASONING`; the misspelling survives only as a backward-compatible
  config-key mapping in `config.py`.
- **Fixed:** `main.py` was removed, along with its hand-edited `sys.path`.
- **Fixed:** `requirements.txt` and `scripts/sync_requirements.py` were removed,
  so `pyproject.toml` is the only dependency source.
- **Fixed:** `.github/copilot-instructions.md` no longer points at the deleted
  roadmap files.
- Re-verified on 2026-11-18: `agent/__init__.py` `__all__` still excludes
  private names; config has no unreachable duplicate except; no `main.py`,
  `requirements.txt` or sync script; typo mapping preserved for compatibility.


---

## Part B: Features and Whether They Follow Standard Approaches

### What the program does

Private Agent is a privacy-focused AI agent that runs in the terminal, uses
local models by default, and works on the project folder it is launched from.

| # | Feature | Where it lives |
|---|---|---|
| 1 | **Model providers:** Ollama, LM Studio, llama.cpp, or an online OpenAI-compatible API, with detection of each model's capabilities (tools, vision, thinking) | `agent/providers.py`, `agent/local_runtimes.py` |
| 2 | **Agent loop:** plan, call tools, check results and retry, within a time budget | `agent/runtime.py` |
| 3 | **Permissions:** three session modes (manual, auto, full) and five access levels for coding tasks | `agent/permissions.py` |
| 4 | **Built-in tools:** read, edit and search files; git; patches; shell commands; web search and fetch; location and weather | `tools/` |
| 5 | **Sandboxed commands:** shell and test commands run in Bubblewrap with resource limits | `code_tasks/__init__.py` |
| 6 | **Coding tasks:** isolated project copy, checkpoints, re-running tests until they pass | `code_tasks/` |
| 7 | **Planning:** saved plans with revisions, approval, resume and leases | `tools/planner.py`, `database/memory.py` |
| 8 | **Memory:** chat history, session summaries, learned facts and context compaction in SQLite | `database/memory.py`, `agent/compaction.py` |
| 9 | **Document search (RAG):** ChromaDB vector search plus BM25 keyword search, 12 chunking strategies, citations | `rag/` |
| 10 | **Skills:** Markdown instruction files picked by relevance | `skills/` |
| 11 | **MCP:** external tool servers, local (stdio, sandboxed) or HTTP | `tools/mcp.py` |
| 12 | **Media:** images, video, webcam, microphone, Whisper transcription | `tools/media.py` |
| 13 | **Interactive terminal:** slash commands, auto-completion, `@file` attachments, session statistics | `agent/interactive.py`, `agent/session_stats.py` |

### Parts that follow standard approaches

- **Model access** uses LangChain (`ChatOllama`, `ChatOpenAI`, `bind_tools`,
  `astream` / `ainvoke`).
- **Document search** combines Chroma and BM25, the usual hybrid design.
  Indexing is incremental and tracks which files have changed.
- **MCP** uses the official `langchain-mcp-adapters` client and supports
  `stdio`, `sse` and `streamable_http`.
- **Sandboxing** uses Bubblewrap with `--unshare-all`, read-only system
  folders and `prlimit`, and refuses to run commands when Bubblewrap is
  missing.
- **Workspace boundary:** `sandbox/linux.py` resolves symlinks with `realpath`
  and checks the path is inside the workspace with `is_relative_to`.
- **Memory search** uses SQLite's built-in full-text search (FTS5).
- **Token counting** uses `tiktoken`, falling back to a character estimate.
- **Shell commands** run with `shell=False` and an argument list.

### Parts that deviate from standard approaches

#### B1. The agent loop is written by hand and duplicated

- The tool-calling loop lives inside the 2,500-line `_run_agent_cli_session`.
- `llm.bind_tools(...)` is called in about 12 places in `runtime.py`, so the
  same loop logic is copied for chat, planning, coding and skills.

**Standard practice:** one reusable loop function, or a framework such as
LangGraph (`create_react_agent`). The project already depends on LangChain.

**Fixed.** All `llm.bind_tools(tools_list)` call sites route through one
`_bind_tools` helper. The model fallback, per-turn preflight, tool-batch
execution and outer turn loop now live in `_invoke_initial_model_with_fallback`,
`_preflight_tool_calls`, `_execute_tool_batch` and `_run_agent_turn` in
`agent/runtime.py`; planning, coding and skills paths share that loop.

Verified with the full suite: 812 passed, 12 skipped, 0 failed.

#### B2. Permission checks are spread across several places

- There is a central `tool_needs_permission` in `agent/permissions.py`.
- `tools/shell.py` also asks the user for approval itself with
  `console.input`.
- `runtime.py` has its own `_tool_needs_permission`.

**Standard practice:** a single policy check that every tool call passes
through. Tools never prompt the user directly.

**Fixed.** `tools/shell.py` no longer prompts for approval itself; approval is
handled by the central `agent.permissions.tool_needs_permission` policy (with
the `_tool_invocation_approved` context). The runtime wrapper was removed, and
the interactive prompt now lives in `prompt_tool_approval` beside the policy.
Direct/unapproved invocations return an explicit error instead of prompting.
Covered by `test_shell_command_requires_explicit_approval` and
`test_permission_mode_policy_matrix`.

Residual follow-up: none identified for B2; session-level allow rules remain a
separate C8 feature.

#### B3. Tool state lives in module globals

- The active database, task, skill folder and code task are global variables
  set through `set_active_*()` functions in `tools/__init__.py`,
  `tools/_task_state.py` and `tools/_sqlite_state.py`.
- `code_tasks.ACTIVE_CODE_TASK` is read directly by `tools/shell.py`.

**Standard practice:** pass a context object into each tool, for example
through LangChain's `RunnableConfig` or injected tool arguments.

**Fixed.** `ToolContext` is now the single process-local source of truth. It
holds the active DB path, task session/id, runtime owner, skill directory,
registry and session, code task, and schema-read set. `tools/memory.py`,
`tools/planner.py`, `create_skill`, `tools/shell.py`, `agent/runtime.py`,
`agent/task_commands.py` and `agent/task_lease.py` read through
`get_tool_context()`; runtime/task-command writes go through
`set_tool_context()`.

`tools/_task_state.py` was removed. `tools/_sqlite_state.py` is reduced to the
shared `SQLITE_LOCK`; the active DB path no longer lives there. The
`set_active_*` compatibility shims were removed: callers and tests now write
directly through `get_tool_context()` / `set_tool_context()`.

`code_tasks.ACTIVE_CODE_TASK` was removed; `run_project_unit_tests`,
`checkpoint_code_task` and `finalize_code_task` now read the active code task
from `get_tool_context().code_task` and still degrade gracefully when no task
is active. New coverage lives in `tests/unit/test_tool_context.py`, including
shared-instance, copy, schema-read and code-task tool cases.

Performance note: `get_tool_context()` returns the shared context object and
performs no SQLite work, so hot tool paths add no database round trips.

Final green run: `812 passed, 12 skipped`. The two previously red
OpenAI-compatible runtime tests exposed an uncaught `httpx.ConnectError` in the
capability probe; `model_capabilities` now treats it as an unknown endpoint
(same as other transport failures), with unit coverage in
`tests/unit/test_model_capabilities.py`.

Residual follow-up: none identified for B3. `ToolContext` is the only
tool-state write path.

#### B4. The risky-command check matches text fragments

**Fixed.** `evaluate_shell_command` now parses the command with `shlex` instead
of matching substrings (commit `9726bd5`), and `tests/unit/test_shell_risk.py`
covers the cases above. The check still only adds a warning to the approval
prompt; Bubblewrap remains the real protection.

#### B5. The workspace root is decided at import time

**Fixed.** `SandboxManager.root_dir` now starts as `None`; no path is chosen at
import time. `SandboxManager.set_root()` runs once during startup with the
configured/project workspace (or the current directory for an empty prompt) and
rejects missing paths, non-directories, and empty values instead of creating
them. `SandboxManager.get_root()` is the checked accessor used by all tools and
runtime code, and fails clearly if startup never configured a root.

Coverage lives in `tests/unit/test_sandbox_root.py` for the unconfigured,
missing, non-directory, empty and valid-root paths.

#### B6. Skill matching uses keyword overlap

**Fixed.** Skill files now support optional YAML front matter:

```markdown
---
name: Python Expert
description: Write secure and robust Python code.
---
# Python Expert
...
```

`skills/loader.py` reads `name` and `description` from the front matter while
keeping the Markdown body as the system prompt. Files without front matter
still load with the previous filename/first-lines fallback, and invalid YAML
is reported and ignored rather than dropping the skill. New skills written by
`create_skill` include front matter automatically.

`skills/manager.py` accepts an optional embedder. When the RAG vector store is
active, the runtime passes its embedding model through `asyncio.to_thread`, so
skill selection uses cosine similarity without blocking the event loop. Skill
embeddings are cached per loaded skill. If the embedding backend is unavailable
or returns malformed data, selection falls back to the existing weighted
keyword matcher. `skills.relevance_embedding_threshold` controls the semantic
cutoff (default `0.55`).

Coverage lives in `tests/unit/test_skills.py` for front-matter loading, invalid
front matter, semantic selection, embedding caching, and keyword fallback. The
semantic path is unit-tested with a fake embedder; it has not been exercised
against a live Ollama embedding endpoint.

#### B7. Web search scrapes DuckDuckGo's HTML page

- `tools/network.py:593` scrapes `html.duckduckgo.com`. This breaks when the
  page layout changes and may violate DuckDuckGo's terms of service.

**Standard practice:** a search API, or a self-hosted SearXNG instance, which
also suits the privacy goal.

**Fixed.** Search now runs through a provider layer in `tools/search.py`. The
`[network] search_provider` setting accepts `auto` (default), `searxng` or
`duckduckgo`; `auto` uses a self-hosted instance when `[network] searxng_url` is
set and otherwise keeps scraping, so a default install behaves exactly as
before. `searxng_search` in `tools/network.py` queries the instance's JSON API
(`?q=...&format=json`) and formats results identically to the scraper, reusing
the existing redirect limit, response-size cap and concurrency semaphore. When
a chosen SearXNG fails for any reason (transport, HTML instead of JSON, size
cap), the failure is logged and, unless `[network] search_fallback` is
`false`, the scraper answers. The returned text then opens with a warning that
names the failed provider and the setting, so privacy is never downgraded
silently; with the fallback disabled the failure is raised as a `ValueError`
and DuckDuckGo is never contacted. `auto` without a configured instance
scrapes directly and adds no warning, because nothing was downgraded.

The instance URL is operator configuration, never model- or page-supplied, so a
loopback instance such as `127.0.0.1:8080` is reachable: `_get_limited_response_sync`
now takes an opt-in validator, and only the SearXNG call relaxes the public-only
DNS policy to scheme/credential checks. Every other caller, including
`fetch_webpage`, keeps the strict default (guarded by a test).

Coverage lives in `tests/unit/test_search.py` (23 tests): JSON parsing and
formatting, the result cap, malformed entries, unconfigured and non-HTTP
endpoints, the provider-selection matrix, unknown providers, the configuration
error for `searxng` without a URL, fallback for transport/parse/OS failures with
its notice, the setting that disables the fallback (asserting the scraper is
never called), and the `web_search` tool wiring end to end. The transport
itself was exercised against a local HTTP server serving SearXNG-shaped JSON;
no live SearXNG or internet search was available here.

#### B8. Sync and async code are mixed

**Fixed by the A7 async cleanup.** Web and location tools use synchronous
`httpx` clients and are dispatched through `asyncio.to_thread`/`ainvoke`.
The network resolver has an async child-process path, and shell/code-task
`subprocess.run` calls live only in synchronous functions. The only
`asyncio.run` in the package is the synchronous CLI boundary in
`agent/runtime.py` (`run_agent_cli`). Coverage lives in
`tests/unit/test_source_hygiene.py`: it fails if `asyncio.run` appears
elsewhere or if `subprocess.run` is called from an `async def`.

#### B9. The document retriever merges results in a basic way

**Fixed.** `rag/retrieval.py` now tokenizes the BM25 corpus and queries with a
casefolded, punctuation-aware tokenizer, so `"error,"` and `"error"` match. It
fuses vector and BM25 rankings with reciprocal rank fusion (`k=60`), dedupes by
source/chunk, and keeps only the top `k` results. BM25 candidates still use a
bounded `heapq.nsmallest`, so retrieval remains O(n log k) for corpus scoring
plus O(k log k) fusion. An all-punctuation corpus is detected before
`BM25Okapi` construction (which otherwise raises `ZeroDivisionError`) and falls
back to vector results.

Coverage lives in `tests/unit/test_rag.py` for tokenizer behavior, RRF
ordering/deduplication, the top-k candidate path, and the no-token fallback.
A synthetic 20,000-document check measured about 3.25 ms per fused query versus
3.08 ms for the old merge (~5% overhead) at top-k 4. The live
`scripts/benchmark_rag.py` run was not possible here because no Ollama endpoint
is available; the harness remains the measurement path for recall@k and MRR.

---

## Part C: Gaps Compared with Claude Code and Codex

This compares Private Agent with agentic coding tools such as Claude Code and
OpenAI Codex CLI. Each "missing" item was checked against this project's code.
The description of Claude Code and Codex is from general knowledge of those
tools, which change often, so check their current docs for exact details.

### What this project already matches

| Capability | Claude Code / Codex | This project |
|---|---|---|
| Tool loop with file read, edit and search, and shell | Yes | Yes |
| Exact-text replacement edits | Yes | Yes (`apply_workspace_patch`, with a diff preview) |
| Several tool calls run at once | Yes | Yes (`asyncio.gather` in `runtime.py`) |
| Permission modes | Yes | Yes: manual, auto, full |
| Plan mode and a to-do list | Yes | Yes: `/plan` and saved plans, with approval |
| Context compaction | Yes | Yes: `/compact` and automatic summarizing |
| MCP servers | Yes | Yes: stdio, SSE and HTTP |
| Skills | Yes | Partly (see C9) |
| Sandboxed commands | Yes | Linux only (see C4) |

### Where this project goes further

- Runs fully locally (Ollama, LM Studio, llama.cpp), and nothing leaves the
  machine by default.
- Keeps memory across sessions: summaries, learned facts, FTS5 search.
- Searches your own documents (RAG) with citations.
- Takes webcam, microphone, image and video input.
- Plans are saved with revisions, approvals and resumable leases, and coding
  requires an approved plan first.

### Missing or different features

#### High impact

**C1. No project instruction file.**
- **Status (updated):** Deferred to the end of the project, alongside multipart documentation work. Project instruction loading is already implemented (6.1); final C1 closure is documentation/status work only.
Claude Code reads `CLAUDE.md` and Codex reads `AGENTS.md` to learn a
project's rules, build commands and conventions. Nothing in this project looks
for such a file. This is probably the cheapest high-value feature to add.

**C2. No non-interactive mode.**
Claude Code has `claude -p` and Codex has `codex exec` for scripts and CI.
This project's only flag is `--verbose-startup` (`cli.py`), and it exits when
not run in an interactive terminal. There is no JSON output and no way to pass
a prompt as an argument.

- **Non-interactive done (step 6.8):** `private-agent -p "prompt"` runs a single
  prompt and exits without a terminal. `--output text|json` selects the final
  answer format, `--permission-mode` overrides the configured mode, and startup
  questions default to their Enter values. Tools that need approval return a
  denial `ToolMessage` advising `--permission-mode full` instead of prompting,
  and online consent, shell history deletion and terminal-only paths fail
  closed. The exit-time session summary is skipped to avoid an extra model call.

**C3. No session resume and no undo.**
- **Resume done (step 6.6):** `private-agent --continue` reopens the most recent
  session, `--resume <session_id>` reopens a named one and `--list-sessions`
  shows the ids, all with the saved message history. A resumed session now
  skips the resume prompt, is not summarized twice, keeps its own episode out
  of the startup memory context, and reclaims its active plan lease when the
  earlier process is gone. Provider and model are still not recorded, so a
  resumed conversation runs on whatever the current startup selects (12.9).
- **Undo done (step 6.7):** `/undo` restores the workspace edits of the last
  agent turn and rewinds that turn's saved conversation messages. Snapshots
  now carry a session/turn manifest (`.agent_snapshots/index.jsonl`), update
  messages in `chat_history` record their `turn_id`, and each file is restored
  atomically and checksum-verified, so a failed restore never leaves half a
  file. Files the turn created are removed again.
- Limits: only workspace edits made through the file tools are recorded (shell
  commands, MCP tools, edits made before this step and paths outside the active
  workspace have no manifest entry; their old `.bak` files stay harmless).
  `/undo` steps back one turn at a time, and when a file cannot be restored the
  conversation is kept and the failure is reported so the undo can be retried.
  Provider and model recording is still step 12.9.
- Claude Code can rewind code and conversation to a checkpoint. Here, edits
  are copied to `.agent_snapshots/` and `/undo` restores them. Git checkpoints
  exist only inside coding tasks.

- **Status (updated):** Done in 6.3: compound shell syntax, per-call timeout, background process tools, the sandbox status indicator, CLI wiring of `background_process_indicator()`, and persistent working-directory state between shell calls are implemented and tested.
- The default timeout is 30 seconds (`config.py:698`), too short for many
  builds and test runs.
- **Fixed:** commands with pipes, `&&`, `||`, redirects and `cd` now run
  through `/bin/bash -c` inside the sandbox (commit `3296c63`).
- It still cannot run background processes such as dev servers or watchers, the
  timeout cannot be set per call, and the shell does not keep state between
  calls.

#### Medium impact

**C6. No subagents.**
- **Status (updated):** Subagent support exists (6.13); broader routing decisions are deferred until C1 documentation work is complete.
Claude Code can hand searches or reviews to separate agents with their own
context. There is nothing like that here.

**C7. No hooks.**
- **Status (updated):** Done in 6.11: configurable PreToolUse, PostToolUse, and SessionStart hooks with matchers, timeouts, and blocking decisions.
Claude Code lets users run their own scripts before or after tool calls, for
example to format code after every edit. There is no hook system here; the
only "hook" mention in the code turns git hooks off
(`code_tasks/checkpoints.py:88`).

**C8. No fine-grained permission rules.**
- **Status (updated):** Done in 6.5: allow/deny permission rules with deny precedence.
Claude Code allows rules like "always allow `npm test`" and "never read
`.env`". Here there are only the three modes plus an MCP allow-list, so Auto
mode asks before every command, including harmless repeated ones.

**C9. Basic skills and no custom commands.**
- **Status (updated):** Done in 5.6 and 6.10: skill front matter/embeddings with keyword fallback and Markdown custom slash commands.
- Skills have no front matter, are matched by keyword overlap and are loaded
  whole (see B6).
- Users cannot add their own slash commands as Markdown files.

**C10. Reading large files.**
- **Status (updated):** Done in 6.2: offset/limit paging with total line count.
`read_local_file` takes only a path, with no offset or line limit
(`tools/schemas.py:15`). Large files are cut off at the byte limit instead of
being read in pages.

**C11. No git write workflow.**
- **Status (updated):** Done in 6.12: branch, stage, commit, and optional pull request tools behind permission checks.
Git access is read-only (`inspect_workspace_git`), and manual git commands are
blocked inside coding tasks. The agent cannot create branches, commits or pull
requests for you, which both other tools do.

**C12. Interrupting a turn.**
- **Status (updated):** Done in 6.4: turn cancellation keeps the session active.
Ctrl+C triggers the summarize-and-exit flow. Claude Code uses Esc to stop the
current action and keep the session, and Esc twice to rewind. This project
does not seem to have a clean way to stop one turn; there are only 3
interrupt handlers in `runtime.py`.

#### Lower impact

**C13. Fewer configuration levels.**
- **Status (updated):** Done in 6.9: user, project, and local settings merge over the base config.
There is one user config file (`~/.private_agent.conf`), not user, project and
local settings files that can be committed alongside the code.

**C14. No status line or output styles.**
- **Status (updated):** Done in 6.15: `status_line()` helper plus CLI rendering of model, context use, and permission mode.
These are cosmetic but help users see the model, context use and permission
mode at a glance.

---

## Part D: Planning Mode

### How it works today

1. **`/plan`** applies planning mode to the **next message only**
   (`planning_mode_armed` in `agent/runtime.py`).
2. For that message the model gets just 5 tools: `read_table_schema`,
   `create_task_plan`, `inspect_task_plan`, `revise_task_plan` and
   `web_search` (`runtime.py:4892`).
3. The model must call `read_table_schema` first, or saving the plan is
   refused (`tools/planner.py:129`). Only one planner tool call is allowed per
   model round (`runtime.py:5686`).
4. The plan is saved in SQLite as `awaiting_approval`. Each plan has a goal, a
   type (research, coding or other), up to 20 steps, assumptions and
   constraints. Each step has dependencies, validation, proof, risks and edge
   cases.
5. The user opens **`/tasks`** and runs `approve <ID>`. For coding plans they
   also pick an access level and a project folder, then type
   `APPROVE <id>`. The plan's goal is then sent as the next request.
6. While the plan runs, the model reports each step with `update_todo_step`.
   The user then confirms each one with `verify <ID> <STEP> <evidence>`,
   followed by `validate` and `complete`.
7. Coding requests are refused unless an approved coding plan exists. A regex
   (`code_tasks/__init__.py:20`) decides what counts as a coding request.

### Compared with Claude Code's plan mode

The Claude Code column is from general knowledge of the tool, which changes
often, so check its current docs for exact behaviour.

| Aspect | Claude Code | This project |
|---|---|---|
| Entering | Shift+Tab; the mode stays on until you leave it | `/plan` covers one message only |
| Looking at code while planning | Read-only file reading, searching, web fetch, exploration subagents | Cannot read files at all; only a web search that returns titles and links |
| Clarifying questions | The model can ask questions before planning | One shot, with no back-and-forth |
| Plan format | Readable Markdown | JSON with a fixed schema (`tools/planner.py:28` prints it with `json.dumps`) |
| Showing the plan | Shown straight away with an approve prompt | Saved first; the user opens `/tasks` to approve it |
| Approval | One choice: approve with edits allowed, approve with manual edit approval, or keep planning | Type `APPROVE <id>`, pick an access level and folder, take a lease |
| Rejecting with feedback | "Keep planning" plus comments | Approval is declined; start again with `/plan` and a new message |
| Editing the plan yourself | Edit the plan text | Not possible; only the model can revise, and only in `/plan` mode |
| Progress tracking | A to-do list the model updates | SQLite steps, plus manual verification by the user |
| Keeping plans | Mostly within the session | Saved with revisions, history log, leases and resume |
| Audit trail | Little | Versioned plans, approvals tied to exact content, event history |

### What this project does better

- **Approval is tied to an exact plan version.** Each revision has a digest; a
  new revision cancels the old approval, and approving checks both the
  revision number and the digest (`approve_task_plan` in
  `database/memory.py`).
- **Step dependencies are checked.** Unknown dependencies, self-references and
  cycles are rejected when a plan is saved, and a step cannot start until the
  steps it depends on are done (`database/memory.py:820` and `:1631`).
- **The model's word is not treated as proof.** "Reported done" and "verified"
  are separate states, and a plan can only be completed when every step is
  verified or skipped.
- **Citations come from real searches.** Research links are added by the
  runtime from actual search results; the model cannot add its own.
- **Plans survive restarts.** Leases with a heartbeat stop two processes from
  running the same plan, and interrupted plans can be resumed.

### Gaps and issues that hurt effectiveness

#### Critical

**D1. The planner cannot see the code.**
In planning mode the model has no `read_local_file`, `search_workspace_text`
or `list_directory`. Coding plans are written without opening a single project
file, so they rest on guesses about structure, names and conventions. This is
the biggest reason plans will be poor. Read-only tools carry no risk, so the
restriction protects nothing.

**D2. The coding-plan gate gets requests wrong in both directions.**
Results of running `is_code_task_request` on sample messages:

| Request | Detected as coding? | Effect |
|---|---|---|
| "what is the best way to create a class in python?" | True | A plain question is blocked until a plan is approved |
| "How would I fix this bug in my code?" | True | Same |
| "Can you add a comment to README file" | True | A one-line edit needs a full plan |
| "write tests for parser.py" | False | Edits go ahead without a plan |
| "Update the parser to handle tabs" | False | Same |
| "rename utils.py to helpers.py" | False | Same |

So "only coding requires an approved plan" does not hold in practice. The
missed cases still go through the normal permission prompts, so this is not a
security hole, but the gate cannot be relied on.

**D3. Planning lasts for one message.**
The model cannot ask clarifying questions, the user cannot push back on the
result, and the plan cannot be refined in conversation. Anything unclear in
the request ends up as a guessed assumption in the plan.

#### High

**D4. Too many steps for small local models.**
- The model must call `read_table_schema` before it may save a plan, and it
  may make only one planner call per round, so saving one plan takes at least
  2–3 model rounds.
- The step schema is large: up to 20 steps, each with 5 lists.
- Small local models (the project's main target) often produce malformed
  nested tool arguments.
- The schema-reading step exposes database internals to the model; the tool's
  own Pydantic schema already enforces the structure.

**D5. Most of the plan is dropped when it runs.**
The approved plan is injected into the prompt as one line holding only each
step's ID, status and description (`runtime.py:5400`). The validation
criteria, risks, edge cases, dependencies, assumptions and constraints the
model had to write are left out.

**D6. Validation criteria are never checked automatically.**
Nothing runs or checks each step's `validation` / `proof` text; the user types
evidence by hand with `verify <ID> <STEP> <evidence>`. For coding plans the
criteria could be turned into test commands and checked.

**D7. Approving and finishing take too many manual steps.**
- After `/plan`: open `/tasks`, find the ID, run `approve <ID>`, pick an
  access level and folder, and type `APPROVE xxxxxxxx`.
- After the work: run `verify` for each step with your own evidence, then
  `validate`, then `complete`.
- This much effort pushes people to skip planning, which then hits the D2
  gate.

**D8. Rejecting a plan loses the work so far.**
Declining an approval tells the user to "submit the updated request with
/plan", and the earlier planning conversation is gone. There is no way to say
"change step 3", and the user cannot edit the plan directly.

#### Medium

**D9. Research during planning is shallow.**
`web_search` returns titles and links, and `fetch_webpage` is not allowed in
planning mode, so plans cite pages the model never read. Each search also
needs its own manual approval, even in Full mode (`runtime.py:5815`).

**D10. The hint for blocked coding requests is misleading.**
The message says to submit "this exact coding request" and then "repeat the
request" (`runtime.py:4960`). But `_approved_coding_task_scope` compares
against the goal the model wrote, ignoring case, so repeating the original
wording usually will not match. Starting the plan through `/tasks approve`
works because it sends the stored goal.

**D11. Only one plan can be active.**
`ACTIVE_TASK_ID` is a single global variable (see B3).

**D12. A whole plan probably runs within one time budget.**
The approved goal is sent as a single request, which appears to share the
600-second `max_task_seconds`. Long plans will likely hit the limit and need a
manual `resume` (not confirmed).

**D13. Plans are shown as raw JSON.**
Plans are shown to the user and to the model as JSON, which is harder to read
than Markdown.

**D14. Planning needs a tool-capable model.**
It stops with an error otherwise. There is no fallback, such as a text plan
that the runtime turns into structure.

---

## Part E: Coverage by Role

The project aims to be five tools at once. This part checks how well it fills
each role and what is missing for each. The ratings are a judgement from
reading the code, not measurements, and the app was not run with a model.

### Summary

| Role | Rating | Strongest part | Biggest gap |
|---|---|---|---|
| Chatbot | Good, about 70% | Local models, streaming, memory across sessions | Plain-text output, no regenerate or edit, one conversation at a time |
| Chatbot with RAG | Good on indexing, partial on answering, about 60% | Hybrid search, 12 chunking strategies, incremental indexing, citations, benchmark script | Only 2 chunks per request, no relevance cut-off, follow-up questions search on the raw text |
| Coding agent | Partial, about 50% | Sandbox, isolated copies, checkpoints, tests re-run until they pass | Planner cannot read code (D1), regex gate (D2), weak shell (C5), Linux only (C4) |
| Research agent | Weak, about 35% | Network consent, citations only from real results, offline handling | 3 search results, DuckDuckGo scraping, no multi-step research loop, no report output |
| Everyday helper | Weak, about 30% | Weather, location, date and time, webcam, microphone, transcription | No calendar, email, notes, reminders or document creation; command execution is Linux only |

### Chatbot

**What it does:** Ollama, LM Studio, llama.cpp or an online API; streaming;
optional thinking with an effort setting; chat history, session summaries and
learned facts in SQLite, with relevant past sessions brought back into the
prompt; `/compact`, `/summarize`, and prompt history with Ctrl+R search.

**Missing:**

- **E1. No Markdown rendering.** Replies are printed as raw text
  (`runtime.py:2676`), and `rich.markdown` is not used anywhere. Code blocks,
  tables and lists show as raw `**` and `|` characters.
- **E2. No regenerate or edit.** The user cannot retry the last answer or edit
  a sent message.
- **E3. One conversation at a time.** No new chat, thread switching or named
  conversations.
- **E4. No export.** Conversations cannot be saved as Markdown.
- **E5. No personas.** The system prompt is set once in config and cannot be
  switched per session.

Covered elsewhere: no non-interactive mode (C2), no session resume (C3).

### Chatbot with RAG

**What it does:** indexes PDF, text, Markdown, code, JSON, CSV, HTML and
SQLite files; follows symlinks safely; re-indexes only changed files; enforces
size limits; 12 chunking strategies including semantic; Chroma plus BM25
hybrid search; source citations; `/maintenance` reindex and reset; a benchmark
script measuring recall@k, MRR and latency.

**Missing or weak:**

- **E6. Too little context per answer.** `rag_context_results` defaults to 2
  chunks (`config.py:77`), about 2,400 characters with the default chunk size.
  Answers that need several sections of a document will be incomplete.
- **E7. No relevance cut-off.** Every request pulls 2 chunks whether or not
  they are relevant (`runtime.py:5290`). "Hello" still gets 2 random chunks,
  which wastes context and can mislead small models.
- **E8. Follow-up questions break retrieval.** The search uses the raw current
  message, so "What about the second one?" searches for exactly those words.
  The question is not rewritten using the conversation.
- **E9. The model cannot search on its own.** There is no `search_documents`
  tool, so the model cannot search again with better words or look up a
  second topic. Retrieval happens once, before the model sees the request.
- **E10. No re-ranking,** and the result merging is basic (see B9).
- **E11. Missing file formats.** No `.docx`, `.xlsx`, `.pptx`, `.epub`, or OCR
  for scanned PDFs; `.html` is read as raw markup.
- **E12. No "answer only from my documents" mode,** and no warning when the
  documents do not contain the answer.
- **E13. One collection only.** No way to pick a document set per question or
  per project, apart from the project-folder override at startup.
- **E14. Embeddings always use Ollama.** RAG does not work without Ollama,
  even with LM Studio or an online model.
- **E15. A hidden extra search for coding requests.** A second query for
  "coding standards" with fixed wording runs on coding requests
  (`runtime.py:647`), which may pull in irrelevant documents.

### Coding agent

**What it does:** reads, searches, patches, renames and deletes files with a
snapshot before each change; symbol search; read-only git inspection;
Bubblewrap-sandboxed commands; isolated project copies; git checkpoints; tests
re-run until they pass; approved plans required.

**Missing:**

- **E16. No language tooling.** No LSP, type checking or "go to definition"
  beyond the regex-based symbol search.
- **E17. No automatic test or lint after edits.** Validation only happens
  inside formal coding tasks, not after normal chat edits.

Covered elsewhere: planner cannot read code (D1), regex gate (D2), no project
instruction file (C1), weak shell (C5), Linux only (C4), no git commits or PRs
(C11), no undo (C3), no paged file reading (C10).

### Research agent

**What it does:** `web_search`, `fetch_webpage` and `download_web_file`;
consent before each network request; connection check first; private-address
(SSRF) protection; citations only from real search results; honest offline
fallback; research plans record which sources are unverified.

**Missing or weak:**

- **E18. Very few results.** `max_search_results` defaults to 3
  (`config.py:132`), scraped from DuckDuckGo's HTML page (B7). Too few and too
  fragile for real research.
- **E19. No multi-step research loop.** Nothing drives a cycle of searching,
  reading several pages, finding gaps, searching again and writing up. It
  depends on the model choosing to call tools, which small local models rarely
  do well.
- **E20. Shallow page reading.** `fetch_webpage` cuts pages at 12,000
  characters (`tools/web.py:51`) with no paging, so long articles lose their
  second half.
- **E21. Downloaded files are not read.** A file from `download_web_file` is
  not added to RAG or summarized.
- **E22. No report output.** No Markdown report with a sources list, and no
  way to save research findings.
- **E23. No source quality checks.** No dates, domain checks or
  cross-checking between sources.
- **E24. No academic or specialist sources.** No arXiv, Semantic Scholar,
  Wikipedia API, GitHub search or documentation sites.
- **E25. Consent on every request** slows multi-step research to a crawl,
  including in Full mode during planning (D9).

Covered elsewhere: planning cannot read pages (D9).

### Everyday helper

**What it does:** date and time; weather (up to 16 days ahead or historical);
location from IP; webcam, images, video, microphone recording and local
Whisper transcription; workspace file operations; learned facts about the
user.

**Missing:**

- **E26. No personal tools.** No calendar, reminders, timers, email, notes or
  contacts. MCP servers could fill this, but none are bundled or documented as
  examples.
- **E27. No document creation.** No tool to write `.docx`, `.xlsx` or PDF, and
  no conversions such as Markdown to PDF.
- **E28. No system or desktop tools.** No clipboard, opening apps or URLs,
  notifications or screenshots.
- **E29. No calculation tool.** No calculator or Python tool, and the shell
  tool is unavailable outside Linux.
- **E30. No scheduling.** No scheduled or recurring tasks.
- **E31. No voice conversation.** Recorded clips can be transcribed, but there
  is no voice mode (no wake word, no spoken replies).
- **E32. Everything is tied to the workspace folder.** File tools cannot reach
  `~/Documents` or `~/Downloads` unless the agent is started in a parent
  folder.

### Gaps that affect all roles

- **E33. One undivided agent.** Every role shares the same loop, tool set and
  system prompt. Every request carries every tool definition, which costs
  context and confuses small models. The fix is not separate modes but one
  runtime that sends only the tools each request needs (see Part F).
- **E34. Mixed output.** User-facing text and diagnostic messages
  (`[RAG State]`, `[Processing]`) are printed together, which clutters the
  chat (see A6).

---

## Part F: Plan for a Single Combined Runtime with Live Status


Goal: the chatbot, RAG chatbot, coding agent, research agent and helper become
**one runtime**, not separate modes the user switches between. Any request
can use any mix of these abilities in the same conversation, for example
"research the latest pytest-asyncio changes, then update our tests to match".
While the agent works, a **status overlay** shows which tools are running, and
agentic tasks get a **timeline with an estimated finish time**.

This replaces the "separate role profiles" idea in E33 and step 8.1.

### F1. Current state

- **The parts are already in one process,** but they are joined by flags and
  special cases inside `_run_agent_cli_session`: `planning_only`,
  `code_task_requested`, `include_private_context`, `active_code_workspace`,
  and others. Each combination has its own copy of the loop (B1).
- **Every request gets every tool,** except in planning mode, which gets 5
  (E33).
- **Abilities cannot be combined in one request.** Planning cannot read code
  (D1), research cannot write code in the same task, and coding needs a
  separate approved plan first (D2).
- **Status display is minimal.** There are 5 `console.status(...)` spinners
  in `runtime.py`, and `_display_tool_request` prints one line per tool call.
  There is no `rich.live`, no `rich.progress`, and no `prompt_toolkit` bottom
  toolbar, so there is no persistent view of what is running.
- **There is no time estimate.** The only timing shown is elapsed time after
  slow steps, and the remaining `max_task_seconds` budget is not shown.

### F2. Target architecture

```
                         ┌──────────────────────────────┐
  user message ────────▶ │ Turn router                  │
                         │ commands → rules → model     │
                         └──────────────┬───────────────┘
                                        │ chosen capability packs
                                        ▼
┌─────────────────┐      ┌──────────────────────────────┐      ┌─────────────────┐
│ Context manager │◀────▶│ Agent loop (one, shared)     │◀────▶│ PermissionPolicy│
│ history, memory,│      │ model call → tool calls →    │      │ (one, shared)   │
│ RAG, files,     │      │ results → repeat             │      └─────────────────┘
│ research notes  │      └──────┬───────────────┬───────┘
└─────────────────┘             │               │
                                ▼               ▼
                  ┌──────────────────┐   ┌──────────────────────┐
                  │ Task orchestrator│   │ Tool executor        │
                  │ plan → approve → │   │ (uses ToolContext)   │
                  │ steps → verify   │   └──────────┬───────────┘
                  └────────┬─────────┘              │
                           │      events            │
                           ▼                        ▼
                  ┌──────────────────────────────────────────┐
                  │ Event bus                                │
                  └───┬──────────┬───────────┬───────────┬───┘
                      ▼          ▼           ▼           ▼
                Status overlay  Run log  Session stats  Hooks / JSON output
```

#### Capability packs

Each ability becomes a **capability pack**: a bundle of tools, a system-prompt
section, limits, and optional checks. Packs are added together for a turn
instead of being separate modes.

| Pack | Tools | Prompt section | Limits and checks |
|---|---|---|---|
| Core (always on) | Date and time, read-only file reading and search, `search_documents`, `ask_user`, `enable_capability` | Base rules, privacy, permission mode | — |
| Knowledge (RAG) | `search_documents`, collection selection | Cite sources; say when documents do not contain the answer | Relevance cut-off (8.3) |
| Research | `web_search`, `fetch_webpage`, `download_web_file`, report writer | Search, read, check gaps, cite only real sources | Consent once per research task; source limits |
| Coding | Edit, patch, rename, delete, shell, tests, git, checkpoints | Project instructions (`AGENTS.md`), test after edits | Approved plan required for edits; tests as verification |
| Helper | Weather, location, media, MCP personal tools, documents | Device and privacy rules | Per-use approval for media |
| Planning | `create_task_plan`, `revise_task_plan`, `update_todo_step` | How to write a plan | Plan approval |
| MCP | Configured MCP tools, grouped by server | Server descriptions | MCP allow-list |

Only the packs a turn needs are sent to the model. Plain chat sends only the
Core pack, which saves context and helps small models (fixes E33 without
splitting the runtime).

#### Turn router

The router picks the packs for each turn, in this order:

1. **Explicit commands** win: `/code`, `/research`, `/plan`, `/ask` (chat
   only). These become shortcuts that force packs, not separate runtimes.
2. **Rules:** `@file` or `/add` attachments add Coding read tools; a URL adds
   Research; an approved active plan adds the packs its steps need; an image
   adds Helper media tools.
3. **Model classification** for everything else: one short, cheap call (or a
   smaller configured model) returns a list of packs and a complexity level
   (`simple`, `multi_step`, `long`). It must return valid JSON, and falls back
   to Core plus Knowledge if it fails.
4. **Escalation during a turn:** the model can call `enable_capability` to add
   a pack it needs, for example when chat turns into "now fix it". The
   `PermissionPolicy` decides whether that needs the user's approval
   (switching to Coding or Research always asks in Manual and Auto modes).

The chosen packs are shown in the status overlay, so routing is never hidden.

#### Task orchestrator

When the router reports `multi_step` or `long`, or the user asks for a plan,
the orchestrator runs the task as a plan instead of one long turn:

1. **Plan** with read-only tools from every chosen pack (fixes D1). The model
   may ask clarifying questions (D3).
2. **Show the plan** as Markdown with one approval choice (D7, D13).
3. **Run each step** with its own packs, context and time budget (D12). A
   research step can feed a coding step.
4. **Verify** each step automatically where possible: tests for coding,
   sources checked for research (D6).
5. **Report** a summary, changed files, sources and anything left unverified.

Simple requests skip the orchestrator and run as one turn, as today. The
existing SQLite plan storage, revisions, digests and leases stay as they are.

#### Context manager

One component builds the prompt from all sources with a token budget per
source, so no single source crowds out the others:

| Source | Example share of the budget |
|---|---|
| System prompt and pack sections | 15% |
| Recent conversation | 35% |
| Active plan and step | 10% |
| RAG chunks | 15% |
| Memory (episodes, learned facts) | 10% |
| Attached files and research notes | 15% |

Unused shares go to recent conversation. The shares are configurable, and the
overlay shows total context use.

#### Event bus

Every component emits typed events instead of printing directly:

| Event | Fields |
|---|---|
| `TurnStarted` / `TurnFinished` | Turn ID, packs, complexity, duration, tokens |
| `ModelCallStarted` / `ModelCallFinished` | Model, input tokens, output tokens, tokens per second |
| `ToolStarted` | Call ID, tool name, pack, short redacted argument summary, start time |
| `ToolProgress` | Call ID, message (for example "downloaded 2.1 MB", "12 tests passed") |
| `ToolFinished` | Call ID, status (ok, error, denied, timed out), duration, output size |
| `ApprovalRequested` / `ApprovalResolved` | Call ID, kind, answer |
| `PlanStepChanged` | Task ID, step ID, old and new status |
| `EstimateUpdated` | Task ID, per-step and total estimate ranges |
| `BudgetWarning` | Time or token budget nearly used |

The status overlay, run log (A6), session statistics, hooks (6.11) and the
JSON output of non-interactive mode (6.8) all read from this bus. This also
removes most of the scattered `console.print` calls (A6, E34).

### F3. Status overlay for tools in use

A live panel pinned to the bottom of the terminal while the agent works.

```
 You: research the pytest-asyncio 1.x changes and update our async tests
 ─────────────────────────────────────────────────────────────────────────
 Agent: Found the 1.0 changelog. The main change is that the event_loop
 fixture was removed, so ...                          (streamed reply text)

╭─ Working · Research + Coding · step 2/4 ───────────────── 01:42 ───────╮
│ ⠹ fetch_webpage      docs.pytest.org/…/changelog.html          0:04     │
│ ⠹ search_workspace   "event_loop" in tests/                    0:01     │
│ ✓ web_search         "pytest-asyncio 1.0 changes"              2.3s     │
│ ✓ read_local_file    tests/conftest.py                         0.1s     │
│ ─────────────────────────────────────────────────────────────────────── │
│ Model qwen3:8b · 14 tok/s · context 7.9k / 12k (66%) · mode Auto        │
╰ Esc stop turn · Ctrl+T timeline · Ctrl+O tool details ──────────────────╯
```

**What it shows:**

- **Phase:** Thinking, Calling tools, Waiting for approval, Verifying, or
  Writing reply.
- **Active packs and current plan step,** if any.
- **Running tools** with a spinner, a short redacted argument summary and
  elapsed time. Tools running at the same time (the existing `asyncio.gather`)
  each get their own line.
- **The last few finished tools** with a status mark (✓ ok, ✗ error, ⊘ denied,
  ⏱ timed out) and duration. Older ones fold into "+12 earlier".
- **Footer:** model, generation speed (already measured by
  `_display_generation_rate`), context use, permission mode, turn elapsed
  time.
- **Key hints:** Esc to stop the turn (6.4), Ctrl+T for the timeline, Ctrl+O
  for full tool details.

**Behaviour:**

- **During a turn:** a `rich.live.Live` region. Streamed reply text renders
  above the panel (with Markdown, 8.2) and the panel stays at the bottom.
- **Between turns:** a one-line `prompt_toolkit` bottom toolbar with model,
  context use, mode and the active plan's progress.
- **Approval prompts** pause the live region, show the prompt, then resume, so
  input and live updates never fight over the terminal.
- **Errors** stay visible in the panel until the turn ends, then go into the
  normal output.
- **Plain mode** (setting, `NO_COLOR`, or a terminal that cannot redraw) falls
  back to one line per event, as today.
- **Non-interactive mode** writes the same events as JSON lines (6.8).
- **Privacy:** argument summaries use the existing redaction
  (`_redact_tool_arguments`).

### F4. Estimated timeline overlay for agentic tasks

For planned tasks, Ctrl+T expands the overlay into a timeline. It opens
automatically when a plan starts.

```
╭─ Task: Update async tests for pytest-asyncio 1.x ────── 04:12 elapsed ─╮
│                                                                         │
│ ✓ 1  Research changelog and migration guide      1:38   (est. 1–3 min)  │
│ ▶ 2  Find affected tests and fixtures             2:34   (est. 2–4 min)  │
│ ○ 3  Update fixtures and tests                    —      (est. 4–8 min)  │
│ ○ 4  Run tests and fix failures                   —      (est. 3–10 min) │
│                                                                         │
│ ████████████░░░░░░░░░░░░░░░░░░  ~35%                                    │
│ Estimated finish: 10–20 min from now (around 14:35–14:45)               │
│ Time budget: 25 min left of 30 · confidence: medium                     │
╰ Ctrl+T collapse ────────────────────────────────────────────────────────╯
```

#### How the estimate is calculated

LLMs are poor at estimating their own task time, so the model's guess is only
one input, and the estimate improves as the task runs.

1. **Starting estimate per step,** from three sources, weighted by how much
   data exists:
   - **History:** the median duration of similar past steps, grouped by pack,
     main tools and model, from a new step-timing table (below). This becomes
     the main source once there are about 5 matching past steps.
   - **Model speed:** expected output tokens × the measured tokens per second
     for the current model, plus the average duration of the tools the step
     is likely to use.
   - **Model's guess:** an optional `estimated_minutes` field on each plan
     step. It gets low weight, and is used mainly when there is no history.
2. **Recalibration while running:** after each step, compare actual and
   estimated time, and scale the remaining steps by a smoothed ratio. Running
   10 minutes slow on step 1 makes later estimates longer.
3. **Ranges, not single numbers:** always show a range (for example 4–8 min)
   and a confidence label (low, medium, high) based on how much history was
   used. With no data at all, show "estimating…" until the first step ends.
4. **Waiting time is separate:** time spent waiting for the user's approval is
   left out of the estimate and shown as "waiting for you".
5. **Budget warnings:** if the estimated finish goes past the remaining
   `max_task_seconds` budget, the overlay warns early and offers to extend the
   budget, pause after the current step, or simplify the plan.

#### Data needed

A new SQLite table, `agent_step_timings`:

| Column | Purpose |
|---|---|
| `task_id`, `step_id`, `revision` | Which step |
| `packs`, `tools_used` | What kind of work it was |
| `model`, `provider` | Speed differs by model |
| `estimated_low_s`, `estimated_high_s` | What was predicted |
| `started_at`, `finished_at`, `waiting_s` | What actually happened |
| `input_tokens`, `output_tokens`, `tool_calls` | Size of the work |
| `outcome` | Done, failed, skipped |

The existing `agent_task_events` table already stores step status changes with
timestamps, so past tasks can partly fill this table when it is created.

#### Non-plan agentic turns

Simple turns that still call several tools show a lighter view: tool rounds
so far, elapsed time and the remaining time budget, without a per-step
estimate.

### F5. Risks and decisions

- **Router mistakes:** a wrong pack choice can hide a needed tool. Mitigations:
  the Core pack always has read-only tools, `enable_capability` allows
  escalation, the chosen packs are shown, and commands can force packs.
- **Extra model call:** classification adds latency. Use rules first, cache
  classifications for similar follow-up messages, and allow a smaller
  classifier model.
- **Terminal handling:** `rich.live` and `prompt_toolkit` must not draw at the
  same time. Keep a strict rule: `Live` only during turns, toolbar only while
  waiting for input, and pause `Live` for every prompt. A full-screen TUI (for
  example Textual) is a possible later step, not needed now.
- **Estimate trust:** wrong estimates erode trust quickly. Always show ranges
  and confidence, and let the user turn the timeline off.
- **Safety stays the same:** combining abilities must not weaken the existing
  rules. Coding edits still need an approved plan, network calls still need
  consent, and media still needs per-use approval. All of these go through the
  single `PermissionPolicy`.

---

## Part G: User Interface

The UI is a prompt-driven terminal app built with `rich` and `prompt_toolkit`.
The startup screens below come from a real run in a pseudo-terminal with a
temporary home folder. No model was reachable, so the conversation screens
are taken from the code.

### What it looks like

**Startup (captured):**

```
Private Agent
No local runtimes found to proceed with local inference. Online configuration is available.
RAG documents directory (Enter to use configured path '/…/.private_agent/rag', or enter
a project-specific path):
[Warning] Provided KB path '/…/.private_agent/rag' is not a valid directory. Skipping
knowledge base initialization.
⠋ Indexing local knowledge base...
Enter workspace root path for security sandbox (Press Enter for current dir):
[Agent Permission Mode]
  1. Manual — ask before every tool action
  2. Auto — read/search freely; ask before edits and command execution
  3. Full — approve all tool calls for this session
Choose permission mode [1/2/3] (default: auto): 2

Model Provider
  1. Online — enter details manually
Choose provider [1-1] (Default: 1):
OpenAI-compatible API base URL :
Online mode will send requests to api.openai.com. Continue? [y/N]:

Model Provider
  1. Online — enter details manually
Choose provider [1-1] (Default: 1):
```

**During a conversation (from the code):**

```
--- Local Ready! Type 'exit' or 'switch' ---
User: explain tests/conftest.py                 ← bold blue, single line
[Processing] Processing model response...
[RAG State] Retrieved 2 document/episodic chunks.
[Tool Request] read_local_file {'file_path': 'tests/conftest.py'}
╭─ Tool output: read_local_file ─────────────╮ ← blue panel, first 500 chars only
│ import pathlib ...                          │
╰─────────────────────────────────────────────╯
AI (qwen3:8b): This file sets up **pytest** fixtures ...   ← raw text, Markdown not rendered
```

At exit, a **Session Statistics** table is printed, which looks good.

**In numbers:**

| Element | Count in code |
|---|---|
| `console.input` prompts | 59 |
| `[y/N]` confirmations | 12 |
| Typed confirmations | `APPROVE <id>`, `DELETE <id>`, `TAKEOVER <id>`, `RESET` |
| Panels / tables | 10 / 9 |
| `Markdown`, `Syntax`, `Live`, `Progress`, `Rule` | 0 |
| Separate `Console()` objects | 11 |
| Different `[Label]` message prefixes | about 57 |
| Colour tags | yellow 166, cyan 75, red 60, green 35 |

### Strengths worth keeping

- Command and `@file` completion while typing, with "directory" and
  "workspace file" labels, and Ctrl+R history search.
- Clean tables for `/status`, the startup Ready summary and Session
  Statistics.
- Typed confirmations for destructive actions (`DELETE <id>`, `RESET`).
- Redacted tool arguments, and plain-text fallback when not attached to a
  terminal.

### Bugs seen in the real run

**G1. No way out of the online-provider loop.**
With no local model, pressing Enter at `Continue? [y/N]` (default No) returns
to a provider menu whose only option is the same online provider. Only Ctrl+C
exits.

**G2. A warning about its own default folder.**
On first run it warns that its own default RAG folder `~/.private_agent/rag`
is not a valid directory, instead of creating it or skipping quietly.

**G3. A spinner after skipping.**
"Indexing local knowledge base..." appears right after it says indexing was
skipped.

**G4. Questions in an odd order.**
It asks for the RAG folder before the workspace, though the project-local
`rag` folder is found relative to the workspace.

### Essentials missing

**G5. No Markdown rendering or code highlighting.**
Replies are streamed as plain text with markup turned off
(`runtime.py:2676`), so code blocks, tables, lists and bold show as raw `**`,
`|` and backticks. This is the biggest visible gap (same as E1).

**G6. No coloured diffs.**
Edit previews (`preview_workspace_patch`) go through the same `Text()` panel
as all tool output, so it is hard to see what an edit changes before
approving it.

**G7. Too many startup questions.**
There are 4–7 questions before the first message (RAG folder, workspace,
mode, provider, URL, key, confirm), with no "remember my choices" and no
command-line flags.

**G8. No live status while working** (see F3; fixed by step 9.7).

**Partly fixed:** the CLI now shows one phase-aware spinner: cyan while a
local model is being loaded into memory or an online request is in flight,
magenta while the model is thinking with `visible_reasoning` off, and blue
while older context or the session summary is being condensed. When
`visible_reasoning` is on, the reasoning streams as dimmed text above the
answer. A context-window gauge beside the spinner and in the
`[Context State]` line fills with the share of the prompt budget in use
(green below 50%, yellow from 50%, red from 80%). The full live panel for
running and finished tools, packs and context use (step 9.7) is still open.

**G9. No way to stop one turn** (see C12; fixed by step 6.4).

**G10. No multi-line input.**
There are no key bindings, so there is no Alt+Enter or Shift+Enter for a new
line.

**G11. No way to expand tool output.**
Previews stop at 500 characters (`TOOL_RESULT_PREVIEW_CHARS`,
`runtime.py:196`), with no way to see the rest of what the model saw.

**G12. Basic approval prompts.**
Prompts are plain `[y/N]` with no "always allow" option (C8), and tool
arguments are shown as a Python dict.

**G13. No arrow-key menus.**
Menus are numbered or typed, and `/tasks` opens a sub-prompt where commands
such as `approve <ID>` are typed and IDs copied by hand.

**G14. Replies and diagnostics are mixed** (see E34; fixed by steps 5.2 and 8.2).

### Cosmetics missing

| ID | Missing | What it would look like |
|---|---|---|
| G15 | **Consistent message style.** About 57 different `[Label]` prefixes in mixed styles; yellow used 166 times for both warnings and normal information | A few levels (info, success, warning, error), each with one colour and one symbol |
| G16 | **One shared console and theme.** 11 separate `Console()` objects and no theme | One console with a named theme (`warning`, `tool`, `user`, …) and a light-terminal option |
| G17 | **An informative prompt.** Just `User:` in bold blue | `qwen3:8b · auto · 42% ›`, or the bottom toolbar (F3) |
| G18 | **Separation between turns.** `AI (model):` header with no divider | A thin `Rule` between turns, aligned headers |
| G19 | **A welcome screen.** Only "Private Agent" | Version, model, workspace and a `/help` hint |
| G20 | **Readable tool requests.** Raw Python `repr` | `read_local_file · tests/conftest.py` |
| G21 | **Per-turn details.** Speed, tokens and time only appear in the exit table | A dimmed line after each reply: `1.8k tokens · 14 tok/s · 6.2s` |
| G22 | **Status symbols.** Results are marked only by panel colour | ✓ ✗ ⊘ ⏱ markers |
| G23 | **Grouped `/help`.** One long list | Groups (Chat, Files, Plans, Memory, Model) in a table |
| G24 | **Messages that fit narrow terminals.** Long warnings wrap mid-sentence | Short messages with details on request |

---

