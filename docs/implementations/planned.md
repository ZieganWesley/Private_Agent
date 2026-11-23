# Code Review: Known Issues, Gaps, Plans and Fix Roadmap

Review date: 2026-10-28

## Part A: Code Quality Issues

### A1. Very large modules (highest priority)

- `src/agent/runtime.py` was 6,507 lines with
  `_run_agent_cli_session` at about 2,600 lines in one function.
- `src/database/memory.py` is 2,926 lines in one
  `PersistentMemory` class.
- **Still open:** `agent/runtime.py` remains large (turn loop and tool
  execution) and `database/memory.py` is a thin facade over nine mixins; the
  large-module refactor needs its own branch.
- **Audit 2026-11-19:** `agent/runtime.py` is still 3,866 lines (the turn loop
  and tool execution remain in it) and `database/memory.py` is a 35-line facade
  over nine mixins. This stays open as the large-module refactor; it needs its
  own branch rather than a single change.
- Ruff complexity checks flag:
  - 38 overly complex functions (`C901`)
  - 22 functions with too many branches (`PLR0912`)
  - 25 functions with too many arguments (`PLR0913`)
  - 15 functions with too many statements (`PLR0915`)

**Standard practice:** small, single-purpose modules. For example, separate
command handlers for `/tasks`, `/plan` and memory upkeep, and a repository
class for each table.

### A2. Settings loaded at import time as global constants

- `config.py` reads `~/.conf` and builds `APP_CONFIG` as soon as
  it is imported, then exposes dozens of module-level constants. `runtime.py`
  imports about 50 of them.
- **Partly addressed (step 3.1):** the schema and all coercion logic now live in
  `src/settings.py` as a typed, frozen `Settings` dataclass
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



### A5. Lint, format and typing are not enforced

**Audit 2026-11-19:** lint and formatting are enforced (`ruff check .` and
`ruff format --check .`, now including `BLE`), but the typing half of this item
is still open: no `[tool.mypy]`/`[tool.pyright]` config exists and neither tool
is a declared dependency, and there is still no CI workflow (step 1.5). Adding a
type checker is a separate decision because it would introduce a dependency and
a baseline of errors to clear.


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
download) and `tests/unit/test_cli.py` (closed stdin).

- There is no `[tool.mypy]` config or other type checking.
- There is still no CI workflow under `.github/workflows/` (step 1.5).

### A6. Logging
  makes problems hard to trace afterwards.

Logging was partially improved (module-level loggers and redacted structured
logging). Remaining: no `[tool.mypy]` config or other type checking; CI status.

- There is no `[tool.mypy]` config or other type checking.
- There is still no CI workflow under `.github/workflows/` (step 1.5).



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

All previously listed deviations (B1–B9) have been addressed; see the fix
roadmap and git history for details.

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


**C3. No session resume and no undo.**

- Limits: only workspace edits made through the file tools are recorded (shell
  commands, MCP tools, edits made before this step and paths outside the active
  workspace have no manifest entry; their old `.bak` files stay harmless).
  `/undo` steps back one turn at a time, and when a file cannot be restored the
  conversation is kept and the failure is reported so the undo can be retried.
  Provider and model recording is still step 12.9.
- Claude Code can rewind code and conversation to a checkpoint. Here, edits
  are copied to `.agent_snapshots/` and `/undo` restores them. Git checkpoints
  exist only inside coding tasks.

**C4. Commands only run on Linux.**
- **Status (updated):** Out of scope for this branch. macOS Seatbelt support exists (6.14), but broader platform equivalence is not being pursued here.
The sandbox needs Bubblewrap and `prlimit`, so on macOS and Windows the shell,
project tests and stdio MCP are turned off. Codex and Claude Code sandbox
commands on macOS (Seatbelt) and Linux, and Codex also on Windows.

**C5. A limited shell tool.**

- The default timeout is 30 seconds (`config.py:698`), too short for many
  builds and test runs.
- It still cannot run background processes such as dev servers or watchers, the
  timeout cannot be set per call, and the shell does not keep state between
  calls.

#### Medium impact

**C6. No subagents.**
- **Status (updated):** Subagent support exists (6.13); broader routing decisions are deferred until C1 documentation work is complete.
Claude Code can hand searches or reviews to separate agents with their own
context. There is nothing like that here.

**C7. No hooks.**


**C8. No fine-grained permission rules.**


**C9. Basic skills and no custom commands.**

- Skills have no front matter, are matched by keyword overlap and are loaded
  whole (see B6).
- Users cannot add their own slash commands as Markdown files.

**C10. Reading large files.**


**C11. No git write workflow.**


**C12. Interrupting a turn.**


#### Lower impact

**C13. Fewer configuration levels.**


**C14. No status line or output styles.**


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

## Part G: User Interface (remaining)

Implemented items (G1, G2, G3, G5, G7, G8, G10, G12, G13, G14, G15/G16,
G17-G19, G20-G23, G24) are complete and documented in
`docs/implementations/ui-architecture.md` and `docs/implementations/implemented.md`.
Only the unfinished items remain, renumbered.

### Bugs seen in the real run

**G1. Questions in an odd order.**
It asks for the RAG folder before the workspace, though the project-local
`rag` folder is found relative to the workspace.

### Essentials missing

**G2. No coloured diffs.**
Edit previews (`preview_workspace_patch`) go through the same `Text()` panel
as all tool output, so it is hard to see what an edit changes before
approving it.

**G3. No way to stop one turn** (see C12).
No cancel/interrupt affordance for an in-flight turn.

**G4. No way to expand tool output.**
Previews stop at 500 characters (`TOOL_RESULT_PREVIEW_CHARS`,
`runtime.py:196`), with no way to see the rest of what the model saw.

---

## Part I: Local Provider Independence (Not Ollama Only)

Goal: the program works with **any installed local provider**, not just
Ollama. If several are installed, it asks which one to use at session start.
When a session or task is resumed and its provider is not available, it falls
back to the same choice prompt.

### I1. How it works today

**Already provider-neutral:**

- **Discovery** (`agent/local_runtimes.py:121`) probes Ollama
  (`/api/tags`) plus up to 7 configured OpenAI-compatible endpoints
  (`/models`) in parallel, with a 1.5-second timeout. The defaults add LM
  Studio (`127.0.0.1:1234/v1`) and llama.cpp (`127.0.0.1:8080/v1`).
- **The provider menu** (`runtime.py:3990`) lists every reachable local
  runtime with its model count, plus the online options, and asks the user to
  choose. So with several running servers, a prompt already appears.
- **Chat** works with OpenAI-compatible local servers through `ChatOpenAI`
  (`_make_local_openai_compatible_chat_model`, `runtime.py:2512`).
- **Session summaries** use whichever chat model the session uses.

**Discovery and selection gaps:**

- **I1j. Only running servers are found.** Discovery detects servers that
  answer on their port. An installed provider that is not running (for example
  Ollama installed but `ollama serve` not started) is invisible, and nothing
  suggests starting it.
- **I1k. Few providers are known.** Only Ollama, LM Studio and llama.cpp have
  defaults. Other common local servers that speak the OpenAI-compatible API
  (for example vLLM, Jan, KoboldCpp, GPT4All, LocalAI, llamafile, MLX-LM and
  text-generation-webui) need manual configuration. Their default ports must
  be checked against each project's docs; several share port 8080, so
  probing a port does not identify which server answered.
- **I1l. Embedding models are listed as chat models** on OpenAI-compatible
  servers: the filter that hides embedding models (`local_runtimes.py:87`)
  only runs for Ollama.
- **I1m. Loopback only.** Only `localhost` and `127.0.0.0/8` addresses are
  accepted (`local_runtimes.py:28`). A provider on the WSL host gateway or on
  another machine in the home network cannot be used as "local" (see H3e).
- **I1n. Always prompts, even with one choice,** and there is no
  `--provider` flag or remembered last choice (see G7).
- **I1o. A provider that stops mid-session** is not handled specially; the
  model call fails and the error is reported, but there is no offer to switch
  to another available provider.

**Resume gaps:**

- **I1p. Nothing records which provider was used.** `chat_history` has no
  provider or model column (`database/memory.py:234`), and `agent_tasks` has
  none either. Conversations can be resumed since step 6.6, but a resumed
  conversation still runs on whatever provider and model the current startup
  selects, and task plans resumed through `/tasks` have no check that the
  provider suits the plan (for example vision or tool support). Step 12.9
  covers recording the provider and falling back on resume.

### I2. Target behaviour

#### Provider registry

Each provider type becomes a **provider adapter** with one interface:

| Method | Purpose |
|---|---|
| `detect_installed()` | Is the provider installed? (binary on `PATH`, known app folder, or configured) |
| `detect_running()` | Does its server answer, and which models does it list? |
| `start()` (optional) | Start the server, only after the user agrees (for example `ollama serve`) |
| `list_models()` | Chat models, with embedding models separated |
| `capabilities(model)` | Tools, vision, thinking, context window; "unknown" when the provider cannot say |
| `chat_model(model, options)` | Build the LangChain chat model |
| `embeddings(model)` (optional) | Build an embeddings client for RAG |
| `hardware_status()` (optional) | GPU and CPU placement, where the provider exposes it |

**Adapters to ship:**

- **Ollama:** native API, all methods.
- **Generic OpenAI-compatible:** chat, `/models` listing, embeddings through
  `/v1/embeddings` when the server supports it.
- **Presets on top of the generic adapter** for LM Studio, llama.cpp
  (`llama-server`), vLLM, Jan, KoboldCpp, GPT4All, LocalAI, llamafile and
  MLX-LM. Each preset has a name, default port, install check and start hint.
  Ports and paths must be verified against each project's docs before
  shipping.

Capabilities for OpenAI-compatible servers come from, in order: what the
server reports (some list a context length), a small built-in table of known
model families, a one-time test call (for example a tiny tool call) cached per
model, and finally user overrides in config.

#### Session start

```
Local providers
  1. Ollama        running   · 6 chat models          (last used)
  2. LM Studio     running   · 2 chat models
  3. llama.cpp     installed · not running            [s] start it
Online
  4. Online — configured in config
Choose provider [1-4] (Default: 1):
```

Rules:

1. **Detect** every known provider: installed, running, or both.
2. **Two or more usable local providers:** always ask, with the last-used
   provider as the default.
3. **One usable local provider:** use it and say so in one line, unless
   `providers.always_ask` is on.
4. **None running but some installed:** offer to start one (with consent), or
   continue online, or show the `doctor` fix.
5. **None installed:** offer online, or show how to install a provider. Never
   loop without an exit (fixes G1).
6. **Flags win:** `--provider lmstudio --model qwen3-8b` skips the prompt;
   if the named provider is missing, fall back to the prompt with a message.
7. **Local first:** online options are listed after local ones and are never
   chosen automatically.

#### Resume with fallback

Each session and each task stores a **provider record**: adapter type, display
name, base URL, model, and the capabilities the work needs (for example tools,
vision).

```
Resuming session "Refactor parser" (last used LM Studio · qwen3-8b)
LM Studio is not running.
  1. Ollama        running · qwen3:8b available
  2. llama.cpp     running · 1 chat model
  3. Start LM Studio
  4. Online — configured in config   (history will be sent online)
Choose provider [1-4]:
```

Rules:

1. **Same provider and model available:** resume without a prompt, and show
   one line saying which provider is used.
2. **Provider available, model missing:** show that provider's model list,
   with the closest name match first.
3. **Provider not available:** show the provider prompt above, with the
   reason, and an option to start it if it is installed.
4. **Capability check:** if the resumed work needs tools or vision and the new
   model lacks them, warn and suggest a model that has them.
5. **Context window check:** if the new model's context is smaller than the
   history, compact the history before continuing (using the existing
   compaction) and say so.
6. **Privacy:** never switch a resumed local session to an online provider
   without an explicit choice, and warn that the conversation history will be
   sent to it.
7. **Record the change:** the session's provider record is updated, and a
   task records the switch as an event in `agent_task_events`.
8. **Mid-session loss:** if the provider stops answering during a session,
   offer the same fallback prompt instead of only reporting an error (I1o).

#### Embeddings without Ollama

- Embeddings get their own setting (`rag.embedding_provider`,
  `rag.embedding_model`), separate from the chat provider.
- Any adapter with an `embeddings()` method can serve it (Ollama, LM Studio,
  llama.cpp with embeddings on, and others).
- The RAG index records which provider and model built it. If they change,
  the index is rebuilt, because vectors from different models cannot be mixed.
- With no embedding provider at all, RAG falls back to keyword search (BM25 or
  SQLite FTS5), as in step 11.6.

---

## Part J: Performance

All numbers were measured on this machine (WSL2, Fedora 44, Python 3.14) with
a temporary home folder. No model was running, so model-call time is not
included. Synthetic data was used for the RAG and memory tests (300 Markdown
files, 5.8 MB, 6,000 chunks; 200 chat messages; 2,000 session summaries).

### J1. Summary

| # | Area | Measured | Severity |
|---|---|---|---|
| J2 | Startup imports | 2.3–2.7 s and about 130 MB before anything runs | High |
| J3 | Tokenizer download | 1.7 MB downloaded on first use, again after every reboot; about 0.4 s | High (also a privacy issue) |
| J4 | Tool definitions in every request | About 4,500 tokens, plus 1,050 for the system prompt | High |
| J5 | DNS lookups | About 250 ms per lookup (49 ms for `localhost`), against 1.4 ms in-process | High for research |
| J6 | Offline internet check | 3 s per web call when offline | Medium |
| J7 | RAG refresh after any file change | All files re-read and re-chunked; BM25 rebuilt (325 ms) | Medium; grows with corpus size |
| J8 | Semantic chunking on refresh | Embedding calls for every file, changed or not | High when `semantic` is used |
| J9 | BM25 query | 5.7 ms per query over 6,000 chunks; grows linearly | Low now |
| J10 | Database | `delete` journal mode; 75 direct database calls in async code | Low now |

**Checked and fine:** RAG change detection (5.5 ms for 300 files), local
runtime discovery (41 ms with none running), opening the database per planner
call (about 1 ms), relevant-episode search over 2,000 summaries (20 ms), saving
a message (0.2 ms), trimming history once the tokenizer is loaded (3 ms for 200
messages).






### J7. RAG refresh rebuilds more than it needs

When any file changes, `initialize_knowledge_base` (`rag/indexing.py:302`)
runs again from the start:

- it re-reads and re-chunks **every** file, not just the changed ones (40 ms
  for 300 files, more for PDFs);
- it rebuilds the BM25 index over all chunks (**325 ms** for 6,000 chunks);
- it creates a new Chroma client.

Change detection itself is cheap (5.5 ms), and only changed chunks are
re-embedded, so embedding cost is already incremental. The rest grows with the
corpus, and PDF text extraction in particular is slow.

**Fix:** keep the parsed chunks for unchanged files (in memory, or stored with
the index state), re-parse only changed files, update BM25 by replacing the
changed files' entries, and reuse the existing Chroma client.

### J8. Semantic chunking calls the embedding model for every file

With `chunking.strategy` set to `semantic`, `_split_into_chunks` is passed the
embedding function (`rag/indexing.py:428`) and embeds every sentence window.
Because every file is re-split on each refresh (J7), **all files are sent to
the embedding model again on every refresh**, even when only one changed. On a
large corpus this can take minutes.

**Fix:** never re-split files whose fingerprint has not changed (follows from
J7).


### J10. Database settings and blocking calls

- The async runtime makes about 75 direct `memory.*` calls on the event loop,
  and only 1 goes through `asyncio.to_thread`. Each is fast, but the slower
  ones (episode search, task listing, retention pruning) block streaming and
  the live overlay while they run.

**Remaining:** move the slower queries off the event loop (step 13.8). WAL
mode and `busy_timeout` remain as configured defaults.

---

## Part K: Hard-Coded Values (Not Read from the Config File)

The config file covers most user-facing settings: 114 keys are read from it.
This part lists values that are still fixed in the code, values that are
defined twice, and values that should stay in code on purpose.


### K2. Sandbox resource limits

`code_tasks/__init__.py:259–262` passes fixed limits to `prlimit`:

| Limit | Value |
|---|---|
| CPU time | 180 s |
| Address space (memory) | 2 GiB |
| Processes | 128 |
| Largest file written | 512 MiB |

A 2 GiB address-space limit is likely to break builds that reserve more
virtual memory than they use (Java, Node, Rust). None of these can be changed
without editing the code.

### K3. Timeouts and leases

| Value | Where |
|---|---|
| Local provider discovery timeout 1.5 s, at most 8 endpoints | `agent/local_runtimes.py:13–14` |
| Git inspection timeout 5 s | `tools/filesystem.py:39` |
| SQLite connect timeout 2 s and busy timeout 2,000 ms | `database/memory.py:64`, `:70`, `tools/planner.py:80` |
| Read timeout for SQLite files indexed by RAG, 5 s | `rag/indexing.py:144` |
| Task lease 180 s, renewed every 30 s | `agent/runtime.py:201–202` |
| Lease defaults of 300 s and 180 s in two different methods, both limited to 30–3,600 s | `database/memory.py:1223`, `:1310` |
| "Slow operation" message threshold of 2 s, repeated 6 times | `runtime.py:2973`, `:3765`, `:3872`, `:5253`, `:5296`, `:5946` |
| Plugin timeout 30 s, at most 100 calls | `plugins.py:19–20` |

### K4. Output and context sizes

| Value | Where |
|---|---|
| Tool output preview 500 characters | `runtime.py:196` |
| Workspace search: 10,000 files, 16 MB, 12,000 characters of output; match lines cut to 500 or 300 characters | `tools/filesystem.py:36–38`, `:748`, `:816` |
| Git output cut to 1,000 characters | `tools/filesystem.py:894` |
| `/add` folder context: 100 files, 10,000 entries scanned, a fixed list of file types | `agent/interactive.py:49–51` |
| Compaction source text capped at 6,000 tokens, regardless of the model's context size | `agent/compaction.py:12` |
| Memory context: episode fields cut to 200, 300, 400, 600, 1,200 or 1,600 characters; 100 episode matches searched; 5 learned items shown | `runtime.py:451–588`, `:575`, `:610` |
| Up to 10 planning citations kept | `runtime.py:5859`, `:5984` |
| `/tasks` lists 50 tasks | `database/memory.py:1010` |
| At most 16 test commands with at most 64 arguments each | `config.py:775` |

`max_memory_context_tokens` is configurable, but the per-field caps inside it
are not.

### K5. RAG

| Value | Where |
|---|---|
| Supported file types | `rag/indexing.py:47` |
| SQLite files: 10 rows per chunk, cells cut to 200 characters | `rag/indexing.py:54–55` |
| File types treated as Markdown or code by `auto` chunking | `rag/chunking.py:28–29` |
| Name patterns that mark embedding models, written twice | `agent/local_runtimes.py:89`, `runtime.py:3108` |

### K6. External services and models

| Value | Where |
|---|---|
| IP location services `ipwho.is` and `ipapi.co` | `tools/location.py:24–26` |
| Open-Meteo geocoding, forecast and archive URLs | `tools/location.py:28–30` |
| Weather limits: 16 days ahead, 90 days back, 1 MB responses | `tools/location.py:17–20` |
| DuckDuckGo search URL | `tools/network.py:602` |
| Tokenizer `cl100k_base`, used for every model | `agent/prompts.py:45` |
| Online base URL `api.openai.com`, written again outside the config defaults | `agent/providers.py:347` |

For a privacy-focused tool, the location and search services should be
swappable, for example for a self-hosted SearXNG or Nominatim.

### K7. Paths

| Value | Where |
|---|---|
| `SNAPSHOT_DIR` resolved against the current folder at import time, not the workspace (it is exported but not used elsewhere) | `sandbox/linux.py:8` |

### K8. Plan limits defined twice

The plan limits are written both in the tool schemas
(`tools/schemas.py:174–232`) and in the database checks
(`database/memory.py:726–871`): 20 steps, 10 items per list, 2,000-character
goals, 1,000-character step descriptions, 64-character step IDs, a
1,000-character approval scope, and research references of up to 1,500
characters each and 10,000 in total. If one changes and not the other, the
schema accepts a plan that the database then rejects.

### K9. Every default written twice

Each setting has its default once in `DEFAULT_CONFIG` and again as the
fallback where it is read. For example, the shell timeout of 30 appears at
`config.py:139` and `config.py:698`, and the Ollama URL at line 37 and line
509. Changing a default means editing both, and they can silently disagree.

### K10. Values that should stay in code

These are correct as fixed values, because a config file should not be able to
weaken safety:

- secret and sensitive-field patterns (`run_logging.py:35–49`,
  `code_tasks/__init__.py:39`);
- private network ranges (`tools/network.py:41`);
- tool permission groups (`agent/permissions.py`, `tools/__init__.py:91–102`);
- schema version, step statuses and weather codes.

The risky-program and system-path lists (`sandbox/linux.py:63–93`) and the RAG file types
could be configurable, but only in the stricter direction: adding words, or
allowing a subset of file types.

---

## Part L: Fix Roadmap

The steps are grouped into phases by topic (Phases 1–15 below). **Do not work
through the phases in number order.** Follow the implementation order in
section L1, which runs from the easiest step to the hardest while respecting
what each step needs first.

### L1. Implementation order: easiest to hardest

**How the order was made:**

- Each step has its own **difficulty**: Trivial (under an hour), Easy (a few
  hours), Medium (one to two days), Hard (several days) or Hardest (a week or
  more, touching much of the code).
- A step is placed in the **stage** of the hardest step it needs first. For
  example, "Stop one turn without exiting" (6.4) is Medium on its own, but it
  needs the shared agent loop (4.2, Hardest), so it is in Stage 5, right after
  4.2. This is why some Easy steps appear late: they unlock only after a hard
  refactor.
- Within a stage, a step always comes after the steps it needs, and easier
  steps come first.
- Every step needs step 1.1 (a working `pytest`), which is not repeated in the
  "Needs first" column.
- Struck-out steps (6.14, 6.15, 8.1, 8.9) were replaced by other steps and are
  left out. Stage 1 is complete; the tables below list the remaining work.

**Summary:**

| Stage | Steps | What it covers |
|---|---|---|
| 1: Trivial | 15 | Typos, docs, config keys, POSIX-only, tokenizer bundling, caching the internet check, WAL mode |
| 2: Easy | 22 | Linting, test skips, quick planning and RAG fixes, Markdown replies, coloured diffs, platform detection, `doctor`, DNS caching |
| 3: Medium | 36 | The remaining test failures and CI, the `Settings` model and everything configurable, logging, search and RAG improvements, lazy imports |
| 4: Hard | 25 | `ToolContext`, splitting commands and memory, async tools, the sandbox interface, provider adapters |
| 5: Hardest | 42 | The shared agent loop and everything built on it: permission rules, session resume, hooks, planning mode, capability packs, router, overlays, orchestrator, macOS sandbox |

#### Stage 2: Easy, independent fixes

16 steps remaining, plus the second half of step 13.3. Stage 1 is complete.

| # | Step | Task | Difficulty | Needs first |
|---|---|---|---|---|
| 22 | 5.5 | Reciprocal rank fusion and better BM25 tokenizing | Easy | — |
| 23 | 6.2 | Paged `read_local_file` (offset and limit) | Easy | — |
| 24 | 7.1 | Read-only tools in planning mode | Easy | — |
| 25 | 7.2 | Save a plan in one model round | Easy | — |
| 26 | 7.3 | Pass the whole plan to the model when it runs | Easy | — |
| 27 | 8.3 | RAG relevance cut-off and more chunks | Easy | — |
| 28 | 8.7 | Documents-only answer mode | Easy | 8.3 |
| 29 | 10.1 | Fix the 4 startup bugs | Easy | — |
| 30 | 10.2 | Markdown replies with code highlighting | Easy | — |
| 31 | 10.3 | Coloured diffs in edit previews | Easy | — |
| 32 | 11.2 | `platform_support.py` detection | Easy | — |
| 33 | 11.10 | Clear webcam and microphone messages | Trivial | 11.2 |
| 34 | 11.3 | `private-agent doctor` command | Easy | 11.2 |
| 35 | 11.4 | WSL warnings and Ollama setup | Easy | 11.2 |
| 36 | 11.12 | Setup docs per platform | Easy | 11.3 |
| 37 | 13.2 | Cache DNS lookups | Easy | — |
| 13.3 | Remaining work: check once per turn and use real request errors as the offline signal  | J6 | — | Five web calls while offline wait at most one timeout |

#### Stage 3: Medium (needs `Settings`, the green test suite or logging)

36 steps.

| # | Step | Task | Difficulty | Needs first |
|---|---|---|---|---|
| 39 | 1.5 | Add the GitHub Actions CI workflow | Easy | 1.3 |
| 40 | 11.11 | CI matrix (Linux, arm64, macOS, core-only) | Easy | 1.5 |
| 41 | 13.10 | Performance checks in CI | Easy | 1.5 |
| 42 | 3.1 | Create the typed `Settings` model | Medium | — |
| 43 | 3.5 | workspace root is set explicitly at startup and validated | Easy | 3.1 |
| 44 | 14.10 | Fix `SNAPSHOT_DIR` | Trivial | 3.5 |
| 45 | 6.1 | Load `AGENTS.md` project instructions | Easy | 3.1 |
| 46 | 6.9 | Project-level config file | Easy | 3.1 |
| 47 | 4.1 | Characterization tests for the agent loop | Medium | 1.3 |
| 48 | 5.2 | Move diagnostics to `logging` | Medium | 1.3 |
| 49 | 8.2 | Keep diagnostics apart from replies (rest of 8.2) | Trivial | 5.2, 10.2 |
| 50 | 5.3 | Narrow the blind `except Exception` blocks | Medium | 5.2 |
| 51 | 5.6 | skill front matter and embedding matching with keyword fallback | Medium | — |
| 52 | 5.7 | Web search provider interface (SearXNG first) | Medium | — |
| 53 | 8.12 | More search results and paged web fetch | Easy | 5.7 |
| 54 | 8.5 | Rewrite follow-up questions before searching | Medium | 8.3 |
| 55 | 8.6 | Re-rank retrieved chunks | Medium | 5.5 |
| 56 | 8.8 | More document formats | Medium | — |
| 57 | 8.10 | Several document collections | Medium | 3.1 |
| 58 | 8.18 | Document creation tools | Medium | — |
| 59 | 10.4 | Fewer startup questions, flags, remembered choices | Medium | 3.1 |
| 60 | 10.5 | One console, theme and message levels | Medium | 5.2 |
| 61 | 10.10 | UI cosmetics | Easy | 10.5 |
| 62 | 11.5 | Lazy imports and optional extras | Medium | — |
| 63 | 11.6 | Keyword-only document search fallback | Medium | 11.5 |
| 64 | 13.6 | Incremental RAG refresh | Medium | — |
| 65 | 13.9 | FTS5 keyword search for large corpora | Medium | 11.6 |
| 66 | 14.3 | Single source of defaults | Medium | 3.1 |
| 67 | 14.4 | Configurable sandbox limits | Easy | 14.3 |
| 68 | 14.5 | Configurable timeouts and leases | Easy | 14.3 |
| 69 | 14.6 | Configurable output and context sizes | Easy | 14.3 |
| 70 | 14.7 | Configurable RAG values | Easy | 14.3 |
| 71 | 14.8 | Configurable external services | Easy | 5.7, 14.3 |
| 72 | 14.9 | Plan limits defined once | Easy | 14.3 |
| 73 | 14.11 | Stricter-only safety lists in config | Easy | 14.3 |

#### Stage 4: Hard (core refactors and what they unlock)

24 steps .

| # | Step | Task | Difficulty | Needs first |
|---|---|---|---|---|
| 74 | 3.2 | Move every module to `Settings`, delete old constants | Hard | 3.1 |
| 75 | 3.4 | Create `ToolContext` and remove tool globals | Hard | 3.1 |
| 76 | 7.9 | More than one active plan | Easy | 3.4 |
| 77 | 8.4 | `search_documents` tool | Easy | 3.4 |
| 78 | 4.4 | Split slash commands into `agent/commands/` | Hard | 4.1 |
| 79 | 4.6 | Explicit `__all__` in `agent/__init__.py` | Easy | 4.4 |
| 80 | 6.10 | Custom slash commands from Markdown | Easy | 4.4 |
| 81 | 10.9 | Arrow-key menus | Medium | 4.4 |
| 82 | 4.5 | Split `PersistentMemory` into repositories | Hard | 1.3 |
| 83 | 4.7 | SQLite connection per thread or a lock | Easy | 4.5 |
| 84 | 12.8 | Store the provider with sessions and tasks | Easy | 4.5 |
| 85 | 13.8 | Slow database calls off the event loop | Easy | 4.7 |
| 86 | 5.1 | Make network and shell tools async | Hard | 1.3 |
| 87 | 11.7 | Sandbox interface with backends | Hard | 5.1 |
| 88 | 11.9 | Termux support | Hard | 10.5, 11.5, 11.7 |
| 89 | 12.1 | Provider adapter interface | Hard | 3.1 |
| 90 | 12.2 | Make Ollama packages optional | Easy | 11.5, 12.1 |
| 91 | 12.6 | Filter embedding models; several servers per type | Easy | 12.1 |
| 92 | 13.5 | Lazy imports and background indexing | Easy | 11.5, 12.2 |
| 93 | ~~12.3~~ | Capability detection for every provider: only the cached probe call and the known-model-family table remain | Medium | 12.1 |
| 94 | 12.4 | Provider presets | Medium | 12.1 |
| 95 | 12.5 | Detect installed-but-stopped providers | Medium | 12.4 |
| 96 | 12.7 | New provider selection rules | Medium | 10.4, 12.5 |
| 97 | 12.11 | Embeddings from any provider | Hard | 11.6, 12.1 |
| 98 | 12.13 | Hardware status and benchmark per adapter | Easy | 12.1, 12.11 |

#### Stage 5: Hardest (the shared agent loop and everything built on it)

42 steps.

| # | Step | Task | Difficulty | Needs first |
|---|---|---|---|---|
| 99 | 4.2 | Extract one shared agent loop | Hardest | 3.4, 4.1 |
| 100 | 6.4 | Stop one turn without exiting | Medium | 4.2 |
| 101 | 7.5 | `/plan` stays on, with `ask_user` | Medium | 4.2, 4.4 |
| 102 | 7.6 | Markdown plan with one-choice approval | Medium | 7.5 |
| 103 | 7.8 | Time budget per plan step | Medium | 4.2 |
| 104 | 7.10 | Text-plan fallback for non-tool models | Medium | 7.6 |
| 105 | 9.1 | Typed event bus | Medium | 4.2 |
| 106 | 9.8 | `agent_step_timings` table | Easy | 4.5, 9.1 |
| 107 | 10.7 | Readable tool requests and full-output view | Easy | 9.1 |
| 108 | 9.9 | Time estimator | Medium | 9.8 |
| 109 | 4.3 | Single `PermissionPolicy` for every tool call | Hard | 4.2 |
| 110 | 6.5 | Allow/deny permission rules | Medium | 4.3 |
| 111 | 12.12 | Configured non-loopback local addresses | Easy | 6.5, 12.1 |
| 112 | 6.8 | Non-interactive `-p` mode | Medium | 4.2, 6.5 |
| 113 | 9.11 | JSON event output | Easy | 6.8, 9.1 |
| 114 | 6.11 | Hooks before and after tool calls | Medium | 4.2, 4.3 |
| 115 | 6.12 | Git write tools (branch, commit, PR) | Medium | 4.3, 6.5 |
| 116 | 7.4 | Replace the regex coding gate | Medium | 4.3 |
| 117 | 8.15 | Auto test and lint after edits, LSP | Medium | 6.11 |
| 118 | 8.19 | Extra approved folders outside the workspace | Medium | 6.5 |
| 119 | 10.8 | Better approval prompts | Medium | 6.5 |
| 120 | 6.3 | Shell: timeouts, pipes, background processes | Hard | 4.3, 5.1 |
| 121 | 7.7 | Automatic step validation | Medium | 6.3 |
| 122 | 8.17 | Calculator / sandboxed Python tool | Medium | 6.3 |
| 123 | 6.6 | Session resume (`--continue`, `--resume`) | Hard | 4.2, 4.5 |
| 124 | 6.7 | `/undo` from snapshots | Medium | 6.6 |
| 125 | 8.11 | Chat commands (`/new`, `/retry`, `/edit`, `/export`) | Medium | 6.6 |
| 126 | 6.13 | Subagents | Hard | 3.4, 4.2 |
| 127 | 9.2 | Capability packs | Hard | 3.4, 4.2 |
| 128 | 13.7 | Stable prompt prefix for prompt caching | Easy | 9.2 |
| 129 | 8.20 | Clipboard, scheduling, voice | Hard | 9.2 |
| 130 | 9.3 | Turn router | Hard | 9.2 |
| 131 | 9.4 | `enable_capability` escalation | Medium | 4.3, 9.3 |
| 132 | 8.13 | Research loop with saved report | Hard | 8.4, 8.12, 9.3 |
| 133 | 8.14 | Source quality checks and specialist sources | Hard | 8.13 |
| 134 | 9.5 | Context manager with per-source budgets | Hard | 9.2 |
| 135 | 9.7 | Live status overlay | Hard | 9.1 |
| 136 | 9.10 | Timeline overlay (Ctrl+T) | Medium | 9.7, 9.9 |
| 137 | 12.9 | Provider fallback on resume | Hard | 6.6, 12.7, 12.8 |
| 138 | 12.10 | Fallback when a provider stops mid-session | Medium | 12.9 |
| 139 | 9.6 | Task orchestrator | Hardest | 7.5, 7.6, 7.8, 9.3 |
| 140 | 11.8 | macOS Seatbelt sandbox | Hardest | 11.7 |

Rules for every step:

- The full test suite passes before the PR is merged.
- No behaviour changes in refactoring steps. Behaviour changes go in their
  own steps.
- Update `README.md` and `.github/copilot-instructions.md` when a step
  changes how things work.

Phases 1–8 are complete; see git history for details.

### Phase 9: Single combined runtime and live status

This phase implements Part F. Steps 9.1–9.6 replace the separate role
profiles of step 8.1, and step 9.7 replaces the simple status line of step
6.15.

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 9.1 | Add a typed event bus. Emit events from the agent loop, model calls, tool executor, approvals and planner. Move session statistics and the run log onto it | F2, A6, E34 | 4.2 | Session statistics are computed only from events, with tests |
| 9.2 | Group tools, prompt sections and limits into capability packs (Core, Knowledge, Research, Coding, Helper, Planning, MCP) | F2, E33 | 3.4, 4.2 | Each tool belongs to exactly one pack, and a turn can bind any set of packs |
| 9.3 | Add the turn router: commands, then rules, then model classification with a safe fallback. Turn `/code`, `/research`, `/plan` and `/ask` into shortcuts that force packs | F2, D2 | 9.2 | Plain chat binds only Core; "research X then change Y" binds Research and Coding, with tests for each route |
| 9.4 | Add the `enable_capability` tool for escalation during a turn, checked by `PermissionPolicy` | F2 | 9.3, 4.3 | Chat can switch to coding after approval, without starting a new turn |
| 9.5 | Add the context manager with a token budget per source, and show context use | F2, E6 | 9.2 | No source goes over its share unless others are unused, with tests |
| 9.6 | Add the task orchestrator: complexity check, plan with read-only tools from all chosen packs, one-choice approval, per-step packs and budgets, automatic verification and a final report | F2, D1, D3, D6, D7, D12 | 9.3, 7.5, 7.6, 7.8 | A multi-step request runs as a plan whose steps use different packs, end to end |
| 9.7 | Add the live status overlay: `rich.live` panel during turns, `prompt_toolkit` bottom toolbar between turns, paused for approvals, plain-mode fallback | F3, C14 | 9.1 | Running tools, finished tools, packs, model speed and context use are shown live; approvals still work |
| 9.8 | Add the `agent_step_timings` table, record every step, and fill it from existing `agent_task_events` history | F4 | 9.1, 4.5 | Every finished step has a timing row |
| 9.9 | Add the estimator: history, model speed and the model's guess, shown as ranges with confidence, recalibrated after each step | F4 | 9.8 | On a replayed set of past tasks, most actual durations fall inside the estimated range |
| 9.10 | Add the timeline overlay (Ctrl+T) with per-step status, elapsed and estimated times, overall finish range, waiting time and budget warnings | F4 | 9.7, 9.9 | A planned task shows its timeline, and a warning appears before the time budget runs out |
| 9.11 | Write events as JSON lines in non-interactive mode | F2, F3 | 9.1, 6.8 | `private-agent -p … --output json` streams tool and step events |

### Phase 10: User interface

Steps 10.1–10.3 are small and are in Stage 2 of the implementation order
(section L1).

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 10.1 | Fix the startup bugs: give the online-provider prompt a way back or out, create the default RAG folder (or skip quietly), drop the spinner when indexing is skipped, and ask for the workspace before the RAG folder | G1, G2, G3, G4 | — | First run with no model and no RAG folder ends cleanly, with a test for each case |
| 10.2 | Render replies as Markdown with code highlighting, including while streaming (same as step 8.2) | G5, E1 | — | Code blocks, tables and lists render correctly |
| 10.3 | Show edit previews as coloured unified diffs with `rich.syntax` | G6 | — | Added and removed lines are coloured in approval prompts |
| 10.4 | Cut the startup questions: remember the last choices, add `--model`, `--provider`, `--mode`, `--workspace` and `--rag` flags, and only ask about what is not set | G7 | 3.1 | Starting with saved settings asks no questions |
| 10.5 | Use one shared console with a named theme, a few message levels (info, success, warning, error) with one colour and symbol each, and replace the ~57 ad-hoc `[Label]` prefixes | G15, G16, G22, G24 | 5.2 | All output goes through one console and theme, and `NO_COLOR` is respected |
| 10.7 | Show tool requests in readable form, and add Ctrl+O (or `/last`) to see full tool output | G11, G20 | 9.1 | The full output of the last tool can be shown on request |
| 10.8 | Better approval prompts: a readable request, coloured diff and choices for yes, no, and always allow for this session or project | G12 | 6.5 | "Always allow" adds a permission rule |
| 10.9 | Arrow-key menus for provider, model, mode and `/tasks`, with typed input still possible | G13 | 4.4 | Menus work with arrow keys and with typed numbers |
| 10.10 | Cosmetics: welcome screen, informative prompt, divider between turns, per-turn tokens and speed line, grouped `/help` | G17, G18, G19, G21, G23 | 10.5 | Each item is shown, and each can be turned off in config |

### Phase 11: Platform support (POSIX only)

Step 11.1 is in Stage 1 and steps 11.2–11.4 in Stage 2 of the implementation
order (section L1), so the app runs cleanly in WSL early.

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 11.2 | Add `platform_support.py` to detect Linux, WSL, macOS and Termux and report available features. Replace direct `platform.system()` checks with it | H3b, H4 | — | Each platform is detected correctly, with tests using fake `/proc` files and environment variables |
| 11.3 | Add the `private-agent doctor` command, with the exact fix command per detected distribution and platform (for example `sudo dnf install bubblewrap`) | H3b, H3d | 11.2 | On this WSL machine, doctor reports the missing `bwrap` with `sudo dnf install bubblewrap` |
| 11.4 | WSL support: warn when the workspace or RAG folder is under `/mnt/<drive>`, document Ollama inside WSL vs. Windows-side Ollama with mirrored networking, and allow an explicitly configured WSL host-gateway address as a local runtime | H3e, H3f | 11.2 | Doctor shows how Ollama is reached, and a `/mnt/c` workspace shows a warning |
| 11.5 | Import optional parts lazily (Ollama client, Chroma, media) and split dependencies into a core plus `rag`, `media` and `all` extras | H3a, H3k | — | `pip install .` without extras starts, and missing features are reported instead of crashing |
| 11.6 | Add a fallback for document search without the `rag` extra: BM25 only, or SQLite FTS5 | H3k | 11.5 | RAG answers with citations on a core-only install |
| 11.7 | Add a sandbox interface with Bubblewrap, Seatbelt and None backends; move the current Bubblewrap code behind it, and use `resource.setrlimit` where `prlimit` is missing | H3h, H4 | 5.1 | The Bubblewrap backend passes the existing sandbox tests unchanged |
| 11.8 | Add the macOS Seatbelt backend (same as step 6.14), and clear messages for camera and microphone permissions | H3h, H3i | 11.7 | Shell, tests and stdio MCP work on macOS |
| 11.9 | Termux support: detect it, keep the workspace in Termux's home folder (warn on `/sdcard`), use a compact layout on narrow screens, and add an explicit off-by-default "unsandboxed, approve every command" session option (project owner to decide whether to ship it) | H3j, H3l, H3m | 11.5, 11.7, 10.5 | The core chat, RAG fallback and online or Termux-local models work on Termux |
| 11.10 | Report missing webcam and microphone access clearly on WSL and Termux | H3g | 11.2 | Media tools explain why they are unavailable instead of failing |
| 11.11 | Add a CI matrix: Ubuntu x86_64, Ubuntu arm64 and macOS, plus a core-only install job that mimics Termux | H3c | 1.5 | All jobs pass on every PR |
| 11.12 | Write setup docs for WSL, macOS and Termux | H4 | 11.3 | Each platform has a tested step-by-step setup in the README |

### Phase 12: Local provider independence

Steps 12.1–12.8 and 12.11 are in Stage 4 and the rest in Stage 5 of the
implementation order (section L1). Step 12.9 (resume fallback) needs session
resume (6.6) first. This phase replaces step 8.9 (embedding providers).

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 12.1 | Add the provider adapter interface and move the current Ollama and OpenAI-compatible code behind it, with no change in behaviour | I1b, I1d, I1g | 3.1 | Existing provider tests pass through the new adapters |
| 12.2 | Import `ollama` and `langchain_ollama` only inside the Ollama adapter, and make them an optional `ollama` extra | I1e | 12.1, 11.5 | The app starts with no Ollama packages installed |
| 12.3 | Capability detection for OpenAI-compatible servers: only the cached test call and the known-model-family table are missing | I1b, I1c | 12.1 | An LM Studio model gets real tool, vision and context values, with tests |
| 12.4 | Add presets (LM Studio, llama.cpp, vLLM, Jan, KoboldCpp, GPT4All, LocalAI, llamafile, MLX-LM), each with verified default port, install check and start hint. Identify the server from its responses, not only its port | I1k | 12.1 | Each preset is detected when running, with recorded test responses |
| 12.5 | Detect installed-but-not-running providers, and offer to start them after consent | I1j | 12.4 | An installed, stopped Ollama is listed as "installed · not running" and can be started from the prompt |
| 12.6 | Filter embedding models from chat lists for every adapter, and allow several servers of the same type (a list of Ollama URLs) | I1i, I1l | 12.1 | Two Ollama servers are both listed, and embedding models are not offered as chat models |
| 12.7 | New startup selection rules: ask when two or more local providers are usable, auto-select a single one, last-used as default, `--provider` and `--model` flags, local before online, no endless loop | I1n, G1, G7 | 12.5, 10.4 | Each rule has a test, including 0, 1 and several providers |
| 12.8 | Store a provider record (adapter, name, URL, model, needed capabilities) with each session and each task | I1p | 4.5 | New sessions and tasks have a provider record, and old rows migrate with "unknown" |
| 12.9 | Resume fallback: same provider and model → resume; model missing → model prompt; provider missing → provider prompt with reason and start option; capability and context checks; explicit consent before going online; record the switch | I1p | 12.7, 12.8, 6.6 | Resuming a session whose provider is stopped shows the fallback prompt, with tests for each case |
| 12.10 | Offer the fallback prompt when the provider stops answering mid-session | I1o | 12.9 | Stopping the server mid-session leads to the prompt instead of a dead session |
| 12.11 | Embeddings from any adapter: separate embedding provider setting, record the provider and model in the RAG index, rebuild when they change, keyword-search fallback | I1a, E14 | 12.1, 11.6 | RAG works with only LM Studio installed, and changing the embedding model triggers a rebuild |
| 12.12 | Allow explicitly configured non-loopback local addresses (WSL host gateway, home network) with a clear warning | I1m, H3e | 12.1, 6.5 | A configured gateway address works after the user confirms it |
| 12.13 | Hardware status through the adapter where available, with a clear "not exposed by this provider" message otherwise; make the RAG benchmark use the embedding adapter | I1f, I1h | 12.1, 12.11 | `/hardware-status` and the benchmark work with every adapter |

### Phase 13: Performance

Steps 13.1, 13.3 and 13.4 are in Stage 1 and step 13.2 in Stage 2 of the
implementation order (section L1). They are small and give the largest
gains.

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 13.2 | Cache validated DNS results for a short time (bounded, rules re-checked on each use), and resolve in a thread with a deadline instead of a subprocess | J5 | — | A second request to the same host does no lookup, and a stuck resolver still times out, with tests for both |
| 13.5 | Import Chroma, Ollama packages and media libraries lazily, and build the RAG index in the background during provider selection | J2 | 11.5, 12.2 | Startup to the first question is under 0.5 s on this machine |
| 13.6 | Incremental RAG refresh: keep chunks for unchanged files, re-parse only changed files, update BM25 in place, reuse the Chroma client | J7, J8 | — | Changing one file in a 300-file corpus refreshes in under 100 ms (excluding embedding), and semantic chunking embeds only the changed file |
| 13.7 | Keep the system prompt and tool definitions identical between turns so local servers can reuse their prompt cache, and send only the needed tools (step 9.2) | J4 | 9.2 | Plain chat sends no tool definitions, and repeated turns keep the same prompt prefix |
| 13.8 | Move slow database calls (episode search, task listing, retention pruning) off the event loop | J10 | 4.7 | No database call over 5 ms runs on the event loop, checked with asyncio debug mode |
| 13.9 | Use SQLite FTS5 for keyword search when the corpus is large | J9 | 11.6 | Keyword search at 20,000 chunks stays under 10 ms |
| 13.10 | Add a performance check to CI: startup import time, a RAG refresh on a fixed corpus, and token counting with the network blocked | J1 | 1.5 | CI fails if startup or refresh time grows by more than 20% |

### Phase 14: Configuration

Builds on the `Settings` model from step 3.1. Steps 14.1 and 14.2 have no
dependencies and are in Stage 1 of the implementation order (section L1).

| Step | Task | Fixes | Depends on | Done when |
|---|---|---|---|---|
| 14.3 | Make the `Settings` model the single source of defaults and remove the repeated fallback values | K9 | 3.1 | Each default appears in exactly one place, checked by a test that compares the model with the generated file |
| 14.4 | Move the sandbox limits (CPU, memory, processes, file size) into config, with safe minimums and maximums, and raise the memory default | K2 | 14.3 | A build that needs more than 2 GiB of address space can run after changing the setting |
| 14.5 | Move timeouts and leases into config, use one lease default, and use one "slow operation" threshold | K3 | 14.3 | No timeout or lease value is written as a literal outside the settings |
| 14.6 | Move output and context sizes into config; derive the compaction cap from the model's context window | K4 | 14.3 | The listed limits can be changed in config, with tests at the limits |
| 14.7 | Move RAG values into config (SQLite rows per chunk, cell size, chunking file types), and keep one shared list of embedding-model name patterns | K5 | 14.3 | Embedding-name patterns are defined once |
| 14.8 | Make external services configurable (location, geocoding, weather, search URLs and limits, tokenizer), and remove the second copy of the online base URL | K6 | 14.3, 5.7 | Each service URL can be replaced by a self-hosted one, with tests |
| 14.9 | Define plan limits once and use them for both the tool schemas and the database checks | K8 | 14.3 | Changing one limit changes both, with a test |
| 14.10 | Remove `SNAPSHOT_DIR` or resolve it from the workspace at use time | K7 | 3.5 | No snapshot path is resolved at import time |
| 14.11 | Allow the risky-command list and RAG file types to be extended in config only in the stricter direction | K10 | 14.3 | Config can add risky words but cannot remove built-in ones, with tests |

### Phase 15: Repository intelligence, architecture memory and metrics

Recorded from issue #13 (Part M). Priorities are the issue's own. These steps
are not yet placed in the L1 stage order: most need Stage 4 work first, so they
follow it.

| Step | Task | Fixes | Priority | Depends on |
|---|---|---|---|---|
| 15.1 | Add Tree-sitter parsing and a project-scoped symbol and dependency index in SQLite | M1a | HIGH | 3.1 |
| 15.2 | Incremental re-indexing, repository version awareness and symbol-change tracking | M1a | HIGH | 15.1 |
| 15.3 | Symbol retrieval tools (`find_symbol`, `find_definition`, `find_references`, `find_callers`, `find_callees`, `find_dependencies`, `trace_execution_flow`) and symbol-first ranking | M1b | HIGH | 15.1, 3.4 |
| 15.4 | Repository knowledge graph with traversal, impact analysis and architectural path tracing | M1c | HIGH | 15.1 |
| 15.5 | Persistent repository maps and structure snapshots for context-efficient prompts | M1d | HIGH | 15.1 |
| 15.6 | Architecture memory subsystem and automatic architecture extraction | M2 | HIGH | 15.1, 4.5 |
| 15.7 | Architectural decision records with cross-session recall | M2 | MEDIUM-HIGH | 15.6 |
| 15.8 | Cross-project awareness and architecture recall inside planning | M2 | HIGH | 15.6, 7.5 |
| 15.9 | Typed memory classification and a ranking engine (frequency, recency, importance, relevance, confidence, feedback) | M3a, M3b | HIGH | 4.5 |
| 15.10 | Memory promotion, decay, archival and compression policies | M3c | MEDIUM | 15.9 |
| 15.11 | Task, tool and retrieval metrics collected from the event bus and task events | M4 | HIGH | 9.1, 4.5 |
| 15.12 | Agent health reports, performance summaries and dashboard export | M4 | MEDIUM | 15.11 |
| 15.13 | OpenVINO NPU-accelerated embeddings for document, memory and repository indexes | M5 | FUTURE | 11.5, 12.11 |
| 15.14 | Local STT and TTS with optional NPU acceleration | M5 | FUTURE | 11.5, 9.2 |
| 15.15 | Hardware-aware CPU/GPU/NPU detection and inference routing | M5 | FUTURE | 12.1, 15.13 |




### Progress tracker

Track progress by stage (section L1), since that is the order of work. The
phase list below shows coverage by topic. Last updated 2026-11-20, after
step 6.8 (non-interactive mode) on `headless-mode` (all 924 tests pass, 12 skipped,
0 failed).

- [ ] Stage 2: Easy (steps 16–37)
- [ ] Stage 3: Medium (steps 38–73)
- [ ] Stage 4: Hard 
- [ ] Stage 5: Hardest (steps 99–140)

By phase:

- [ ] Phase 1: Safety net — 1.1, 1.2, 1.3, 1.4 and 1.6 done]
- [ ] Phase 3: Remove global state (3.1–3.5]
- [ ] Phase 4: Restructure the core (4.1–4.7)
- [ ] Phase 5: Improve individual features (5.1–5.9]
- [ ] Phase 6: Features from Claude Code and Codex (6.1–6.15]
- [ ] Phase 7: Planning mode (7.1–7.10)
- [ ] Phase 8: Role coverage (8.1–8.20]
- [ ] Phase 9: Single combined runtime and live status (9.1–9.11)
- [ ] Phase 10: User interface (10.1–10.10]
- [ ] Phase 11: Platform support, POSIX only (11.1–11.12]
- [ ] Phase 12: Local provider independence (12.1–12.13]
- [ ] Phase 13: Performance (13.1–13.10]
- [ ] Phase 14: Configuration (14.1–14.11]
- [ ] Phase 15: Repository intelligence, architecture memory and metrics (15.1–15.15)

---

## Part M: Repository Intelligence, Architecture Memory and Engineering Metrics

This part records the enhancement proposal in
[issue #13](https://github.com/ZieganWesley/Private_Agent/issues/13),
"Implement Repository Knowledge Graph, Architecture Memory and Agent
Intelligence Layer", opened 2026-11-12 and labelled `enhancement`. Unlike Parts
A–K it is a forward-looking feature set rather than a defect list, so it is
categorised the same way here: what it asks for, what exists today, what is
missing, and the priority the issue assigns.

The issue's own status marks are kept throughout: `[x]` implemented, `[/]`
partial or in progress, `[ ]` not implemented, `[!]` implemented but needing
enhancement, a refactor or fixes.

### M1. Repository intelligence and code understanding

Turns the repository from text in a vector index into a structured model of
files, symbols and their relationships.

#### M1a. Tree-sitter repository indexing (priority: HIGH)

| Requested | Status |
|---|---|
| Tree-sitter parsing and a project-scoped repository-intelligence SQLite database | `[ ]` |
| Extract classes, interfaces, functions, methods, imports, dependencies, call relationships, inheritance relationships, entry points and configuration files | `[ ]` |
| Incremental re-indexing of modified files only, repository version awareness, symbol-change tracking between runs | `[/]` |

What exists today: project-scoped memory in SQLite, and RAG indexing with
fingerprint-based change detection (J7), so unchanged files are not re-parsed.
There is no parser: `tree-sitter` is not a dependency and nothing extracts
symbols or relationships automatically. The nearest existing feature is the
regex-based `search_workspace_symbols` declaration search (E16).

#### M1b. Symbol-aware retrieval (priority: HIGH)

| Requested API | Status |
|---|---|
| `find_symbol`, `find_definition` | `[/]` (exact declaration search only) |
| `find_references`, `find_callers`, `find_callees` | `[ ]` |
| `find_dependencies`, `trace_execution_flow` | `[ ]` |
| Symbol-aware ranking before vector retrieval, and symbol retrieval inside planning | `[ ]` |

Retrieval today is hybrid BM25 plus vector over documents (B9, E6–E10). It has
no notion of a definition, a reference or a call edge, so "who calls this"
cannot be answered from the index.

#### M1c. Repository knowledge graph (priority: HIGH)

- A graph of files, symbols, components, services, modules and APIs, stored in
  SQLite.
- Traversal, impact analysis, architectural path tracing and visual dependency
  graphs.

Status: `[ ]`. There is no graph data model; document retrieval and memory
retrieval are independent of each other.

#### M1d. Repository mapping (priority: HIGH)

- Aider-style repository maps, lightweight repository summaries, stored
  structure snapshots, context-efficient abstractions.

Status: `[/]`. The agent can summarise a repository on request, and RAG can
retrieve code and documents, but nothing is stored: there are no persistent
maps or structure snapshots.

### M2. Architecture memory

Keeps an understanding of the system itself, separately from conversations.

| Capability | Status | Notes |
|---|---|---|
| Dedicated architecture memory subsystem, independent of chat history | `[ ]` | Long-term session memory exists (A1, Part F) but is conversation-shaped |
| Automatic extraction of application architecture, layer and domain boundaries, services, APIs, data flows, event workflows, infrastructure dependencies and technology stack | `[/]` | Only on demand, through model reasoning; no extraction pipeline or architecture registry |
| Architectural decision records (decisions, tradeoffs, rejected alternatives, migrations, operational constraints) with recall across sessions | `[ ]` | Not available |
| Cross-session project awareness, architecture-aware onboarding, architecture recall injected into planning | `[/]` | Project memory and session persistence exist; no comparison between projects and no injection into plans (D5 drops most plan context) |

### M3. Memory ranking and lifecycle management

#### M3a. Memory classification (priority: MEDIUM)

| Memory type | Status |
|---|---|
| Working memory | `[/]` |
| Long-term memory | `[/]` |
| Episodic memory | `[ ]` as a typed class (episodic summaries exist) |
| Project memory classification | `[ ]` |
| Architecture memory | `[ ]` |
| User memory classification | `[ ]` (learned facts exist, untyped) |

The database stores chat history, session summaries, learned items and task
plans, but memory is not strongly typed, so ranking and lifecycle rules cannot
be applied per type.

#### M3b. Ranking engine (priority: HIGH)

- Dimensions: frequency, recency, importance, relevance, confidence and user
  feedback.

Status: `[/]`. Retrieval exists through RAG and FTS5 episode search, but there
is no explicit score, no weighting between dimensions and no prioritised
recall.

#### M3c. Promotion, decay and compression (priority: MEDIUM)

- Promote important memories, archive stale ones, apply decay policies and
  compress memories automatically.

Status: `[/]` in a narrow form only. Retention and expiry settings exist
(`conversation_retention_days`, `episode_retention_days`,
`registered_skill_retention_days`, `task_stale_after_days`,
`learned_item_expiry_days`), which prune old rows. There is no promotion path,
no decay scoring, and no compression beyond context compaction.

#### M3d. Intelligent context selection (priority: HIGH)

- Dynamic memory prioritisation, context budgeting and ranking strategies.

Status: `[/]`. RAG already reduces what is sent, and fixed per-field caps apply
(K4), but there is no per-source token budget (planned in 9.5) and no ranking
strategy.

### M4. Agent metrics and observability

| Metric group | Requested | Status |
|---|---|---|
| Tasks | Completed, failed, abandoned, retried, completion duration | `[/]`: the task workflow and `agent_task_events` exist, but there is no historical telemetry |
| Tools | Usage count, success rate, error rate, duration, reliability | `[ ]`: the run log records individual tool events, but nothing aggregates them |
| Retrieval | Retrieval effectiveness, memory hit rate, symbol retrieval success, graph utilisation, context relevance | `[ ]`: only `scripts/benchmark_rag.py` measures recall@k and MRR |
| Agent performance | Planning success rate, verification success rate, human intervention rate, autonomous completion rate, confidence scoring | `[/]`: the planning and verification workflow exists, but it is not measurable |
| Reporting | Metrics database, agent health reports, performance summaries, dashboards | `[ ]` |

`agent/session_stats.py` reports per-session tokens, speed and elapsed time at
exit. That is the whole of the current telemetry.

### M5. Intel AI-PC support

Status: deferred to a future implementation. NPU/OpenVINO acceleration,
local STT/TTS and hardware-aware routing are out of scope for the
m-section branch and remain FUTURE.

| Item | Status | Notes |
|---|---|---|
| OpenVINO NPU-accelerated embeddings for memory, repository and vector indexing | `[ ]` | Priority FUTURE; embeddings are Ollama-only today (I1a) |
| Local speech-to-text with OpenVINO/NPU execution | `[ ]` | The `media` extra already transcribes locally with `faster-whisper` on CPU |
| Local text-to-speech providers and voice interaction | `[ ]` | — |
| CPU/GPU/NPU detection and hardware-aware inference routing | `[/]` | Capability detection and `/hardware-status` exist but are Ollama-only; no NPU orchestration |

### M6. Overall status in issue #13

The issue's own summary, mapped to the sections above:

| Bucket | Items | Sections |
|---|---|---|
| Already implemented | SQLite persistent memory; hybrid BM25 + vector RAG; session persistence; project-scoped memory; planning and task workflows; MCP; sandboxed tools; capability detection; coding workflow | Parts B, D, E, I |
| Partially implemented | Repository summarisation; cross-session memory; context reduction through RAG; project awareness; verification and planning workflows | M1d, M2, M3, D6 |
| Needs enhancement or refactor | Memory ranking and prioritisation; architecture persistence; context selection strategy; repository abstraction layer; planning observability; task execution telemetry | M2, M3, M4 |
| Not yet implemented | Tree-sitter indexing; symbol graph; repository knowledge graph; architectural decision memory; agent metrics and dashboards; memory lifecycle management; NPU/OpenVINO acceleration; STT/TTS subsystems | M1, M2c, M3c, M4, M5 |

### M7. How Part M relates to the existing roadmap

- **New capability, not a duplicate:** M1 has no equivalent in Parts A–K,
  though it extends E16 (language tooling, step 8.15) and would strengthen D1
  by giving planning mode structured repository knowledge.
- **Depends on existing refactors:** M1 and M2 need `ToolContext` (3.4) and the
  memory repositories (4.5); M4 needs the typed event bus (9.1) and step
  timings (9.8); M5 needs lazy imports (11.5) and provider adapters (12.1).
- **Roadmap:** the proposal is captured as Phase 15 in Part L. It is a large body
  of work and should not start before Stage 4, because every item depends on
  the shared loop, `ToolContext` or the memory split.
