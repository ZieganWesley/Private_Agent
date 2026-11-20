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
