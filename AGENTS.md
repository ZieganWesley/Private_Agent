# AGENTS.md — Private Agent

## Project Context
Private Agent is a local-first, privacy-focused terminal AI agent. It runs chat
models through local runtimes (Ollama, LM Studio, llama.cpp) or approved
OpenAI-compatible online endpoints, keeps memory and task plans in SQLite,
supports local RAG, skills, and permission-gated tools. Nothing leaves the
machine unless an online provider or research request is approved.

## Source Layout
- `src/cli.py` — entry point and CLI flags.
- `src/agent/runtime.py` — session loop, provider selection, turn execution.
- `src/agent/*.py` — permissions, providers, prompts, compaction, memory context.
- `src/ui/` — Rich terminal UI: console, theme, status bar, spinner, transcript.
- `src/tools/` — filesystem, network, media, memory, planning, RAG tools.
- `src/rag/` — indexing and retrieval (lazy imports).
- `src/database/` — SQLite memory.
- `src/sandbox/` — Linux/macOS command isolation.
- `src/code_tasks/` — isolated coding workspaces and tests.
- `src/config/` — settings and defaults.

## Implementing Features
1. Keep heavy imports lazy; import inside functions or behind `__getattr__`.
2. Respect permission modes and workspace boundaries; never bypass sandbox.
3. Add tools through `src/tools` and register in `AVAILABLE_TOOLS`.
4. Keep UI output through `src/ui` facades; do not call `console.print` directly
   in business modules.
5. Preserve exception handling; report failures concisely and continue safely.
6. Bound in-memory context: trim `chat_history`, compact when threshold is met,
   and avoid storing unbounded tool output.
7. Add tests in `tests/unit` or `tests/integration` and run `pytest -q`.

## Feature List
See [docs/FEATURES.md](docs/FEATURES.md).

## Performance and Memory
- Lazy-load optional/heavy modules (RAG, media, skills, learning).
- Use sets for membership checks and avoid copies in hot paths.
- Keep context compaction enabled; cap history and truncate stored messages.
- Prefer bounded queues/deques and explicit cleanup for caches.
- Profile with `python -X importtime` and `cProfile` before optimizing.

## Developer Checks
- `pytest -q` — full suite.
- `python scripts/ui_check.py` — UI-only self-check (development aid, not shipped).
- `private-agent --doctor` — environment diagnostics.
