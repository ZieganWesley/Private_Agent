# Features

## Core
- Local-first chat with Ollama, LM Studio, and llama.cpp.
- Online providers with permission gating.
- Tool use with filesystem, network, media, memory, and planning tools.
- Context compaction and bounded history.
- Local RAG indexing and retrieval.
- Skills and slash commands.
- Persistent memory and session summaries.
- Permission modes and sandboxed execution.
- Coding workspace isolation and test running.

## UI
- Rich terminal UI with live status bar.
- Accessible mode via `PRIVATE_AGENT_ACCESSIBLE=1`.
- NO_COLOR support.
- Context gauge and turn summaries.
- UI-only check trigger (`--ui-check`).

## Security
- Workspace boundary enforcement.
- Permission modes: auto, prompt, allowlist.
- Network permission persistence.
- Sandbox command isolation.

## Testing
- Unit and integration tests under `tests/`.
- UI accessibility and boundary tests.
- Sandbox and provider tests.
