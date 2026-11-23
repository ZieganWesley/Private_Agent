# Architecture

See [AGENTS.md](../AGENTS.md) for source layout and implementation guidance.

Core flow:
1. CLI starts session.
2. Provider selected (local or online).
3. Turn loop builds context, runs tools, compacts history.
4. UI renders status and results.
5. Memory and plans persist in SQLite.
