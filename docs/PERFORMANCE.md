# Performance and Memory

- Lazy imports reduce startup time.
- Context compaction and hard caps bound memory.
- Profiling: `python -X importtime`, `cProfile`.
- Repository indexing uses parallel parsing (see docs/profiling.md).
- Avoid unbounded caches and large stored messages.
