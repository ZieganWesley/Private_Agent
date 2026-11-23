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

## Not yet implemented (remaining)

### Part J — Performance

- **J7/J8 (partly addressed):** `rag/indexing.py` keeps parsed chunks for
  unchanged files in `_CHUNK_CACHE` (plus `clear_chunk_cache()`); semantic
  chunking still re-embeds unchanged files on refresh.

### Part K — Hard-coded values

Hard-coded values from Part K remain tracked for a dedicated configuration
pass (Phase 14): sandbox limits (K2), timeouts/leases (K3), output/context
sizes (K4), RAG values and embedding-name patterns (K5), external service
URLs (K6), import-time snapshot path (K7), duplicated plan limits (K8),
duplicated defaults (K9), and stricter-only config extensions (K10). Similar
unlisted values include network timeouts, memory-context caps, and
tool-output caps, which follow the same K3/K4 pattern and are tracked with
them.
