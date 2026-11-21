# Debug and Verbose Logging Audit

## Findings
- Debug logging is opt-in via `DEBUG_LOG_ENABLED` and writes owner-only JSONL
  files under `~/.private_agent/logs`.
- Sensitive fields such as API keys, tokens, prompts, arguments, raw content,
  and credentials are redacted or omitted by `run_logging._safe_field_value`.
- Console output is logged as character count rather than full text to avoid
  leaking conversation content.
- Verbose startup prints diagnostics to the terminal; debug mode records run
  metadata to the log file.

## Fixes
- No product behavior changed.
- Logging behavior is covered by `tests/unit/test_config.py` and source-hygiene
  tests.

## Follow-ups
- Add structured debug events around tool execution and model selection where
  useful without logging secrets or full prompts.
