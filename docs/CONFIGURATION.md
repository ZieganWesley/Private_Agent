# Configuration

Configuration lives in `src/config/settings.py` and environment variables.

Key toggles:
- `PRIVATE_AGENT_ACCESSIBLE=1` — accessible UI.
- `NO_COLOR=1` — disable color.
- `PRIVATE_AGENT_UI_CHECK=1` — enable UI check.
- `CONTEXT_COMPACTION_THRESHOLD` — compaction trigger.
- `MAX_HISTORY_MESSAGES` — hard history cap.
- `MAX_CONTEXT_TOKENS` — context budget.

See `src/config/settings.py` for defaults and full list.
