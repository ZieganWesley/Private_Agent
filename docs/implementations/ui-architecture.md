# UI Architecture

The terminal UI for the Private Agent CLI lives in `src/private_agent/ui/`.
It is the **single owner** of all terminal chrome: status ribbons, spinners,
turn summaries, and labelled notices. Business modules (in particular
`agent/runtime.py`) must not call `console.print` for chrome; they go through
`SessionUI` (or the leaf helpers in this package).

## Surfaces

| Surface | Module | Purpose |
|---|---|---|
| Shared console | `ui/console.py` | Re-exports the one `Console` + palettes/builders from `private_agent.console`. |
| Theme tokens | `ui/theme.py` | Glyphs (`GAUGE_FILLED`/`GAUGE_EMPTY`), widths, separators, `TURN_LABEL`, `NO_COLOR_ENV`. |
| Context math | `ui/state.py` | `StatusState` (immutable snapshot), `context_percent`, `context_bar`, `context_style`, `context_state_line`. |
| Ribbon | `ui/ribbon.py` | `status_ribbon(state, width=)` — the canonical `model=… \| context=… \| …` line. |
| Spinner | `ui/spinner.py` | `StatusIndicator` + phase messages + reasoning extraction. |
| Status footer | `ui/status_bar.py` | `StatusBar`: Mode A (live TTY footer) / Mode B (reprint ribbon as final line). |
| Turn report | `ui/turn.py` | `TurnReport`, `GenerationStats` → the single `[Turn]` summary line. |
| Facade | `ui/session_ui.py` | `SessionUI`: the only object the runtime talks to for chrome. |

`agent/status.py` is a backward-compat shim that re-exports the above; new code
should import from `private_agent.ui`.

## SessionUI contract

`SessionUI(console=..., status_bar=..., verbose_timings=...)` is constructed once
per session in `agent/runtime.py` and bound before the main loop:

```python
session_ui = SessionUI(console=console, status_bar=STATUS_BAR,
                      verbose_timings=VERBOSE_TIMINGS)
session_ui.bind(model=selected_model, permission=permission_mode)
```

Methods the runtime uses:

| Method | When |
|---|---|
| `bind(model=, permission=, output_tokens=)` | Once at startup. |
| `set_context(used, maximum)` | After each context estimate. |
| `turn_started(user_input)` → `TurnReport` | Start of a turn (starts footer). |
| `turn_finished(report)` | End of a turn (prints `[Turn]`, stops footer). |
| `set_generation(stats)` | After a generation completes. |
| `rule()` | Print the dim separator before the prompt. |
| `ribbon_text(width=)` | Full ribbon for the prompt's `bottom_toolbar`. |
| `print(*args, **kwargs)` / `print_markup(text)` | Passthrough chrome output. |

`footer_text()` (compact `model · permission`) was removed; use `ribbon_text()`.

## Footer modes (StatusBar)

- **Mode A** (interactive TTY, colour on): a one-line `rich.live.Live` footer
  repainted on `update()`. `start()` opens it, `stop()` closes it.
- **Mode B** (non-TTY / `NO_COLOR` / dumb terminal): no live region;
  `stop()` reprints the canonical ribbon as the final line of the turn.
- **Disabled** (`status_bar=False`): `start`/`stop`/`update` are no-ops.

Call sites never branch on the mode; the mode is an internal detail of
`StatusBar`.

## Ownership rule

1. **One owner.** All terminal chrome goes through `SessionUI` (or, for leaf
   rendering, through the `ui/` helpers). `agent/runtime.py` must not call
   `console.print` for chrome. Passthroughs are `SessionUI.print` /
   `SessionUI.print_markup`.
2. **One gauge.** `context_bar`, `GAUGE_FILLED`/`GAUGE_EMPTY`, `▰`/`▱` may
   appear only under `src/private_agent/ui/` (enforced by
   `tests/unit/test_ui_boundaries.py`).
3. **One summary.** Each turn prints exactly one `[Turn]` line
   (`TURN_LABEL`), no `[Generation]`/`[Completed]`/`[Latency]`, no `Working`
   panel, no `ETA`.
4. **Ribbon last.** In Mode B the ribbon is the final line of the turn.
5. **Verbose timings.** Per-call `[Generation]`/`[Completed]` lines are gated
   behind `VERBOSE_TIMINGS`.

## Where to change

| To change… | Edit |
|---|---|
| Glyphs, widths, separators, labels | `ui/theme.py` |
| Context rounding / gauge math | `ui/state.py` (`_context_use`, `context_bar`, `context_percent`) |
| Ribbon segment order / dropping | `ui/state.py` (`StatusState.segments`) + `ui/ribbon.py` |
| Footer mode behaviour | `ui/status_bar.py` |
| Turn summary wording | `ui/turn.py` (`TurnReport.summary_line`) |
| Spinner phases / messages | `ui/spinner.py` |
| Facade API / state holders | `ui/session_ui.py` |
| Chrome call sites in the agent | `agent/runtime.py` via `session_ui.*` |
| Backward-compat re-exports | `agent/status.py` (shim) |

## Guard tests

- `tests/unit/test_ui_boundaries.py` — gauge tokens only under `ui/`, no old
  chrome labels in `runtime.py`, one `[Turn]` line per scripted turn with the
  ribbon last.
- `tests/unit/test_status.py` — ribbon/context math.