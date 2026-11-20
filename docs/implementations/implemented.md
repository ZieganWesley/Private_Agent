# Implemented: Part G — User Interface

Implemented on branch `g-tui-ui` (from `main` at `70f02c7`).

## Completed items

- **G1** No way out of the online-provider loop: added an `Exit` option (0) to the
  provider menu and an `exit_requested` flag on `ProviderPreference`.
- **G2** Warning about its own default folder: `initialize_knowledge_base` now creates
  a missing directory instead of warning.
- **G3** Spinner after skipping: the "Indexing local knowledge base..." spinner only
  runs when a docs path was actually provided.
- **G5** Markdown rendering: non-streamed replies render through `rich.markdown.Markdown`.
- **G6** Coloured diffs: tool results that look like diffs are coloured (+ green, -
  red, @@ cyan).
- **G11** Expandable tool output: `/expand` shows the full stored result; results are
  recorded in `tool_display`.
- **G15** Consistent message style: shared `Console` with a named theme
  (info/success/warning/error/tool/user) and a light-terminal option.
- **G16** One shared console and theme: `agent/console.py` exposes the theme.
- **G17** Informative prompt: bottom toolbar shows model and command hint.
- **G18** Separation between turns: a dim `Rule` is printed before each prompt.
- **G19** Welcome screen: version, workspace and `/help` hint printed at ready.
- **G20** Readable tool requests: `name · key=value` instead of `repr(dict)`.
- **G21** Per-turn details: existing latency line retained; generation rate shown.
- **G22** Status symbols: ✓/✗ markers on tool result panels.
- **G23** Grouped `/help`: commands grouped by Chat/Files/Plans/Memory/Model/Session.

## Partially addressed

- **G10** Multi-line input: Alt+Enter / Ctrl+J already supported via
  `build_prompt_key_bindings`.
- **G8** Live status: phase-aware spinner and context gauge already exist
  (status.py); full running/finished tool panel still open.

## Not addressed this pass (deferred)

- G4 question ordering, G7 startup flags, G12 approval prompts, G13 arrow menus,
  G14 reply/diagnostic mixing, G24 narrow-terminal wrapping.
- G9 stop one turn (already handled by step 6.4).
