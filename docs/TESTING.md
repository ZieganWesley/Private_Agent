# Testing

```bash
pytest -q
```

UI-only:
```bash
private-agent --ui-check
```

Notes:
- Sandbox network tests may skip when `bwrap` is unavailable.
- Keep tests passing; do not weaken assertions.
