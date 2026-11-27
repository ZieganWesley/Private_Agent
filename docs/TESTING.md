# Testing

```bash
pytest -q
```

UI-only:
```bash
python scripts/ui_check.py
```

Notes:
- Sandbox network tests may skip when `bwrap` is unavailable.
- Keep tests passing; do not weaken assertions.
