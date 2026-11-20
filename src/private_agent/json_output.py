"""JSON event output for non-interactive mode (Part F7)."""

from __future__ import annotations

import json
import sys
from typing import Any


def emit_event(event: Any) -> None:
    """Emit an event as a JSON line to stdout."""
    payload = event if isinstance(event, dict) else _to_dict(event)
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def _to_dict(event: Any) -> dict[str, Any]:
    if hasattr(event, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(event)
    if hasattr(event, "__dict__"):
        return dict(event.__dict__)
    return {"value": str(event)}
