"""The /status panel, built from the session's live state.

Keeping the assembly here leaves the session loop with one call and lets
the UI self-check (``python scripts/ui_check.py``) and tests build the same panel
without a running session.
"""

from __future__ import annotations

from typing import Any

from ui.status_view import StatusSection, status_panel


def _context_rows(context_value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(context_value, dict) or not context_value.get("total_tokens"):
        return (("Last turn", "not measured yet"),)
    rows = [
        ("Window", f"{context_value['total_tokens']} tokens"),
        (
            "Used last turn",
            f"{context_value.get('sum_used', 0)} tokens "
            f"({float(context_value.get('usage_percent', 0.0)):.1f}%)",
        ),
    ]
    truncated = context_value.get("truncated") or []
    if truncated:
        rows.append(("Trimmed", ", ".join(str(name) for name in truncated)))
    return tuple(rows)


def _network_consent_label(state: Any) -> str:
    """Describe the session's one-time full-mode internet decision."""
    if not isinstance(state, dict):
        return "not requested yet"
    if state.get("granted"):
        return "granted for this session"
    if state.get("denied"):
        return "denied for this session"
    return "not requested yet"


def session_status_panel(
    *,
    provider_type: Any,
    local_runtime_name: str,
    selected_model: str,
    permission_mode: str,
    workspace: str,
    rag_description: str,
    queued_context_present: bool,
    queued_context_files: int,
    queued_context_bytes: int,
    context_value: Any,
    network_consent_state: Any = None,
) -> Any:
    """Return the one renderable behind ``/status``."""
    provider_label = (
        f"Local ({local_runtime_name})" if provider_type == "local" else "Online"
    )
    session_rows: list[tuple[str, str]] = [
        ("Provider kind", str(provider_type).title()),
        ("Provider", provider_label),
        ("Model", selected_model),
        ("Permission mode", permission_mode.title()),
    ]
    if permission_mode.strip().lower() == "full":
        session_rows.append(
            ("Internet access", _network_consent_label(network_consent_state))
        )
    session_rows.append(("Workspace", workspace))
    return status_panel(
        "Session Status",
        (
            StatusSection(
                "Session",
                tuple(session_rows),
                ("Setting", "Value"),
            ),
            StatusSection(
                "Context", _context_rows(context_value), ("Measure", "Value")
            ),
            StatusSection(
                "Workspace",
                (
                    ("RAG", rag_description),
                    (
                        "Queued folder context",
                        f"{queued_context_files} file(s), {queued_context_bytes:,} bytes"
                        if queued_context_present
                        else "none",
                    ),
                ),
                ("Source", "State"),
            ),
        ),
    )


__all__ = ["session_status_panel"]
