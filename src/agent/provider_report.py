"""Inspection output for resolved providers and endpoint telemetry (P6)."""

from __future__ import annotations

from typing import Any

from providers.endpoint_telemetry import format_endpoint_report

_EMBEDDING_ROUTE: dict[str, Any] = {}


def record_embedding_route(provider: str, model: str, base_url: str) -> None:
    """Remember the resolved embedding route for inspection (no content)."""
    _EMBEDDING_ROUTE.update(
        {
            "provider": str(provider or "none"),
            "model": str(model or ""),
            "base_url": str(base_url or ""),
        }
    )


def embedding_route() -> dict[str, Any]:
    return dict(_EMBEDDING_ROUTE)


def reset_provider_report() -> None:
    _EMBEDDING_ROUTE.clear()


def handle_provider_command(text: str, print_fn) -> bool:
    """Handle ``/provider`` and ``/provider usage``; return whether handled."""
    if str(text or "").strip().lower() not in {"/provider", "/provider usage"}:
        return False
    print_provider_report(print_fn)
    return True


def print_provider_report(print_fn) -> None:
    """Print the inspection report, swallowing optional-surface failures."""
    try:
        print_fn(format_provider_report())
    except Exception as exc:  # noqa: BLE001 - optional inspection surface
        print_fn(f"[warning][Provider] {exc}[/warning]")


def format_provider_report() -> str:
    route = embedding_route()
    if route:
        lines = [
            "Provider routing:",
            f"  embeddings: provider={route.get('provider', 'none')} "
            f"model={route.get('model', '') or '(default)'} "
            f"base_url={route.get('base_url', '') or '(configured)'}",
        ]
    else:
        lines = ["Provider routing: embeddings not resolved yet."]
    lines.append(format_endpoint_report())
    return "\n".join(lines)


__all__ = [
    "embedding_route",
    "format_provider_report",
    "handle_provider_command",
    "print_provider_report",
    "record_embedding_route",
    "reset_provider_report",
]
