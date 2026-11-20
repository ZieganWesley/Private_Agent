"""Tool for escalating to an additional capability pack (Part F3)."""

from langchain_core.tools import tool

from ..capabilities import enable_capability as _enable


@tool
def enable_capability(name: str) -> str:
    """Placeholder: report an additional capability pack (not yet wired)."""
    return _enable(name)
