"""Data containers handed between startup, provider selection and the session loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..database import PersistentMemory


@dataclass(frozen=True)
class SessionSetup:
    """State produced by the interactive startup sequence."""

    memory: PersistentMemory
    local_runtimes: list[Any]
    hardware_status_by_url: dict[str, dict[str, Any]]
    isolation_warning: str | None
    vectorstore: Any
    docs_input: str
    is_rag_active: bool
    rag_signature: Any
    rag_last_check: float
    skills_folder_input: str
    loaded_skills: dict[str, Any]
    mode_label: str
    session_id: str
    network_consent_state: dict[str, bool]
    thinking_enabled: bool
    thinking_effort: str
    permission_mode: str


@dataclass(frozen=True)
class ProviderSelection:
    """State produced by one pass of the provider/model selection menu."""

    provider_type: str
    selected_model: str
    capabilities: dict[str, Any]
    llm: Any
    llm_with_tools: Any
    tools_list: list[Any]
    selected_local_runtime: Any
    include_private_context: bool
    startup_memory_context: str
    chat_history: list[Any]
    context_summary: str
    session_summary_saved: bool
    available_models: list[Any]


@dataclass
class ProviderPreference:
    """Choices that must survive a failed selection pass."""

    online_preference: bool | None = None
    online_selection: dict[str, Any] | None = field(default=None)
    exit_requested: bool = False
