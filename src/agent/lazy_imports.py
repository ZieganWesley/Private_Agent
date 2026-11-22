"""Lazy import helpers used to defer heavy module loading at startup."""

from __future__ import annotations

import importlib
from typing import Any


class _LazyProxy:
    __slots__ = ("_module", "_attr", "_value", "_loaded")

    def __init__(self, module_path: str, attr_name: str):
        self._module = module_path
        self._attr = attr_name
        self._value = None
        self._loaded = False

    def _load(self):
        if not self._loaded:
            module = importlib.import_module(self._module)
            self._value = getattr(module, self._attr)
            self._loaded = True
        return self._value

    def __getattr__(self, item: str):
        return getattr(self._load(), item)

    def __call__(self, *args: Any, **kwargs: Any):
        return self._load()(*args, **kwargs)

    def __contains__(self, item: Any) -> bool:
        return item in self._load()

    def __getitem__(self, item: Any) -> Any:
        return self._load()[item]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._load()[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())

    def items(self):
        return self._load().items()

    def get(self, *args: Any, **kwargs: Any):
        return self._load().get(*args, **kwargs)

    def pop(self, *args: Any, **kwargs: Any):
        return self._load().pop(*args, **kwargs)

    def add(self, *args: Any, **kwargs: Any):
        return self._load().add(*args, **kwargs)

    def discard(self, *args: Any, **kwargs: Any):
        return self._load().discard(*args, **kwargs)

    def __repr__(self) -> str:
        return repr(self._load())


_LAZY_SPECS: dict[str, tuple[str, str]] = {
    "AVAILABLE_TOOLS": ("tools", "AVAILABLE_TOOLS"),
    "LOCAL_MEDIA_TOOL_NAMES": ("tools", "LOCAL_MEDIA_TOOL_NAMES"),
    "LOCAL_ONLY_NETWORK_TOOL_NAMES": ("tools", "LOCAL_ONLY_NETWORK_TOOL_NAMES"),
    "MCP_TOOL_NAMES": ("tools", "MCP_TOOL_NAMES"),
    "MCP_TOOL_SOURCES": ("tools", "MCP_TOOL_SOURCES"),
    "MCP_TOOL_TRANSPORTS": ("tools", "MCP_TOOL_TRANSPORTS"),
    "NETWORK_TOOL_NAMES": ("tools", "NETWORK_TOOL_NAMES"),
    "approved_tool_invocation": ("tools", "approved_tool_invocation"),
    "clear_schema_reads": ("tools", "clear_schema_reads"),
    "close_mcp_sandbox_dirs": ("tools", "close_mcp_sandbox_dirs"),
    "close_outbound_http_clients": ("tools", "close_outbound_http_clients"),
    "describe_tool_catalog": ("tools", "describe_tool_catalog"),
    "get_tool_context": ("tools", "get_tool_context"),
    "has_internet_connection": ("tools", "has_internet_connection"),
    "load_configured_mcp_tools": ("tools", "load_configured_mcp_tools"),
    "ollama_client_kwargs": ("tools", "ollama_client_kwargs"),
    "set_tool_context": ("tools", "set_tool_context"),
    "approve_local_capture": ("tools.media", "approve_local_capture"),
    "clear_captured_images": ("tools.media", "clear_captured_images"),
    "_format_interactive_help": ("agent.interactive", "format_help"),
    "handle_session_command": ("agent.interactive", "handle_session_command"),
    "include_file_context": ("agent.interactive", "include_file_context"),
    "include_folder_context": ("agent.interactive", "include_folder_context"),
    "load_custom_slash_commands": ("agent.interactive", "load_custom_slash_commands"),
    "prompt_user_input": ("agent.interactive", "prompt_user_input"),
    "_discover_available_local_runtimes": ("agent.model_selection", "_discover_available_local_runtimes"),
    "_effective_context_window": ("agent.model_selection", "_effective_context_window"),
    "_make_selected_local_chat_model": ("agent.model_selection", "_make_selected_local_chat_model"),
    "inspect_model_capabilities": ("agent.model_selection", "inspect_model_capabilities"),
    "inspect_openai_compatible_model_capabilities": ("agent.model_selection", "inspect_openai_compatible_model_capabilities"),
    "select_online_model": ("agent.model_selection", "select_online_model"),
    "_display_tool_error": ("agent.tool_display", "_display_tool_error"),
    "_display_tool_request": ("agent.tool_display", "_display_tool_request"),
    "_display_tool_result": ("agent.tool_display", "_display_tool_result"),
    "_local_image_input_error": ("agent.tool_display", "_local_image_input_error"),
    "_redact_sensitive_values": ("agent.tool_display", "_redact_sensitive_values"),
    "_redact_tool_arguments": ("agent.tool_display", "_redact_tool_arguments"),
    "_tool_log_redactions": ("agent.tool_display", "_tool_log_redactions"),
    "_tool_result_is_error": ("agent.tool_display", "_tool_result_is_error"),
    "_append_captured_images_from_tool_results": ("agent.tool_display", "_append_captured_images_from_tool_results"),
    "CodeTaskWorkspace": ("code_tasks", "CodeTaskWorkspace"),
    "checkpoint_code_task": ("code_tasks", "checkpoint_code_task"),
    "finalize_code_task": ("code_tasks", "finalize_code_task"),
    "run_project_unit_tests": ("code_tasks", "run_project_unit_tests"),
}




class _LazyModule:
    def __init__(self, module_path: str):
        self._module_path = module_path
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._module_path, package=__package__)
        return self._module

    def __getattr__(self, item: str):
        return getattr(self._load(), item)

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            setattr(self._load(), key, value)

    def __delattr__(self, item: str) -> None:
        delattr(self._load(), item)

    def __repr__(self) -> str:
        return repr(self._load())

def install_lazy_globals(namespace: dict) -> None:
    for name, (module, attr) in _LAZY_SPECS.items():
        namespace[name] = _LazyProxy(module, attr)
    # _planning_search_citations is now imported directly; keep spec removed.
    namespace.setdefault("_planning_search_citations", None)
    namespace["code_tasks_module"] = _LazyModule("code_tasks")
