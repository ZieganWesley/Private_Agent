"""Compatibility facade for the agent runtime and its public helpers.

The runtime was split into focused modules (``streaming``, ``task_commands``,
``memory_context`` and friends). This package keeps every helper importable from
``agent`` by falling back to the module that now owns the name.
"""

from . import learning as _learning
from . import memory_context as _memory_context
from . import model_selection as _model_selection
from . import runtime as _runtime
from . import streaming as _streaming
from . import task_commands as _task_commands
from . import task_lease as _task_lease
from . import tool_display as _tool_display

_HELPER_MODULES = (
    _tool_display,
    _model_selection,
    _streaming,
    _learning,
    _memory_context,
    _task_commands,
    _task_lease,
)

__all__ = list(
    dict.fromkeys(
        name
        for module in (_runtime, *_HELPER_MODULES)
        for name in vars(module)
        if not name.startswith("_")
    )
)


def __getattr__(name: str):
    for module in (_runtime, *_HELPER_MODULES):
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
