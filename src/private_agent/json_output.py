"""Backward-compatible shim: implementation moved to private_agent.ui.json_output."""
import sys

from private_agent.ui.json_output import *  # noqa: F403

_impl = sys.modules['private_agent.ui.json_output']
sys.modules[__name__] = _impl
