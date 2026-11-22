"""Backward-compatible shim: implementation moved to private_agent.agent.context_manager."""
import sys

from private_agent.agent.context_manager import *  # noqa: F403

_impl = sys.modules['private_agent.agent.context_manager']
sys.modules[__name__] = _impl
