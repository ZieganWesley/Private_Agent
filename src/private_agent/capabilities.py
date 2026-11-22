"""Backward-compatible shim: implementation moved to private_agent.agent.capabilities."""
import sys

from private_agent.agent.capabilities import *  # noqa: F403

_impl = sys.modules['private_agent.agent.capabilities']
sys.modules[__name__] = _impl
