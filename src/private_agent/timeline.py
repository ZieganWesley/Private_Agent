"""Backward-compatible shim: implementation moved to private_agent.agent.timeline."""
import sys

from private_agent.agent.timeline import *  # noqa: F403

_impl = sys.modules['private_agent.agent.timeline']
sys.modules[__name__] = _impl
