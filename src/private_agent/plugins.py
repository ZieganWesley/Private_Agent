"""Backward-compatible shim: implementation moved to private_agent.core.plugins."""
import sys

from private_agent.core.plugins import *  # noqa: F403

_impl = sys.modules['private_agent.core.plugins']
sys.modules[__name__] = _impl
