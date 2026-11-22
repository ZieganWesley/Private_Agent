"""Backward-compatible shim: implementation moved to private_agent.core.hooks."""
import sys

from private_agent.core.hooks import *  # noqa: F403

_impl = sys.modules['private_agent.core.hooks']
sys.modules[__name__] = _impl
