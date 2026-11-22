"""Backward-compatible shim: implementation moved to private_agent.core.events."""
import sys

from private_agent.core.events import *  # noqa: F403

_impl = sys.modules['private_agent.core.events']
sys.modules[__name__] = _impl
