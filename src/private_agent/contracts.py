"""Backward-compatible shim: implementation moved to private_agent.core.contracts."""
import sys

from private_agent.core.contracts import *  # noqa: F403

_impl = sys.modules['private_agent.core.contracts']
sys.modules[__name__] = _impl
