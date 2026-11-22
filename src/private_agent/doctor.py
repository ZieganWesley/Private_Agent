"""Backward-compatible shim: implementation moved to private_agent.diagnostics.doctor."""
import sys

from private_agent.diagnostics.doctor import *  # noqa: F403

_impl = sys.modules['private_agent.diagnostics.doctor']
sys.modules[__name__] = _impl
