"""Backward-compatible shim: implementation moved to private_agent.agent.runtime_f_integration."""
import sys

from private_agent.agent.runtime_f_integration import *  # noqa: F403

_impl = sys.modules['private_agent.agent.runtime_f_integration']
sys.modules[__name__] = _impl
