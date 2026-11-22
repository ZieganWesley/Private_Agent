"""Backward-compatible shim: implementation moved to private_agent.utils.run_logging."""
import sys

from private_agent.utils.run_logging import *  # noqa: F403

_impl = sys.modules['private_agent.utils.run_logging']
sys.modules[__name__] = _impl
