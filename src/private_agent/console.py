"""Backward-compatible shim: implementation moved to private_agent.ui.console."""
import sys

from private_agent.ui.console import *  # noqa: F403

_impl = sys.modules['private_agent.ui.console']
sys.modules[__name__] = _impl
