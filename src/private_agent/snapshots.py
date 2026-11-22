"""Backward-compatible shim: implementation moved to private_agent.code_tasks.snapshots."""
import sys

from private_agent.code_tasks.snapshots import *  # noqa: F403

_impl = sys.modules['private_agent.code_tasks.snapshots']
sys.modules[__name__] = _impl
