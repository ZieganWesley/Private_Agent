"""Backward-compatible shim: implementation moved to private_agent.config.settings."""
import sys

from private_agent.config.settings import *  # noqa: F403

_impl = sys.modules['private_agent.config.settings']
sys.modules[__name__] = _impl
