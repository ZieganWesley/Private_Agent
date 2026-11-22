"""Backward-compatible shim: implementation moved to private_agent.ui.chat_ui."""
import sys

from private_agent.ui.chat_ui import *  # noqa: F403

_impl = sys.modules['private_agent.ui.chat_ui']
sys.modules[__name__] = _impl
