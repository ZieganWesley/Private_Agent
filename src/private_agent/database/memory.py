"""SQLite-backed conversation history and episodic memory."""

from .connection import ConnectionMixin
from .conversations import ConversationMixin
from .episodes import EpisodeMixin
from .learning import LearningMixin
from .maintenance import MaintenanceMixin
from .skills import SkillMixin
from .task_actions import TaskActionMixin
from .tasks import TaskMixin


class PersistentMemory(
    ConnectionMixin,
    TaskMixin,
    TaskActionMixin,
    LearningMixin,
    ConversationMixin,
    EpisodeMixin,
    SkillMixin,
    MaintenanceMixin,
):
    """SQLite-backed store for conversations, tasks, learning and episodes."""

    SCHEMA_TABLE_ALLOWLIST = {
        "chat_history",
        "agent_tasks",
        "agent_plan_revisions",
        "agent_todo_steps",
        "agent_task_events",
        "agent_task_leases",
        "agent_task_actions",
        "agent_learned_items",
        "agent_learning_settings",
    }
