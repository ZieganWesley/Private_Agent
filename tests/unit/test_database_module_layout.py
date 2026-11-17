"""Guards for the split of PersistentMemory into per-concern mixins."""

import inspect
import pathlib

import pytest

import private_agent.database as database_package
from private_agent.database import (
    connection,
    conversations,
    episodes,
    learning,
    maintenance,
    memory,
    skills,
    task_actions,
    tasks,
)

DATABASE_DIR = pathlib.Path(inspect.getfile(memory)).parent


@pytest.mark.parametrize(
    ("method", "mixin"),
    [
        (memory.PersistentMemory.__init__, connection.ConnectionMixin),
        (memory.PersistentMemory._create_table, connection.ConnectionMixin),
        (memory.PersistentMemory.close, connection.ConnectionMixin),
        (memory.PersistentMemory.create_task_plan, tasks.TaskMixin),
        (memory.PersistentMemory.begin_task_action, task_actions.TaskActionMixin),
        (memory.PersistentMemory.complete_task, tasks.TaskMixin),
        (memory.PersistentMemory.add_learned_item, learning.LearningMixin),
        (memory.PersistentMemory.get_relevant_learned_items, learning.LearningMixin),
        (memory.PersistentMemory.save_message, conversations.ConversationMixin),
        (memory.PersistentMemory.prune_history, conversations.ConversationMixin),
        (
            memory.PersistentMemory.search_relevant_episodic_memories,
            episodes.EpisodeMixin,
        ),
        (memory.PersistentMemory.register_skill_file, skills.SkillMixin),
        (memory.PersistentMemory.maintain_database, maintenance.MaintenanceMixin),
        (memory.PersistentMemory.backup_to, maintenance.MaintenanceMixin),
    ],
)
def test_repository_methods_live_in_their_own_mixin(method, mixin):
    assert method.__qualname__.split(".")[0] == mixin.__name__
    assert inspect.getmodule(method) is inspect.getmodule(mixin)


def test_composed_class_keeps_the_public_import_path():
    assert database_package.PersistentMemory is memory.PersistentMemory
    assert memory.PersistentMemory.__mro__[1:9] == (
        connection.ConnectionMixin,
        tasks.TaskMixin,
        task_actions.TaskActionMixin,
        learning.LearningMixin,
        conversations.ConversationMixin,
        episodes.EpisodeMixin,
        skills.SkillMixin,
        maintenance.MaintenanceMixin,
    )


def test_schema_allowlist_stays_on_the_composed_class():
    assert "agent_task_actions" in memory.PersistentMemory.SCHEMA_TABLE_ALLOWLIST


def test_database_modules_stay_below_the_previous_monolith_size():
    oversized = {
        path.name: len(path.read_text().splitlines())
        for path in DATABASE_DIR.glob("*.py")
        if len(path.read_text().splitlines()) > 1250
    }
    assert oversized == {}
