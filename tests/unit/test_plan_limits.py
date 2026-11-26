"""Plan limits live once and both the schema and database enforce them (K8/14.9)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import plan_limits
from database.tasks import TaskMixin
from tools.schemas import (
    CreateTaskPlanInput,
    ReviseTaskPlanInput,
    TaskPlanStepInput,
)


def _plan(**overrides) -> dict:
    plan = {
        "goal": "Ship the feature",
        "task_type": "coding",
        "steps": [
            {
                "step_id": "step-1",
                "description": "Do the work",
                "dependencies": [],
                "validation": [],
                "proof": [],
                "risks": [],
                "edge_cases": [],
            }
        ],
    }
    plan.update(overrides)
    return plan


def _steps(count: int, description: str = "Do the work") -> list[dict]:
    return [
        {
            "step_id": f"step-{index}",
            "description": description,
            "dependencies": [],
        }
        for index in range(1, count + 1)
    ]


def test_limit_values_are_centralized() -> None:
    assert plan_limits.PLAN_MAX_GOAL_CHARS == 2000
    assert plan_limits.PLAN_MAX_STEPS == 20
    assert plan_limits.PLAN_MAX_STEP_DESCRIPTION_CHARS == 1000
    assert plan_limits.PLAN_MAX_STEP_DEPENDENCIES == 20
    assert plan_limits.PLAN_MAX_STEP_ID_CHARS == 64
    assert plan_limits.PLAN_MAX_TEXT_ITEMS == 10
    assert plan_limits.PLAN_MAX_TEXT_ITEM_CHARS == 500
    assert plan_limits.PLAN_MAX_REFERENCE_CHARS == 1500
    assert plan_limits.PLAN_MAX_CITATIONS == 10
    assert plan_limits.PLAN_MAX_CITATION_CHARS == 10000


def _max_length(field) -> int | None:
    return next(
        (
            metadata.max_length
            for metadata in field.metadata
            if hasattr(metadata, "max_length")
        ),
        None,
    )


def test_schemas_use_the_shared_limits() -> None:
    fields = CreateTaskPlanInput.model_fields
    assert _max_length(fields["goal"]) == plan_limits.PLAN_MAX_GOAL_CHARS
    assert _max_length(fields["steps"]) == plan_limits.PLAN_MAX_STEPS
    assert _max_length(fields["assumptions"]) == plan_limits.PLAN_MAX_TEXT_ITEMS
    assert _max_length(fields["research_references"]) == plan_limits.PLAN_MAX_CITATIONS
    step_fields = TaskPlanStepInput.model_fields
    assert (
        _max_length(step_fields["description"])
        == plan_limits.PLAN_MAX_STEP_DESCRIPTION_CHARS
    )
    assert (
        _max_length(step_fields["dependencies"])
        == plan_limits.PLAN_MAX_STEP_DEPENDENCIES
    )
    assert _max_length(step_fields["proof"]) == plan_limits.PLAN_MAX_TEXT_ITEMS
    assert "expected_revision" in ReviseTaskPlanInput.model_fields


def test_schema_enforces_goal_and_step_boundaries() -> None:
    CreateTaskPlanInput(**_plan(goal="g" * plan_limits.PLAN_MAX_GOAL_CHARS))
    with pytest.raises(ValidationError):
        CreateTaskPlanInput(**_plan(goal="g" * (plan_limits.PLAN_MAX_GOAL_CHARS + 1)))

    CreateTaskPlanInput(**_plan(steps=_steps(plan_limits.PLAN_MAX_STEPS)))
    with pytest.raises(ValidationError):
        CreateTaskPlanInput(**_plan(steps=_steps(plan_limits.PLAN_MAX_STEPS + 1)))


def test_schema_enforces_reference_and_citation_boundaries() -> None:
    step = _steps(1)
    at_limit = ["r" * 1000 for _ in range(plan_limits.PLAN_MAX_CITATIONS)]
    CreateTaskPlanInput(**_plan(steps=step, research_references=at_limit))

    with pytest.raises(ValidationError):
        CreateTaskPlanInput(
            **_plan(
                steps=step,
                research_references=at_limit + ["extra"],
            )
        )
    with pytest.raises(ValidationError):
        CreateTaskPlanInput(
            **_plan(
                steps=step,
                research_references=["r" * (plan_limits.PLAN_MAX_REFERENCE_CHARS + 1)],
            )
        )
    with pytest.raises(ValidationError):
        CreateTaskPlanInput(
            **_plan(
                steps=step,
                research_references=["r" * 1500 for _ in range(7)],
            )
        )


def test_database_enforces_the_same_step_boundaries() -> None:
    normalized = TaskMixin._normalize_plan_steps(_steps(plan_limits.PLAN_MAX_STEPS))
    assert len(normalized) == plan_limits.PLAN_MAX_STEPS

    with pytest.raises(ValueError, match=f"1 and {plan_limits.PLAN_MAX_STEPS}"):
        TaskMixin._normalize_plan_steps(_steps(plan_limits.PLAN_MAX_STEPS + 1))

    TaskMixin._normalize_plan_steps(
        _steps(1, "d" * plan_limits.PLAN_MAX_STEP_DESCRIPTION_CHARS)
    )
    with pytest.raises(ValueError, match="1 to 1000"):
        TaskMixin._normalize_plan_steps(
            _steps(1, "d" * (plan_limits.PLAN_MAX_STEP_DESCRIPTION_CHARS + 1))
        )


def test_database_enforces_the_same_citation_boundaries() -> None:
    at_limit = ["r" * 1000 for _ in range(plan_limits.PLAN_MAX_CITATIONS)]
    assert TaskMixin._normalize_research_references(at_limit) == at_limit

    with pytest.raises(ValueError, match="At most 10 research references"):
        TaskMixin._normalize_research_references(at_limit + ["extra"])
    with pytest.raises(ValueError, match="1 to 1500"):
        TaskMixin._normalize_research_references(
            ["r" * (plan_limits.PLAN_MAX_REFERENCE_CHARS + 1)]
        )
    with pytest.raises(ValueError, match="10000-character plan budget"):
        TaskMixin._normalize_research_references(["r" * 1500 for _ in range(7)])
