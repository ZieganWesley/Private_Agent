"""Task orchestrator over capability packs (Part F4)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskStep:
    index: int
    description: str
    packs: set[str] = field(default_factory=set)
    status: str = "pending"  # pending/running/done/failed
    result: str = ""


@dataclass
class TaskPlan:
    goal: str
    steps: list[TaskStep] = field(default_factory=list)
    revision: int = 0

    def add_step(self, description: str, packs: set[str] | None = None) -> TaskStep:
        step = TaskStep(
            index=len(self.steps), description=description, packs=packs or set()
        )
        self.steps.append(step)
        return step

    def mark(self, index: int, status: str, result: str = "") -> None:
        if 0 <= index < len(self.steps):
            self.steps[index].status = status
            self.steps[index].result = result

    def is_complete(self) -> bool:
        return bool(self.steps) and all(s.status in ("done",) for s in self.steps)

    def summary(self) -> str:
        lines = [f"Goal: {self.goal}"]
        for s in self.steps:
            lines.append(f"  [{s.status}] {s.index}. {s.description}")
        return "\n".join(lines)


def build_plan(goal: str, step_descriptions: list[str]) -> TaskPlan:
    plan = TaskPlan(goal=goal)
    for desc in step_descriptions:
        plan.add_step(desc)
    return plan


def verify_step(step: TaskStep, evidence: str = "") -> bool:
    """Auto-verify a step; coding steps require test pass, research steps require sources."""
    if "test" in step.description.lower() or "code" in step.description.lower():
        return "pass" in evidence.lower() or "ok" in evidence.lower()
    if "research" in step.description.lower():
        return "http" in evidence.lower() or "source" in evidence.lower()
    return True
