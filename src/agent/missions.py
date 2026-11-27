"""Mission controller: plan, act, observe, verify, and guard.

The mission controller does not execute tools or model calls itself.  It owns the
durable state machine and tells the existing turn loop whether to continue.  A
step is complete only with evidence, and the whole mission is complete only when
every step is verified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_NO_PROGRESS_REPEATS = 3


class MissionLeaseError(RuntimeError):
    """Raised when another controller already owns the mission lease."""


@dataclass
class MissionPlanStep:
    step_id: str
    description: str
    acceptance: tuple[str, ...] = ()
    status: str = "pending"
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "acceptance": list(self.acceptance),
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass
class MissionController:
    """State machine over the durable mission row."""

    memory: Any
    mission_id: str
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES
    no_progress_repeats: int = DEFAULT_NO_PROGRESS_REPEATS
    _signatures: dict[str, int] = field(default_factory=dict, repr=False)
    _consecutive_failures: int = 0

    @classmethod
    def start(
        cls,
        memory: Any,
        session_id: str,
        goal: str,
        steps: list[dict[str, Any]] | list[MissionPlanStep],
        *,
        max_iterations: int = 20,
        tool_call_budget: int = 60,
        deadline: float | None = None,
        lease_owner: str | None = None,
    ) -> MissionController:
        normalized = []
        for index, step in enumerate(steps, start=1):
            if isinstance(step, MissionPlanStep):
                normalized.append(step.to_dict())
                continue
            if not isinstance(step, dict):
                raise TypeError("Mission steps must be mappings.")
            description = str(step.get("description", "")).strip()
            if not description:
                raise ValueError("Every mission step needs a description.")
            acceptance = step.get("acceptance", [])
            if isinstance(acceptance, str):
                acceptance = [acceptance]
            normalized.append(
                MissionPlanStep(
                    step_id=str(step.get("step_id") or f"step-{index}"),
                    description=description,
                    acceptance=tuple(
                        str(item).strip() for item in acceptance if str(item).strip()
                    ),
                ).to_dict()
            )
        if not normalized:
            raise ValueError("A mission needs at least one step.")
        mission = memory.create_mission(
            session_id,
            goal,
            max_iterations=max_iterations,
            tool_call_budget=tool_call_budget,
            deadline=deadline,
            plan={"steps": normalized},
        )
        controller = cls(memory, mission["mission_id"])
        if lease_owner and not memory.acquire_mission_lease(
            mission["mission_id"], lease_owner
        ):
            controller.fail("Another controller owns this mission lease.")
            raise MissionLeaseError(
                "Another controller is already running this mission."
            )
        controller._record("planned", {"steps": len(normalized)})
        return controller

    def _record(self, event: str, payload: dict | None = None) -> None:
        self.memory.append_mission_event(self.mission_id, event, payload or {})

    def mission(self) -> dict:
        mission = self.memory.get_mission(self.mission_id)
        if mission is None:
            raise ValueError("Mission not found.")
        return mission

    @property
    def status(self) -> str:
        return self.mission()["status"]

    def _steps(self) -> list[dict]:
        return list(self.mission().get("plan", {}).get("steps", []))

    def _set_steps(
        self,
        steps: list[dict],
        *,
        status: str | None = None,
        progress: dict | None = None,
        last_error: str | None = None,
        completion_evidence: str | None = None,
    ) -> dict:
        plan = dict(self.mission().get("plan", {}))
        plan["steps"] = steps
        return self.memory.update_mission(
            self.mission_id,
            status=status,
            plan=plan,
            progress=progress,
            last_error=last_error,
            completion_evidence=completion_evidence,
        )

    def begin_iteration(self) -> bool:
        """Return whether another iteration is allowed; fail when a guard trips."""
        mission = self.mission()
        if mission["status"] in {"completed", "failed", "stopped", "paused"}:
            return False
        if mission["status"] == "waiting_user":
            return False
        deadline = mission.get("deadline")
        if isinstance(deadline, (int, float)) and time.time() >= deadline:
            self.fail("Mission deadline reached before the next iteration.")
            return False
        if mission["iteration"] >= mission["max_iterations"]:
            self.fail("Mission iteration budget exhausted.")
            return False
        if mission["tool_calls_used"] >= mission["tool_call_budget"]:
            self.fail("Mission tool-call budget exhausted.")
            return False
        self.memory.update_mission(
            self.mission_id,
            status="active",
            iteration=mission["iteration"] + 1,
        )
        self._record("iteration_started", {"iteration": mission["iteration"] + 1})
        return True

    def next_step(self) -> dict | None:
        """Return the first pending step and mark it active."""
        steps = self._steps()
        for step in steps:
            if step.get("status") in {"pending", "active"}:
                if step.get("status") != "active":
                    step["status"] = "active"
                    self._set_steps(steps, status="active")
                    self._record(
                        "step_started",
                        {"step_id": step.get("step_id")},
                    )
                return step
        return None

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Any = None,
        *,
        success: bool,
        error: str | None = None,
    ) -> bool:
        """Record a tool result; return False when the no-progress guard trips."""
        mission = self.mission()
        used = mission["tool_calls_used"] + 1
        signature = f"{tool_name}:{arguments!r}"
        repeats = self._signatures.get(signature, 0) + 1
        self._signatures[signature] = repeats
        self._consecutive_failures = 0 if success else self._consecutive_failures + 1
        progress = dict(mission.get("progress", {}))
        progress["last_tool"] = str(tool_name)[:120]
        progress["tool_calls_used"] = used
        self.memory.update_mission(
            self.mission_id,
            tool_calls_used=used,
            progress=progress,
        )
        self._record(
            "tool_result",
            {
                "tool": str(tool_name)[:120],
                "success": bool(success),
                "error": (str(error)[:500] if error else None),
            },
        )
        if used >= mission["tool_call_budget"]:
            self.fail("Mission tool-call budget exhausted.")
            return False
        if repeats >= self.no_progress_repeats:
            self.fail(
                f"No progress: {tool_name} repeated with the same arguments "
                f"{repeats} times."
            )
            return False
        if self._consecutive_failures >= self.max_consecutive_failures:
            self.fail(f"{self._consecutive_failures} consecutive tool failures.")
            return False
        return True

    def complete_step(self, step_id: str, evidence: str) -> dict:
        """Mark one step complete; evidence is mandatory."""
        evidence = str(evidence or "").strip()
        if not evidence:
            raise ValueError("Step completion requires evidence.")
        steps = self._steps()
        for step in steps:
            if step.get("step_id") == step_id:
                step["status"] = "completed"
                step["evidence"] = evidence[:4000]
                plan = self._set_steps(steps, status="active")
                self._record(
                    "step_completed",
                    {"step_id": step_id, "evidence": evidence[:500]},
                )
                return plan
        raise ValueError(f"Unknown mission step: {step_id}")

    def complete_mission(self, summary: str, evidence: str) -> dict:
        """Complete only when every step is verified."""
        summary = str(summary or "").strip()
        evidence = str(evidence or "").strip()
        if not summary or not evidence:
            raise ValueError("Mission completion needs a summary and evidence.")
        steps = self._steps()
        incomplete = [
            step.get("step_id") for step in steps if step.get("status") != "completed"
        ]
        if incomplete:
            raise ValueError(
                "Mission still has incomplete steps: " + ", ".join(incomplete)
            )
        progress = dict(self.mission().get("progress", {}))
        progress["summary"] = summary[:4000]
        mission = self._set_steps(
            steps,
            status="completed",
            progress=progress,
            completion_evidence=evidence[:8000],
        )
        self._record(
            "completed",
            {"summary": summary[:1000], "evidence": evidence[:1000]},
        )
        return mission

    def steer(self, text: str) -> None:
        """Record a user instruction without changing mission state."""
        instruction = str(text or "").strip()
        if instruction:
            self._record("steered", {"text": instruction[:500]})

    def pause(self, reason: str = "Paused by the user.") -> dict:
        mission = self.memory.update_mission(
            self.mission_id, status="paused", last_error=reason
        )
        self._record("paused", {"reason": reason})
        return mission

    def resume(self) -> dict:
        mission = self.mission()
        if mission["status"] not in {"paused", "waiting_user"}:
            raise ValueError("Only a paused or waiting mission can resume.")
        updated = self.memory.update_mission(self.mission_id, status="active")
        self._record("resumed", {})
        return updated

    def stop(self, reason: str = "Stopped by the user.") -> dict:
        mission = self.memory.update_mission(
            self.mission_id, status="stopped", last_error=reason
        )
        self._record("stopped", {"reason": reason})
        return mission

    def fail(self, reason: str) -> dict:
        mission = self.memory.update_mission(
            self.mission_id, status="failed", last_error=reason
        )
        self._record("failed", {"reason": reason})
        return mission

    def release_lease(self, owner: str) -> bool:
        return self.memory.release_mission_lease(self.mission_id, owner)

    def has_pending_steps(self) -> bool:
        return any(step.get("status") != "completed" for step in self._steps())


__all__ = [
    "DEFAULT_MAX_CONSECUTIVE_FAILURES",
    "DEFAULT_NO_PROGRESS_REPEATS",
    "MissionController",
    "MissionLeaseError",
    "MissionPlanStep",
]
