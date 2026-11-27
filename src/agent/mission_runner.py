"""Drive a durable mission across automatic model iterations."""

from __future__ import annotations

from .missions import MissionController, MissionLeaseError


class MissionAutoRunner:
    """Plan -> work -> verify -> next iteration, using model output evidence."""

    def __init__(self, controller: MissionController, owner: str):
        self.controller = controller
        self.owner = owner
        if not controller.memory.acquire_mission_lease(controller.mission_id, owner):
            raise MissionLeaseError(
                "Another controller is already running this mission."
            )

    def next_prompt(self) -> str | None:
        if not self.controller.begin_iteration():
            self.controller.release_lease(self.owner)
            return None
        step = self.controller.next_step()
        if step is None:
            self.controller.release_lease(self.owner)
            return None
        return self._prompt(step)

    def _prompt(self, step: dict) -> str:
        mission = self.controller.mission()
        acceptance = ", ".join(step.get("acceptance", ())) or "the requested outcome"
        return (
            f"[Mission {mission['mission_id']} · iteration "
            f"{mission['iteration']}/{mission['max_iterations']}] "
            f"Goal: {mission['goal']}. Current step "
            f"'{step['step_id']}': {step['description']}. "
            f"Acceptance: {acceptance}. Work this step with tools, verify the "
            "acceptance criteria, then reply with the evidence and no tool "
            "calls when the step is complete."
        )

    def resume_prompt(self) -> str | None:
        """Return the current step's prompt after a pause, without a new iteration."""
        mission = self.controller.mission()
        if mission["status"] != "active":
            return None
        step = self.controller.next_step()
        return self._prompt(step) if step is not None else None

    def observe(
        self, text: str, *, had_tool_calls: bool, interrupted: bool = False
    ) -> str | None:
        """Consume one completed model turn and return the next prompt."""
        mission = self.controller.mission()
        if mission["status"] in {"stopped", "failed"}:
            self.controller.release_lease(self.owner)
            return None
        if mission["status"] in {"paused", "waiting_user"}:
            # Keep the lease so /resume can continue this mission.
            return None
        step = self.controller.next_step()
        if step is None:
            if not self.controller.has_pending_steps():
                self.controller.complete_mission(
                    text.strip()[:1000] or "All mission steps completed.",
                    text.strip()[:8000] or "Every step reported completion.",
                )
            self.controller.release_lease(self.owner)
            return None
        evidence = str(text or "").strip()
        if interrupted or had_tool_calls or not evidence:
            return self._prompt(step)
        self.controller.complete_step(step["step_id"], evidence[:4000])
        if not self.controller.has_pending_steps():
            self.controller.complete_mission(evidence[:1000], evidence[:8000])
            self.controller.release_lease(self.owner)
            return None
        if not self.controller.begin_iteration():
            self.controller.release_lease(self.owner)
            return None
        next_step = self.controller.next_step()
        if next_step is None:
            self.controller.release_lease(self.owner)
            return None
        return self._prompt(next_step)


__all__ = ["MissionAutoRunner"]
