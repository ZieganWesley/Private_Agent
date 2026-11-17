"""Task plans, leases, actions, steps and validation records."""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone

from ..config import (
    MAX_SUMMARY_CHARS,
)

_LOGGER = logging.getLogger("private_agent.database.memory")


class TaskMixin:
    """Task plans, leases, actions, steps and validation records."""

    @staticmethod
    def _plan_digest(plan: dict) -> tuple[str, str]:
        plan_json = json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return plan_json, hashlib.sha256(plan_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _decode_episode_plan(
        plan_json: str | None,
        episode_id: int,
    ) -> dict | None:
        if not plan_json:
            return None
        try:
            plan = json.loads(plan_json)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            _LOGGER.warning(
                "Skipping malformed structured episode plan id=%s", episode_id
            )
            return None
        if (
            not isinstance(plan, dict)
            or not isinstance(plan.get("steps"), list)
            or any(not isinstance(step, dict) for step in plan["steps"])
        ):
            _LOGGER.warning(
                "Skipping invalid structured episode plan id=%s", episode_id
            )
            return None
        return plan

    @staticmethod
    def _decode_task_plan(plan_json: str, task_id: str, revision: int) -> dict:
        try:
            plan = json.loads(plan_json)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
            _LOGGER.warning(
                "Malformed saved task plan id=%s revision=%s error=%s",
                task_id,
                revision,
                type(exc).__name__,
            )
            raise ValueError(
                f"Saved task plan {task_id} revision {revision} is malformed."
            ) from exc
        if (
            not isinstance(plan, dict)
            or not isinstance(plan.get("steps"), list)
            or any(not isinstance(step, dict) for step in plan["steps"])
        ):
            _LOGGER.warning(
                "Invalid saved task plan id=%s revision=%s",
                task_id,
                revision,
            )
            raise ValueError(
                f"Saved task plan {task_id} revision {revision} is invalid."
            )
        return plan

    @staticmethod
    def _normalize_text_items(value: object, label: str) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError(f"{label} must be a list of text items.")
        if len(value) > 10:
            raise ValueError(f"{label} must contain at most 10 text items.")
        normalized = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, str):
                raise TypeError(f"{label} item {index} must be text.")
            text = item.strip()
            if not text or len(text) > 500:
                raise ValueError(
                    f"{label} item {index} must contain 1 to 500 characters."
                )
            normalized.append(text)
        return normalized

    @staticmethod
    def _normalize_research_references(value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Research references must be a list of text citations.")
        if len(value) > 10:
            raise ValueError("Research references must contain at most 10 citations.")
        references = []
        total_chars = 0
        for index, item in enumerate(value, start=1):
            if not isinstance(item, str):
                raise TypeError(f"Research reference {index} must be text.")
            reference = item.strip()
            if not reference or len(reference) > 1500:
                raise ValueError(
                    f"Research reference {index} must contain 1 to 1500 characters."
                )
            total_chars += len(reference)
            references.append(reference)
        if total_chars > 10000:
            raise ValueError(
                "Research citations exceed the 10000-character plan budget."
            )
        return references

    @classmethod
    def _normalize_plan_steps(cls, steps: list[dict]) -> list[dict]:
        if not steps or len(steps) > 20:
            raise ValueError("A plan must contain between 1 and 20 steps.")
        normalized = []
        for index, step in enumerate(steps, start=1):
            step_data = (
                step
                if isinstance(step, dict)
                else step.model_dump()
                if callable(getattr(step, "model_dump", None))
                else {}
            )
            description = str(step_data.get("description", "")).strip()
            if not description or len(description) > 1000:
                raise ValueError(
                    f"Plan step {index} must contain 1 to 1000 characters."
                )
            step_id = str(step_data.get("step_id") or f"step-{index}").strip()
            if len(step_id) > 64 or not step_id:
                raise ValueError(f"Plan step {index} has an invalid ID.")
            dependencies = step_data.get("dependencies", [])
            if not isinstance(dependencies, list) or len(dependencies) > 20:
                raise ValueError(
                    f"Plan step {index} dependencies must be a list of step IDs."
                )
            if any(not isinstance(item, str) for item in dependencies):
                raise ValueError(f"Plan step {index} dependency IDs must be text.")
            dependency_ids = [item.strip() for item in dependencies]
            if any(not item or len(item) > 64 for item in dependency_ids):
                raise ValueError(f"Plan step {index} has an invalid dependency ID.")
            if len(set(dependency_ids)) != len(dependency_ids):
                raise ValueError(f"Plan step {index} has duplicate dependencies.")
            normalized.append(
                {
                    "step_id": step_id,
                    "description": description,
                    "dependencies": dependency_ids,
                    "validation": cls._normalize_text_items(
                        step_data.get("validation", []),
                        f"Plan step {index} validation",
                    ),
                    "proof": cls._normalize_text_items(
                        step_data.get("proof", []),
                        f"Plan step {index} proof",
                    ),
                    "risks": cls._normalize_text_items(
                        step_data.get("risks", []),
                        f"Plan step {index} risks",
                    ),
                    "edge_cases": cls._normalize_text_items(
                        step_data.get("edge_cases", []),
                        f"Plan step {index} edge cases",
                    ),
                }
            )
        if len({step["step_id"] for step in normalized}) != len(normalized):
            raise ValueError("Plan step IDs must be unique.")
        step_ids = {step["step_id"] for step in normalized}
        dependencies = {step["step_id"]: step["dependencies"] for step in normalized}
        for step in normalized:
            if step["step_id"] in step["dependencies"]:
                raise ValueError(
                    f"Plan step {step['step_id']} cannot depend on itself."
                )
            missing = set(step["dependencies"]) - step_ids
            if missing:
                raise ValueError(
                    f"Plan step {step['step_id']} references unknown dependencies: "
                    + ", ".join(sorted(missing))
                )

        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("Plan step dependencies must not contain a cycle.")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in dependencies[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in dependencies:
            visit(step_id)
        return normalized

    def create_task_plan(
        self,
        session_id: str,
        goal: str,
        task_type: str,
        steps: list[dict],
        *,
        assumptions: list[str] | None = None,
        constraints: list[str] | None = None,
        research_questions: list[str] | None = None,
        research_references: list[str] | None = None,
    ) -> dict:
        """Persist a new task and its first immutable plan revision."""
        goal = str(goal).strip()
        task_type = str(task_type).strip().lower()
        if not session_id or not goal or len(goal) > 2000:
            raise ValueError("Task goal must contain 1 to 2000 characters.")
        if task_type not in {"research", "coding", "other"}:
            raise ValueError("Task type must be research, coding, or other.")
        normalized_steps = self._normalize_plan_steps(steps)
        plan = {
            "goal": goal,
            "task_type": task_type,
            "steps": normalized_steps,
            "assumptions": self._normalize_text_items(assumptions, "Assumptions"),
            "constraints": self._normalize_text_items(constraints, "Constraints"),
            "research_questions": self._normalize_text_items(
                research_questions,
                "Research questions",
            ),
            "research_references": self._normalize_research_references(
                research_references
            ),
        }
        plan_json, digest = self._plan_digest(plan)
        task_id = uuid.uuid4().hex
        with self._lock, self._write_transaction():
            self.conn.execute(
                "INSERT INTO agent_tasks "
                "(task_id, originating_session_id, goal, task_type, status, "
                "current_revision) VALUES (?, ?, ?, ?, 'awaiting_approval', 1)",
                (task_id, session_id, goal, task_type),
            )
            self.conn.execute(
                "INSERT INTO agent_plan_revisions "
                "(task_id, revision, digest, plan_json) VALUES (?, 1, ?, ?)",
                (task_id, digest, plan_json),
            )
            self.conn.executemany(
                "INSERT INTO agent_todo_steps "
                "(task_id, revision, step_id, position, description) "
                "VALUES (?, 1, ?, ?, ?)",
                [
                    (task_id, step["step_id"], index, step["description"])
                    for index, step in enumerate(normalized_steps, start=1)
                ],
            )
            self._record_task_event(
                task_id,
                1,
                "created",
                {"goal": goal, "task_type": task_type},
            )
            self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def revise_task_plan(
        self,
        task_id: str,
        goal: str,
        task_type: str,
        steps: list[dict],
        *,
        expected_revision: int,
        assumptions: list[str] | None = None,
        constraints: list[str] | None = None,
        research_questions: list[str] | None = None,
        research_references: list[str] | None = None,
    ) -> dict:
        """Append an immutable revision and invalidate any previous approval."""
        goal = str(goal).strip()
        task_type = str(task_type).strip().lower()
        if not goal or len(goal) > 2000:
            raise ValueError("Task goal must contain 1 to 2000 characters.")
        if task_type not in {"research", "coding", "other"}:
            raise ValueError("Task type must be research, coding, or other.")
        normalized_steps = self._normalize_plan_steps(steps)
        plan = {
            "goal": goal,
            "task_type": task_type,
            "steps": normalized_steps,
            "assumptions": self._normalize_text_items(assumptions, "Assumptions"),
            "constraints": self._normalize_text_items(constraints, "Constraints"),
            "research_questions": self._normalize_text_items(
                research_questions,
                "Research questions",
            ),
            "research_references": self._normalize_research_references(
                research_references
            ),
        }
        plan_json, digest = self._plan_digest(plan)
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT current_revision FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise ValueError("Task not found.")
            if task[0] != expected_revision:
                raise ValueError(
                    "Task plan changed since it was read; refresh before revising."
                )
            revision = expected_revision + 1
            self.conn.execute(
                "INSERT INTO agent_plan_revisions "
                "(task_id, revision, digest, plan_json) VALUES (?, ?, ?, ?)",
                (task_id, revision, digest, plan_json),
            )
            self.conn.executemany(
                "INSERT INTO agent_todo_steps "
                "(task_id, revision, step_id, position, description) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        task_id,
                        revision,
                        step["step_id"],
                        index,
                        step["description"],
                    )
                    for index, step in enumerate(normalized_steps, start=1)
                ],
            )
            self.conn.execute(
                "UPDATE agent_tasks SET goal = ?, task_type = ?, status = "
                "'awaiting_approval', current_revision = ?, "
                "approved_revision = NULL, approved_digest = NULL, "
                "approved_scope = NULL, validation_revision = NULL, "
                "validation_evidence = NULL, updated_at = CURRENT_TIMESTAMP, "
                "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (goal, task_type, revision, task_id),
            )
            self._record_task_event(
                task_id,
                revision,
                "revised",
                {"previous_revision": expected_revision},
            )
            self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def list_tasks(self, limit: int = 50) -> list[dict]:
        limit = max(1, min(int(limit), 100))
        with self._lock:
            rows = self.conn.execute(
                "SELECT task_id FROM agent_tasks ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        tasks = []
        for row in rows:
            try:
                task = self.get_task(row[0])
            except ValueError as exc:
                _LOGGER.warning(
                    "Skipping malformed task from listing id=%s error=%s",
                    row[0],
                    str(exc)[:160],
                )
                continue
            if task is not None:
                tasks.append(task)
        return tasks

    def count_task_plans(self) -> tuple[int, int]:
        """Return total task plans and plans with current approval."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(CASE WHEN task.approved_revision = task.current_revision "
                "AND task.approved_digest = revision.digest THEN 1 ELSE 0 END), 0) "
                "FROM agent_tasks AS task "
                "LEFT JOIN agent_plan_revisions AS revision "
                "ON revision.task_id = task.task_id "
                "AND revision.revision = task.current_revision"
            ).fetchone()
        return int(row[0]), int(row[1])

    def mark_stale_tasks(self, stale_after_days: int) -> list[str]:
        """Expire old task approvals without deleting the task or its recovery state."""
        stale_after_days = max(0, int(stale_after_days))
        if not stale_after_days:
            return []
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=stale_after_days)
        ).strftime("%Y-%m-%d %H:%M:%S")
        now = time.time()
        with self._lock, self._write_transaction():
            rows = self.conn.execute(
                "SELECT task.task_id, task.current_revision, task.status "
                "FROM agent_tasks AS task WHERE task.status IN "
                "('draft', 'awaiting_approval', 'active', 'paused', 'interrupted') "
                "AND COALESCE(task.checkpoint_at, task.updated_at) < ? "
                "AND NOT EXISTS (SELECT 1 FROM agent_task_leases AS lease "
                "WHERE lease.task_id = task.task_id AND lease.expires_at > ?)",
                (cutoff, now),
            ).fetchall()
            for task_id, revision, previous_status in rows:
                self.conn.execute(
                    "UPDATE agent_tasks SET status = 'stale', "
                    "approved_revision = NULL, approved_digest = NULL, "
                    "approved_scope = NULL, updated_at = CURRENT_TIMESTAMP, "
                    "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (task_id,),
                )
                self.conn.execute(
                    "DELETE FROM agent_task_leases WHERE task_id = ?",
                    (task_id,),
                )
                self._record_task_event(
                    task_id,
                    revision,
                    "staled",
                    {"previous_status": previous_status},
                )
                self._refresh_task_resume_capsule(task_id)
        return [row[0] for row in rows]

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            task = self.conn.execute(
                "SELECT task_id, originating_session_id, goal, task_type, status, "
                "current_revision, approved_revision, approved_digest, "
                "approved_scope, created_at, updated_at, checkpoint_at, "
                "user_rating, resume_state_json, validation_revision, "
                "validation_evidence "
                "FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None:
                return None
            revision = task[5]
            plan = self.conn.execute(
                "SELECT digest, plan_json, created_at FROM agent_plan_revisions "
                "WHERE task_id = ? AND revision = ?",
                (task_id, revision),
            ).fetchone()
            steps = self.conn.execute(
                "SELECT step_id, position, description, status, evidence, updated_at "
                "FROM agent_todo_steps WHERE task_id = ? AND revision = ? "
                "ORDER BY position",
                (task_id, revision),
            ).fetchall()
        if plan is None:
            _LOGGER.warning(
                "Missing saved task plan id=%s revision=%s",
                task_id,
                revision,
            )
            raise ValueError(
                f"Saved task plan {task_id} revision {revision} is missing."
            )
        plan_data = self._decode_task_plan(plan[1], task_id, revision)
        step_details = {item["step_id"]: item for item in plan_data.get("steps", [])}
        return {
            "task_id": task[0],
            "session_id": task[1],
            "goal": task[2],
            "task_type": task[3],
            "status": task[4],
            "revision": revision,
            "approved_revision": task[6],
            "approved_digest": task[7],
            "approved_scope": task[8],
            "created_at": task[9],
            "updated_at": task[10],
            "checkpoint_at": task[11],
            "user_rating": task[12],
            "resume_capsule": self._decode_resume_capsule(task[13], task_id),
            "validation_revision": task[14],
            "validation_evidence": task[15],
            "digest": plan[0],
            "plan": plan_data,
            "plan_created_at": plan[2],
            "steps": [
                {
                    "step_id": step[0],
                    "position": step[1],
                    "description": step[2],
                    "status": step[3],
                    "evidence": step[4],
                    "updated_at": step[5],
                    **{
                        key: value
                        for key, value in step_details.get(step[0], {}).items()
                        if key not in {"step_id", "description"}
                    },
                }
                for step in steps
            ],
        }

    def approve_task_plan(
        self,
        task_id: str,
        revision: int,
        digest: str,
        scope: str,
        *,
        owner_id: str | None = None,
    ) -> dict:
        """Record explicit runtime approval for one immutable plan and scope."""
        if not scope or len(scope) > 1000:
            raise ValueError("Approval scope must contain 1 to 1000 characters.")
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT current_revision, status FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            plan = self.conn.execute(
                "SELECT digest FROM agent_plan_revisions "
                "WHERE task_id = ? AND revision = ?",
                (task_id, revision),
            ).fetchone()
            if task is None or plan is None:
                raise ValueError("Task or plan revision not found.")
            lease = self.conn.execute(
                "SELECT owner_id FROM agent_task_leases WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if lease and lease[0] != owner_id:
                raise ValueError(
                    "Task approval blocked: another process owns the task lease."
                )
            if task[0] != revision or plan[0] != digest:
                raise ValueError("Approval does not match the current plan.")
            if task[1] not in {"awaiting_approval", "paused", "interrupted"}:
                raise ValueError(
                    f"Task cannot be approved while status is '{task[1]}'."
                )
            self.conn.execute(
                "UPDATE agent_tasks SET status = 'active', "
                "approved_revision = ?, approved_digest = ?, "
                "approved_scope = ?, updated_at = CURRENT_TIMESTAMP, "
                "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (revision, digest, scope, task_id),
            )
            self._record_task_event(
                task_id,
                revision,
                "approved",
                {"scope": scope, "digest": digest},
            )
            self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def acquire_task_lease(
        self,
        task_id: str,
        owner_id: str,
        session_id: str,
        *,
        lease_seconds: int = 300,
        allow_takeover: bool = False,
    ) -> bool:
        """Acquire or renew exclusive task execution ownership."""
        if not owner_id or not session_id:
            raise ValueError("Task lease requires an owner and session ID.")
        if not 30 <= lease_seconds <= 3600:
            raise ValueError("Task lease duration must be between 30 and 3600 seconds.")
        now = time.time()
        expires_at = now + lease_seconds
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT current_revision, status FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise ValueError("Task not found.")
            lease = self.conn.execute(
                "SELECT owner_id, expires_at FROM agent_task_leases WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if (
                lease
                and lease[0] != owner_id
                and (lease[1] > now or not allow_takeover)
            ):
                return False
            self.conn.execute(
                "INSERT INTO agent_task_leases "
                "(task_id, owner_id, session_id, expires_at, updated_at) "
                "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(task_id) DO UPDATE SET "
                "owner_id = excluded.owner_id, session_id = excluded.session_id, "
                "expires_at = excluded.expires_at, updated_at = CURRENT_TIMESTAMP",
                (task_id, owner_id, session_id, expires_at),
            )
            self._record_task_event(
                task_id,
                task[0],
                "lease_acquired",
                {"session_id": session_id, "expires_at": expires_at},
            )
        return True

    def release_task_lease(self, task_id: str, owner_id: str) -> bool:
        with self._lock, self._write_transaction():
            cursor = self.conn.execute(
                "DELETE FROM agent_task_leases WHERE task_id = ? AND owner_id = ?",
                (task_id, owner_id),
            )
            if cursor.rowcount:
                self._record_task_event(
                    task_id,
                    0,
                    "lease_released",
                    {"owner_id": owner_id},
                )
            return cursor.rowcount == 1

    def get_task_lease(self, task_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT owner_id, session_id, expires_at FROM agent_task_leases "
                "WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "owner_id": row[0],
            "session_id": row[1],
            "expires_at": row[2],
            "expired": row[2] <= time.time(),
        }

    def update_task_step(
        self,
        task_id: str,
        revision: int,
        step_id: str,
        status: str,
        evidence: str = "",
        owner_id: str | None = None,
    ) -> dict:
        """Persist a model-reported todo update for the approved active plan."""
        allowed_statuses = {
            "pending",
            "in_progress",
            "blocked",
            "reported_done",
        }
        evidence = str(evidence).strip()
        if len(evidence) > 2000:
            raise ValueError(
                "Progress evidence exceeds the 2000-character application limit; "
                "prepare a concise, meaning-preserving summary."
            )
        if status not in allowed_statuses:
            raise ValueError(
                "Step status must be pending, in_progress, blocked, "
                "reported_done, or skipped."
            )
        if status in {"reported_done", "blocked"} and not evidence:
            raise ValueError("A reported completion or blocked step needs evidence.")
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT current_revision, approved_revision, approved_digest, "
                "status FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            plan = self.conn.execute(
                "SELECT digest, plan_json FROM agent_plan_revisions "
                "WHERE task_id = ? AND revision = ?",
                (task_id, revision),
            ).fetchone()
            if (
                task is None
                or plan is None
                or task[0] != revision
                or task[1] != revision
                or task[2] != plan[0]
                or task[3] != "active"
            ):
                raise ValueError("No active approval for this plan revision.")
            if owner_id is not None:
                lease = self.conn.execute(
                    "SELECT owner_id, expires_at FROM agent_task_leases "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if not lease or lease[0] != owner_id or lease[1] <= time.time():
                    raise ValueError(
                        "Progress update blocked: this process does not hold "
                        "the active task lease."
                    )
            step = self.conn.execute(
                "SELECT position FROM agent_todo_steps WHERE task_id = ? "
                "AND revision = ? AND step_id = ?",
                (task_id, revision, step_id),
            ).fetchone()
            if step is None:
                raise ValueError("Todo step not found in the active plan.")
            if status in {"in_progress", "reported_done"}:
                plan_steps = json.loads(plan[1]).get("steps", [])
                dependencies = next(
                    (
                        item.get("dependencies", [])
                        for item in plan_steps
                        if item.get("step_id") == step_id
                    ),
                    [],
                )
                dependency_states = {
                    row[0]: row[1]
                    for row in self.conn.execute(
                        "SELECT step_id, status FROM agent_todo_steps "
                        "WHERE task_id = ? AND revision = ?",
                        (task_id, revision),
                    ).fetchall()
                }
                unmet_dependencies = [
                    dependency
                    for dependency in dependencies
                    if dependency_states.get(dependency)
                    not in {"reported_done", "verified", "skipped"}
                ]
                if unmet_dependencies:
                    names = ", ".join(unmet_dependencies)
                    raise ValueError(
                        f"Step dependencies must be reported complete first: {names}"
                    )
            cursor = self.conn.execute(
                "UPDATE agent_todo_steps SET status = ?, evidence = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE task_id = ? "
                "AND revision = ? AND step_id = ? AND status != 'verified'",
                (status, evidence, task_id, revision, step_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Todo step not found in the active plan.")
            self.conn.execute(
                "UPDATE agent_tasks SET updated_at = CURRENT_TIMESTAMP, "
                "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (task_id,),
            )
            self._record_task_event(
                task_id,
                revision,
                "step_updated",
                {"step_id": step_id, "status": status, "evidence": evidence},
            )
            self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def verify_task_step(
        self,
        task_id: str,
        revision: int,
        step_id: str,
        evidence: str,
        *,
        owner_id: str | None = None,
    ) -> dict:
        """Mark a model-reported step verified through the user/runtime flow."""
        evidence = str(evidence).strip()
        if not evidence:
            raise ValueError("Verification requires concrete evidence.")
        if len(evidence) > 2000:
            raise ValueError(
                "Verification evidence exceeds the 2000-character application "
                "limit; prepare a concise, meaning-preserving summary."
            )
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT current_revision, status FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None or task[0] != revision or task[1] != "active":
                raise ValueError("Only steps in the active revision can be verified.")
            lease = self.conn.execute(
                "SELECT owner_id FROM agent_task_leases WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if lease and lease[0] != owner_id:
                raise ValueError(
                    "Step verification blocked: another process owns the task lease."
                )
            cursor = self.conn.execute(
                "UPDATE agent_todo_steps SET status = 'verified', evidence = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE task_id = ? "
                "AND revision = ? AND step_id = ? "
                "AND status = 'reported_done'",
                (evidence, task_id, revision, step_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("Step is not reported complete or does not exist.")
            self.conn.execute(
                "UPDATE agent_tasks SET updated_at = CURRENT_TIMESTAMP, "
                "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (task_id,),
            )
            self._record_task_event(
                task_id,
                revision,
                "step_verified",
                {"step_id": step_id, "evidence": evidence},
            )
            self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def record_task_validation(
        self,
        task_id: str,
        revision: int,
        evidence: str,
        *,
        owner_id: str | None = None,
    ) -> dict:
        """Persist verified runtime evidence for the current task revision."""
        evidence = str(evidence).strip()
        if not evidence:
            raise ValueError("Task validation requires concrete evidence.")
        if len(evidence) > 2000:
            raise ValueError(
                "Task validation evidence exceeds the 2000-character application "
                "limit; prepare a concise, meaning-preserving summary."
            )
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT current_revision, status FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None or task[0] != revision or task[1] != "active":
                raise ValueError(
                    "Validation evidence must match the active task revision."
                )
            lease = self.conn.execute(
                "SELECT owner_id FROM agent_task_leases WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if lease and lease[0] != owner_id:
                raise ValueError(
                    "Task validation blocked: another process owns the task lease."
                )
            self.conn.execute(
                "UPDATE agent_tasks SET validation_revision = ?, "
                "validation_evidence = ?, updated_at = CURRENT_TIMESTAMP, "
                "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (revision, evidence, task_id),
            )
            self._record_task_event(
                task_id,
                revision,
                "task_validated",
                {"evidence": evidence},
            )
            self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def set_task_status(
        self,
        task_id: str,
        status: str,
        *,
        owner_id: str | None = None,
    ) -> dict:
        """Apply user/runtime task lifecycle transitions; never model approval."""
        transitions = {
            "paused": {"active", "awaiting_approval"},
            "interrupted": {"active"},
            "awaiting_approval": {"active", "paused", "interrupted", "stale"},
            "stale": {"draft", "awaiting_approval", "active", "paused", "interrupted"},
            "cancelled": {
                "draft",
                "awaiting_approval",
                "active",
                "paused",
                "interrupted",
                "stale",
            },
            "abandoned": {
                "draft",
                "awaiting_approval",
                "active",
                "paused",
                "interrupted",
                "cancelled",
                "stale",
            },
        }
        if status not in transitions:
            raise ValueError("Unsupported task lifecycle transition.")
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT current_revision, status FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise ValueError("Task not found.")
            if task[1] not in transitions[status]:
                raise ValueError(f"Cannot change task from '{task[1]}' to '{status}'.")
            lease = self.conn.execute(
                "SELECT owner_id, expires_at FROM agent_task_leases WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if lease and lease[1] > time.time() and lease[0] != owner_id:
                raise ValueError(
                    "Task transition blocked: another process still holds "
                    "the active task lease."
                )
            if status in {
                "paused",
                "interrupted",
                "awaiting_approval",
                "cancelled",
                "abandoned",
                "stale",
            }:
                self.conn.execute(
                    "UPDATE agent_tasks SET status = ?, approved_revision = NULL, "
                    "approved_digest = NULL, approved_scope = NULL, "
                    "updated_at = CURRENT_TIMESTAMP, "
                    "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                    (status, task_id),
                )
                self.conn.execute(
                    "DELETE FROM agent_task_leases WHERE task_id = ?",
                    (task_id,),
                )
            self._record_task_event(
                task_id, task[0], status, {"previous_status": task[1]}
            )
            self._refresh_task_resume_capsule(task_id)
        return self.get_task(task_id)

    def complete_task(
        self,
        task_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict:
        """Complete only when every current-revision step is explicitly verified."""
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT current_revision, status FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise ValueError("Task not found.")
            if task[1] != "active":
                raise ValueError("Only an active task can be completed.")
            unresolved_actions = self.conn.execute(
                "SELECT COUNT(*) FROM agent_task_actions "
                "WHERE task_id = ? AND status IN "
                "('in_flight', 'outcome_unknown')",
                (task_id,),
            ).fetchone()[0]
            if unresolved_actions:
                raise ValueError(
                    "Task cannot be completed while tool actions have "
                    "unresolved outcomes."
                )
            lease = self.conn.execute(
                "SELECT owner_id FROM agent_task_leases WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if lease and lease[0] != owner_id:
                raise ValueError(
                    "Task completion blocked: another process owns the task lease."
                )
            remaining = self.conn.execute(
                "SELECT COUNT(*) FROM agent_todo_steps WHERE task_id = ? "
                "AND revision = ? AND status NOT IN ('verified', 'skipped')",
                (task_id, task[0]),
            ).fetchone()[0]
            if remaining:
                raise ValueError(
                    "Every current plan step must be verified or explicitly skipped."
                )
            details = self.conn.execute(
                "SELECT task.originating_session_id, task.goal, task.task_type, "
                "revision.plan_json FROM agent_tasks AS task "
                "JOIN agent_plan_revisions AS revision "
                "ON revision.task_id = task.task_id "
                "AND revision.revision = task.current_revision "
                "WHERE task.task_id = ?",
                (task_id,),
            ).fetchone()
            verified_steps = self.conn.execute(
                "SELECT step_id, description, status, evidence "
                "FROM agent_todo_steps WHERE task_id = ? AND revision = ? "
                "ORDER BY position",
                (task_id, task[0]),
            ).fetchall()
            plan = json.loads(details[3])
            plan_steps = {item["step_id"]: item for item in plan.get("steps", [])}
            episode_topic = self._sanitize_episode_text(details[1].splitlines()[0])
            episode_description = (
                f"Completed {details[2]} task: "
                f"{sum(row[2] == 'verified' for row in verified_steps)} verified, "
                f"{sum(row[2] == 'skipped' for row in verified_steps)} skipped."
            )
            episode_task_summary = self._sanitize_episode_text(details[1])
            episode_plan = {
                "assumptions": [
                    self._sanitize_episode_text(value)
                    for value in plan.get("assumptions", [])
                ],
                "constraints": [
                    self._sanitize_episode_text(value)
                    for value in plan.get("constraints", [])
                ],
                "research_questions": [
                    self._sanitize_episode_text(value)
                    for value in plan.get("research_questions", [])
                ],
                "steps": [
                    {
                        "step_id": row[0],
                        "description": self._sanitize_episode_text(row[1]),
                        "status": row[2],
                        "dependencies": plan_steps.get(row[0], {}).get(
                            "dependencies", []
                        ),
                        "validation": [
                            self._sanitize_episode_text(value)
                            for value in plan_steps.get(row[0], {}).get(
                                "validation", []
                            )
                        ],
                        "proof": [
                            self._sanitize_episode_text(value)
                            for value in plan_steps.get(row[0], {}).get("proof", [])
                        ],
                        "risks": [
                            self._sanitize_episode_text(value)
                            for value in plan_steps.get(row[0], {}).get("risks", [])
                        ],
                        "edge_cases": [
                            self._sanitize_episode_text(value)
                            for value in plan_steps.get(row[0], {}).get(
                                "edge_cases", []
                            )
                        ],
                    }
                    for row in verified_steps
                ],
            }
            episode_outcome = (
                "Completed after all steps were verified or explicitly skipped."
            )
            episode_content = (
                f"Topic: {episode_topic}\n"
                f"Description: {episode_description}\n"
                f"{episode_description}\n{episode_outcome}"
            )
            if len(episode_content) > MAX_SUMMARY_CHARS:
                raise ValueError(
                    "Structured episode summary exceeds the configured "
                    f"{MAX_SUMMARY_CHARS}-character budget."
                )
            self.conn.execute(
                "UPDATE agent_tasks SET status = 'completed', "
                "updated_at = CURRENT_TIMESTAMP, "
                "checkpoint_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (task_id,),
            )
            self.conn.execute(
                "DELETE FROM agent_task_leases WHERE task_id = ?",
                (task_id,),
            )
            episode_cursor = self.conn.execute(
                "INSERT INTO chat_history "
                "(session_id, role, content, episode_topic, "
                "episode_description, episode_task_summary, "
                "episode_plan_json, episode_outcome, episode_task_id) "
                "VALUES (?, 'summary', ?, ?, ?, ?, ?, ?, ?)",
                (
                    details[0],
                    episode_content,
                    episode_topic,
                    episode_description,
                    episode_task_summary,
                    json.dumps(episode_plan, ensure_ascii=False),
                    episode_outcome,
                    task_id,
                ),
            )
            self._record_task_event(
                task_id, task[0], "completed", {"verified_steps_only": True}
            )
            self._refresh_task_resume_capsule(task_id)
        completed = self.get_task(task_id)
        completed["episode_id"] = episode_cursor.lastrowid
        return completed

    def rate_completed_task(self, task_id: str, rating: int) -> dict:
        """Persist subjective user feedback for a successfully completed task."""
        if (
            not isinstance(rating, int)
            or isinstance(rating, bool)
            or not 0 <= rating <= 5
        ):
            raise ValueError("Task completeness rating must be an integer from 0 to 5.")
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT current_revision, status FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None or task[1] != "completed":
                raise ValueError("Only a completed task can receive a rating.")
            self.conn.execute(
                "UPDATE agent_tasks SET user_rating = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE task_id = ?",
                (rating, task_id),
            )
            self._record_task_event(
                task_id,
                task[0],
                "user_rated",
                {"rating": rating, "rating_is_subjective": True},
            )
        return self.get_task(task_id)

    def delete_task_records(self, task_id: str) -> dict:
        """Delete a non-active task and its linked learning, episode, and skill metadata."""
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT status FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise ValueError("Task not found.")
            if task[0] == "active":
                raise ValueError(
                    "Pause or cancel an active task before deleting its records."
                )
            skill_rows = self.conn.execute(
                "SELECT id, skill_file_path FROM chat_history "
                "WHERE is_skill_file = 1 AND skill_task_id = ?",
                (task_id,),
            ).fetchall()
            episode_cursor = self.conn.execute(
                "DELETE FROM chat_history WHERE role = 'summary' "
                "AND episode_task_id = ?",
                (task_id,),
            )
            self.conn.execute(
                "DELETE FROM chat_history WHERE is_skill_file = 1 "
                "AND skill_task_id = ?",
                (task_id,),
            )
            self.conn.execute(
                "DELETE FROM agent_learned_items WHERE source_task_id = ?",
                (task_id,),
            )
            self.conn.execute(
                "DELETE FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            )
        return {
            "task_id": task_id,
            "episodes_deleted": episode_cursor.rowcount,
            "skills_deleted": len(skill_rows),
            "skill_paths": [row[1] for row in skill_rows],
        }

    def get_task_linked_skill_paths(self, task_id: str) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT skill_file_path FROM chat_history "
                "WHERE is_skill_file = 1 AND skill_task_id = ? "
                "ORDER BY id",
                (task_id,),
            ).fetchall()
        return [str(row[0]) for row in rows if row[0]]
