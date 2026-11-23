"""Task action journaling, leases and restart reconciliation."""

import hashlib
import json
import logging
import secrets
import time

from config import (
    MAX_RESUME_STATE_CHARS,
)

_LOGGER = logging.getLogger("private_agent.database.memory")


class TaskActionMixin:
    """Task action journaling, leases and restart reconciliation."""

    def _record_task_event(
        self,
        task_id: str,
        revision: int,
        event_type: str,
        details: dict,
    ) -> None:
        details_json = json.dumps(details, ensure_ascii=False, sort_keys=True)
        event_id = hashlib.sha256(
            f"{task_id}:{revision}:{event_type}:{details_json}".encode()
        ).hexdigest()
        self.conn.execute(
            "INSERT OR IGNORE INTO agent_task_events "
            "(event_id, task_id, revision, event_type, details_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                event_id,
                task_id,
                revision,
                event_type,
                details_json,
            ),
        )

    def _refresh_task_resume_capsule(self, task_id: str) -> None:
        task = self.conn.execute(
            "SELECT goal, current_revision, status, checkpoint_at "
            "FROM agent_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if task is None:
            return
        revision = int(task[1])
        plan_row = self.conn.execute(
            "SELECT plan_json FROM agent_plan_revisions "
            "WHERE task_id = ? AND revision = ?",
            (task_id, revision),
        ).fetchone()
        step_rows = self.conn.execute(
            "SELECT step_id, status, evidence FROM agent_todo_steps "
            "WHERE task_id = ? AND revision = ? ORDER BY position",
            (task_id, revision),
        ).fetchall()
        action_rows = self.conn.execute(
            "SELECT action_id, tool_name, status FROM agent_task_actions "
            "WHERE task_id = ? AND status IN ('in_flight', 'outcome_unknown') "
            "ORDER BY started_at",
            (task_id,),
        ).fetchall()
        if plan_row is None:
            _LOGGER.warning(
                "Missing saved task plan id=%s revision=%s",
                task_id,
                revision,
            )
            raise ValueError(
                f"Saved task plan {task_id} revision {revision} is missing."
            )
        plan = self._decode_task_plan(plan_row[0], task_id, revision)
        unresolved = [
            {
                "action_id": row[0],
                "tool_name": row[1],
                "status": row[2],
            }
            for row in action_rows
        ]
        if unresolved:
            next_safe_action = "Reconcile unresolved tool actions before continuing."
        elif task[2] != "active":
            next_safe_action = (
                "Review the saved task and obtain approval before execution."
            )
        else:
            next_step = next(
                (
                    row
                    for row in step_rows
                    if row[1] in {"in_progress", "pending", "blocked"}
                ),
                None,
            )
            next_safe_action = (
                f"Continue step {next_step[0]}."
                if next_step
                else "Review verification evidence and complete the task."
            )
        capsule = {
            "goal": task[0],
            "assumptions": plan.get("assumptions", []),
            "constraints": plan.get("constraints", []),
            "unresolved_questions": plan.get("research_questions", []),
            "revision": revision,
            "todo": [
                {
                    "step_id": row[0],
                    "status": row[1],
                    "evidence_ref": (
                        f"task:{task_id}:revision:{revision}:step:{row[0]}"
                        if row[2]
                        else None
                    ),
                }
                for row in step_rows
            ],
            "checkpoint_at": task[3],
            "unresolved_actions": unresolved,
            "next_safe_action": (
                "Task is completed; no execution action remains."
                if task[2] == "completed"
                else "Task was cancelled; review its history before starting new work."
                if task[2] in {"cancelled", "abandoned"}
                else next_safe_action
            ),
        }
        capsule_json = json.dumps(
            capsule,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(capsule_json) > MAX_RESUME_STATE_CHARS:
            raise ValueError(
                "Task resume capsule exceeds the configured "
                f"{MAX_RESUME_STATE_CHARS}-character limit. Reduce plan "
                "constraints or steps before saving progress."
            )
        self.conn.execute(
            "UPDATE agent_tasks SET resume_state_json = ? WHERE task_id = ?",
            (capsule_json, task_id),
        )

    @staticmethod
    def _decode_resume_capsule(
        capsule_json: str | None,
        task_id: str,
    ) -> dict | None:
        if not capsule_json:
            return None
        try:
            capsule = json.loads(capsule_json)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            _LOGGER.warning("Skipping malformed task resume capsule id=%s", task_id)
            return None
        if (
            not isinstance(capsule, dict)
            or not isinstance(capsule.get("todo"), list)
            or not isinstance(capsule.get("unresolved_actions"), list)
        ):
            _LOGGER.warning("Skipping invalid task resume capsule id=%s", task_id)
            return None
        return capsule

    def begin_task_action(
        self,
        task_id: str,
        owner_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
        *,
        lease_seconds: int = 180,
    ) -> str:
        """Durably record action intent before execution; never persist arguments."""
        if not tool_call_id or not tool_name:
            raise ValueError("Task action requires a tool call ID and name.")
        if not 30 <= lease_seconds <= 3600:
            raise ValueError("Task lease duration must be between 30 and 3600 seconds.")
        args_json = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        args_digest = hashlib.sha256(args_json.encode("utf-8")).hexdigest()
        action_id = secrets.token_hex(16)
        with self._lock, self._write_transaction():
            task = self.conn.execute(
                "SELECT current_revision, approved_revision, approved_digest, status "
                "FROM agent_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            lease = self.conn.execute(
                "SELECT owner_id, expires_at FROM agent_task_leases WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if not (
                task
                and task[3] == "active"
                and task[0] == task[1]
                and lease
                and lease[0] == owner_id
                and lease[1] > time.time()
            ):
                raise ValueError(
                    "Action blocked: this process does not hold the active "
                    "approved task lease."
                )
            unresolved = self.conn.execute(
                "SELECT tool_name, status FROM agent_task_actions "
                "WHERE task_id = ? AND status IN "
                "('in_flight', 'outcome_unknown') LIMIT 1",
                (task_id,),
            ).fetchone()
            if unresolved:
                raise ValueError(
                    "Action blocked: unresolved action "
                    f"'{unresolved[0]}' has status '{unresolved[1]}'. "
                    "A user must reconcile it before another task action."
                )
            plan = self.conn.execute(
                "SELECT digest FROM agent_plan_revisions "
                "WHERE task_id = ? AND revision = ?",
                (task_id, task[0]),
            ).fetchone()
            if not plan or plan[0] != task[2]:
                raise ValueError("Action blocked: active plan approval is stale.")
            existing = self.conn.execute(
                "SELECT action_id, status FROM agent_task_actions "
                "WHERE task_id = ? AND tool_call_id = ?",
                (task_id, tool_call_id),
            ).fetchone()
            if existing:
                raise ValueError(
                    f"Tool call {tool_call_id} already has durable action state "
                    f"'{existing[1]}'; reconcile before retrying."
                )
            self.conn.execute(
                "INSERT INTO agent_task_actions "
                "(action_id, task_id, revision, tool_call_id, tool_name, "
                "arguments_digest, status) "
                "VALUES (?, ?, ?, ?, ?, ?, 'in_flight')",
                (
                    action_id,
                    task_id,
                    task[0],
                    tool_call_id,
                    tool_name,
                    args_digest,
                ),
            )
            self.conn.execute(
                "UPDATE agent_task_leases SET expires_at = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE task_id = ? AND owner_id = ?",
                (time.time() + lease_seconds, task_id, owner_id),
            )
            self._record_task_event(
                task_id,
                task[0],
                "action_intent",
                {
                    "action_id": action_id,
                    "tool_name": tool_name,
                    "arguments_digest": args_digest,
                    "status": "in_flight",
                },
            )
            self._refresh_task_resume_capsule(task_id)
        return action_id

    def finish_task_action(
        self,
        action_id: str,
        status: str,
        outcome_digest: str | None = None,
    ) -> dict:
        if status not in {"outcome_observed", "outcome_unknown"}:
            raise ValueError("Task action outcome status is invalid.")
        with self._lock, self._write_transaction():
            action = self.conn.execute(
                "SELECT task_id, revision, status FROM agent_task_actions "
                "WHERE action_id = ?",
                (action_id,),
            ).fetchone()
            if action is None:
                raise ValueError("Task action record not found.")
            if action[2] != "in_flight":
                raise ValueError("Task action is no longer in flight.")
            self.conn.execute(
                "UPDATE agent_task_actions SET status = ?, outcome_digest = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE action_id = ?",
                (status, outcome_digest, action_id),
            )
            self._record_task_event(
                action[0],
                action[1],
                "action_outcome",
                {"action_id": action_id, "status": status},
            )
            self._refresh_task_resume_capsule(action[0])
        return {
            "action_id": action_id,
            "task_id": action[0],
            "revision": action[1],
            "status": status,
        }

    def list_task_actions(
        self,
        task_id: str,
        *,
        unresolved_only: bool = False,
    ) -> list[dict]:
        query = (
            "SELECT action_id, revision, tool_call_id, tool_name, "
            "arguments_digest, status, started_at, updated_at "
            "FROM agent_task_actions WHERE task_id = ?"
        )
        if unresolved_only:
            query += " AND status IN ('in_flight', 'outcome_unknown')"
        query += " ORDER BY started_at"
        with self._lock:
            rows = self.conn.execute(query, (task_id,)).fetchall()
        return [
            {
                "action_id": row[0],
                "revision": row[1],
                "tool_call_id": row[2],
                "tool_name": row[3],
                "arguments_digest": row[4],
                "status": row[5],
                "started_at": row[6],
                "updated_at": row[7],
            }
            for row in rows
        ]

    def resolve_task_action(
        self,
        action_id: str,
        resolution: str,
        *,
        owner_id: str | None = None,
    ) -> dict:
        """Record a user's explicit reconciliation of an uncertain action."""
        if resolution not in {"verified", "not_executed"}:
            raise ValueError("Resolution must be 'verified' or 'not_executed'.")
        with self._lock, self._write_transaction():
            action = self.conn.execute(
                "SELECT task_id, revision, status FROM agent_task_actions "
                "WHERE action_id = ?",
                (action_id,),
            ).fetchone()
            if action is None:
                raise ValueError("Task action record not found.")
            if action[2] not in {"in_flight", "outcome_unknown"}:
                raise ValueError("Only unresolved task actions can be reconciled.")
            lease = self.conn.execute(
                "SELECT owner_id, expires_at FROM agent_task_leases WHERE task_id = ?",
                (action[0],),
            ).fetchone()
            if lease and lease[1] > time.time() and lease[0] != owner_id:
                raise ValueError(
                    "Action reconciliation blocked: another process still "
                    "holds the active task lease."
                )
            self.conn.execute(
                "UPDATE agent_task_actions SET status = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE action_id = ?",
                (resolution, action_id),
            )
            self._record_task_event(
                action[0],
                action[1],
                "action_reconciled",
                {"action_id": action_id, "resolution": resolution},
            )
            self._refresh_task_resume_capsule(action[0])
        return {
            "action_id": action_id,
            "task_id": action[0],
            "revision": action[1],
            "status": resolution,
        }

    def reconcile_task_actions_after_restart(self) -> int:
        """Mark orphaned in-flight actions unknown; never guess they completed."""
        with self._lock, self._write_transaction():
            now = time.time()
            rows = self.conn.execute(
                "SELECT action.action_id, action.task_id, action.revision "
                "FROM agent_task_actions AS action "
                "LEFT JOIN agent_task_leases AS lease "
                "ON lease.task_id = action.task_id "
                "WHERE action.status = 'in_flight' "
                "AND (lease.task_id IS NULL OR lease.expires_at <= ?)",
                (now,),
            ).fetchall()
            for action_id, task_id, revision in rows:
                self.conn.execute(
                    "UPDATE agent_task_actions SET status = 'outcome_unknown', "
                    "updated_at = CURRENT_TIMESTAMP WHERE action_id = ?",
                    (action_id,),
                )
                self._record_task_event(
                    task_id,
                    revision,
                    "action_reconciliation_required",
                    {"action_id": action_id, "status": "outcome_unknown"},
                )
                self._refresh_task_resume_capsule(task_id)
            self.conn.execute(
                "DELETE FROM agent_task_leases WHERE expires_at <= ?",
                (now,),
            )
        return len(rows)
