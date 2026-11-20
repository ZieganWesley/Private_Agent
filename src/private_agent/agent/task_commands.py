"""The /tasks command handler and its task-scope helpers."""

from __future__ import annotations

import ipaddress
import json
import logging
import pathlib
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ..config import CODE_OUTPUT_ROOT
from ..database import PersistentMemory
from ..sandbox import SandboxManager
from ..tools import get_tool_context, set_tool_context
from .console import console
from .learning import _offer_skill_capture
from .permissions import coding_allows_command_network as _coding_allows_command_network
from .permissions import coding_uses_isolated_copy as _coding_uses_isolated_copy
from .permissions import select_coding_access_level as _select_coding_access_level
from .task_lease import TASK_LEASE_SECONDS

logger = logging.getLogger(__name__)


def _task_age_label(created_at: str) -> str:
    logger.debug(
        "_task_age_label called with created_at="
        + (created_at[:20] if created_at else "")
        + r"\..."
    )
    try:
        created = datetime.fromisoformat(created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_seconds = max(
            0,
            (
                datetime.now(timezone.utc) - created.astimezone(timezone.utc)
            ).total_seconds(),
        )
    except (TypeError, ValueError):
        return "age unavailable"
    if age_seconds < 3600:
        return f"{int(age_seconds // 60)}m old"
    if age_seconds < 86400:
        return f"{int(age_seconds // 3600)}h old"
    return f"{int(age_seconds // 86400)}d old"


def _planning_search_citations(result: str) -> list[str]:
    logger.debug(
        f"_planning_search_citations called with result length={len(result) if result else 0}"
    )
    """Extract distinct HTTPS citations and reject malformed or local destinations."""
    citations = []
    seen_urls = set()
    for block in result.split("\n---\n"):
        title_match = re.search(r"(?m)^Title:\s*(.+)$", block)
        url_match = re.search(r"(?m)^URL:\s*(https://\S+)\s*$", block)
        if not title_match or not url_match:
            continue
        title = " ".join(title_match.group(1).split())
        raw_url = url_match.group(1).rstrip(".,;)")
        try:
            parsed = urlsplit(raw_url)
            host = parsed.hostname
            port = parsed.port
            if (
                parsed.scheme.lower() != "https"
                or not host
                or parsed.username is not None
                or parsed.password is not None
                or host.lower() == "localhost"
                or host.lower().endswith(".local")
                or host.lower().endswith(".localhost")
                or "." not in host
                and ":" not in host
            ):
                continue
            try:
                if not ipaddress.ip_address(host).is_global:
                    continue
            except ValueError:
                pass
            normalized_host = f"[{host.lower()}]" if ":" in host else host.lower()
            normalized_url = urlunsplit(
                (
                    "https",
                    normalized_host + (f":{port}" if port is not None else ""),
                    parsed.path or "/",
                    parsed.query,
                    "",
                )
            )
        except ValueError:
            continue
        if (
            title
            and len(title) <= 500
            and len(normalized_url) <= 900
            and normalized_url not in seen_urls
        ):
            seen_urls.add(normalized_url)
            citations.append(
                f"Untrusted planning search citation: {title}; URL: {normalized_url}"
            )
        if len(citations) >= 10:
            break
    return citations


def _task_plan_markdown(task: dict) -> str:
    """Render a task plan as Markdown for approval and feedback."""
    lines = [f"# {task['goal']}", ""]
    lines.append(
        f"- ID: `{task['task_id']}`\n- Type: {task['task_type']}\n"
        f"- Revision: {task['revision']}\n- Status: {task['status']}"
    )
    plan = task.get("plan") or {}
    for label, key in (
        ("Assumptions", "assumptions"),
        ("Constraints", "constraints"),
    ):
        items = plan.get(key, [])
        if items:
            lines.append(f"\n## {label}")
            lines.extend(f"- {item}" for item in items)
    lines.append("\n## Steps")
    for step in task.get("steps", []):
        lines.append(f"{step['position']}. **{step['description']}** ({step['status']})")
        for label, key in (
            ("Depends on", "dependencies"),
            ("Validation", "validation"),
            ("Proof", "proof"),
            ("Risks", "risks"),
            ("Edge cases", "edge_cases"),
        ):
            items = step.get(key, [])
            if items:
                lines.append(f"   - {label}: " + "; ".join(items))
    return "\n".join(lines)


SHELL_METACHARACTERS = frozenset(";&|<>$`\\\n\r")
VALIDATION_EXECUTABLES = frozenset(
    {"pytest", "python", "python3", "npm", "pnpm", "yarn", "cargo", "go", "make"}
)


def _validated_validation_command(value: str, root: pathlib.Path) -> list[str] | None:
    """Return a safe argv for a validation command, or None when unsafe."""
    stripped = value.strip()
    if not stripped or any(char in stripped for char in SHELL_METACHARACTERS):
        return None
    try:
        argv = shlex.split(stripped, posix=True)
    except ValueError:
        return None
    if not argv or len(argv) > 64:
        return None
    executable = pathlib.Path(argv[0]).name.lower()
    if executable not in VALIDATION_EXECUTABLES:
        return None
    # Keep interpreters and package runners restricted to test commands.
    if executable in {"python", "python3"}:
        if len(argv) < 3 or argv[1] != "-m" or argv[2] not in {"pytest", "unittest"}:
            return None
    elif executable in {"npm", "pnpm", "yarn"}:
        if len(argv) < 2 or argv[1] not in {"test", "run"}:
            return None
        if argv[1] == "run" and (len(argv) < 3 or argv[2] != "test"):
            return None
    elif executable == "cargo" and (len(argv) < 2 or argv[1] != "test"):
        return None
    elif executable == "go" and (len(argv) < 2 or argv[1] != "test"):
        return None
    elif executable == "make" and (len(argv) < 2 or argv[1] != "test"):
        return None

    # Reject absolute paths and traversal outside the selected workspace.
    for token in argv[1:]:
        if token.startswith("-"):
            continue
        candidate = pathlib.Path(token)
        if candidate.is_absolute():
            try:
                candidate.resolve().relative_to(root.resolve())
            except ValueError:
                return None
        elif ".." in candidate.parts:
            return None
    return argv


def _run_automatic_step_validation(
    memory: PersistentMemory,
    task: dict,
) -> None:
    """Run validation commands found in step validation lists and prefill evidence."""
    try:
        root = SandboxManager.get_root()
    except RuntimeError:
        return
    if not root.exists() or not root.is_dir():
        return
    for step in task.get("steps", []):
        if step.get("status") in {"verified", "skipped"}:
            continue
        for item in step.get("validation", []):
            argv = _validated_validation_command(item, root)
            if argv is None:
                console.print(
                    f"[yellow]Skipping unsafe or unsupported validation command: "
                    f"{item}[/yellow]"
                )
                continue
            console.print(
                f"[cyan]Running automatic validation for {step['step_id']}: {item}[/cyan]"
            )
            try:
                completed = subprocess.run(
                    argv,
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                console.print(
                    f"[yellow]Automatic validation could not run: {exc}[/yellow]"
                )
                continue
            output = (completed.stdout + completed.stderr).strip()
            snippet = output[-1000:]
            if completed.returncode == 0:
                try:
                    memory.verify_task_step(
                        task["task_id"],
                        task["revision"],
                        step["step_id"],
                        (
                            f"Automated validation passed: {item}\n"
                            f"{snippet}"
                        ).strip(),
                        owner_id=get_tool_context().runtime_owner_id,
                    )
                    console.print(
                        f"[green]Step {step['step_id']} verified automatically.[/green]"
                    )
                except (ValueError, OSError) as exc:
                    console.print(
                        f"[yellow]Could not save automatic evidence: {exc}[/yellow]"
                    )
            else:
                console.print(
                    f"[red]Automatic validation failed for {step['step_id']} "
                    f"(exit {completed.returncode}).[/red]"
                )
                if snippet:
                    console.print(snippet)


def _run_task_management(
    memory: PersistentMemory,
    session_id: str,
    skills_directory: str | None = None,
) -> str | None:
    """Review and control durable SQLite task plans from the interactive CLI."""
    active_task_id = get_tool_context().task_id
    approved_for_execution_id: str | None = None

    def progress_summary(task: dict) -> str:
        steps = task["steps"]
        verified = sum(step["status"] in {"verified", "skipped"} for step in steps)
        blocked = sum(step["status"] == "blocked" for step in steps)
        current = next(
            (step["step_id"] for step in steps if step["status"] == "in_progress"),
            next(
                (step["step_id"] for step in steps if step["status"] == "pending"),
                "none",
            ),
        )
        unverified = sum(step["status"] == "reported_done" for step in steps)
        remaining = len(steps) - verified
        return (
            f"done {verified}/{len(steps)}; current {current}; "
            f"blocked {blocked}; reported-unverified {unverified}; "
            f"remaining {remaining}"
        )

    def show_tasks() -> None:
        tasks = memory.list_tasks()
        if not tasks:
            console.print("[yellow]No saved task plans are stored in SQLite.[/yellow]")
            return
        table = Table(title="Saved task plans", box=None, expand=True)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Status")
        table.add_column("Type")
        table.add_column("Revision")
        table.add_column("Progress")
        table.add_column("Goal", ratio=3, overflow="ellipsis")
        for task in tasks:
            table.add_row(
                task["task_id"][:12],
                task["status"],
                task["task_type"],
                str(task["revision"]),
                progress_summary(task),
                task["goal"],
            )
        console.print(table)

    def resolve_task(identifier: str) -> dict | None:
        task = memory.get_task(identifier)
        if task is not None:
            return task
        matches = [
            item
            for item in memory.list_tasks()
            if item["task_id"].startswith(identifier)
        ]
        if len(matches) > 1:
            console.print(
                "[yellow]Task ID prefix is ambiguous; use a longer ID.[/yellow]"
            )
            return None
        return matches[0] if matches else None

    def show_task(task: dict) -> None:
        plan = task["plan"]
        resume_capsule = task.get("resume_capsule") or {}
        research_validation_current = bool(
            task.get("validation_revision") == task["revision"]
            and task.get("validation_evidence")
        )
        plan_notes = []
        for label, key in (
            ("Assumptions", "assumptions"),
            ("Constraints", "constraints"),
            ("Research questions", "research_questions"),
            ("Consent-approved research references", "research_references"),
        ):
            items = plan.get(key, [])
            if items:
                plan_notes.append(f"{label}: " + "; ".join(items))
        step_lines = []
        for step in task["steps"]:
            step_lines.append(
                f"{step['position']}. [{step['status']}] {step['description']}"
            )
            for label, key in (
                ("Depends on", "dependencies"),
                ("Validation", "validation"),
                ("Proof", "proof"),
                ("Risks", "risks"),
                ("Edge cases", "edge_cases"),
            ):
                items = step.get(key, [])
                if items:
                    step_lines.append(f"   {label}: " + "; ".join(items))
            if step["evidence"]:
                step_lines.append(f"   Evidence: {step['evidence']}")
        console.print(
            Panel(
                f"[bold]{task['goal']}[/bold]\n"
                f"ID: {task['task_id']} | type: {task['task_type']} | "
                f"status: {task['status']} | revision: {task['revision']}\n"
                f"Created: {task['created_at']} ({_task_age_label(task['created_at'])}) "
                f"| updated: {task['updated_at']}\n"
                f"Progress: {progress_summary(task)}\n"
                f"Next safe action: {resume_capsule.get('next_safe_action', 'unavailable')}\n"
                f"Approval: {'current' if task['approved_revision'] == task['revision'] else 'not current'} "
                f"| scope: {task['approved_scope'] or 'none'}\n\n"
                + (
                    (
                        "Research evidence status: validation evidence recorded; "
                        "consented citations are references only and have not "
                        "been independently verified.\n\n"
                    )
                    if task["task_type"] == "research" and research_validation_current
                    else (
                        "Research evidence status: INCOMPLETE. Plan approval "
                        "does not verify facts or sources; research questions "
                        "and citations must be checked during execution and "
                        "validation evidence recorded with `/tasks validate`.\n\n"
                    )
                    if task["task_type"] == "research"
                    else ""
                )
                + (
                    f"Task validation: {task['validation_evidence']}\n\n"
                    if task.get("validation_revision") == task["revision"]
                    and task.get("validation_evidence")
                    else ""
                )
                + ("\n".join(plan_notes) + "\n\n" if plan_notes else "")
                + "\n".join(step_lines),
                title=f"Task {task['task_id']}",
                border_style="cyan",
            )
        )

    show_tasks()
    while True:
        command = console.input(
            "[cyan]Tasks[/cyan] [list/view/approve/resume/pause/cancel/"
            "verify/validate/reconcile/complete/discard/delete/done]: "
        ).strip()
        if command.lower() in {"done", "exit", "back", "q"}:
            break
        if command.lower() == "list":
            show_tasks()
            continue
        parts = command.split(maxsplit=3)
        action = parts[0].lower() if parts else ""
        if (
            action
            in {
                "approve",
                "resume",
                "pause",
                "cancel",
                "verify",
                "validate",
                "reconcile",
                "complete",
                "discard",
                "delete",
            }
            and not sys.stdin.isatty()
        ):
            console.print(
                "[yellow]Saved task plans are read-only in non-interactive "
                "sessions; no approval or state change was applied.[/yellow]"
            )
            continue
        if action == "view" and len(parts) == 2:
            task = resolve_task(parts[1])
            if task is None:
                console.print("[yellow]Task plan not found.[/yellow]")
            else:
                show_task(task)
                unresolved = memory.list_task_actions(
                    task["task_id"],
                    unresolved_only=True,
                )
                if unresolved:
                    console.print(
                        "[yellow]Unresolved tool actions require verification "
                        "before retry:[/yellow]"
                    )
                    for action_record in unresolved:
                        console.print(
                            f"- {action_record['tool_name']} "
                            f"({action_record['status']}, "
                            f"action {action_record['action_id'][:12]})"
                        )
            continue
        if (
            action
            not in {
                "approve",
                "resume",
                "pause",
                "cancel",
                "verify",
                "reconcile",
                "validate",
                "complete",
                "discard",
                "delete",
            }
            or len(parts) < 2
        ):
            console.print(
                "Use list, view <ID>, approve <ID>, resume <ID>, pause <ID>, "
                "cancel <ID>, verify <ID> <STEP_ID> <evidence>, "
                "validate <ID> <evidence>, complete <ID>, "
                "reconcile <ID> <ACTION_ID> completed|not-run, discard <ID>, "
                "delete <ID>, or done."
            )
            continue

        task_id = parts[1]
        task = resolve_task(task_id)
        if task is None:
            console.print("[yellow]Task plan not found.[/yellow]")
            continue
        task_id = task["task_id"]

        try:
            if action == "approve":
                if task["status"] != "awaiting_approval":
                    console.print(
                        "[yellow]Only an awaiting-approval plan can be approved. "
                        "Resume a paused/interrupted plan first.[/yellow]"
                    )
                    continue
                show_task(task)
                console.print(Markdown(_task_plan_markdown(task)))
                approval_choice = console.input(
                    "Approval choice [1 approve, 2 approve and allow edits, "
                    "3 give feedback and keep planning] (default 1): "
                ).strip() or "1"
                if approval_choice == "3":
                    feedback = console.input(
                        "Feedback to keep planning: "
                    ).strip()
                    console.print(
                        "[yellow]Keeping planning. Resubmit the updated "
                        "request with /plan, including this feedback: "
                        f"{feedback}[/yellow]"
                    )
                    continue
                if approval_choice not in {"1", "2", "3"}:
                    console.print(
                        "[yellow]Plan approval declined; no task actions ran. "
                        f"To retry, enter approve {task_id[:12]} and type the exact "
                        f"confirmation APPROVE {task_id[:8]}. To change the strategy, "
                        "leave Tasks and submit the updated request with /plan.[/yellow]"
                    )
                    continue
                allow_edits = approval_choice == "2"
                if task["task_type"] == "coding":
                    access_level = _select_coding_access_level(
                        console,
                        sys.stdin.isatty,
                    )
                    if _coding_uses_isolated_copy(access_level):
                        source_input = console.input(
                            "[cyan]Existing project directory to copy into the "
                            "isolated output workspace (Enter for a new project): [/cyan]"
                        ).strip()
                    else:
                        allowed_root = SandboxManager.get_root().resolve()
                        source_input = console.input(
                            "[cyan]Project directory to edit directly (Enter for "
                            f"the selected workspace {allowed_root}): [/cyan]"
                        ).strip()
                        requested_source = pathlib.Path(source_input).expanduser()
                        source_path = (
                            (
                                requested_source
                                if requested_source.is_absolute()
                                else allowed_root / requested_source
                            ).resolve()
                            if source_input
                            else allowed_root
                        )
                        if not source_path.is_dir() or not source_path.is_relative_to(
                            allowed_root
                        ):
                            console.print(
                                "[yellow]Direct access must target an existing "
                                "directory inside the workspace selected at "
                                "session start. Approval cancelled.[/yellow]"
                            )
                            continue
                        source_input = str(source_path)
                    if _coding_allows_command_network(access_level):
                        console.print(
                            "[yellow]This access level enables network access "
                            "for approved sandboxed commands, including package "
                            "installation. Commands remain filesystem-limited to "
                            "the selected workspace, but network access may include "
                            "local/private network destinations. Shell commands "
                            "still require per-command approval.[/yellow]"
                        )
                    else:
                        console.print(
                            "[cyan]This access level keeps code-task command "
                            "network access disabled.[/cyan]"
                        )
                    scope = json.dumps(
                        {
                            "task_type": "coding",
                            "goal": task["goal"],
                            "output_root": CODE_OUTPUT_ROOT,
                            "source_path": source_input or None,
                            "access_level": access_level,
                            "network_access": _coding_allows_command_network(
                                access_level
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                else:
                    scope = f"{task['task_type']}: goal={task['goal']}"
                if task["task_type"] == "research" and not (
                    task.get("validation_revision") == task["revision"]
                    and task.get("validation_evidence")
                ):
                    console.print(
                        "[yellow]This approval accepts the plan for execution "
                        "only. It does not mark the research findings or listed "
                        "sources as verified; evidence must be checked and "
                        "recorded separately.[/yellow]"
                    )
                risky_approval = (
                    task["task_type"] == "research"
                    or (
                        task["task_type"] == "coding"
                        and (
                            access_level
                            in {"full_hitl", "full_governed", "full_monitored"}
                            or allow_edits
                        )
                    )
                )
                if risky_approval:
                    confirmation = console.input(
                        f"Approve exact revision {task['revision']}? "
                        f"Type APPROVE {task_id[:8]}: "
                    ).strip()
                    if confirmation != f"APPROVE {task_id[:8]}":
                        console.print(
                            "[yellow]Plan approval declined; no task actions ran. "
                            f"To retry, enter approve {task_id[:12]} and type the exact "
                            f"confirmation APPROVE {task_id[:8]}. To change the strategy, "
                            "leave Tasks and submit the updated request with /plan.[/yellow]"
                        )
                        continue
                else:
                    confirmation = console.input(
                        "Approve this plan? [Y/n/APPROVE "
                        f"{task_id[:8]}] (default Y): "
                    ).strip()
                    if confirmation.lower() not in {
                        "",
                        "y",
                        "yes",
                        f"approve {task_id[:8]}".lower(),
                    }:
                        console.print(
                            "[yellow]Plan approval declined; no task actions ran.[/yellow]"
                        )
                        continue
                owner_id = get_tool_context().runtime_owner_id
                lease = memory.get_task_lease(task_id)
                takeover = False
                if lease and lease["owner_id"] != owner_id:
                    if not lease["expired"]:
                        expires = time.strftime(
                            "%Y-%m-%d %H:%M:%S",
                            time.localtime(lease["expires_at"]),
                        )
                        console.print(
                            "[yellow]Another process holds this task lease until "
                            f"{expires}. Wait for it to expire before taking over.[/yellow]"
                        )
                        continue
                    takeover_confirmation = console.input(
                        "The previous task owner lease has expired. Type "
                        f"TAKEOVER {task_id[:8]} to claim it: "
                    ).strip()
                    if takeover_confirmation != f"TAKEOVER {task_id[:8]}":
                        console.print("[yellow]Task takeover declined.[/yellow]")
                        continue
                    takeover = True
                if not memory.acquire_task_lease(
                    task_id,
                    owner_id,
                    session_id,
                    lease_seconds=TASK_LEASE_SECONDS,
                    allow_takeover=takeover,
                ):
                    console.print(
                        "[yellow]Task lease acquisition failed; no approval "
                        "or execution was started.[/yellow]"
                    )
                    continue
                try:
                    memory.approve_task_plan(
                        task_id,
                        task["revision"],
                        task["digest"],
                        scope,
                        owner_id=owner_id,
                    )
                except (ValueError, OSError):
                    memory.release_task_lease(task_id, owner_id)
                    raise
                active_task_id = task_id
                approved_for_execution_id = task_id
                _tool_context = get_tool_context()
                _tool_context.task_session_id = session_id
                _tool_context.task_id = task_id
                set_tool_context(_tool_context)
                console.print(
                    "[green]This plan revision is approved and active. "
                    "Existing tool permissions and confirmations still apply.[/green]"
                )
            elif action == "resume":
                if task["status"] not in {"active", "paused", "interrupted", "stale"}:
                    console.print(
                        "[yellow]Only active, paused, interrupted, or stale tasks "
                        "can be returned to review.[/yellow]"
                    )
                    continue
                show_task(task)
                console.print(
                    "[cyan]Before continuing, recheck the workspace/repository "
                    "state, research freshness, available tools, permissions, and "
                    "plan assumptions. Revise the plan if any scope or assumption "
                    "changed; fresh approval is required before execution.[/cyan]"
                )
                confirmation = console.input(
                    "Move this task back to review? Type REVIEW: "
                ).strip()
                if confirmation != "REVIEW":
                    console.print("[yellow]Task resume cancelled.[/yellow]")
                    continue
                memory.set_task_status(
                    task_id,
                    "awaiting_approval",
                    owner_id=get_tool_context().runtime_owner_id,
                )
                console.print(
                    "[cyan]Task restored from SQLite and returned to approval. "
                    "Review it, then approve the current revision to continue.[/cyan]"
                )
            elif action in {"pause", "cancel", "discard"}:
                target_status = {
                    "pause": "paused",
                    "cancel": "cancelled",
                    "discard": "abandoned",
                }[action]
                confirm_text = {
                    "pause": "PAUSE",
                    "cancel": "CANCEL",
                    "discard": "DISCARD",
                }[action]
                confirmation = console.input(
                    f"Confirm task {action}. Type {confirm_text}: "
                ).strip()
                if confirmation != confirm_text:
                    console.print(f"[yellow]Task {action} cancelled.[/yellow]")
                    continue
                memory.set_task_status(
                    task_id,
                    target_status,
                    owner_id=get_tool_context().runtime_owner_id,
                )
                if active_task_id == task_id:
                    active_task_id = None
                    _tool_context = get_tool_context()
                    _tool_context.task_session_id = session_id
                    _tool_context.task_id = None
                    set_tool_context(_tool_context)
                console.print(f"[green]Task status saved as {target_status}.[/green]")
            elif action == "verify":
                if len(parts) != 4:
                    console.print(
                        "Usage: verify <ID> <STEP_ID> <concrete verification evidence>"
                    )
                    continue
                memory.verify_task_step(
                    task_id,
                    task["revision"],
                    parts[2],
                    parts[3],
                    owner_id=get_tool_context().runtime_owner_id,
                )
                console.print("[green]Step verification saved in SQLite.[/green]")
            elif action == "validate":
                _run_automatic_step_validation(memory, task)
                if len(parts) < 3:
                    console.print(
                        "Usage: validate <ID> <concrete task-level validation evidence>"
                    )
                    continue
                if task["task_type"] == "research":
                    references = task["plan"].get("research_references", [])
                    if references:
                        console.print(
                            Panel(
                                "\n".join(references)
                                + "\n\nCheck that each source is relevant, "
                                "authoritative for the claim, and current enough "
                                "for the task. These URLs came from search results "
                                "but have not been independently verified.",
                                title="Research source review",
                                border_style="yellow",
                            )
                        )
                    else:
                        console.print(
                            "[yellow]No consent-approved web citations are "
                            "attached. Validate using identified local evidence "
                            "or record which claims remain unverified.[/yellow]"
                        )
                confirmation = console.input(
                    "Record this task-level validation evidence after checking "
                    "the listed claims/sources? Type VALIDATE: "
                ).strip()
                if confirmation != "VALIDATE":
                    console.print("[yellow]Task validation cancelled.[/yellow]")
                    continue
                validated_task = memory.record_task_validation(
                    task_id,
                    task["revision"],
                    " ".join(parts[2:]),
                    owner_id=get_tool_context().runtime_owner_id,
                )
                console.print(
                    "[green]Task validation saved for revision "
                    f"{validated_task['validation_revision']}.[/green]"
                )
            elif action == "reconcile":
                if len(parts) != 4 or parts[3] not in {"completed", "not-run"}:
                    console.print("Usage: reconcile <ID> <ACTION_ID> completed|not-run")
                    continue
                action_matches = [
                    item
                    for item in memory.list_task_actions(
                        task_id,
                        unresolved_only=True,
                    )
                    if item["action_id"].startswith(parts[2])
                ]
                if len(action_matches) != 1:
                    console.print(
                        "[yellow]Use an unambiguous unresolved action ID from "
                        "view <ID>.[/yellow]"
                    )
                    continue
                resolution = "verified" if parts[3] == "completed" else "not_executed"
                confirmation = console.input(
                    f"Record action {action_matches[0]['action_id'][:12]} as "
                    f"{parts[3]} based on your verification? Type RECONCILE: "
                ).strip()
                if confirmation != "RECONCILE":
                    console.print("[yellow]Action reconciliation declined.[/yellow]")
                    continue
                memory.resolve_task_action(
                    action_matches[0]["action_id"],
                    resolution,
                    owner_id=get_tool_context().runtime_owner_id,
                )
                console.print("[green]Action reconciliation saved in SQLite.[/green]")
            elif action == "complete":
                confirmation = console.input(
                    "Mark task complete only if every step is verified. Type COMPLETE: "
                ).strip()
                if confirmation != "COMPLETE":
                    console.print("[yellow]Task completion cancelled.[/yellow]")
                    continue
                completed_task = memory.complete_task(
                    task_id,
                    owner_id=get_tool_context().runtime_owner_id,
                )
                console.print("[green]Verified task marked complete.[/green]")
                if completed_task:
                    _offer_skill_capture(
                        memory,
                        completed_task,
                        session_id,
                        skills_directory,
                    )
                if active_task_id == task_id:
                    active_task_id = None
                    _tool_context = get_tool_context()
                    _tool_context.task_session_id = session_id
                    _tool_context.task_id = None
                    set_tool_context(_tool_context)
            elif action == "delete":
                if task["status"] == "active":
                    console.print(
                        "[yellow]Pause or cancel an active task before deleting it.[/yellow]"
                    )
                    continue
                skill_paths = memory.get_task_linked_skill_paths(task_id)
                skill_root = (
                    pathlib.Path(skills_directory).expanduser().resolve()
                    if skills_directory
                    else None
                )
                safe_skill_paths = []
                unsafe_skill_paths = []
                for raw_path in skill_paths:
                    candidate = pathlib.Path(raw_path).expanduser()
                    try:
                        resolved = candidate.resolve(strict=False)
                        exists = resolved.exists()
                        if exists and (
                            skill_root is None
                            or not resolved.is_relative_to(skill_root)
                            or candidate.is_symlink()
                            or not resolved.is_file()
                        ):
                            unsafe_skill_paths.append(raw_path)
                        else:
                            safe_skill_paths.append(resolved)
                    except OSError:
                        unsafe_skill_paths.append(raw_path)
                if unsafe_skill_paths:
                    console.print(
                        "[yellow]Cannot delete task records while linked skill "
                        "files cannot be safely resolved inside the configured "
                        "skills directory. No records were deleted.[/yellow]"
                    )
                    continue
                confirmation = console.input(
                    "Delete this task, its linked episodes, learned items, skill "
                    f"registrations, and skill files? Type DELETE {task_id[:8]}: "
                ).strip()
                if confirmation != f"DELETE {task_id[:8]}":
                    console.print("[yellow]Task deletion cancelled.[/yellow]")
                    continue
                deleted = memory.delete_task_records(task_id)
                failed_skill_deletions = []
                for skill_path in safe_skill_paths:
                    try:
                        skill_path.unlink(missing_ok=True)
                    except OSError as exc:
                        failed_skill_deletions.append(
                            f"{skill_path}: {type(exc).__name__}: {exc}"
                        )
                if active_task_id == task_id:
                    active_task_id = None
                    _tool_context = get_tool_context()
                    _tool_context.task_session_id = session_id
                    _tool_context.task_id = None
                    set_tool_context(_tool_context)
                console.print(
                    "[green]Task records deleted: "
                    f"{deleted['episodes_deleted']} linked episode(s), "
                    f"{deleted['skills_deleted']} skill registration(s), and "
                    "linked learned items.[/green]"
                )
                if failed_skill_deletions:
                    console.print(
                        "[red]Some linked skill files could not be removed; "
                        "delete or review them manually:[/red]"
                    )
                    for failure in failed_skill_deletions:
                        console.print(f"- {failure}")
        except (ValueError, sqlite3.Error, OSError) as exc:
            console.print(f"[red][Task state error] {type(exc).__name__}: {exc}[/red]")
    return approved_for_execution_id


def _approved_coding_task_scope(
    memory: PersistentMemory,
) -> dict | None:
    """Return the active approved coding plan selected through ToolContext."""
    task_id = get_tool_context().task_id
    if not task_id:
        return None
    task = memory.get_task(task_id)
    if not (
        task
        and task["status"] == "active"
        and task["task_type"] == "coding"
        and task["approved_revision"] == task["revision"]
        and task["approved_digest"] == task["digest"]
    ):
        return None
    try:
        scope = json.loads(task["approved_scope"])
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(scope, dict):
        scope.setdefault("access_level", "isolated_governed")
        scope.setdefault("network_access", False)
    if (
        not isinstance(scope, dict)
        or scope.get("task_type") != "coding"
        or scope.get("goal") != task["goal"]
        or scope.get("output_root") != CODE_OUTPUT_ROOT
        or scope.get("access_level")
        not in {
            "full_hitl",
            "full_governed",
            "full_monitored",
            "isolated_governed",
            "isolated_hitl",
        }
        or scope.get("source_path") is not None
        and not isinstance(scope.get("source_path"), str)
    ):
        return None
    access_level = scope["access_level"]
    if not _coding_uses_isolated_copy(access_level):
        try:
            selected_source = pathlib.Path(scope["source_path"]).resolve()
            if not selected_source.is_dir() or not selected_source.is_relative_to(
                SandboxManager.get_root().resolve()
            ):
                return None
        except (KeyError, OSError, RuntimeError, TypeError):
            return None
    if scope.get("network_access") is not _coding_allows_command_network(access_level):
        return None
    return scope
