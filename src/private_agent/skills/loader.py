"""Load Markdown skill profiles into the runtime registry."""

import pathlib

import yaml

from private_agent.console import console

from ..config import SKILL_DESCRIPTION_PREVIEW_CHARS, SKILL_MAX_ITERATIONS


class AgentSkill:
    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        max_iterations: int = SKILL_MAX_ITERATIONS,
        embedding: list[float] | None = None,
    ):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.embedding = embedding


def _split_front_matter(content: str) -> tuple[dict, str]:
    """Return optional YAML metadata and the Markdown body."""
    lines = content.lstrip("\ufeff").splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, content
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            metadata = yaml.safe_load("".join(lines[1:index]))
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError("skill front matter must be a YAML mapping")
            return metadata, "".join(lines[index + 1 :]).lstrip("\n")
    return {}, content


def _metadata_text(metadata: dict, key: str) -> str:
    value = metadata.get(key)
    return value.strip() if isinstance(value, str) else ""


def _description_preview(content: str) -> str:
    preview = content.strip()[:SKILL_DESCRIPTION_PREVIEW_CHARS]
    return f"{preview}..." if preview else ""


def load_skills_from_folder(folder_path: str) -> dict[str, AgentSkill]:
    """Load Markdown skill profiles from the designated skills directory."""
    skills = {}
    path = pathlib.Path(folder_path)
    if not path.exists() or not path.is_dir():
        return skills

    for markdown_file in path.glob("*.md"):
        try:
            content = markdown_file.read_text(encoding="utf-8")
            try:
                metadata, body = _split_front_matter(content)
            except (ValueError, yaml.YAMLError) as exc:
                console.print(
                    f"[yellow][Warning] Ignoring invalid front matter in "
                    f"{markdown_file.name}: {exc}[/yellow]"
                )
                metadata, body = {}, content
            skill_name = _metadata_text(metadata, "name") or (
                markdown_file.stem.replace("_", " ").title()
            )
            description = _metadata_text(metadata, "description") or (
                _description_preview(body)
            )
            skills[markdown_file.stem] = AgentSkill(
                name=skill_name,
                description=description,
                system_prompt=f"[Skill Markdown Profile: {skill_name}]\n{body}",
                max_iterations=SKILL_MAX_ITERATIONS,
            )
        except OSError as exc:
            console.print(
                f"[red][Warning] Failed loading markdown skill file "
                f"{markdown_file.name}: {exc}[/red]"
            )
    return skills
