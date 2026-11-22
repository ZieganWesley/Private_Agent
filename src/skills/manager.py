"""Create validated skills and select relevant profiles."""

import math
import pathlib
import re
from collections.abc import Callable, Sequence

import yaml

from config import (
    MAX_SKILL_DESCRIPTION_CHARS,
    MAX_SKILL_INSTRUCTION_CHARS,
    MAX_SKILL_NAME_CHARS,
    SKILL_RELEVANCE_EMBEDDING_THRESHOLD,
    SKILL_RELEVANCE_EXACT_MATCH_SCORE,
    SKILL_RELEVANCE_PREFIX_MATCH_SCORE,
    SKILL_RELEVANCE_THRESHOLD,
    SKILL_RELEVANCE_WORD_MATCH_SCORE,
)

from .loader import AgentSkill

Embedder = Callable[[str], Sequence[float]] | object


def create_skill_file(
    folder_path: str,
    name: str,
    description: str,
    instructions: str,
) -> pathlib.Path:
    """Create one validated Markdown skill without replacing an existing file."""
    normalized_name = re.sub(r"[\s-]+", "_", name.strip().lower())
    if (
        not normalized_name
        or len(normalized_name) > MAX_SKILL_NAME_CHARS
        or not re.fullmatch(r"[a-z0-9_]+", normalized_name)
    ):
        raise ValueError(
            f"Skill name must be 1-{MAX_SKILL_NAME_CHARS} characters using letters, numbers, "
            "spaces, hyphens, or underscores."
        )
    description = description.strip()
    instructions = instructions.strip()
    if not description or len(description) > MAX_SKILL_DESCRIPTION_CHARS:
        raise ValueError(
            f"Skill description must contain 1-{MAX_SKILL_DESCRIPTION_CHARS} characters."
        )
    if not instructions or len(instructions) > MAX_SKILL_INSTRUCTION_CHARS:
        raise ValueError(
            f"Skill instructions must contain 1-{MAX_SKILL_INSTRUCTION_CHARS} characters."
        )

    directory = pathlib.Path(folder_path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{normalized_name}.md"
    metadata = yaml.safe_dump(
        {"name": name.strip(), "description": description},
        sort_keys=False,
        allow_unicode=True,
    ).strip()
    content = f"---\n{metadata}\n---\n\n# {name.strip()}\n\n{instructions}\n"
    try:
        with target.open("x", encoding="utf-8") as skill_file:
            skill_file.write(content)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Skill '{normalized_name}' already exists; no file was changed."
        ) from exc
    return target


def _embed_texts(embedder: Embedder, texts: list[str]) -> list[Sequence[float]]:
    embed_documents = getattr(embedder, "embed_documents", None)
    if callable(embed_documents):
        return embed_documents(texts)
    embed_query = getattr(embedder, "embed_query", None)
    if callable(embed_query):
        return [embed_query(text) for text in texts]
    if callable(embedder):
        return [embedder(text) for text in texts]
    raise TypeError("embedder must be callable or expose embed_documents/embed_query")


def _embed_query(embedder: Embedder, text: str) -> Sequence[float]:
    embed_query = getattr(embedder, "embed_query", None)
    if callable(embed_query):
        return embed_query(text)
    return _embed_texts(embedder, [text])[0]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _match_by_embedding(
    user_input: str,
    loaded_skills: dict[str, AgentSkill],
    embedder: Embedder,
) -> AgentSkill | None:
    try:
        query_embedding = _embed_query(embedder, user_input)
        best_skill = None
        highest_score = 0.0
        for skill in loaded_skills.values():
            if skill.embedding is None:
                skill.embedding = list(
                    _embed_texts(
                        embedder,
                        [f"{skill.name}\n{skill.description}"],
                    )[0]
                )
            score = _cosine_similarity(query_embedding, skill.embedding)
            if score > highest_score:
                highest_score = score
                best_skill = skill
    except Exception:  # noqa: BLE001 - embedding backends are optional; fall back to keyword matching
        # Embedding providers are optional; any transport or shape failure
        # falls back to the deterministic keyword matcher below.
        return None
    return (
        best_skill
        if best_skill is not None
        and highest_score >= SKILL_RELEVANCE_EMBEDDING_THRESHOLD
        else None
    )


def _match_by_keywords(
    user_input: str,
    loaded_skills: dict[str, AgentSkill],
) -> AgentSkill | None:
    query_tokens = set(user_input.lower().split())
    if not query_tokens:
        return None

    best_skill = None
    highest_score = 0.0
    for key, skill in loaded_skills.items():
        key_tokens = set(key.lower().replace("_", " ").split())
        name_tokens = set(skill.name.lower().split())
        description_tokens = set(skill.description.lower().split())

        score = 0.0
        for token in query_tokens:
            if token in key_tokens:
                score += SKILL_RELEVANCE_EXACT_MATCH_SCORE
            if token in name_tokens:
                score += SKILL_RELEVANCE_PREFIX_MATCH_SCORE
            if token in description_tokens:
                score += SKILL_RELEVANCE_WORD_MATCH_SCORE

        if score > highest_score:
            highest_score = score
            best_skill = skill

    return best_skill if highest_score >= SKILL_RELEVANCE_THRESHOLD else None


def match_skill_by_relevancy(
    user_input: str,
    loaded_skills: dict[str, AgentSkill],
    embedder: Embedder | None = None,
) -> AgentSkill | None:
    """Select a relevant skill using embeddings, then keyword overlap."""
    if not loaded_skills:
        return None

    if embedder is not None:
        semantic_match = _match_by_embedding(user_input, loaded_skills, embedder)
        if semantic_match is not None:
            return semantic_match
    return _match_by_keywords(user_input, loaded_skills)
