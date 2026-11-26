"""Per-model capability and limit catalog for OpenAI-compatible endpoints.

Every consumer that needs to know what a model can do asks this catalog instead
of classifying a model name or reading a single listing field.  A record keeps
the endpoint identity, role, context/input/output limits, modalities and the
provenance of each field so a guessed value is never mistaken for a value the
server reported.

Lookups are lazy and single-flight: the first caller for a model key runs the
probe, concurrent callers wait for that probe, and unrelated keys are never
blocked by it.  Records expire after a TTL and are refreshed on the next lookup.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from utils.run_logging import RUN_LOGGER, log_event

PROVENANCE_PROBE = "probe"
PROVENANCE_DECLARED = "declared"
PROVENANCE_DEFAULT = "default"

ROLE_CHAT = "chat"
ROLE_EMBEDDING = "embedding"
ROLE_RERANKER = "reranker"
ROLE_FILL_IN_THE_MIDDLE = "fill-in-the-middle"
ROLE_UNKNOWN = "unknown"
CHAT_ROLES = (ROLE_CHAT, ROLE_FILL_IN_THE_MIDDLE)

VALID_ROLES = (
    ROLE_CHAT,
    ROLE_EMBEDDING,
    ROLE_RERANKER,
    ROLE_FILL_IN_THE_MIDDLE,
    ROLE_UNKNOWN,
)

CAPABILITY_FLAGS = (
    "tools",
    "function_calls",
    "structured_output",
    "thinking",
    "vision",
    "audio",
)

CONTEXT_FIELDS = (
    "loaded_context_length",
    "context_length",
    "max_model_len",
    "max_context_length",
    "max_context_tokens",
)
INPUT_FIELDS = (
    "input_token_limit",
    "max_input_tokens",
    "prompt_token_limit",
)
OUTPUT_FIELDS = (
    "output_token_limit",
    "max_output_tokens",
    "max_completion_tokens",
    "max_tokens",
)
MODALITY_FIELDS = (
    "modalities",
    "input_modalities",
)

_EMBEDDING_MARKERS = ("embed", "embedding", "bge", "e5", "nomic-embed")
_RERANK_MARKERS = ("rerank", "reranker", "cross-encoder", "text-ranking")
_TYPE_ROLES = {
    "embedding": ROLE_EMBEDDING,
    "embeddings": ROLE_EMBEDDING,
    "embed": ROLE_EMBEDDING,
    "feature-extraction": ROLE_EMBEDDING,
    "reranker": ROLE_RERANKER,
    "rerank": ROLE_RERANKER,
    "text-ranking": ROLE_RERANKER,
    "llm": ROLE_CHAT,
    "vlm": ROLE_CHAT,
    "chat": ROLE_CHAT,
    "completion": ROLE_CHAT,
    "infill": ROLE_FILL_IN_THE_MIDDLE,
    "fill-in-the-middle": ROLE_FILL_IN_THE_MIDDLE,
    "fim": ROLE_FILL_IN_THE_MIDDLE,
}

DEFAULT_CATALOG_TTL_SECONDS = 300.0
DEFAULT_MAX_CATALOG_ENTRIES = 512


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value > 0 else None


def _clamp(value: int | None, ceiling: int) -> int:
    candidates = [
        candidate
        for candidate in (value, ceiling)
        if isinstance(candidate, int)
        and not isinstance(candidate, bool)
        and candidate > 0
    ]
    return min(candidates) if candidates else max(1, int(ceiling))


@dataclass(frozen=True)
class RequestLimits:
    """Effective request limits after clamping to configured ceilings."""

    context: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ModelRecord:
    """What one endpoint reports (or defaults) for one model."""

    model_id: str
    base_url: str = ""
    endpoint_name: str = ""
    provider: str = "openai-compatible"
    role: str = ROLE_UNKNOWN
    context_length: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    modalities: tuple[str, ...] = ()
    flags: Mapping[str, bool | int | None] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)
    fetched_at: float = 0.0
    expires_at: float | None = None

    def provenance_for(self, field_name: str) -> str:
        return str(self.provenance.get(field_name, PROVENANCE_DEFAULT))

    def is_expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        current = time.monotonic() if now is None else float(now)
        return current >= self.expires_at

    def capabilities(self) -> dict[str, bool | int | None]:
        """Return the legacy capability dict, now derived from this record."""
        result: dict[str, bool | int | None] = dict.fromkeys(CAPABILITY_FLAGS)
        for key in CAPABILITY_FLAGS:
            value = self.flags.get(key)
            result[key] = value if isinstance(value, bool) else None
        result["function_calls"] = result["tools"]
        result["context_window"] = self.context_length
        return result

    def effective_limits(
        self,
        *,
        ceiling_context: int,
        ceiling_output: int,
    ) -> RequestLimits:
        context = _clamp(self.context_length, ceiling_context)
        output = _clamp(self.output_tokens, ceiling_output)
        input_tokens = _clamp(self.input_tokens, context)
        return RequestLimits(
            context=context,
            input_tokens=min(input_tokens, context),
            output_tokens=min(output, context),
        )


def _entry_values(entry: Mapping[str, Any] | None, fields: Iterable[str]) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    for name in fields:
        value = _positive_int(entry.get(name))
        if value is not None:
            return value
    return None


def _first_positive(
    candidates: Iterable[tuple[str, int | None]],
) -> tuple[int | None, str]:
    for provenance, value in candidates:
        if value is not None:
            return int(value), provenance
    return None, PROVENANCE_DEFAULT


def _normalized_modalities(values: Iterable[Any]) -> tuple[str, ...]:
    result = []
    for value in values:
        if not isinstance(value, str):
            continue
        name = value.strip().lower()
        if name and name not in result:
            result.append(name)
    return tuple(result)


def modalities_from_entry(entry: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Read input modalities from a model listing or props entry."""
    if not isinstance(entry, Mapping):
        return ()
    architecture = entry.get("architecture")
    if isinstance(architecture, Mapping):
        values = architecture.get("input_modalities")
        if isinstance(values, list):
            modalities = _normalized_modalities(values)
            if modalities:
                return modalities
    for name in MODALITY_FIELDS:
        value = entry.get(name)
        if isinstance(value, list):
            modalities = _normalized_modalities(value)
            if modalities:
                return modalities
        if isinstance(value, Mapping):
            enabled = _normalized_modalities(
                key for key, flag in value.items() if flag is True
            )
            if enabled:
                return enabled
    return ()


def _declared_role(entry: Mapping[str, Any] | None) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    for name in ("type", "model_type", "role", "pipeline_tag"):
        value = entry.get(name)
        if isinstance(value, str):
            role = _TYPE_ROLES.get(value.strip().lower())
            if role is not None:
                return role
    return None


def classify_role(
    model_id: str,
    entries: Sequence[Mapping[str, Any] | None] = (),
) -> tuple[str, str]:
    """Classify a model as chat/embedding/reranker, with provenance.

    Server metadata wins; name markers are only a visible last-resort default.
    An unknown model stays ``chat`` with ``default`` provenance, matching the
    long-standing behaviour of treating every non-embedding model as chat.
    """
    for entry in entries:
        role = _declared_role(entry)
        if role is not None:
            return role, PROVENANCE_PROBE

    lowered = str(model_id or "").lower()
    if any(marker in lowered for marker in _RERANK_MARKERS):
        return ROLE_RERANKER, PROVENANCE_DEFAULT
    if any(marker in lowered for marker in _EMBEDDING_MARKERS):
        return ROLE_EMBEDDING, PROVENANCE_DEFAULT
    return ROLE_CHAT, PROVENANCE_DEFAULT


def merge_flag_sources(
    sources: Sequence[tuple[str, Mapping[str, Any] | None]],
) -> tuple[dict[str, bool | None], dict[str, str]]:
    """Merge capability flags from ``(provenance, dict)`` sources in order.

    The first valid value wins, matching the established precedence, and the
    provenance of each accepted flag is recorded.
    """
    flags: dict[str, bool | None] = {name: None for name in CAPABILITY_FLAGS}
    provenance: dict[str, str] = {}
    for source_provenance, source in sources:
        if not isinstance(source, Mapping):
            continue
        for name in CAPABILITY_FLAGS:
            if name == "function_calls" or flags[name] is not None:
                continue
            value = source.get(name)
            if isinstance(value, bool):
                flags[name] = value
                provenance[name] = source_provenance
    flags["function_calls"] = flags["tools"]
    provenance["function_calls"] = provenance.get("tools", PROVENANCE_DEFAULT)
    return flags, provenance


def build_model_record(
    model_id: str,
    *,
    base_url: str = "",
    endpoint_name: str = "",
    provider: str = "openai-compatible",
    role_entries: Sequence[Mapping[str, Any] | None] = (),
    declared_role: str | None = None,
    context_candidates: Iterable[tuple[str, int | None]] = (),
    input_candidates: Iterable[tuple[str, int | None]] = (),
    output_candidates: Iterable[tuple[str, int | None]] = (),
    modality_candidates: Iterable[tuple[str, tuple[str, ...]]] = (),
    flags: Mapping[str, bool | int | None | None] | None = None,
    flag_provenance: Mapping[str, str] | None = None,
    ttl_seconds: float = DEFAULT_CATALOG_TTL_SECONDS,
    now: float | None = None,
) -> ModelRecord:
    """Build one catalog record from already-parsed probe sources."""
    current = time.monotonic() if now is None else float(now)
    if isinstance(declared_role, str) and declared_role in VALID_ROLES:
        role = declared_role
        role_provenance = PROVENANCE_DECLARED
    else:
        role, role_provenance = classify_role(model_id, role_entries)

    context_length, context_provenance = _first_positive(context_candidates)
    input_tokens, input_provenance = _first_positive(input_candidates)
    output_tokens, output_provenance = _first_positive(output_candidates)

    modalities: tuple[str, ...] = ()
    modality_provenance = PROVENANCE_DEFAULT
    for provenance, values in modality_candidates:
        if values:
            modalities = tuple(values)
            modality_provenance = provenance
            break
    if not modalities and role in CHAT_ROLES:
        # Every chat endpoint accepts text; say so, but mark it as a default.
        modalities = ("text",)

    provenance: dict[str, str] = {
        "role": role_provenance,
        "context_length": context_provenance,
        "input_tokens": input_provenance,
        "output_tokens": output_provenance,
        "modalities": modality_provenance,
    }
    if flag_provenance:
        provenance.update(
            {
                key: value
                for key, value in flag_provenance.items()
                if key in CAPABILITY_FLAGS
            }
        )

    return ModelRecord(
        model_id=str(model_id),
        base_url=str(base_url or "").rstrip("/"),
        endpoint_name=str(endpoint_name or ""),
        provider=str(provider or "openai-compatible"),
        role=role,
        context_length=context_length,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        modalities=modalities,
        flags=dict(flags or {}),
        provenance=provenance,
        fetched_at=current,
        expires_at=(
            current + float(ttl_seconds)
            if ttl_seconds and math.isfinite(float(ttl_seconds))
            else None
        ),
    )


def record_from_listing_entry(
    entry: Mapping[str, Any],
    *,
    model_id: str | None = None,
    base_url: str = "",
    endpoint_name: str = "",
    provider: str = "openai-compatible",
    ttl_seconds: float = DEFAULT_CATALOG_TTL_SECONDS,
    now: float | None = None,
) -> ModelRecord:
    """Build a record from one ``/v1/models`` entry."""
    identifier = model_id
    if not identifier:
        for name in ("id", "model", "name", "key", "model_key"):
            value = entry.get(name)
            if isinstance(value, str) and value.strip():
                identifier = value.strip()
                break
    identifier = identifier or "unknown"
    modalities = modalities_from_entry(entry)
    return build_model_record(
        identifier,
        base_url=base_url,
        endpoint_name=endpoint_name,
        provider=provider,
        role_entries=(entry,),
        context_candidates=((PROVENANCE_PROBE, _entry_values(entry, CONTEXT_FIELDS)),),
        input_candidates=((PROVENANCE_PROBE, _entry_values(entry, INPUT_FIELDS)),),
        output_candidates=((PROVENANCE_PROBE, _entry_values(entry, OUTPUT_FIELDS)),),
        modality_candidates=(
            (PROVENANCE_PROBE if modalities else PROVENANCE_DEFAULT, modalities),
        ),
        ttl_seconds=ttl_seconds,
        now=now,
    )


def iter_model_entries(body: Any) -> tuple[Mapping[str, Any], ...]:
    """Return mapping entries from an OpenAI-compatible or Ollama listing."""
    if not isinstance(body, Mapping):
        return ()
    entries = body.get("models", body.get("data", []))
    if not isinstance(entries, list):
        return ()
    return tuple(entry for entry in entries if isinstance(entry, Mapping))


def default_model_record(
    model_id: str,
    *,
    base_url: str = "",
    endpoint_name: str = "",
    provider: str = "openai-compatible",
    ttl_seconds: float = DEFAULT_CATALOG_TTL_SECONDS,
    now: float | None = None,
) -> ModelRecord:
    """Return an all-default record for a model no probe could describe."""
    return build_model_record(
        model_id,
        base_url=base_url,
        endpoint_name=endpoint_name,
        provider=provider,
        ttl_seconds=ttl_seconds,
        now=now,
    )


class ModelCatalog:
    """Bounded, TTL-aware, single-flight store of :class:`ModelRecord` values."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_CATALOG_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_CATALOG_ENTRIES,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = (
            float(ttl_seconds)
            if isinstance(ttl_seconds, (int, float)) and ttl_seconds > 0
            else DEFAULT_CATALOG_TTL_SECONDS
        )
        self.max_entries = max(1, int(max_entries))
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._records: dict[tuple[str, str], ModelRecord] = {}
        self._inflight: dict[tuple[str, str], threading.Event] = {}

    @staticmethod
    def key(model_id: str, base_url: str = "") -> tuple[str, str]:
        return (str(base_url or "").rstrip("/"), str(model_id or ""))

    def put(self, record: ModelRecord) -> None:
        """Insert or replace a record, evicting the oldest when bounded."""
        key = self.key(record.model_id, record.base_url)
        with self._lock:
            self._records[key] = record
            if len(self._records) > self.max_entries:
                oldest = min(
                    self._records,
                    key=lambda entry: self._records[entry].fetched_at,
                )
                self._records.pop(oldest, None)

    def get(
        self,
        model_id: str,
        base_url: str = "",
        *,
        allow_stale: bool = True,
        now: float | None = None,
    ) -> ModelRecord | None:
        """Return a cached record without probing."""
        key = self.key(model_id, base_url)
        current = self._clock() if now is None else float(now)
        with self._lock:
            record = self._records.get(key)
        if record is None:
            return None
        if record.is_expired(current) and not allow_stale:
            return None
        return record

    def records(self) -> tuple[ModelRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def invalidate(self, model_id: str, base_url: str = "") -> None:
        with self._lock:
            self._records.pop(self.key(model_id, base_url), None)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def lookup(
        self,
        model_id: str,
        base_url: str = "",
        *,
        probe: Callable[[], ModelRecord | Iterable[ModelRecord] | None] | None = None,
        allow_stale: bool = True,
        wait_timeout: float | None = 30.0,
    ) -> ModelRecord | None:
        """Return a fresh record, refreshing it lazily with a single probe.

        Only the key being refreshed is serialized; other model keys continue
        to be served from the catalog while a probe is in flight.
        """
        key = self.key(model_id, base_url)
        current = self._clock()
        with self._lock:
            record = self._records.get(key)
            if record is not None and not record.is_expired(current):
                return record
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                owner = True
            else:
                owner = False

        if not owner:
            event.wait(wait_timeout)
            with self._lock:
                refreshed = self._records.get(key)
            if refreshed is not None:
                return refreshed
            return record if allow_stale else None

        try:
            produced = probe() if probe is not None else None
            observed = self._normalize_probe(produced)
            if observed:
                with self._lock:
                    for candidate in observed:
                        self.put(candidate)
                refreshed = self._find(model_id, base_url, observed)
                if refreshed is not None:
                    return refreshed
            return record if allow_stale else None
        except Exception as exc:  # noqa: BLE001 - probes are optional and must not fail a turn
            log_event(
                RUN_LOGGER,
                "model_catalog.probe_failed",
                level=logging.DEBUG,
                model=model_id,
                error_type=type(exc).__name__,
            )
            return record if allow_stale else None
        finally:
            if owner:
                with self._lock:
                    self._inflight.pop(key, None)
                event.set()

    def _normalize_probe(
        self, produced: ModelRecord | Iterable[ModelRecord] | None
    ) -> tuple[ModelRecord, ...]:
        if produced is None:
            return ()
        if isinstance(produced, ModelRecord):
            return (produced,)
        try:
            return tuple(item for item in produced if isinstance(item, ModelRecord))
        except TypeError:
            return ()

    @staticmethod
    def _find(
        model_id: str,
        base_url: str,
        records: Sequence[ModelRecord],
    ) -> ModelRecord | None:
        normalized_url = str(base_url or "").rstrip("/")
        for record in records:
            if (
                record.model_id == model_id
                and record.base_url.rstrip("/") == normalized_url
            ):
                return record
        for record in records:
            if record.model_id == model_id:
                return record
        return None


__all__ = [
    "CAPABILITY_FLAGS",
    "CHAT_ROLES",
    "CONTEXT_FIELDS",
    "DEFAULT_CATALOG_TTL_SECONDS",
    "DEFAULT_MAX_CATALOG_ENTRIES",
    "INPUT_FIELDS",
    "ModelCatalog",
    "ModelRecord",
    "OUTPUT_FIELDS",
    "PROVENANCE_DECLARED",
    "PROVENANCE_DEFAULT",
    "PROVENANCE_PROBE",
    "ROLE_CHAT",
    "ROLE_EMBEDDING",
    "ROLE_FILL_IN_THE_MIDDLE",
    "ROLE_RERANKER",
    "ROLE_UNKNOWN",
    "RequestLimits",
    "VALID_ROLES",
    "build_model_record",
    "classify_role",
    "default_model_record",
    "iter_model_entries",
    "merge_flag_sources",
    "modalities_from_entry",
    "record_from_listing_entry",
]
