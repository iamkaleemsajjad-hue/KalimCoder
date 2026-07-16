"""
src/data/schema.py — Canonical training data schema for KalimCoder.

Every pipeline stage (normalise, clean, score, dedup, mix, write) operates on
``CanonicalExample`` objects exclusively.  Source-specific row formats are
converted to ``CanonicalExample`` inside the DatasetSource implementations
and never leak into the rest of the pipeline.

Versioning
----------
Bump ``SCHEMA_VERSION`` whenever a mandatory field is added or removed.
The version is stored in every output parquet file so downstream consumers
can detect and reject incompatible data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

TASK_TYPES: frozenset[str] = frozenset({
    "instruction",    # user asks → model writes or explains code
    "completion",     # model continues partial code
    "qa",             # technical question-and-answer
    "debugging",      # bug fix / patch / error analysis
    "documentation",  # docstring / README / inline comment generation
})

UNKNOWN: str = "unknown"

# ---------------------------------------------------------------------------
# Canonical example dataclass
# ---------------------------------------------------------------------------


@dataclass
class CanonicalExample:
    """Versioned, source-agnostic training example.

    All fields are required at construction time.  Use :func:`make_example`
    as a convenient factory that fills defaults and generates the ``id``.

    Attributes
    ----------
    id:
        Deterministic SHA-256 hex digest derived from ``instruction + output``.
        Stable across runs; used by the deduplication stage.
    schema_version:
        Must match :data:`SCHEMA_VERSION`.
    dataset:
        Registry entry name (from ``configs/datasets.yaml``).
    task_type:
        One of :data:`TASK_TYPES`.
    language:
        Detected programming language (e.g. ``"python"``) or ``"unknown"``.
    repository:
        Source repository URL when available, else empty string.
    license:
        SPDX licence identifier (e.g. ``"MIT"``, ``"Apache-2.0"``) or
        ``"unknown"``.
    instruction:
        The human turn / prompt / task description.
    input:
        Optional supplementary context (code snippet, stdin, etc.).
    output:
        The model's target response (code, explanation, patch, etc.).
    quality_score:
        Float in ``[0.0, 1.0]`` assigned by :class:`~src.data.quality.QualityScorer`.
        Defaults to ``0.0`` until the quality stage runs.
    metadata:
        Free-form source-specific extras.  Serialised to a JSON string in
        parquet; deserialised back to a dict on read.
    """

    # Identity
    id: str
    schema_version: str

    # Provenance
    dataset: str
    task_type: str
    language: str
    repository: str
    license: str

    # Content
    instruction: str
    input: str
    output: str

    # Quality
    quality_score: float

    # Source-specific extras
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_id(instruction: str, output: str) -> str:
    """Return a deterministic SHA-256 hex digest for a (instruction, output) pair."""
    payload = (instruction + "\x00" + output).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def make_example(
    *,
    dataset: str,
    instruction: str,
    output: str,
    input: str = "",
    task_type: str = "instruction",
    language: str = UNKNOWN,
    repository: str = "",
    license: str = UNKNOWN,
    quality_score: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> CanonicalExample:
    """Convenience factory that generates ``id`` and fills ``schema_version``."""
    return CanonicalExample(
        id=make_id(instruction, output),
        schema_version=SCHEMA_VERSION,
        dataset=dataset,
        task_type=task_type,
        language=language,
        repository=repository,
        license=license,
        instruction=instruction,
        input=input,
        output=output,
        quality_score=quality_score,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def canonical_to_dict(ex: CanonicalExample) -> dict[str, Any]:
    """Convert to a flat dict suitable for parquet / JSONL serialisation.

    ``metadata`` is JSON-encoded to a string so it fits in a single parquet
    string column without requiring a nested schema.
    """
    d = asdict(ex)
    d["metadata"] = json.dumps(d["metadata"], ensure_ascii=False, separators=(",", ":"))
    return d


def dict_to_canonical(d: dict[str, Any]) -> CanonicalExample:
    """Reconstruct a :class:`CanonicalExample` from a parquet row dict."""
    meta_raw = d.get("metadata", "{}")
    if isinstance(meta_raw, str):
        try:
            meta: dict = json.loads(meta_raw)
        except (json.JSONDecodeError, ValueError):
            meta = {}
    elif isinstance(meta_raw, dict):
        meta = meta_raw
    else:
        meta = {}

    return CanonicalExample(
        id=str(d.get("id", "")),
        schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
        dataset=str(d.get("dataset", UNKNOWN)),
        task_type=str(d.get("task_type", "instruction")),
        language=str(d.get("language", UNKNOWN)),
        repository=str(d.get("repository", "")),
        license=str(d.get("license", UNKNOWN)),
        instruction=str(d.get("instruction", "")),
        input=str(d.get("input", "")),
        output=str(d.get("output", "")),
        quality_score=float(d.get("quality_score", 0.0)),
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_canonical(ex: CanonicalExample) -> list[str]:
    """Return a list of validation error strings; empty list means valid."""
    errors: list[str] = []
    if not ex.id:
        errors.append("id is empty")
    if ex.schema_version != SCHEMA_VERSION:
        errors.append(
            f"schema_version mismatch: got {ex.schema_version!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )
    if ex.task_type not in TASK_TYPES:
        errors.append(
            f"unknown task_type {ex.task_type!r}; valid: {sorted(TASK_TYPES)}"
        )
    if not ex.instruction.strip():
        errors.append("instruction is empty or whitespace-only")
    if not ex.output.strip():
        errors.append("output is empty or whitespace-only")
    if not (0.0 <= ex.quality_score <= 1.0):
        errors.append(f"quality_score {ex.quality_score} is outside [0.0, 1.0]")
    return errors
