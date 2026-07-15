"""
cleaner.py — Filter, deduplicate, and quality-check dataset examples.

Provides two layers:

1. **Low-level helpers** (original API, unchanged):
   ``deduplicate``, ``filter_by_length``, ``apply_filters``

2. **Pipeline primitives** (new):
   Individual stateless functions that each accept a
   ``datasets.Dataset`` and return a cleaned ``datasets.Dataset``.

3. **Orchestrator**:
   ``CleaningConfig`` + ``apply_pipeline()`` — apply all enabled rules
   in a fixed, reproducible order and return both the cleaned dataset
   and a structured ``CleaningStats`` report.

None of the functions modify their input dataset in-place; every
operation returns a *new* ``datasets.Dataset`` object.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate chars-per-token (GPT-2 / Qwen empirical mean)
_CHARS_PER_TOKEN: float = 4.0

# Columns always considered as text / code for deep cleaning
_TEXT_COLUMN_HINTS: frozenset[str] = frozenset(
    {
        "text", "content", "code", "prompt", "response", "input", "output",
        "instruction", "answer", "question", "body", "solution",
        "func_code_string", "whole_func_string", "func_documentation_string",
        "docstring", "problem_statement", "patch", "hint",
    }
)


# ---------------------------------------------------------------------------
# Original low-level helpers (public API, unchanged)
# ---------------------------------------------------------------------------


def deduplicate(records: list[dict], key: str = "prompt") -> list[dict]:
    """Remove duplicate examples based on a hash of the given key."""
    seen: set[str] = set()
    unique = []
    for rec in records:
        h = hashlib.md5(rec.get(key, "").encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(rec)
    return unique


def filter_by_length(
    records: list[dict],
    min_chars: int = 50,
    max_chars: int = 8000,
    key: str = "response",
) -> list[dict]:
    """Keep only records where the response length is within [min_chars, max_chars]."""
    return [r for r in records if min_chars <= len(r.get(key, "")) <= max_chars]


def apply_filters(records: list[dict], filters: list[Callable]) -> list[dict]:
    """Apply a list of filter functions sequentially."""
    for fn in filters:
        records = [r for r in records if fn(r)]
    return records


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CleaningConfig:
    """Configures which cleaning rules are applied and their parameters.

    Each boolean flag enables/disables a specific rule.  Parameters
    control thresholds for the rules that need them.

    Attributes
    ----------
    text_columns:
        Column names to treat as text/code.  When empty the cleaner uses
        ``_TEXT_COLUMN_HINTS`` to detect suitable columns automatically.
    remove_empty_outputs:
        Drop rows where any *output* column (``output``, ``response``,
        ``code``, ``content``) is ``None``, empty, or whitespace-only.
    remove_empty_instructions:
        Drop rows where any *instruction* column (``instruction``,
        ``prompt``, ``question``, ``input``) is ``None``, empty, or
        whitespace-only.
    strip_whitespace:
        Strip leading/trailing whitespace from every string cell.
    normalize_line_endings:
        Normalise ``\r\n`` and ``\r`` to ``\n`` in every string cell.
    remove_duplicate_rows:
        Hash the full row (all columns) and drop exact duplicates, keeping
        the first occurrence.
    remove_duplicate_code:
        Hash the concatenation of all detected text columns per row and
        drop rows whose code fingerprint has already been seen.
    max_tokens:
        If > 0, drop rows where the total estimated token count across all
        text columns exceeds this limit.  Estimation: ``len(text) / 4``.
    """

    text_columns: list[str] = field(default_factory=list)

    # Rule toggles
    remove_empty_outputs: bool = True
    remove_empty_instructions: bool = True
    strip_whitespace: bool = True
    normalize_line_endings: bool = True
    remove_duplicate_rows: bool = True
    remove_duplicate_code: bool = True

    # Token limit (0 = disabled)
    max_tokens: int = 8_192


# ---------------------------------------------------------------------------
# Cleaning statistics
# ---------------------------------------------------------------------------


@dataclass
class CleaningStats:
    """Tracks how many rows each rule removed."""

    original_rows: int = 0
    after_strip_whitespace: int = 0
    after_normalize_line_endings: int = 0
    removed_empty_outputs: int = 0
    removed_empty_instructions: int = 0
    removed_long_samples: int = 0
    removed_duplicate_rows: int = 0
    removed_duplicate_code: int = 0
    final_rows: int = 0

    @property
    def total_removed(self) -> int:
        return self.original_rows - self.final_rows

    @property
    def retention_pct(self) -> float:
        if self.original_rows == 0:
            return 0.0
        return round(100.0 * self.final_rows / self.original_rows, 2)

    def as_dict(self) -> dict:
        return {
            "original_rows": self.original_rows,
            "final_rows": self.final_rows,
            "total_removed": self.total_removed,
            "retention_pct": self.retention_pct,
            "removed_empty_outputs": self.removed_empty_outputs,
            "removed_empty_instructions": self.removed_empty_instructions,
            "removed_long_samples": self.removed_long_samples,
            "removed_duplicate_rows": self.removed_duplicate_rows,
            "removed_duplicate_code": self.removed_duplicate_code,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_text_columns(
    column_names: list[str],
    explicit: list[str],
) -> list[str]:
    """Return the ordered list of columns to treat as text/code."""
    if explicit:
        # Only keep columns that actually exist
        return [c for c in explicit if c in column_names]
    return [
        c for c in column_names
        if c.lower() in _TEXT_COLUMN_HINTS
        or any(hint in c.lower() for hint in _TEXT_COLUMN_HINTS)
    ]


_OUTPUT_COLS: frozenset[str] = frozenset(
    {"output", "response", "code", "content", "answer", "solution",
     "func_code_string", "whole_func_string", "patch"}
)
_INSTRUCTION_COLS: frozenset[str] = frozenset(
    {"instruction", "prompt", "question", "input", "problem_statement",
     "func_documentation_string", "docstring", "hint", "body"}
)


def _is_empty(value: object) -> bool:
    """Return True for None, empty string, or whitespace-only string."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _row_fingerprint(row: dict, columns: Sequence[str]) -> str:
    """SHA-256 of the concatenated text-column values for one row."""
    parts = [str(row.get(c) or "") for c in sorted(columns)]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def _full_row_fingerprint(row: dict) -> str:
    """SHA-256 of the full JSON-serialisable row (all columns).

    Uses SHA-256 (not ``hash()``) to guarantee identical results across
    processes, Python versions, and PYTHONHASHSEED settings.
    """
    serialised = json.dumps(row, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialised).hexdigest()


# ---------------------------------------------------------------------------
# Pipeline rule functions  (Dataset → Dataset)
# ---------------------------------------------------------------------------


def rule_strip_whitespace(
    dataset: "Dataset",  # type: ignore[name-defined]  # noqa: F821
    text_cols: list[str],
) -> "Dataset":
    """Strip leading/trailing whitespace from every text column value."""

    def _strip(batch: dict) -> dict:
        for col in text_cols:
            batch[col] = [
                v.strip() if isinstance(v, str) else v
                for v in batch[col]
            ]
        return batch

    return dataset.map(_strip, batched=True, desc="strip_whitespace")


def rule_normalize_line_endings(
    dataset: "Dataset",  # type: ignore[name-defined]  # noqa: F821
    text_cols: list[str],
) -> "Dataset":
    """Normalise \\r\\n and \\r to \\n in every text column."""
    _CRLF = re.compile(r"\r\n|\r")

    def _normalise(batch: dict) -> dict:
        for col in text_cols:
            batch[col] = [
                _CRLF.sub("\n", v) if isinstance(v, str) else v
                for v in batch[col]
            ]
        return batch

    return dataset.map(_normalise, batched=True, desc="normalize_line_endings")


def rule_remove_empty_outputs(
    dataset: "Dataset",  # type: ignore[name-defined]  # noqa: F821
) -> "Dataset":
    """Drop rows where any output-type column is missing or empty."""
    present_output_cols = [
        c for c in dataset.column_names if c.lower() in _OUTPUT_COLS
    ]
    if not present_output_cols:
        logger.debug("rule_remove_empty_outputs: no output columns found, skipping.")
        return dataset

    def _keep(row: dict) -> bool:
        return not any(_is_empty(row.get(c)) for c in present_output_cols)

    return dataset.filter(_keep, desc="remove_empty_outputs")


def rule_remove_empty_instructions(
    dataset: "Dataset",  # type: ignore[name-defined]  # noqa: F821
) -> "Dataset":
    """Drop rows where any instruction-type column is missing or empty."""
    present_instr_cols = [
        c for c in dataset.column_names if c.lower() in _INSTRUCTION_COLS
    ]
    if not present_instr_cols:
        logger.debug(
            "rule_remove_empty_instructions: no instruction columns found, skipping."
        )
        return dataset

    def _keep(row: dict) -> bool:
        return not any(_is_empty(row.get(c)) for c in present_instr_cols)

    return dataset.filter(_keep, desc="remove_empty_instructions")


def rule_remove_long_samples(
    dataset: "Dataset",  # type: ignore[name-defined]  # noqa: F821
    text_cols: list[str],
    max_tokens: int,
) -> "Dataset":
    """Drop rows whose total estimated token count exceeds *max_tokens*.

    Token estimate: ``sum(len(col_value) for text cols) / 4.0``
    """
    if max_tokens <= 0 or not text_cols:
        return dataset

    max_chars = max_tokens * _CHARS_PER_TOKEN

    def _keep(row: dict) -> bool:
        total_chars = sum(
            len(str(row[c])) for c in text_cols if row.get(c) is not None
        )
        return total_chars <= max_chars

    return dataset.filter(_keep, desc="remove_long_samples")


def rule_remove_duplicate_rows(
    dataset: "Dataset",  # type: ignore[name-defined]  # noqa: F821
) -> "Dataset":
    """Drop rows that are identical across all columns (full-row dedup).

    Uses a single batched ``to_dict()`` call rather than per-row
    ``dataset[i]`` fetches to avoid O(N) Arrow deserialisation overhead.
    """
    cols = dataset.column_names
    # Materialise all columns at once — one vectorised copy
    col_data = {c: dataset[c] for c in cols}
    n = len(dataset)

    seen: set[str] = set()
    keep_indices: list[int] = []

    for i in range(n):
        row = {c: col_data[c][i] for c in cols}
        fp = _full_row_fingerprint(row)
        if fp not in seen:
            seen.add(fp)
            keep_indices.append(i)

    return dataset.select(keep_indices)


def rule_remove_duplicate_code(
    dataset: "Dataset",  # type: ignore[name-defined]  # noqa: F821
    text_cols: list[str],
) -> "Dataset":
    """Drop rows whose code/text fingerprint (across all text columns) has
    already been seen, keeping the first occurrence.

    Uses batched column access rather than per-row ``dataset[i]`` fetches.
    """
    if not text_cols:
        logger.debug("rule_remove_duplicate_code: no text columns found, skipping.")
        return dataset

    # Materialise only the text columns needed
    col_data = {c: dataset[c] for c in text_cols}
    n = len(dataset)

    seen: set[str] = set()
    keep_indices: list[int] = []

    for i in range(n):
        row = {c: col_data[c][i] for c in text_cols}
        fp = _row_fingerprint(row, text_cols)
        if fp not in seen:
            seen.add(fp)
            keep_indices.append(i)

    return dataset.select(keep_indices)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def apply_pipeline(
    dataset: "Dataset",  # type: ignore[name-defined]  # noqa: F821
    config: CleaningConfig,
) -> tuple["Dataset", CleaningStats]:
    """Apply all enabled cleaning rules in a fixed, reproducible order.

    The original *dataset* is **never modified**.  Each rule produces a new
    ``datasets.Dataset`` object.

    Parameters
    ----------
    dataset:
        Input HF ``Dataset`` (single split, not a ``DatasetDict``).
    config:
        :class:`CleaningConfig` controlling which rules run and with what
        parameters.

    Returns
    -------
    tuple[Dataset, CleaningStats]
        The cleaned dataset and a stats object describing how many rows
        each rule removed.

    Rule execution order
    --------------------
    1.  ``strip_whitespace``           (normalisation — must come first)
    2.  ``normalize_line_endings``     (normalisation)
    3.  ``remove_empty_outputs``       (filter)
    4.  ``remove_empty_instructions``  (filter)
    5.  ``remove_long_samples``        (filter — after normalisation)
    6.  ``remove_duplicate_rows``      (dedup — after normalisation)
    7.  ``remove_duplicate_code``      (dedup — after row dedup)
    """
    stats = CleaningStats(original_rows=len(dataset))
    ds = dataset

    text_cols = _detect_text_columns(ds.column_names, config.text_columns)
    logger.debug("Detected text columns: %s", text_cols)

    # ── 1. Strip whitespace ──────────────────────────────────────────────────
    if config.strip_whitespace and text_cols:
        ds = rule_strip_whitespace(ds, text_cols)
        stats.after_strip_whitespace = len(ds)
        logger.debug("After strip_whitespace: %d rows", len(ds))
    else:
        stats.after_strip_whitespace = len(ds)

    # ── 2. Normalise line endings ────────────────────────────────────────────
    if config.normalize_line_endings and text_cols:
        ds = rule_normalize_line_endings(ds, text_cols)
        stats.after_normalize_line_endings = len(ds)
        logger.debug("After normalize_line_endings: %d rows", len(ds))
    else:
        stats.after_normalize_line_endings = len(ds)

    # ── 3. Remove empty outputs ──────────────────────────────────────────────
    if config.remove_empty_outputs:
        before = len(ds)
        ds = rule_remove_empty_outputs(ds)
        stats.removed_empty_outputs = before - len(ds)
        logger.debug("Removed %d empty-output rows.", stats.removed_empty_outputs)

    # ── 4. Remove empty instructions ─────────────────────────────────────────
    if config.remove_empty_instructions:
        before = len(ds)
        ds = rule_remove_empty_instructions(ds)
        stats.removed_empty_instructions = before - len(ds)
        logger.debug(
            "Removed %d empty-instruction rows.", stats.removed_empty_instructions
        )

    # ── 5. Remove long samples ───────────────────────────────────────────────
    if config.max_tokens > 0 and text_cols:
        before = len(ds)
        ds = rule_remove_long_samples(ds, text_cols, config.max_tokens)
        stats.removed_long_samples = before - len(ds)
        logger.debug(
            "Removed %d rows exceeding %d token limit.",
            stats.removed_long_samples,
            config.max_tokens,
        )

    # ── 6. Remove duplicate rows ─────────────────────────────────────────────
    if config.remove_duplicate_rows:
        before = len(ds)
        ds = rule_remove_duplicate_rows(ds)
        stats.removed_duplicate_rows = before - len(ds)
        logger.debug("Removed %d full-row duplicates.", stats.removed_duplicate_rows)

    # ── 7. Remove duplicate code ─────────────────────────────────────────────
    if config.remove_duplicate_code and text_cols:
        before = len(ds)
        ds = rule_remove_duplicate_code(ds, text_cols)
        stats.removed_duplicate_code = before - len(ds)
        logger.debug(
            "Removed %d code-fingerprint duplicates.", stats.removed_duplicate_code
        )

    stats.final_rows = len(ds)
    return ds, stats

