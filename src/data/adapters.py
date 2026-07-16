"""
src/data/adapters.py — Per-dataset schema adapters for KalimCoder.

Each adapter maps a raw row ``dict`` to a :class:`~src.data.schema.CanonicalExample`
(or ``None`` to silently drop the row).  Adapters are keyed by the ``adapter``
field in ``configs/datasets.yaml`` (falling back to the dataset ``name``).

Adding a new dataset
--------------------
1. Write an ``_adapt_<name>`` function following the pattern below.
2. Register it in :data:`ADAPTER_REGISTRY`.
3. Set ``adapter: <name>`` in ``configs/datasets.yaml`` for the dataset entry.

The generic fallback :func:`_adapt_generic` covers most instruction-format
datasets without any registration.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from src.data.schema import (
    UNKNOWN,
    CanonicalExample,
    make_example,
)

logger = logging.getLogger(__name__)

# Type alias
Adapter = Callable[[dict[str, Any], str, str, str], Optional[CanonicalExample]]
"""
Each adapter has the signature::

    adapter(row, dataset_name, license, task_type) -> CanonicalExample | None

Parameters
----------
row:
    Raw dict from the source.
dataset_name:
    Registry entry name (stored in ``CanonicalExample.dataset``).
license:
    SPDX identifier from the registry entry.
task_type:
    Canonical task type from the registry entry.
"""


# ---------------------------------------------------------------------------
# Coerce helper
# ---------------------------------------------------------------------------


def _coerce(value: Any) -> str:
    """Return a stripped string; empty string for None / non-str values."""
    if value is None:
        return ""
    return str(value).strip()


def _detect_language(row: dict, field_hints: tuple[str, ...] = ("language", "lang", "programming_language")) -> str:
    """Extract language from a row dict; returns UNKNOWN if not found."""
    for f in field_hints:
        val = _coerce(row.get(f))
        if val:
            return val.lower()
    return UNKNOWN


# ---------------------------------------------------------------------------
# Dataset-specific adapters
# ---------------------------------------------------------------------------


# ── OPC SFT Stage 1 ─────────────────────────────────────────────────────────
# Schema: { "instruction": str, "output": str, ... }
def _adapt_opc_sft_stage1(
    row: dict, dataset: str, license: str, task_type: str
) -> Optional[CanonicalExample]:
    instr = _coerce(row.get("instruction") or row.get("prompt") or row.get("text"))
    out = _coerce(row.get("output") or row.get("response"))
    if not instr or not out:
        return None
    return make_example(
        dataset=dataset,
        instruction=instr,
        output=out,
        task_type=task_type or "instruction",
        license=license,
    )


# ── The Stack v2 ────────────────────────────────────────────────────────────
# Schema: { "content": str, "lang": str, ... }
def _adapt_the_stack_v2(
    row: dict, dataset: str, license: str, task_type: str
) -> Optional[CanonicalExample]:
    content = _coerce(row.get("content") or row.get("text") or row.get("code"))
    if not content:
        return None
    lang = _detect_language(row, ("lang", "language", "programming_language"))
    instr = f"Complete the following {lang} code:" if lang != UNKNOWN else "Complete the following code:"
    repo = _coerce(row.get("repository_name") or row.get("repo_name") or row.get("repo") or "")
    repo_url = f"https://github.com/{repo}" if repo and "/" in repo else ""
    return make_example(
        dataset=dataset,
        instruction=instr,
        output=content,
        task_type=task_type or "completion",
        language=lang,
        repository=repo_url,
        license=_coerce(row.get("license") or license),
    )


# ── CodeSearchNet ────────────────────────────────────────────────────────────
# Schema: { "func_code_string": str, "func_documentation_string": str, ... }
def _adapt_code_search_net(
    row: dict, dataset: str, license: str, task_type: str
) -> Optional[CanonicalExample]:
    code = _coerce(
        row.get("func_code_string")
        or row.get("whole_func_string")
        or row.get("code")
    )
    doc = _coerce(
        row.get("func_documentation_string")
        or row.get("docstring")
        or row.get("summary")
    )
    if not code:
        return None
    lang = _detect_language(row, ("language", "lang"))
    if doc:
        instr = f"Write a function that does the following:\n{doc}"
        actual_task = "instruction"
    else:
        instr = f"Complete the following {lang} function:" if lang != UNKNOWN else "Complete the following function:"
        actual_task = "completion"
    return make_example(
        dataset=dataset,
        instruction=instr,
        output=code,
        task_type=task_type or actual_task,
        language=lang,
        license=license,
    )


# ── SWE-bench Verified ───────────────────────────────────────────────────────
# Schema: { "problem_statement": str, "patch": str, ... }
def _adapt_swe_bench_verified(
    row: dict, dataset: str, license: str, task_type: str
) -> Optional[CanonicalExample]:
    problem = _coerce(row.get("problem_statement") or row.get("text") or row.get("issue"))
    patch = _coerce(row.get("patch") or row.get("solution") or row.get("output"))
    if not problem or not patch:
        return None
    repo = _coerce(row.get("repo") or row.get("repository") or "")
    repo_url = f"https://github.com/{repo}" if repo else ""
    return make_example(
        dataset=dataset,
        instruction=f"Fix the following software issue:\n{problem}",
        output=patch,
        task_type=task_type or "debugging",
        language="python",   # SWE-bench is Python-focused
        repository=repo_url,
        license=license,
    )


# ── Generic heuristic ────────────────────────────────────────────────────────
_INSTRUCTION_FIELDS = (
    "instruction", "prompt", "question", "problem_statement",
    "text", "body", "hint", "query",
)
_OUTPUT_FIELDS = (
    "output", "response", "answer", "solution", "code",
    "content", "patch", "func_code_string",
)
_INPUT_FIELDS = ("input", "context", "auxiliary", "stdin")


def _adapt_generic(
    row: dict, dataset: str, license: str, task_type: str
) -> Optional[CanonicalExample]:
    """Best-effort heuristic adapter for unknown dataset schemas."""
    instr = next(
        (_coerce(row.get(f)) for f in _INSTRUCTION_FIELDS if _coerce(row.get(f))),
        "",
    )
    out = next(
        (_coerce(row.get(f)) for f in _OUTPUT_FIELDS if _coerce(row.get(f))),
        "",
    )
    inp = next(
        (_coerce(row.get(f)) for f in _INPUT_FIELDS if _coerce(row.get(f))),
        "",
    )
    if not instr or not out:
        return None
    lang = _detect_language(row)
    return make_example(
        dataset=dataset,
        instruction=instr,
        input=inp,
        output=out,
        task_type=task_type or "instruction",
        language=lang,
        license=license,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: dict[str, Adapter] = {
    "opc_sft_stage1":     _adapt_opc_sft_stage1,
    "the_stack_v2":       _adapt_the_stack_v2,
    "code_search_net":    _adapt_code_search_net,
    "swe_bench_verified": _adapt_swe_bench_verified,
    "generic":            _adapt_generic,
}


def get_adapter(name: str | None, fallback_name: str | None = None) -> Adapter:
    """Look up and return an adapter by name.

    Lookup order:
    1. *name* (the ``adapter`` field from ``configs/datasets.yaml``)
    2. *fallback_name* (the dataset ``name`` field)
    3. :func:`_adapt_generic`

    Parameters
    ----------
    name:
        Adapter hint from the registry entry (may be ``None``).
    fallback_name:
        Dataset name to try if *name* is ``None`` or not found.
    """
    for key in (name, fallback_name):
        if key and key in ADAPTER_REGISTRY:
            return ADAPTER_REGISTRY[key]
    return _adapt_generic
