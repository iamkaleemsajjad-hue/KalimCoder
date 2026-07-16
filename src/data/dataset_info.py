"""
src/data/dataset_info.py — LLaMA Factory dataset_info.json manager.

This module owns all logic for creating and updating ``data/dataset_info.json``,
which is the dataset registry that LLaMA Factory reads at training time.

It is intentionally a library module (not a script) so that
``scripts/build_training_dataset.py`` and any future tool can call it
with a single function call — no manual JSON editing required.

LLaMA Factory dataset_info.json format (alpaca)
-------------------------------------------------
{
    "<dataset_name>": {
        "file_name": "<path/to/file.jsonl>",
        "formatting": "alpaca",
        "columns": {
            "prompt":    "<instruction_column>",
            "query":     "<input_column>",
            "response":  "<output_column>",
            "system":    "<system_column>"   # optional
        }
    }
}

Usage
-----
    from src.data.dataset_info import register_dataset, write_dataset_info

    register_dataset(
        dataset_name="kalimcoder_sft",
        file_path=Path("datasets/instruction/kalimcoder_sft.jsonl"),
    )
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR: Path = _PROJECT_ROOT / "data"
_DATASET_INFO_FILENAME = "dataset_info.json"


# ---------------------------------------------------------------------------
# Default column mapping (CanonicalExample → LLaMA Factory Alpaca columns)
# ---------------------------------------------------------------------------

_DEFAULT_COLUMNS: dict[str, str] = {
    "prompt":   "instruction",
    "query":    "input",
    "response": "output",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_entry(
    file_path: str | Path,
    formatting: str = "alpaca",
    columns: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a single LLaMA Factory dataset_info entry dict.

    Parameters
    ----------
    file_path:
        Path to the dataset file (.json or .jsonl).  Stored as a string
        relative to the project root if possible, otherwise absolute.
    formatting:
        LLaMA Factory formatting type.  Use ``"alpaca"`` for instruction /
        input / output triplets, ``"sharegpt"`` for multi-turn conversations.
    columns:
        Column-name mapping.  Defaults to the CanonicalExample layout:
        ``{"prompt": "instruction", "query": "input", "response": "output"}``.

    Returns
    -------
    dict
        Entry dict ready to be inserted into ``dataset_info.json``.
    """
    resolved = Path(file_path).resolve()
    try:
        rel = resolved.relative_to(_PROJECT_ROOT)
        path_str = str(rel).replace("\\", "/")
    except ValueError:
        # file_path is outside the project — keep absolute
        path_str = str(resolved).replace("\\", "/")

    return {
        "file_name": path_str,
        "formatting": formatting,
        "columns": columns or dict(_DEFAULT_COLUMNS),
    }


def load_dataset_info(data_dir: Path | None = None) -> dict[str, Any]:
    """Read and return the current ``dataset_info.json`` as a dict.

    Returns an empty dict if the file does not yet exist.

    Parameters
    ----------
    data_dir:
        Directory containing ``dataset_info.json``.
        Defaults to ``<project_root>/data/``.
    """
    info_path = _resolve_info_path(data_dir)
    if not info_path.exists():
        return {}
    try:
        return json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("dataset_info.json is malformed (%s) — treating as empty.", exc)
        return {}


def register_dataset(
    dataset_name: str,
    file_path: str | Path,
    formatting: str = "alpaca",
    columns: dict[str, str] | None = None,
    data_dir: Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Register a dataset in ``data/dataset_info.json``.

    If the file already contains an entry for *dataset_name* and
    ``overwrite=False``, the existing entry is left unchanged.

    Parameters
    ----------
    dataset_name:
        The key used in ``dataset_info.json`` and in training YAML
        (e.g. ``"kalimcoder_sft"``).
    file_path:
        Path to the dataset file on disk.
    formatting:
        LLaMA Factory formatting type (``"alpaca"`` or ``"sharegpt"``).
    columns:
        Column-name mapping override.
    data_dir:
        Directory where ``dataset_info.json`` lives.
        Defaults to ``<project_root>/data/``.
    overwrite:
        When ``True`` (default), always update the entry even if it exists.

    Returns
    -------
    Path
        Absolute path to the written ``dataset_info.json``.
    """
    info_path = _resolve_info_path(data_dir)
    current = load_dataset_info(data_dir)

    if dataset_name in current and not overwrite:
        logger.info(
            "dataset_info.json already contains %r — skipping (overwrite=False).",
            dataset_name,
        )
        return info_path

    entry = build_entry(file_path=file_path, formatting=formatting, columns=columns)
    current[dataset_name] = entry

    _write_atomic(info_path, current)
    logger.info(
        "Registered %r in %s  (file_name=%r)",
        dataset_name, info_path, entry["file_name"],
    )
    return info_path


def unregister_dataset(
    dataset_name: str,
    data_dir: Path | None = None,
) -> bool:
    """Remove a dataset entry from ``dataset_info.json``.

    Parameters
    ----------
    dataset_name:
        Key to remove.
    data_dir:
        Directory of ``dataset_info.json``.

    Returns
    -------
    bool
        ``True`` if the entry was present and removed, ``False`` if not found.
    """
    info_path = _resolve_info_path(data_dir)
    current = load_dataset_info(data_dir)
    if dataset_name not in current:
        return False
    del current[dataset_name]
    _write_atomic(info_path, current)
    logger.info("Unregistered %r from %s", dataset_name, info_path)
    return True


def is_registered(
    dataset_name: str,
    data_dir: Path | None = None,
) -> bool:
    """Return ``True`` if *dataset_name* is in ``dataset_info.json``."""
    return dataset_name in load_dataset_info(data_dir)


def get_dataset_file(
    dataset_name: str,
    data_dir: Path | None = None,
) -> Path | None:
    """Return the resolved file path for a registered dataset, or ``None``."""
    info = load_dataset_info(data_dir)
    entry = info.get(dataset_name)
    if entry is None:
        return None
    raw_path = entry.get("file_name", "")
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_info_path(data_dir: Path | None) -> Path:
    base = data_dir or _DEFAULT_DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / _DATASET_INFO_FILENAME


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write *data* as JSON to *path* atomically (temp-file + os.replace)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)
