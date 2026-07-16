"""
src/data/sources/jsonl.py — JSONLSource.

Lazy-reads one or more ``.jsonl`` files without loading them into memory
at once.  No HuggingFace dependency required — works with pure Python.

Use when
--------
* You have locally generated synthetic data, evaluation sets, or curated
  prompts stored as ``.jsonl`` files.
* You want to inject custom data into the pipeline without uploading to HF Hub.
* You are testing the pipeline with small local datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator, Iterator

from src.data.adapters import get_adapter
from src.data.schema import CanonicalExample, validate_canonical
from src.data.sources.base import DatasetSource

logger = logging.getLogger(__name__)


class JSONLSource(DatasetSource):
    """Lazy JSONL file reader.

    Parameters
    ----------
    paths:
        One or more ``.jsonl`` file paths.  Processed in order.
    dataset_name:
        Registry entry name.
    adapter_hint:
        Adapter name; falls back to *dataset_name* then generic.
    license:
        SPDX identifier.
    task_type:
        Canonical task type.
    encoding:
        File encoding (default ``"utf-8"``).
    """

    def __init__(
        self,
        paths: Path | list[Path],
        dataset_name: str,
        adapter_hint: str | None = None,
        license: str = "unknown",
        task_type: str = "instruction",
        encoding: str = "utf-8",
    ) -> None:
        self._paths = [paths] if isinstance(paths, Path) else list(paths)
        self._dataset_name = dataset_name
        self._adapter = get_adapter(adapter_hint, dataset_name)
        self._license = license
        self._task_type = task_type
        self._encoding = encoding

    # ------------------------------------------------------------------
    # DatasetSource interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._dataset_name

    @property
    def estimated_rows(self) -> int | None:
        """Count lines across all files (fast line-count, no JSON parsing)."""
        try:
            total = 0
            for p in self._paths:
                with p.open("r", encoding=self._encoding, errors="replace") as fh:
                    total += sum(1 for line in fh if line.strip())
            return total
        except Exception:
            return None

    @property
    def supports_streaming(self) -> bool:
        return True  # reads line-by-line

    def iter_raw_rows(self) -> Iterator[dict]:
        """Yield raw dicts from all configured JSONL files, one per line."""
        for path in self._paths:
            if not path.exists():
                logger.warning("JSONLSource: file not found: %s", path)
                continue
            with path.open("r", encoding=self._encoding, errors="replace") as fh:
                for lineno, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.debug(
                            "JSONLSource[%s:%d]: JSON parse error — %s",
                            path.name, lineno, exc,
                        )

    def iter_canonical_rows(self) -> Generator[CanonicalExample, None, None]:
        """Yield :class:`~src.data.schema.CanonicalExample` objects."""
        dropped = 0
        for raw_row in self.iter_raw_rows():
            try:
                example = self._adapter(
                    raw_row,
                    self._dataset_name,
                    self._license,
                    self._task_type,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Adapter error: %s", exc)
                dropped += 1
                continue

            if example is None:
                dropped += 1
                continue

            errors = validate_canonical(example)
            if errors:
                dropped += 1
                continue

            yield example

        if dropped:
            logger.info("JSONLSource[%r]: dropped %d invalid rows.", self._dataset_name, dropped)
