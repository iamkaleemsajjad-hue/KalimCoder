"""
src/data/sources/local_arrow.py — LocalArrowSource.

Reads existing HuggingFace Arrow datasets from disk (``datasets/raw/<name>/``).
This is the backward-compatible source that lets the new streaming pipeline
consume data that was already downloaded by ``scripts/download_datasets.py``.

Use when
--------
* You have run ``make download-data`` and have Arrow files in ``datasets/raw/``.
* You want to reprocess existing data with the new quality scoring / dedup
  pipeline without re-downloading.
* You are running ``run_pipeline.py --offline``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterator

from src.data.adapters import get_adapter
from src.data.schema import CanonicalExample, validate_canonical
from src.data.sources.base import DatasetSource

logger = logging.getLogger(__name__)


class LocalArrowSource(DatasetSource):
    """Reads a HuggingFace Arrow dataset saved to disk via ``save_to_disk()``.

    Parameters
    ----------
    path:
        Absolute path to the Arrow dataset directory (contains
        ``dataset_info.json`` or ``dataset_dict.json``).
    dataset_name:
        Registry entry name.  Used in :class:`~src.data.schema.CanonicalExample`
        and for adapter lookup.
    adapter_hint:
        Adapter name from the registry entry.  Falls back to *dataset_name*.
    license:
        SPDX identifier.
    task_type:
        Canonical task type.
    """

    def __init__(
        self,
        path: Path,
        dataset_name: str,
        adapter_hint: str | None = None,
        license: str = "unknown",
        task_type: str = "instruction",
    ) -> None:
        self._path = path
        self._dataset_name = dataset_name
        self._adapter = get_adapter(adapter_hint, dataset_name)
        self._license = license
        self._task_type = task_type
        self._dataset = None  # lazy-loaded

    # ------------------------------------------------------------------
    # DatasetSource interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._dataset_name

    @property
    def estimated_rows(self) -> int | None:
        try:
            ds = self._load()
            return len(ds)
        except Exception:
            return None

    @property
    def supports_streaming(self) -> bool:
        return False  # reads from Arrow cache (already local)

    def iter_raw_rows(self) -> Iterator[dict]:
        """Yield raw row dicts from the Arrow dataset."""
        ds = self._load()
        yield from ds

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
                logger.debug("Adapter error on row from %r: %s", self._dataset_name, exc)
                dropped += 1
                continue

            if example is None:
                dropped += 1
                continue

            errors = validate_canonical(example)
            if errors:
                logger.debug(
                    "Invalid canonical row from %r: %s", self._dataset_name, errors
                )
                dropped += 1
                continue

            yield example

        if dropped:
            logger.info(
                "LocalArrowSource[%r]: dropped %d invalid rows.", self._dataset_name, dropped
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self):
        """Lazy-load (and cache) the HF dataset from disk."""
        if self._dataset is not None:
            return self._dataset

        # Local import to avoid the datasets/directory shadowing issue
        import sys
        from pathlib import Path as _Path
        _proj = str(_Path(__file__).resolve().parents[3])
        _shadow = ["", ".", str(_Path(".").resolve()), _proj]
        for _e in _shadow:
            while _e in sys.path:
                sys.path.remove(_e)
        try:
            import datasets as hf_datasets
        finally:
            if _proj not in sys.path:
                sys.path.insert(0, _proj)

        raw = hf_datasets.load_from_disk(str(self._path))
        if isinstance(raw, hf_datasets.DatasetDict):
            split = list(raw.keys())[0]
            logger.debug("LocalArrowSource[%r]: using split %r.", self._dataset_name, split)
            self._dataset = raw[split]
        else:
            self._dataset = raw
        logger.info(
            "LocalArrowSource[%r]: loaded %d rows.", self._dataset_name, len(self._dataset)
        )
        return self._dataset
