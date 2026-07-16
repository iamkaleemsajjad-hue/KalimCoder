"""
src/data/sources/parquet.py — ParquetSource.

Reads sharded parquet files from ``datasets/processed/<name>/`` using
PyArrow's :class:`pyarrow.parquet.ParquetFile` for memory-efficient
columnar reads.  The files must conform to the
:class:`~src.data.schema.CanonicalExample` schema (i.e. they were written
by :class:`~src.data.writer.ShardedWriter`).

Use when
--------
* Running ``build_training_dataset.py --processed-dir``.
* Running ``tokenize_dataset.py``.
* Running ``validate_dataset.py --streaming``.
* Re-ingesting already-processed data for a second mixing / filtering pass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterator

from src.data.schema import CanonicalExample, dict_to_canonical
from src.data.sources.base import DatasetSource

logger = logging.getLogger(__name__)


class ParquetSource(DatasetSource):
    """Memory-efficient reader for sharded CanonicalExample parquet files.

    Parameters
    ----------
    directory:
        Directory containing ``*.parquet`` files (train and/or val shards).
    glob:
        File glob pattern (default ``"*.parquet"``).
    batch_size:
        Number of rows per Arrow record batch (controls memory per read).
    dataset_name:
        Human-readable name used in log messages and ``CanonicalExample.dataset``
        override.  Defaults to the directory name.
    """

    def __init__(
        self,
        directory: Path,
        glob: str = "*.parquet",
        batch_size: int = 1000,
        dataset_name: str | None = None,
    ) -> None:
        self._directory = directory
        self._glob = glob
        self._batch_size = batch_size
        self._dataset_name = dataset_name or directory.name

    # ------------------------------------------------------------------
    # DatasetSource interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._dataset_name

    @property
    def estimated_rows(self) -> int | None:
        try:
            import pyarrow.parquet as pq
            total = 0
            for p in sorted(self._directory.glob(self._glob)):
                pf = pq.ParquetFile(p)
                total += pf.metadata.num_rows
            return total
        except Exception:
            return None

    @property
    def supports_streaming(self) -> bool:
        return True  # batch-reads — never loads entire file

    def iter_raw_rows(self) -> Iterator[dict]:
        """Yield raw dicts from all matching parquet files."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for ParquetSource. "
                "Install with:  pip install pyarrow"
            )

        files = sorted(self._directory.glob(self._glob))
        if not files:
            logger.warning("ParquetSource[%r]: no parquet files in %s", self._dataset_name, self._directory)
            return

        for pf_path in files:
            logger.debug("ParquetSource: reading %s", pf_path.name)
            pf = pq.ParquetFile(pf_path)
            for batch in pf.iter_batches(batch_size=self._batch_size):
                batch_dict = batch.to_pydict()
                n_rows = len(next(iter(batch_dict.values()), []))
                for i in range(n_rows):
                    yield {col: vals[i] for col, vals in batch_dict.items()}

    def iter_canonical_rows(self) -> Generator[CanonicalExample, None, None]:
        """Yield :class:`~src.data.schema.CanonicalExample` objects.

        Since the parquet files were written by :class:`~src.data.writer.ShardedWriter`
        they are already in canonical format — no adapter is needed.
        """
        dropped = 0
        for raw in self.iter_raw_rows():
            try:
                example = dict_to_canonical(raw)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ParquetSource: failed to deserialise row: %s", exc)
                dropped += 1
                continue
            yield example
        if dropped:
            logger.info(
                "ParquetSource[%r]: dropped %d unreadable rows.",
                self._dataset_name, dropped,
            )
