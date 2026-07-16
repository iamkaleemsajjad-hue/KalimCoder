"""
src/data/writer.py — Sharded parquet / JSONL writer for the streaming pipeline.

:class:`ShardedWriter` buffers :class:`~src.data.schema.CanonicalExample` objects
in memory and flushes to disk when the buffer reaches ``shard_size`` rows.
Writes are **atomic** — data goes to a ``.tmp`` file first, then
:func:`os.replace` renames it to the final path, so a crash never produces
a corrupt or partial shard.

Output layout::

    datasets/processed/<dataset_name>/
        train-00001.parquet
        train-00002.parquet
        val-00001.parquet
        _metadata.json

Train / validation split
    ``val_ratio`` fraction of each buffer is reserved for validation.
    The split is deterministic (``seed``-controlled shuffle before split).
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.schema import CanonicalExample, canonical_to_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WriterStats
# ---------------------------------------------------------------------------


@dataclass
class WriterStats:
    """Aggregate statistics returned by :meth:`ShardedWriter.close`."""
    train_rows: int = 0
    val_rows: int = 0
    train_files: list[str] = field(default_factory=list)
    val_files: list[str] = field(default_factory=list)
    bytes_written: int = 0
    shards_flushed: int = 0


# ---------------------------------------------------------------------------
# ShardedWriter
# ---------------------------------------------------------------------------


class ShardedWriter:
    """Buffers examples and writes sharded parquet (or JSONL) files atomically.

    Parameters
    ----------
    out_dir:
        Output directory (created if it does not exist).
    shard_size:
        Number of rows per output shard file.
    val_ratio:
        Fraction of rows reserved for the validation split ``[0, 0.5]``.
    seed:
        Random seed for reproducible val sampling.
    fmt:
        Output format: ``"parquet"`` (default) or ``"jsonl"``.
    compress:
        Parquet compression codec (``"snappy"``, ``"zstd"``, ``"none"``).
    dataset_name:
        Used for log messages and ``_metadata.json``.
    """

    def __init__(
        self,
        out_dir: Path,
        shard_size: int = 100_000,
        val_ratio: float = 0.05,
        seed: int = 42,
        fmt: str = "parquet",
        compress: str = "snappy",
        dataset_name: str = "",
    ) -> None:
        self._out_dir = out_dir
        self._shard_size = max(1, shard_size)
        self._val_ratio = max(0.0, min(val_ratio, 0.5))
        self._rng = random.Random(seed)
        self._fmt = fmt.lower()
        self._compress = compress
        self._dataset_name = dataset_name or out_dir.name

        # Buffer
        self._buffer: list[dict] = []

        # Stats
        self._stats = WriterStats()
        self._train_shard_idx = 0
        self._val_shard_idx = 0

        out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, ex: CanonicalExample) -> None:
        """Buffer one example.  Flushes automatically when the buffer is full."""
        self._buffer.append(canonical_to_dict(ex))
        if len(self._buffer) >= self._shard_size:
            self.flush()

    def flush(self) -> tuple[Path | None, Path | None]:
        """Force-write the current buffer to disk.

        Returns
        -------
        tuple[Path | None, Path | None]
            ``(train_path, val_path)`` — either may be ``None`` if empty.
        """
        if not self._buffer:
            return None, None

        # Shuffle buffer deterministically before split
        self._rng.shuffle(self._buffer)

        n_val = round(len(self._buffer) * self._val_ratio)
        n_train = len(self._buffer) - n_val

        train_rows = self._buffer[:n_train]
        val_rows = self._buffer[n_train:]   # tail of the buffer, not [n_val:]
        self._buffer = []

        train_path = val_path = None

        if train_rows:
            self._train_shard_idx += 1
            train_path = self._write_shard(
                train_rows, "train", self._train_shard_idx
            )
            if train_path:
                self._stats.train_rows += len(train_rows)
                self._stats.train_files.append(str(train_path))
                self._stats.bytes_written += train_path.stat().st_size

        if val_rows:
            self._val_shard_idx += 1
            val_path = self._write_shard(
                val_rows, "val", self._val_shard_idx
            )
            if val_path:
                self._stats.val_rows += len(val_rows)
                self._stats.val_files.append(str(val_path))
                self._stats.bytes_written += val_path.stat().st_size

        self._stats.shards_flushed += 1
        logger.debug(
            "Flushed shard %d: train=%d, val=%d → %s",
            self._stats.shards_flushed,
            len(train_rows),
            len(val_rows),
            self._out_dir.name,
        )
        return train_path, val_path

    def close(self) -> WriterStats:
        """Flush any remaining buffer and write ``_metadata.json``."""
        if self._buffer:
            self.flush()
        self._write_metadata()
        logger.info(
            "ShardedWriter[%r] closed: train=%d rows (%d files), val=%d rows (%d files), %.1f MB.",
            self._dataset_name,
            self._stats.train_rows,
            len(self._stats.train_files),
            self._stats.val_rows,
            len(self._stats.val_files),
            self._stats.bytes_written / (1024 * 1024),
        )
        return self._stats

    @property
    def buffered_count(self) -> int:
        """Number of examples currently in the buffer."""
        return len(self._buffer)

    @property
    def train_rows_written(self) -> int:
        return self._stats.train_rows

    @property
    def val_rows_written(self) -> int:
        return self._stats.val_rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shard_path(self, split: str, idx: int) -> Path:
        ext = "parquet" if self._fmt == "parquet" else "jsonl"
        return self._out_dir / f"{split}-{idx:05d}.{ext}"

    def _write_shard(
        self, rows: list[dict], split: str, idx: int
    ) -> Path | None:
        """Atomically write *rows* to a shard file."""
        path = self._shard_path(split, idx)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            if self._fmt == "parquet":
                self._write_parquet(rows, tmp)
            else:
                self._write_jsonl(rows, tmp)
            os.replace(tmp, path)
            return path
        except Exception as exc:
            logger.error(
                "ShardedWriter: failed to write %s — %s", path.name, exc, exc_info=True
            )
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            return None

    def _write_parquet(self, rows: list[dict], path: Path) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for parquet output. "
                "Install with:  pip install pyarrow"
            )
        # Build a pyarrow table from the list of dicts
        if not rows:
            return
        # Use the keys from the first row; all rows share the same schema
        keys = list(rows[0].keys())
        columns: dict[str, list] = {k: [] for k in keys}
        for row in rows:
            for k in keys:
                columns[k].append(row.get(k))

        table = pa.table(columns)
        codec = None if self._compress == "none" else self._compress
        pq.write_table(table, str(path), compression=codec)

    @staticmethod
    def _write_jsonl(rows: list[dict], path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _write_metadata(self) -> None:
        """Write a ``_metadata.json`` summary file."""
        meta: dict[str, Any] = {
            "dataset": self._dataset_name,
            "format": self._fmt,
            "compression": self._compress,
            "train_rows": self._stats.train_rows,
            "val_rows": self._stats.val_rows,
            "train_files": [Path(p).name for p in self._stats.train_files],
            "val_files": [Path(p).name for p in self._stats.val_files],
            "shards_flushed": self._stats.shards_flushed,
            "bytes_written": self._stats.bytes_written,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta_path = self._out_dir / "_metadata.json"
        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp, meta_path)
