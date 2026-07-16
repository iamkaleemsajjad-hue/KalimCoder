"""
src/data/sources/base.py — Abstract base class for all dataset sources.

All pipeline sources must subclass :class:`DatasetSource` and implement:

* :meth:`iter_raw_rows` — yield raw ``dict`` rows from the underlying storage.
* :meth:`iter_canonical_rows` — apply the per-source adapter and yield
  :class:`~src.data.schema.CanonicalExample` objects.
* :attr:`estimated_rows` — approximate row count or ``None`` for streaming.
* :attr:`supports_streaming` — ``True`` if no full materialisation is needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator, Iterator

from src.data.schema import CanonicalExample


class DatasetSource(ABC):
    """Abstract base for all dataset sources in the KalimCoder pipeline.

    Contract
    --------
    * :meth:`iter_raw_rows` must be a lazy generator — no buffering of the
      entire dataset in memory at any point.
    * :meth:`iter_canonical_rows` must silently skip invalid rows (log +
      continue) rather than raising, to keep the pipeline fault-tolerant.
    * Implementations must be re-entrant: calling :meth:`iter_raw_rows`
      twice produces independent iterators from the start.
    """

    @abstractmethod
    def iter_raw_rows(self) -> Iterator[dict]:
        """Yield raw ``dict`` rows from the underlying storage.

        Each call starts a fresh iterator from the beginning of the source.
        """
        ...

    @abstractmethod
    def iter_canonical_rows(self) -> Generator[CanonicalExample, None, None]:
        """Yield :class:`~src.data.schema.CanonicalExample` objects.

        Applies the per-source adapter internally.  Invalid or empty rows
        are logged at DEBUG level and skipped — never raised.
        """
        ...

    @property
    @abstractmethod
    def estimated_rows(self) -> int | None:
        """Approximate total row count, or ``None`` for streaming sources."""
        ...

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """``True`` if this source streams without materialising to disk."""
        ...

    @property
    def name(self) -> str:
        """Human-readable identifier used in log messages."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"streaming={self.supports_streaming}, "
            f"estimated_rows={self.estimated_rows})"
        )
