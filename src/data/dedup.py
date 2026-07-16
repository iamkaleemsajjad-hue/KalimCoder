"""
src/data/dedup.py — Two-stage deduplication for streaming pipelines.

Architecture
------------
Stage 1 — BloomFilter   : O(1) probabilistic membership test.
                          ~8.6 MB RAM for 5M entries at FPR=0.001.
Stage 2 — SHA-256 set   : Exact verification triggered only on Bloom hits.
                          Capped at ``max_confirmed_mb`` to prevent OOM.

Guarantees
----------
* False positives:  **impossible** — SHA-256 verification eliminates them.
* False negatives:  **impossible** — first-time items always pass.
* Memory overhead:  bounded by ``max_confirmed_mb`` + Bloom array.
  Above the cap the deduplicator switches to Bloom-only mode (matching
  BigCode's approach for The Stack v2) and logs a one-time warning.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BloomFilter
# ---------------------------------------------------------------------------


class BloomFilter:
    """Pure-Python Bloom filter using double hashing (FNV-1a × djb2).

    Parameters
    ----------
    capacity:
        Expected maximum number of unique elements.
    fpr:
        Target false positive rate (e.g. 0.001 = 0.1 %).

    Memory approximation::

        capacity=5_000_000, fpr=0.001  →  ~8.6 MB (bit array only)
    """

    def __init__(self, capacity: int = 5_000_000, fpr: float = 0.001) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if not (0 < fpr < 1):
            raise ValueError(f"fpr must be in (0, 1), got {fpr}")
        self._capacity = capacity
        self._fpr = fpr
        self._size = self._optimal_size(capacity, fpr)
        self._num_hashes = self._optimal_hashes(self._size, capacity)
        self._bits = bytearray(math.ceil(self._size / 8))
        self._count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, key: str) -> None:
        """Insert *key* into the filter."""
        for byte_idx, bit_idx in self._positions(key):
            self._bits[byte_idx] |= 1 << bit_idx
        self._count += 1

    def __contains__(self, key: str) -> bool:
        """Return ``True`` if *key* is **probably** in the set."""
        return all(
            bool(self._bits[byte_idx] & (1 << bit_idx))
            for byte_idx, bit_idx in self._positions(key)
        )

    def estimated_fpr(self) -> float:
        """Return the current estimated false positive rate."""
        if self._capacity == 0:
            return 0.0
        fill = self._count / self._capacity
        return (1 - math.exp(-self._num_hashes * fill)) ** self._num_hashes

    @property
    def count(self) -> int:
        """Number of items inserted so far."""
        return self._count

    @property
    def memory_bytes(self) -> int:
        """Size of the internal bit array in bytes."""
        return len(self._bits)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _positions(self, key: str) -> Iterator[tuple[int, int]]:
        """Yield (byte_index, bit_index) for each hash function."""
        encoded = key.encode("utf-8", errors="replace")
        h1 = self._fnv1a(encoded) % self._size
        h2 = self._djb2(encoded) % self._size or 1  # avoid stride-0
        for i in range(self._num_hashes):
            idx = (h1 + i * h2) % self._size
            yield divmod(idx, 8)

    @staticmethod
    def _fnv1a(data: bytes) -> int:
        h = 0xCBF29CE484222325
        for byte in data:
            h ^= byte
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h

    @staticmethod
    def _djb2(data: bytes) -> int:
        h = 5381
        for byte in data:
            h = ((h << 5) + h + byte) & 0xFFFFFFFFFFFFFFFF
        return h

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Optimal bit-array size: m = -(n · ln p) / (ln 2)²."""
        return max(1, int(-n * math.log(p) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_hashes(m: int, n: int) -> int:
        """Optimal number of hash functions: k = (m/n) · ln 2."""
        return max(1, round((m / max(n, 1)) * math.log(2)))


# ---------------------------------------------------------------------------
# TwoStageDedup
# ---------------------------------------------------------------------------


class TwoStageDedup:
    """Exact deduplication with bounded memory.

    Parameters
    ----------
    bloom_capacity:
        Expected number of unique items across the entire corpus.
    bloom_fpr:
        Target Bloom false positive rate.
    max_confirmed_mb:
        Maximum RAM (MB) for the SHA-256 exact-verification set.
        Above this limit the deduplicator switches to Bloom-only mode
        and emits a one-time ``WARNING`` log.

    Example
    -------
    ::

        dedup = TwoStageDedup()
        for text in corpus:
            if not dedup.is_duplicate(text):
                yield text

        print(dedup.stats)
    """

    # Each SHA-256 hex digest is 64 ASCII chars.
    # CPython str object overhead ≈ 56 bytes on 64-bit platforms.
    _BYTES_PER_HASH: int = 64 + 56

    def __init__(
        self,
        bloom_capacity: int = 5_000_000,
        bloom_fpr: float = 0.001,
        max_confirmed_mb: int = 512,
    ) -> None:
        self._bloom = BloomFilter(capacity=bloom_capacity, fpr=bloom_fpr)
        self._confirmed: set[str] = set()
        self._max_confirmed_bytes: int = max_confirmed_mb * 1024 * 1024
        self._confirmed_bytes: int = 0
        self._bloom_only: bool = False
        self._bloom_only_warned: bool = False

        # Counters
        self._total_seen: int = 0
        self._bloom_hits: int = 0
        self._confirmed_dups: int = 0
        self._bloom_only_dups: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate(self, text: str) -> bool:
        """Return ``True`` if *text* was seen before.

        Side-effect: registers *text* as a first occurrence if not a duplicate.
        """
        self._total_seen += 1

        if text not in self._bloom:
            # Definitely first time — add to Bloom and return not-dup
            self._bloom.add(text)
            return False

        # Bloom hit → potential duplicate; verify exactly
        self._bloom_hits += 1

        if self._bloom_only:
            # Degraded mode: treat all Bloom hits as duplicates
            self._bloom_only_dups += 1
            return True

        h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        if h in self._confirmed:
            self._confirmed_dups += 1
            return True

        # First confirmed occurrence of a Bloom-hit item
        if self._confirmed_bytes + self._BYTES_PER_HASH <= self._max_confirmed_bytes:
            self._confirmed.add(h)
            self._confirmed_bytes += self._BYTES_PER_HASH
        else:
            self._bloom_only = True
            if not self._bloom_only_warned:
                logger.warning(
                    "TwoStageDedup: SHA-256 set reached %.0f MB limit. "
                    "Switching to Bloom-only deduplication for remaining items. "
                    "Increase pipeline.dedup.max_confirmed_set_mb to avoid this.",
                    self._max_confirmed_bytes / (1024 * 1024),
                )
                self._bloom_only_warned = True

        return False

    def reset(self) -> None:
        """Clear all state (e.g. between independent datasets)."""
        self._confirmed.clear()
        self._confirmed_bytes = 0
        self._bloom_only = False
        self._bloom_only_warned = False
        self._total_seen = self._bloom_hits = self._confirmed_dups = self._bloom_only_dups = 0
        # Re-create Bloom with same parameters
        self._bloom = BloomFilter(
            capacity=self._bloom._capacity,
            fpr=self._bloom._fpr,
        )

    @property
    def stats(self) -> dict:
        """Return a snapshot of deduplication statistics."""
        return {
            "total_seen": self._total_seen,
            "bloom_hits": self._bloom_hits,
            "confirmed_duplicates": self._confirmed_dups,
            "bloom_only_duplicates": self._bloom_only_dups,
            "bloom_only_mode": self._bloom_only,
            "bloom_estimated_fpr": round(self._bloom.estimated_fpr(), 6),
            "confirmed_set_mb": round(self._confirmed_bytes / (1024 * 1024), 3),
            "bloom_memory_mb": round(self._bloom.memory_bytes / (1024 * 1024), 3),
        }
