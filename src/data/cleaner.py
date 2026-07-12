"""
cleaner.py — Filter, deduplicate, and quality-check dataset examples.
"""

from __future__ import annotations

import hashlib
from typing import Callable


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
