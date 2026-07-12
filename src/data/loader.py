"""
loader.py — Load datasets from disk (JSONL) or Hugging Face Hub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from datasets import Dataset, load_dataset as hf_load_dataset


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_hf_dataset(repo_id: str, split: str = "train", **kwargs) -> Dataset:
    """Load a dataset from Hugging Face Hub."""
    return hf_load_dataset(repo_id, split=split, **kwargs)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Memory-efficient JSONL iterator for large files."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
