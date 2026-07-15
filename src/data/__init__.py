"""
data/ — Dataset loading, cleaning, formatting, and tokenization utilities.

Modules:
  registry.py    — Dataset registry: load configs/datasets.yaml, validate schema
  loader.py      — Load raw datasets from disk or HF Hub
  cleaner.py     — Filter and deduplicate examples
  formatter.py   — Convert raw records to instruction-response format
  tokenizer.py   — Tokenize and pack sequences for training
"""

from src.data.registry import DatasetEntry, get_enabled_datasets, load_registry

__all__ = [
    "DatasetEntry",
    "load_registry",
    "get_enabled_datasets",
]
