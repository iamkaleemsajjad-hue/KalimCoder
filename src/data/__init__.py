"""
data/ — Dataset loading, cleaning, formatting, and tokenization utilities.

Legacy modules (preserved for backwards compatibility):
  registry.py    — Dataset registry: load configs/datasets.yaml, validate schema
  loader.py      — Load raw datasets from disk or HF Hub
  cleaner.py     — Filter and deduplicate examples
  formatter.py   — Convert raw records to instruction-response format
  tokenizer.py   — Tokenize and pack sequences for training

Streaming pipeline modules (new):
  schema.py       — CanonicalExample dataclass + helpers
  sources/        — DatasetSource ABC and per-source implementations
  adapters.py     — Per-dataset row→CanonicalExample adapters
  quality.py      — QualityScorer: multi-signal quality scoring
  dedup.py        — TwoStageDedup: Bloom + SHA-256 deduplication
  streaming.py    — build_pipeline(): lazy generator stage chain
  writer.py       — ShardedWriter: atomic parquet shard output
  mixer.py        — DatasetMixer: enforces mixture ratios
  manifest.py     — ExperimentManifest: reproducibility metadata
  state.py        — StateManager: crash-safe shard checkpointing
  dataset_info.py — LLaMA Factory dataset_info.json manager
"""

# ── Registry (existing public API — preserved) ────────────────────────────────
from src.data.registry import DatasetEntry, get_enabled_datasets, load_registry

# ── Schema ────────────────────────────────────────────────────────────────────
from src.data.schema import (
    SCHEMA_VERSION,
    CanonicalExample,
    canonical_to_dict,
    dict_to_canonical,
    make_example,
    validate_canonical,
)

# ── Sources ───────────────────────────────────────────────────────────────────
from src.data.sources import (
    DatasetSource,
    GitRepositorySource,
    HuggingFaceSource,
    JSONLSource,
    LocalArrowSource,
    ParquetSource,
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
from src.data.adapters import ADAPTER_REGISTRY, get_adapter
from src.data.dedup import BloomFilter, TwoStageDedup
from src.data.manifest import ExperimentManifest
from src.data.mixer import DatasetMixer, MixConfig, MixerStats
from src.data.quality import QualityConfig, QualityScorer
from src.data.state import ShardState, StateManager
from src.data.streaming import PipelineStats, StreamingCleanConfig, build_pipeline
from src.data.writer import ShardedWriter, WriterStats

# ── Training dataset registration ─────────────────────────────────────────────
from src.data.dataset_info import (
    build_entry,
    get_dataset_file,
    is_registered,
    load_dataset_info,
    register_dataset,
    unregister_dataset,
)

__all__ = [
    # Registry
    "DatasetEntry",
    "load_registry",
    "get_enabled_datasets",
    # Schema
    "SCHEMA_VERSION",
    "CanonicalExample",
    "make_example",
    "canonical_to_dict",
    "dict_to_canonical",
    "validate_canonical",
    # Sources
    "DatasetSource",
    "HuggingFaceSource",
    "LocalArrowSource",
    "JSONLSource",
    "ParquetSource",
    "GitRepositorySource",
    # Adapters
    "ADAPTER_REGISTRY",
    "get_adapter",
    # Quality
    "QualityConfig",
    "QualityScorer",
    # Dedup
    "BloomFilter",
    "TwoStageDedup",
    # Pipeline
    "StreamingCleanConfig",
    "PipelineStats",
    "build_pipeline",
    # Writer
    "ShardedWriter",
    "WriterStats",
    # Mixer
    "MixConfig",
    "DatasetMixer",
    "MixerStats",
    # State
    "ShardState",
    "StateManager",
    # Manifest
    "ExperimentManifest",
    # Dataset info (LLaMA Factory registration)
    "build_entry",
    "load_dataset_info",
    "register_dataset",
    "unregister_dataset",
    "is_registered",
    "get_dataset_file",
]

