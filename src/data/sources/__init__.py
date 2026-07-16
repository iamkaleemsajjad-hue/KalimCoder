"""
src/data/sources/__init__.py — DatasetSource package.

Re-exports all source implementations for convenient imports::

    from src.data.sources import HuggingFaceSource, LocalArrowSource

Implementations
---------------
HuggingFaceSource   : Streams from the HuggingFace Hub with shard fallback.
LocalArrowSource    : Reads existing Arrow datasets from disk (datasets/raw/).
JSONLSource         : Lazy-reads one or more .jsonl files.
ParquetSource       : Reads sharded parquet files from datasets/processed/.
GitRepositorySource : Stub for future GitHub repository ingestion.
"""

from src.data.sources.base import DatasetSource
from src.data.sources.git_repo import GitRepositorySource
from src.data.sources.huggingface import HuggingFaceSource
from src.data.sources.jsonl import JSONLSource
from src.data.sources.local_arrow import LocalArrowSource
from src.data.sources.parquet import ParquetSource

__all__ = [
    "DatasetSource",
    "HuggingFaceSource",
    "LocalArrowSource",
    "JSONLSource",
    "ParquetSource",
    "GitRepositorySource",
]
