"""
src/data/sources/huggingface.py — HuggingFaceSource.

Streams datasets from the HuggingFace Hub with automatic shard-at-a-time
fallback when streaming is unavailable.

Streaming mode (``streaming=True``)
    Uses ``datasets.load_dataset(..., streaming=True)`` which returns an
    ``IterableDataset``.  No data is written to disk.  Compatible with gated
    datasets when ``HF_TOKEN`` is set.

Fallback mode (``streaming=False`` or when streaming raises)
    Downloads the full dataset into the HF cache
    (``~/.cache/huggingface/datasets/``), then reads it shard-by-shard.
    The HF cache is shared across runs; HF's own resume logic handles
    partial downloads.

Retry policy
    Transient network failures are retried up to ``retry_count`` times with
    exponential backoff starting at ``retry_backoff_s`` seconds.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Generator, Iterator

from src.data.adapters import get_adapter
from src.data.schema import CanonicalExample, validate_canonical
from src.data.sources.base import DatasetSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.path bootstrap helper
# ---------------------------------------------------------------------------
_PROJECT_ROOT: str = str(Path(__file__).resolve().parents[3])


def _import_hf_datasets():
    """Import HuggingFace datasets while avoiding the local datasets/ shadow."""
    _shadow = ["", ".", str(Path(".").resolve()), _PROJECT_ROOT]
    for _e in _shadow:
        while _e in sys.path:
            sys.path.remove(_e)
    try:
        import datasets as hf_datasets
        return hf_datasets
    finally:
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# HuggingFaceSource
# ---------------------------------------------------------------------------


class HuggingFaceSource(DatasetSource):
    """Streams or downloads a dataset from the HuggingFace Hub.

    Parameters
    ----------
    repo_id:
        HuggingFace repository identifier (e.g. ``"bigcode/the-stack-v2"``).
    dataset_name:
        Registry entry name (used in CanonicalExample and adapter lookup).
    split:
        Dataset split (default ``"train"``).
    config:
        Optional dataset configuration / subset name (``name`` kwarg for
        ``load_dataset``).
    streaming:
        Whether to use streaming mode (default ``True``).
    adapter_hint:
        Adapter name; falls back to *dataset_name* then generic.
    license:
        SPDX identifier for the dataset licence.
    task_type:
        Canonical task type.
    retry_count:
        Number of retries on transient failures.
    retry_backoff_s:
        Base backoff interval in seconds (doubles each retry).
    hf_token:
        HuggingFace access token for gated datasets.  Falls back to the
        ``HF_TOKEN`` environment variable.
    """

    def __init__(
        self,
        repo_id: str,
        dataset_name: str,
        split: str = "train",
        config: str | None = None,
        streaming: bool = True,
        adapter_hint: str | None = None,
        license: str = "unknown",
        task_type: str = "instruction",
        retry_count: int = 3,
        retry_backoff_s: float = 5.0,
        hf_token: str | None = None,
    ) -> None:
        self._repo_id = repo_id
        self._dataset_name = dataset_name
        self._split = split
        self._config = config
        self._streaming = streaming
        self._adapter = get_adapter(adapter_hint, dataset_name)
        self._license = license
        self._task_type = task_type
        self._retry_count = retry_count
        self._retry_backoff_s = retry_backoff_s
        self._hf_token = hf_token
        self._estimated_rows: int | None = None
        self._actual_streaming: bool = streaming  # may flip to False on error

    # ------------------------------------------------------------------
    # DatasetSource interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._dataset_name

    @property
    def estimated_rows(self) -> int | None:
        return self._estimated_rows  # unknown for streaming; set after load

    @property
    def supports_streaming(self) -> bool:
        return self._actual_streaming

    def iter_raw_rows(self) -> Iterator[dict]:
        """Yield raw dicts from the HF dataset (streaming or materialised)."""
        hf = _import_hf_datasets()

        kwargs: dict = dict(
            path=self._repo_id,
            split=self._split,
            trust_remote_code=True,
        )
        if self._config:
            kwargs["name"] = self._config
        if self._hf_token:
            kwargs["token"] = self._hf_token

        dataset = self._load_with_retry(hf, kwargs)

        if hasattr(dataset, "__len__"):
            self._estimated_rows = len(dataset)
        else:
            self._estimated_rows = None  # IterableDataset has no len

        yield from dataset

    def iter_canonical_rows(self) -> Generator[CanonicalExample, None, None]:
        """Yield :class:`~src.data.schema.CanonicalExample` objects."""
        dropped = 0
        yielded = 0
        for raw_row in self.iter_raw_rows():
            try:
                example = self._adapter(
                    raw_row,
                    self._dataset_name,
                    self._license,
                    self._task_type,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "HuggingFaceSource[%r]: adapter error: %s",
                    self._dataset_name, exc,
                )
                dropped += 1
                continue

            if example is None:
                dropped += 1
                continue

            errors = validate_canonical(example)
            if errors:
                logger.debug(
                    "HuggingFaceSource[%r]: invalid row — %s",
                    self._dataset_name, errors,
                )
                dropped += 1
                continue

            yielded += 1
            yield example

        logger.info(
            "HuggingFaceSource[%r]: yielded %d, dropped %d.",
            self._dataset_name, yielded, dropped,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_with_retry(self, hf, kwargs: dict):
        """Attempt to load the dataset with retry + streaming fallback."""
        streaming_kwargs = dict(kwargs)
        if self._streaming:
            streaming_kwargs["streaming"] = True

        last_exc: Exception | None = None

        # First attempt: streaming (if enabled)
        if self._streaming:
            for attempt in range(1, self._retry_count + 1):
                try:
                    logger.info(
                        "HuggingFaceSource[%r]: loading (streaming, attempt %d/%d).",
                        self._dataset_name, attempt, self._retry_count,
                    )
                    return hf.load_dataset(**streaming_kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "HuggingFaceSource[%r]: streaming attempt %d failed: %s",
                        self._dataset_name, attempt, exc,
                    )
                    if attempt < self._retry_count:
                        wait = self._retry_backoff_s * (2 ** (attempt - 1))
                        logger.info("Retrying in %.1fs …", wait)
                        time.sleep(wait)

            # Streaming failed — fall back to full download
            logger.warning(
                "HuggingFaceSource[%r]: streaming unavailable (%s). "
                "Falling back to full download.",
                self._dataset_name, last_exc,
            )
            self._actual_streaming = False

        # Fallback: full download with retry
        for attempt in range(1, self._retry_count + 1):
            try:
                logger.info(
                    "HuggingFaceSource[%r]: downloading (attempt %d/%d).",
                    self._dataset_name, attempt, self._retry_count,
                )
                dataset = hf.load_dataset(**kwargs)
                if isinstance(dataset, hf.DatasetDict):
                    split = self._split if self._split in dataset else list(dataset.keys())[0]
                    return dataset[split]
                return dataset
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.error(
                    "HuggingFaceSource[%r]: download attempt %d failed: %s",
                    self._dataset_name, attempt, exc,
                )
                if attempt < self._retry_count:
                    wait = self._retry_backoff_s * (2 ** (attempt - 1))
                    time.sleep(wait)

        raise RuntimeError(
            f"HuggingFaceSource[{self._dataset_name!r}]: "
            f"all {self._retry_count} attempts failed."
        ) from last_exc
