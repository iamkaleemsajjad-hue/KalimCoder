"""
src/data/streaming.py — Core streaming pipeline orchestrator.

:func:`build_pipeline` composes all pipeline stages as a lazy generator
chain.  It operates on any :class:`~src.data.sources.base.DatasetSource` and
never materialises the full dataset in memory.

Stage order::

    source.iter_canonical_rows()
        → clean_stage()          # applies CleaningConfig rules
        → quality_stage()        # scores and filters by threshold
        → dedup_stage()          # two-stage Bloom + SHA-256 dedup
        → yield CanonicalExample

Usage::

    from src.data.streaming import build_pipeline, PipelineStats

    gen = build_pipeline(source, cleaning_cfg, quality_cfg, dedup)
    try:
        while True:
            example = next(gen)
            writer.write(example)
    except StopIteration as e:
        stats: PipelineStats = e.value

Design notes
------------
* The pipeline uses ``StopIteration.value`` to return :class:`PipelineStats`
  without adding a return channel to the generator protocol.
* :class:`~src.data.cleaner.CleaningConfig` rules are applied inline; the
  existing ``cleaner.py`` is not imported directly to preserve the cleaner's
  independent contract.
* Each stage is a separate generator function for testability — you can unit-
  test them independently with any Python iterable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Generator, Iterable, Iterator

from src.data.dedup import TwoStageDedup
from src.data.quality import QualityConfig, QualityScorer
from src.data.schema import CanonicalExample
from src.data.sources.base import DatasetSource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PipelineStats
# ---------------------------------------------------------------------------


@dataclass
class PipelineStats:
    """Statistics collected by :func:`build_pipeline` for one source pass."""

    source_rows: int = 0
    dropped_cleaning: int = 0
    dropped_quality: int = 0
    dropped_dedup: int = 0
    yielded: int = 0

    # Quality score distribution (Welford online algorithm)
    _quality_count: int = field(default=0, repr=False)
    _quality_mean: float = field(default=0.0, repr=False)
    _quality_m2: float = field(default=0.0, repr=False)
    _quality_min: float = field(default=1.0, repr=False)
    _quality_max: float = field(default=0.0, repr=False)

    def update_quality(self, score: float) -> None:
        """Welford online update for mean and variance."""
        self._quality_count += 1
        delta = score - self._quality_mean
        self._quality_mean += delta / self._quality_count
        self._quality_m2 += delta * (score - self._quality_mean)
        self._quality_min = min(self._quality_min, score)
        self._quality_max = max(self._quality_max, score)

    @property
    def quality_score_mean(self) -> float:
        return round(self._quality_mean, 4)

    @property
    def quality_score_variance(self) -> float:
        if self._quality_count < 2:
            return 0.0
        return round(self._quality_m2 / (self._quality_count - 1), 4)

    def to_dict(self) -> dict:
        return {
            "source_rows": self.source_rows,
            "dropped_cleaning": self.dropped_cleaning,
            "dropped_quality": self.dropped_quality,
            "dropped_dedup": self.dropped_dedup,
            "yielded": self.yielded,
            "quality_score_mean": self.quality_score_mean,
            "quality_score_variance": self.quality_score_variance,
            "quality_score_min": round(self._quality_min, 4) if self._quality_count else None,
            "quality_score_max": round(self._quality_max, 4) if self._quality_count else None,
        }


# ---------------------------------------------------------------------------
# Cleaning config (mirrors src/data/cleaner.py — no cross-import)
# ---------------------------------------------------------------------------


@dataclass
class StreamingCleanConfig:
    """Inline cleaning rules for the streaming pipeline.

    These mirror :class:`~src.data.cleaner.CleaningConfig` so that the
    streaming pipeline does not import the cleaner's HF-dependent code.
    """
    min_chars: int = 10
    max_chars: int = 32_768
    min_instruction_chars: int = 3
    strip_html: bool = True
    strip_extra_whitespace: bool = True
    blocked_substrings: list[str] = field(default_factory=list)
    blocked_patterns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._compiled_patterns: list[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self.blocked_patterns
        ]

    _compiled_patterns: list = field(default_factory=list, init=False, repr=False)

    def _tag_pattern(self) -> re.Pattern:
        return re.compile(r"<[^>]+>")

    def is_blocked(self, text: str) -> bool:
        for sub in self.blocked_substrings:
            if sub.lower() in text.lower():
                return True
        for pat in self._compiled_patterns:
            if pat.search(text):
                return True
        return False

    def clean_text(self, text: str) -> str:
        if self.strip_html:
            text = self._tag_pattern().sub("", text)
        if self.strip_extra_whitespace:
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()
        return text


# ---------------------------------------------------------------------------
# Stage generators
# ---------------------------------------------------------------------------


def clean_stage(
    stream: Iterable[CanonicalExample],
    config: StreamingCleanConfig,
    stats: PipelineStats,
) -> Generator[CanonicalExample, None, None]:
    """Apply inline cleaning rules; drop examples that fail any rule."""
    from dataclasses import replace

    for example in stream:
        output = config.clean_text(example.output)
        instruction = config.clean_text(example.instruction)
        inp = config.clean_text(example.input) if example.input else ""

        # Length checks on the output (the primary training signal)
        if len(output) < config.min_chars or len(output) > config.max_chars:
            stats.dropped_cleaning += 1
            continue

        if len(instruction) < config.min_instruction_chars:
            stats.dropped_cleaning += 1
            continue

        if config.is_blocked(output) or config.is_blocked(instruction):
            stats.dropped_cleaning += 1
            continue

        yield replace(example, output=output, instruction=instruction, input=inp)


def quality_stage(
    stream: Iterable[CanonicalExample],
    scorer: QualityScorer,
    stats: PipelineStats,
) -> Generator[CanonicalExample, None, None]:
    """Annotate quality scores and drop examples below the threshold."""
    for example in stream:
        annotated = scorer.annotate(example)
        stats.update_quality(annotated.quality_score)
        if not scorer.passes(annotated):
            stats.dropped_quality += 1
            continue
        yield annotated


def dedup_stage(
    stream: Iterable[CanonicalExample],
    dedup: TwoStageDedup,
    stats: PipelineStats,
) -> Generator[CanonicalExample, None, None]:
    """Drop examples whose output was seen in a previous shard."""
    for example in stream:
        if dedup.is_duplicate(example.output):
            stats.dropped_dedup += 1
            continue
        yield example


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_pipeline(
    source: DatasetSource,
    clean_config: StreamingCleanConfig | None = None,
    quality_config: QualityConfig | None = None,
    dedup: TwoStageDedup | None = None,
    enable_cleaning: bool = True,
    enable_quality: bool = True,
    enable_dedup: bool = True,
) -> Generator[CanonicalExample, None, PipelineStats]:
    """Compose all pipeline stages as a single lazy generator.

    Parameters
    ----------
    source:
        Any :class:`~src.data.sources.base.DatasetSource` implementation.
    clean_config:
        Inline cleaning rules.  Defaults to :class:`StreamingCleanConfig` with
        sensible defaults.
    quality_config:
        Quality scoring config.  Defaults to :class:`~src.data.quality.QualityConfig`.
    dedup:
        :class:`~src.data.dedup.TwoStageDedup` instance (shared across shards
        for cross-shard deduplication).  If ``None``, deduplication is skipped
        regardless of *enable_dedup*.
    enable_cleaning:
        Set ``False`` to skip the cleaning stage (useful for already-clean data).
    enable_quality:
        Set ``False`` to skip scoring/filtering (scores will remain 0.0).
    enable_dedup:
        Set ``False`` to skip deduplication.

    Yields
    ------
    CanonicalExample
        Cleaned, scored, and deduplicated examples.

    Returns
    -------
    PipelineStats
        Available as ``StopIteration.value`` after the generator is exhausted.

    Example
    -------
    ::

        gen = build_pipeline(source, enable_dedup=False)
        try:
            while True:
                writer.write(next(gen))
        except StopIteration as e:
            stats = e.value
    """
    cfg_clean = clean_config or StreamingCleanConfig()
    scorer = QualityScorer(quality_config) if enable_quality else None
    stats = PipelineStats()

    # 1 — Source (canonical rows)
    stream: Iterator[CanonicalExample] = source.iter_canonical_rows()

    # Count source rows without consuming the generator early
    def _counting(s: Iterator[CanonicalExample]) -> Generator[CanonicalExample, None, None]:
        for ex in s:
            stats.source_rows += 1
            yield ex

    pipeline = _counting(stream)

    # 2 — Clean
    if enable_cleaning:
        pipeline = clean_stage(pipeline, cfg_clean, stats)  # type: ignore[assignment]

    # 3 — Quality score + filter
    if enable_quality and scorer is not None:
        pipeline = quality_stage(pipeline, scorer, stats)  # type: ignore[assignment]

    # 4 — Dedup
    if enable_dedup and dedup is not None:
        pipeline = dedup_stage(pipeline, dedup, stats)  # type: ignore[assignment]

    # 5 — Yield and accumulate count
    for example in pipeline:
        stats.yielded += 1
        yield example

    logger.info(
        "Pipeline[%r]: source=%d  cleaned_drop=%d  quality_drop=%d  "
        "dedup_drop=%d  yielded=%d  quality_mean=%.3f",
        source.name,
        stats.source_rows,
        stats.dropped_cleaning,
        stats.dropped_quality,
        stats.dropped_dedup,
        stats.yielded,
        stats.quality_score_mean,
    )

    return stats  # available as StopIteration.value
