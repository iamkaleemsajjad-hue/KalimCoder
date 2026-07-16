"""
src/data/mixer.py — Dataset mixture builder for the KalimCoder pipeline.

:class:`DatasetMixer` enforces configurable per-source ratios across multiple
:class:`~src.data.schema.CanonicalExample` streams.  Rather than materialising
all examples in memory, it uses a round-robin quota tracker that decides at each
step which source to pull from next.

Strategies
----------
approximate
    Best-effort proportional sampling.  Pulls from sources in proportion to
    their remaining quota.  Stops when all sources are exhausted.  Actual
    ratios may differ slightly from the configured values for small datasets
    where one source runs out before its quota is met.

exact
    Hard quota per source.  Stops pulling from a source once its allocation
    is reached (even if more examples are available).  Requires
    ``total_examples`` to be set.

oversample
    Like ``exact`` but repeats small sources cyclically (using
    ``itertools.cycle``) until their quota is reached.  Introduces repetition
    artefacts for small datasets — use with caution.
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator

from src.data.schema import CanonicalExample
from src.data.writer import ShardedWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MixConfig:
    """Mixture configuration loaded from ``configs/mixture.yaml``.

    Parameters
    ----------
    ratios:
        Mapping of dataset name → target proportion.
        Values do **not** need to sum to 1.0 — they are normalised internally.
    strategy:
        One of ``"approximate"``, ``"exact"``, ``"oversample"``.
    total_examples:
        Target total examples after mixing (``None`` = use all available).
    seed:
        Random seed for shuffle operations.
    """

    ratios: dict[str, float]
    strategy: str = "approximate"
    total_examples: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        valid = {"approximate", "exact", "oversample"}
        if self.strategy not in valid:
            raise ValueError(
                f"Invalid mixture strategy {self.strategy!r}; valid: {sorted(valid)}"
            )
        if not self.ratios:
            raise ValueError("MixConfig.ratios must not be empty.")
        if any(v < 0 for v in self.ratios.values()):
            raise ValueError("Mixture ratios must be non-negative.")
        total = sum(self.ratios.values())
        if total <= 0:
            raise ValueError("Sum of mixture ratios must be positive.")
        # Normalise to [0, 1]
        self.ratios = {k: v / total for k, v in self.ratios.items()}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class MixerStats:
    """Statistics collected during a mixing run."""
    per_source_written: dict[str, int] = field(default_factory=dict)
    per_source_target: dict[str, int] = field(default_factory=dict)
    actual_ratios: dict[str, float] = field(default_factory=dict)
    total_written: int = 0


# ---------------------------------------------------------------------------
# DatasetMixer
# ---------------------------------------------------------------------------


class DatasetMixer:
    """Enforces configurable dataset mixture ratios.

    Parameters
    ----------
    config:
        :class:`MixConfig` instance describing target ratios and strategy.
    writer:
        The :class:`~src.data.writer.ShardedWriter` to send examples to.
        Pass ``None`` to operate as a pure iterator (``run()`` yields instead
        of writing).

    Example usage::

        config = MixConfig(ratios={"stack": 0.4, "opc": 0.6})
        mixer  = DatasetMixer(config, writer)
        mixer.register("stack", stack_source.iter_canonical_rows())
        mixer.register("opc",   opc_source.iter_canonical_rows())
        stats  = mixer.run()
    """

    def __init__(
        self,
        config: MixConfig,
        writer: ShardedWriter | None = None,
    ) -> None:
        self._cfg = config
        self._writer = writer
        self._streams: dict[str, Iterator[CanonicalExample]] = {}
        self._rng = random.Random(config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, name: str, stream: Iterator[CanonicalExample]) -> None:
        """Register *stream* under *name* for mixing."""
        if name in self._streams:
            raise ValueError(f"DatasetMixer: source {name!r} is already registered.")
        if name not in self._cfg.ratios:
            logger.warning(
                "DatasetMixer: %r is registered but not in ratios config — "
                "it will receive a ratio of 0 and be skipped.",
                name,
            )
        self._streams[name] = stream

    def run(self) -> MixerStats:
        """Execute the mixture strategy and return statistics.

        If a :class:`~src.data.writer.ShardedWriter` was provided at
        construction, examples are written directly.  Otherwise they are
        consumed and discarded (useful for dry-runs / testing with side-effects
        in the generator).
        """
        strategy = self._cfg.strategy
        if strategy == "approximate":
            return self._run_approximate()
        elif strategy == "exact":
            return self._run_exact()
        elif strategy == "oversample":
            return self._run_oversample()
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

    def iter_mixed(self) -> Generator[CanonicalExample, None, None]:
        """Like :meth:`run` but yields examples instead of writing them.

        Useful for chaining with downstream pipeline stages.
        Uses the ``approximate`` strategy regardless of config.
        """
        yield from self._approximate_iter()

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _run_approximate(self) -> MixerStats:
        stats = MixerStats()
        for name in self._streams:
            stats.per_source_written[name] = 0

        for example in self._approximate_iter():
            name = example.dataset
            stats.per_source_written[name] = stats.per_source_written.get(name, 0) + 1
            stats.total_written += 1
            if self._writer:
                self._writer.write(example)

        self._finalise_stats(stats)
        return stats

    def _approximate_iter(self) -> Generator[CanonicalExample, None, None]:
        """Interleave streams proportionally using a weighted random picker."""
        active: dict[str, Iterator[CanonicalExample]] = dict(self._streams)
        exhausted: set[str] = set()

        while active:
            weights = {
                name: self._cfg.ratios.get(name, 0.0)
                for name in active
                if name not in exhausted
            }
            if not weights or all(w == 0 for w in weights.values()):
                break
            names = list(weights.keys())
            ws = [weights[n] for n in names]
            chosen = self._rng.choices(names, weights=ws, k=1)[0]
            try:
                example = next(active[chosen])
                yield example
            except StopIteration:
                exhausted.add(chosen)
                active.pop(chosen, None)

    def _run_exact(self) -> MixerStats:
        stats = MixerStats()
        total = self._cfg.total_examples
        if total is None:
            raise ValueError(
                "DatasetMixer exact strategy requires total_examples to be set."
            )
        targets = {
            name: round(ratio * total)
            for name, ratio in self._cfg.ratios.items()
        }
        stats.per_source_target = dict(targets)
        written: dict[str, int] = {name: 0 for name in targets}

        for example in self._approximate_iter():
            name = example.dataset
            if written.get(name, 0) >= targets.get(name, 0):
                continue
            written[name] = written.get(name, 0) + 1
            stats.per_source_written[name] = written[name]
            stats.total_written += 1
            if self._writer:
                self._writer.write(example)

            if all(written.get(n, 0) >= t for n, t in targets.items()):
                break

        self._finalise_stats(stats)
        return stats

    def _run_oversample(self) -> MixerStats:
        """Cycle small sources until per-source quotas are met."""
        stats = MixerStats()
        total = self._cfg.total_examples
        if total is None:
            raise ValueError(
                "DatasetMixer oversample strategy requires total_examples to be set."
            )
        targets = {
            name: round(ratio * total)
            for name, ratio in self._cfg.ratios.items()
        }
        stats.per_source_target = dict(targets)

        # Wrap streams in cycle for oversampling
        cycled = {
            name: itertools.cycle(stream)
            for name, stream in self._streams.items()
        }
        written: dict[str, int] = {name: 0 for name in targets}

        while any(written.get(n, 0) < t for n, t in targets.items()):
            remaining = {
                n: t - written.get(n, 0)
                for n, t in targets.items()
                if written.get(n, 0) < t
            }
            if not remaining:
                break
            names = list(remaining.keys())
            ws = [remaining[n] for n in names]
            chosen = self._rng.choices(names, weights=ws, k=1)[0]
            try:
                example = next(cycled[chosen])
                written[chosen] = written.get(chosen, 0) + 1
                stats.per_source_written[chosen] = written[chosen]
                stats.total_written += 1
                if self._writer:
                    self._writer.write(example)
            except StopIteration:
                targets.pop(chosen, None)
                remaining.pop(chosen, None)

        self._finalise_stats(stats)
        return stats

    @staticmethod
    def _finalise_stats(stats: MixerStats) -> None:
        total = stats.total_written
        if total > 0:
            stats.actual_ratios = {
                name: round(count / total, 4)
                for name, count in stats.per_source_written.items()
            }
        logger.info(
            "DatasetMixer finished: %d total examples; per-source: %s",
            stats.total_written,
            {k: v for k, v in stats.per_source_written.items()},
        )
