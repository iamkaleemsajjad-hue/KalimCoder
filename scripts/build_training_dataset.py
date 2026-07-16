"""
scripts/build_training_dataset.py — Build the final SFT corpus for LLaMA Factory.

Reads every processed dataset under ``datasets/processed/<name>/train-*.parquet``
(written by ``scripts/run_pipeline.py``), converts each row to Alpaca format,
applies mixture ratios from ``configs/mixture.yaml``, shuffles and writes:

    datasets/instruction/kalimcoder_sft.jsonl   ← JSONL for LLaMA Factory
    data/dataset_info.json                       ← auto-registered
    datasets/instruction/build_stats.json        ← statistics
    datasets/instruction/build_report.md         ← Markdown report

The script replaces the old ``datasets/cleaned/ → datasets/final/`` path and
is the canonical connection point between the streaming pipeline and training.

Usage
-----
    python scripts/build_training_dataset.py
    python scripts/build_training_dataset.py --name opc_sft_stage1
    python scripts/build_training_dataset.py --max-samples 100000
    python scripts/build_training_dataset.py --dry-run
    python scripts/build_training_dataset.py --no-shuffle
    python scripts/build_training_dataset.py --processed-dir datasets/processed
    python scripts/build_training_dataset.py --out-dir datasets/instruction
    python scripts/build_training_dataset.py --mixture configs/mixture.yaml
    python scripts/build_training_dataset.py --dataset-name my_sft  # output name

Output JSONL format (one JSON object per line, Alpaca):
    {"instruction": "...", "input": "", "output": "...", "_source": "..."}
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# Path bootstrap — strip shadow entries before any project imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

_shadow = ["", ".", str(Path(".").resolve()), str(_PROJECT_ROOT)]
for _e in _shadow:
    while _e in sys.path:
        sys.path.remove(_e)

try:
    import pyarrow.parquet as pq
    import yaml
    from tqdm import tqdm
except ImportError as exc:
    print(
        f"[ERROR] Missing dependency: {exc}\n"
        "Install with:  pip install pyarrow pyyaml tqdm\n"
        "Or:            pip install -e '.[train]'"
    )
    sys.exit(1)
finally:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports (after path bootstrap)
# ---------------------------------------------------------------------------
from src.data.dataset_info import register_dataset
from src.data.registry import DatasetEntry, get_enabled_datasets, load_registry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PROCESSED_BASE: Path = _PROJECT_ROOT / "datasets" / "processed"
_INSTRUCTION_BASE: Path = _PROJECT_ROOT / "datasets" / "instruction"
_MIXTURE_CFG: Path = _PROJECT_ROOT / "configs" / "mixture.yaml"
_LOG_DIR: Path = _PROJECT_ROOT / "logs" / "build"
_DEFAULT_DATASET_NAME = "kalimcoder_sft"

_CHARS_PER_TOKEN: float = 4.0
_TOKEN_BUCKETS = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(log_dir: Path, verbose: bool) -> logging.Logger:
    from src.utils.logging import configure_pipeline_logging  # noqa: PLC0415
    return configure_pipeline_logging(
        log_dir=log_dir,
        log_prefix="build",
        logger_name="build_dataset",
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Mixture config loader
# ---------------------------------------------------------------------------


def _load_mixture_cfg(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_ratios(mixture_cfg: dict) -> dict[str, float]:
    """Extract and normalise dataset ratios from mixture.yaml."""
    raw = mixture_cfg.get("mixture", {}).get("ratios", {})
    if not raw:
        return {}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Alpaca adapter (reads CanonicalExample columns from parquet)
# ---------------------------------------------------------------------------

def _coerce(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _row_to_alpaca(row: dict) -> Optional[dict[str, str]]:
    """Convert a CanonicalExample row dict to an Alpaca training example.

    Returns ``None`` for rows that should be dropped (empty instruction or output).
    The parquet columns (instruction, input, output, dataset, quality_score …)
    map directly to LLaMA Factory's expected Alpaca format.
    """
    instruction = _coerce(row.get("instruction"))
    output = _coerce(row.get("output"))
    if not instruction or not output:
        return None
    return {
        "instruction": instruction,
        "input":       _coerce(row.get("input", "")),
        "output":      output,
        "_source":     _coerce(row.get("dataset", "unknown")),
    }


# ---------------------------------------------------------------------------
# Parquet shard reader
# ---------------------------------------------------------------------------


def _iter_parquet_shards(
    dataset_dir: Path,
    glob: str = "train-*.parquet",
    batch_size: int = 1000,
) -> Iterator[dict]:
    """Yield raw row dicts from all matching parquet shards under *dataset_dir*."""
    shards = sorted(dataset_dir.glob(glob))
    if not shards:
        return
    for shard_path in shards:
        try:
            pf = pq.ParquetFile(shard_path)
            for batch in pf.iter_batches(batch_size=batch_size):
                batch_dict = batch.to_pydict()
                n = len(next(iter(batch_dict.values()), []))
                for i in range(n):
                    yield {col: vals[i] for col, vals in batch_dict.items()}
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("build_dataset").warning(
                "Skipping corrupted shard %s: %s", shard_path, exc
            )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _tok_bucket(n: int) -> str:
    for i in range(len(_TOKEN_BUCKETS) - 1):
        if n <= _TOKEN_BUCKETS[i + 1]:
            return f"{_TOKEN_BUCKETS[i]}-{_TOKEN_BUCKETS[i + 1]}"
    return f"{_TOKEN_BUCKETS[-1]}+"


@dataclass
class SourceStats:
    name: str
    raw_rows: int = 0
    converted: int = 0
    dropped: int = 0
    sampled: int = 0

    @property
    def drop_pct(self) -> float:
        return round(100.0 * self.dropped / max(self.raw_rows, 1), 2)


@dataclass
class BuildStats:
    generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    seed: int = 42
    dataset_name: str = _DEFAULT_DATASET_NAME
    output_file: str = ""
    sources: list[SourceStats] = field(default_factory=list)
    total_written: int = 0
    elapsed_s: float = 0.0
    token_histogram: dict[str, int] = field(default_factory=dict)

    @property
    def total_source_rows(self) -> int:
        return sum(s.raw_rows for s in self.sources)

    @property
    def total_dropped(self) -> int:
        return sum(s.dropped for s in self.sources)

    def as_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "seed": self.seed,
            "dataset_name": self.dataset_name,
            "output_file": self.output_file,
            "total_source_rows": self.total_source_rows,
            "total_written": self.total_written,
            "total_dropped": self.total_dropped,
            "elapsed_s": round(self.elapsed_s, 2),
            "sources": [
                {
                    "name": s.name,
                    "raw_rows": s.raw_rows,
                    "converted": s.converted,
                    "dropped": s.dropped,
                    "sampled": s.sampled,
                    "drop_pct": s.drop_pct,
                }
                for s in self.sources
            ],
            "token_histogram": self.token_histogram,
        }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _build_markdown(stats: BuildStats) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# Training Dataset Build Report — `{stats.dataset_name}`")
    a("")
    a(f"- **Generated:** {stats.generated_at}")
    a(f"- **Random seed:** {stats.seed}")
    a(f"- **Output file:** `{stats.output_file}`")
    a(f"- **Elapsed:** {stats.elapsed_s:.1f}s")
    a("")
    a("## Overview")
    a("")
    a("| Metric | Count |")
    a("|--------|-------|")
    a(f"| Total source rows | {stats.total_source_rows:,} |")
    a(f"| Total written | {stats.total_written:,} |")
    a(f"| Total dropped (adapter/quality) | {stats.total_dropped:,} |")
    a("")
    a("## Per-Source Breakdown")
    a("")
    a("| Source | Raw | Converted | Dropped | Sampled | Drop% |")
    a("|--------|-----|-----------|---------|---------|-------|")
    for s in stats.sources:
        a(f"| `{s.name}` | {s.raw_rows:,} | {s.converted:,} | {s.dropped:,} | {s.sampled:,} | {s.drop_pct:.1f}% |")
    a("")
    if stats.token_histogram:
        a("## Token Length Distribution")
        a("")
        a("| Token bucket | Row count |")
        a("|-------------|-----------|")
        for bucket, cnt in sorted(stats.token_histogram.items()):
            a(f"| {bucket} | {cnt:,} |")
        a("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core build logic
# ---------------------------------------------------------------------------


def _build_token_histogram(examples: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ex in examples:
        text = (ex.get("instruction") or "") + (ex.get("output") or "")
        bucket = _tok_bucket(math.ceil(len(text) / _CHARS_PER_TOKEN))
        counts[bucket] = counts.get(bucket, 0) + 1
    return {k: v for k, v in counts.items() if v > 0}


def _process_source(
    dataset_dir: Path,
    dataset_name: str,
    max_per_source: Optional[int],
    logger: logging.Logger,
) -> tuple[list[dict], SourceStats]:
    """Read all train parquet shards for one dataset and convert to Alpaca."""
    stats = SourceStats(name=dataset_name)
    examples: list[dict] = []

    for raw_row in _iter_parquet_shards(dataset_dir):
        stats.raw_rows += 1
        alpaca = _row_to_alpaca(raw_row)
        if alpaca is None:
            stats.dropped += 1
        else:
            examples.append(alpaca)
            stats.converted += 1

    # Apply per-source cap (from mixture quota or --max-samples)
    if max_per_source is not None and len(examples) > max_per_source:
        examples = examples[:max_per_source]

    stats.sampled = len(examples)
    logger.info(
        "  %s: raw=%d converted=%d dropped=%d sampled=%d",
        dataset_name, stats.raw_rows, stats.converted, stats.dropped, stats.sampled,
    )
    return examples, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_training_dataset.py",
        description=(
            "Merge processed parquet datasets into a single Alpaca JSONL file "
            "for LLaMA Factory SFT training."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from all processed datasets
  python scripts/build_training_dataset.py

  # Build only one source
  python scripts/build_training_dataset.py --name opc_sft_stage1

  # Cap total examples
  python scripts/build_training_dataset.py --max-samples 500000

  # Dry run (no disk writes)
  python scripts/build_training_dataset.py --dry-run

  # Custom output name
  python scripts/build_training_dataset.py --dataset-name my_sft_v2
""",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=None, metavar="PATH",
        help="Root of processed datasets (default: datasets/processed).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, metavar="PATH",
        help="Output directory (default: datasets/instruction).",
    )
    parser.add_argument(
        "--mixture", type=Path, default=None, metavar="PATH",
        help="Mixture config YAML (default: configs/mixture.yaml).",
    )
    parser.add_argument(
        "--dataset-name", type=str, default=_DEFAULT_DATASET_NAME, metavar="NAME",
        help=f"LLaMA Factory dataset name (default: {_DEFAULT_DATASET_NAME}).",
    )
    parser.add_argument(
        "--name", type=str, default=None, metavar="NAME",
        help="Process only the processed dataset with this source name.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, metavar="N",
        help="Maximum total examples to write (applied after mixture sampling).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="SEED",
        help="Random seed for shuffling (default: 42).",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", default=False,
        help="Skip the global shuffle step.",
    )
    parser.add_argument(
        "--no-register", action="store_true", default=False,
        help="Skip writing data/dataset_info.json.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Run all logic but do not write any files.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable DEBUG-level console output.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logger = _configure_logging(_LOG_DIR, verbose=args.verbose)

    processed_root: Path = args.processed_dir or _PROCESSED_BASE
    out_dir: Path = args.out_dir or _INSTRUCTION_BASE
    mixture_path: Path = args.mixture or _MIXTURE_CFG
    dataset_name: str = args.dataset_name
    seed: int = args.seed

    # ── Validate processed root ───────────────────────────────────────────────
    if not processed_root.exists():
        logger.error(
            "Processed dataset directory not found: %s\n"
            "Run 'python scripts/run_pipeline.py' first.",
            processed_root,
        )
        return 1

    # ── Discover source directories ───────────────────────────────────────────
    candidate_dirs = sorted(
        d for d in processed_root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    )
    if not candidate_dirs:
        logger.error("No processed dataset directories found under %s.", processed_root)
        return 1

    if args.name:
        candidate_dirs = [d for d in candidate_dirs if d.name == args.name]
        if not candidate_dirs:
            logger.error(
                "Dataset %r not found under %s. Available: %s",
                args.name, processed_root,
                [d.name for d in sorted(processed_root.iterdir()) if d.is_dir()],
            )
            return 1

    # ── Load mixture ratios ────────────────────────────────────────────────────
    mixture_cfg = _load_mixture_cfg(mixture_path)
    ratios = _parse_ratios(mixture_cfg)
    max_total = mixture_cfg.get("mixture", {}).get("total_examples") or args.max_samples
    logger.info("Mixture ratios: %s", ratios or "(uniform — no mixture.yaml)")
    if max_total:
        logger.info("Max total examples: %d", max_total)

    t0 = time.monotonic()
    logger.info(
        "Building %r from %d source(s): %s%s",
        dataset_name, len(candidate_dirs),
        [d.name for d in candidate_dirs],
        " [DRY-RUN]" if args.dry_run else "",
    )

    # ── Process each source ───────────────────────────────────────────────────
    all_examples: list[dict] = []
    all_stats: list[SourceStats] = []

    for ds_dir in tqdm(candidate_dirs, desc="Sources", unit="ds", file=sys.stdout):
        name = ds_dir.name

        # Compute per-source cap from mixture ratios + max_total
        max_per_source: Optional[int] = None
        if ratios and max_total:
            ratio = ratios.get(name, 1.0 / len(candidate_dirs))
            max_per_source = max(1, round(max_total * ratio))
        elif max_total:
            max_per_source = max(1, round(max_total / len(candidate_dirs)))

        logger.info("[LOAD] %s (max_per_source=%s)", name, max_per_source)
        examples, stats = _process_source(
            ds_dir, name, max_per_source, logger
        )
        all_examples.extend(examples)
        all_stats.append(stats)

    if not all_examples:
        logger.error("No examples were produced from any dataset. Aborting.")
        return 1

    logger.info("Total examples before shuffle: %d", len(all_examples))

    # ── Apply global max_samples cap ─────────────────────────────────────────
    if max_total and len(all_examples) > max_total:
        all_examples = all_examples[:max_total]
        logger.info("Capped to %d examples.", max_total)

    # ── Shuffle ───────────────────────────────────────────────────────────────
    if not args.no_shuffle:
        logger.info("Shuffling %d examples (seed=%d) ...", len(all_examples), seed)
        rng = random.Random(seed)
        rng.shuffle(all_examples)

    # ── Build statistics ──────────────────────────────────────────────────────
    token_hist = _build_token_histogram(all_examples)
    output_file = out_dir / f"{dataset_name}.jsonl"
    build_stats = BuildStats(
        seed=seed,
        dataset_name=dataset_name,
        output_file=str(output_file),
        sources=all_stats,
        total_written=len(all_examples),
        elapsed_s=time.monotonic() - t0,
        token_histogram=token_hist,
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info(
            "[DRY-RUN] Would write %d examples to %s",
            len(all_examples), output_file,
        )
        _print_summary(build_stats, logger)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    logger.info("Writing %d examples → %s ...", len(all_examples), output_file)
    tmp_out = output_file.with_suffix(".jsonl.tmp")
    with tmp_out.open("w", encoding="utf-8") as fh:
        for ex in all_examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    os.replace(tmp_out, output_file)
    logger.info("JSONL written: %s  (%.1f MB)", output_file,
                output_file.stat().st_size / 1024 / 1024)

    # Write stats JSON
    stats_path = out_dir / "build_stats.json"
    stats_path.write_text(
        json.dumps(build_stats.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Stats  → %s", stats_path)

    # Write Markdown report
    md_path = out_dir / "build_report.md"
    md_path.write_text(_build_markdown(build_stats), encoding="utf-8")
    logger.info("Report → %s", md_path)

    # Register in data/dataset_info.json
    if not args.no_register:
        info_path = register_dataset(
            dataset_name=dataset_name,
            file_path=output_file,
        )
        logger.info("Registered in %s", info_path)

    _print_summary(build_stats, logger)
    return 0


def _print_summary(stats: BuildStats, logger: logging.Logger) -> None:
    _W = 72
    sep = "  " + "-" * _W

    def _row(a: str, b: str, c: str, d: str, e: str) -> str:
        return f"  {a:<22} {b:>10} {c:>10} {d:>9} {e:>8}"

    lines = [
        "",
        "  " + "=" * _W,
        f"  BUILD SUMMARY — {stats.dataset_name}",
        "  " + "=" * _W,
        _row("Source", "Raw", "Converted", "Sampled", "Drop%"),
        sep,
    ]
    for s in stats.sources:
        lines.append(
            _row(s.name[:22], f"{s.raw_rows:,}", f"{s.converted:,}",
                 f"{s.sampled:,}", f"{s.drop_pct:.1f}%")
        )
    lines += [
        sep,
        f"  Total written : {stats.total_written:>10,} examples",
        f"  Output file   : {stats.output_file}",
        "  " + "=" * _W,
        "",
    ]
    if stats.token_histogram:
        lines.insert(-1, "  Token histogram:")
        for bucket, cnt in sorted(stats.token_histogram.items()):
            lines.insert(-1, f"    {bucket:<16} {cnt:>8,} rows")
        lines.insert(-1, "")

    logger.info("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
