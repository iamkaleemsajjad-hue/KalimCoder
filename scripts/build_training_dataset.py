"""
scripts/build_training_dataset.py — Dataset merger for the KalimCoder pipeline.

Reads every cleaned Arrow dataset under ``datasets/cleaned/``, normalises each
sample into the canonical Alpaca instruction format::

    {
        "instruction": "...",
        "input":       "",
        "output":      "..."
    }

then shuffles, splits into train/validation, and saves to ``datasets/final/``.

Design decisions
----------------
* **Per-source adapters** — each dataset has a named adapter function that maps
  its idiosyncratic schema to the canonical format.  Unknown schemas fall back
  to a heuristic adapter.
* **Provenance tracking** — a ``_source`` column is added so downstream tooling
  can weight or filter by origin.
* **Configurable split** — ``--val-ratio`` controls how much goes to validation
  (default 5 %).
* **Reproducible** — ``--seed`` makes shuffling deterministic.
* **Non-destructive** — cleaned datasets are never modified; final datasets are
  written to a separate directory.
* **Statistics** — per-source counts, global token-length histograms, and a
  Markdown summary are saved alongside the final data.

Usage
-----
    python scripts/build_training_dataset.py
    python scripts/build_training_dataset.py --val-ratio 0.05 --seed 42
    python scripts/build_training_dataset.py --name opc_sft_stage1
    python scripts/build_training_dataset.py --cleaned-dir datasets/cleaned
    python scripts/build_training_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Path bootstrap — strip shadow entries before importing HF library
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

_shadow = ["", ".", str(Path(".").resolve()), str(_PROJECT_ROOT)]
for _e in _shadow:
    while _e in sys.path:
        sys.path.remove(_e)

try:
    import datasets as hf_datasets
    from tqdm import tqdm
except ImportError as exc:
    print(
        f"[ERROR] Missing dependency: {exc}\n"
        "Install with:  pip install datasets tqdm\n"
        "Or:            pip install -e '.[train]'"
    )
    sys.exit(1)
finally:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CLEANED_BASE: Path = _PROJECT_ROOT / "datasets" / "cleaned"
_FINAL_BASE: Path = _PROJECT_ROOT / "datasets" / "final"
_LOG_DIR: Path = _PROJECT_ROOT / "logs" / "build"

_CHARS_PER_TOKEN: float = 4.0

# Canonical output columns
_INSTRUCTION_COL = "instruction"
_INPUT_COL = "input"
_OUTPUT_COL = "output"
_SOURCE_COL = "_source"

_CANONICAL_COLUMNS = [_INSTRUCTION_COL, _INPUT_COL, _OUTPUT_COL, _SOURCE_COL]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> logging.Logger:
    """Delegates to the shared pipeline logging utility."""
    from src.utils.logging import configure_pipeline_logging  # noqa: PLC0415
    return configure_pipeline_logging(
        log_dir=_LOG_DIR,
        log_prefix="build",
        logger_name="build_dataset",
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Schema adapters
#
# Each adapter accepts a raw dict (one row) and must return a dict with keys:
#   instruction (str), input (str), output (str)
# Return None to drop the row entirely.
# ---------------------------------------------------------------------------

Adapter = Callable[[dict[str, Any]], Optional[dict[str, str]]]


def _coerce(value: Any) -> str:
    """Return a stripped string, or empty string for None / non-str values."""
    if value is None:
        return ""
    return str(value).strip()


# ── Adapter: opc_sft_stage1 ─────────────────────────────────────────────────
# Schema: { "instruction": str, "output": str, ... }
def _adapt_opc_sft_stage1(row: dict) -> Optional[dict[str, str]]:
    instr = _coerce(row.get("instruction") or row.get("prompt") or row.get("text"))
    out = _coerce(row.get("output") or row.get("response"))
    if not instr or not out:
        return None
    return {"instruction": instr, "input": "", "output": out}


# ── Adapter: the_stack_v2 ────────────────────────────────────────────────────
# Schema: { "content": str, "lang": str, ... }
def _adapt_the_stack_v2(row: dict) -> Optional[dict[str, str]]:
    content = _coerce(row.get("content") or row.get("text") or row.get("code"))
    if not content:
        return None
    lang = _coerce(row.get("lang") or row.get("language") or "code")
    instr = f"Complete the following {lang} code:"
    return {"instruction": instr, "input": "", "output": content}


# ── Adapter: code_search_net ─────────────────────────────────────────────────
# Schema: { "func_code_string": str, "func_documentation_string": str, ... }
def _adapt_code_search_net(row: dict) -> Optional[dict[str, str]]:
    code = _coerce(
        row.get("func_code_string")
        or row.get("whole_func_string")
        or row.get("code")
    )
    doc = _coerce(
        row.get("func_documentation_string")
        or row.get("docstring")
        or row.get("summary")
    )
    if not code:
        return None
    if doc:
        instr = f"Write a function that does the following:\n{doc}"
    else:
        lang = _coerce(row.get("language") or "code")
        instr = f"Complete the following {lang} function:"
    return {"instruction": instr, "input": "", "output": code}


# ── Adapter: swe_bench_verified ───────────────────────────────────────────────
# Schema: { "problem_statement": str, "patch": str, ... }
def _adapt_swe_bench_verified(row: dict) -> Optional[dict[str, str]]:
    problem = _coerce(row.get("problem_statement") or row.get("text"))
    patch = _coerce(row.get("patch") or row.get("solution") or row.get("output"))
    if not problem or not patch:
        return None
    instr = f"Fix the following software issue:\n{problem}"
    return {"instruction": instr, "input": "", "output": patch}


# ── Generic heuristic adapter ─────────────────────────────────────────────────
# Applied to any dataset that does not match a known name.
_INSTRUCTION_FIELDS = ("instruction", "prompt", "question", "problem_statement",
                       "text", "body", "hint")
_OUTPUT_FIELDS = ("output", "response", "answer", "solution", "code",
                  "content", "patch", "func_code_string")
_INPUT_FIELDS = ("input", "context", "auxiliary")


def _adapt_generic(row: dict) -> Optional[dict[str, str]]:
    """Best-effort heuristic adapter for unknown dataset schemas."""
    instr = ""
    for f in _INSTRUCTION_FIELDS:
        val = _coerce(row.get(f))
        if val:
            instr = val
            break

    out = ""
    for f in _OUTPUT_FIELDS:
        val = _coerce(row.get(f))
        if val:
            out = val
            break

    inp = ""
    for f in _INPUT_FIELDS:
        val = _coerce(row.get(f))
        if val:
            inp = val
            break

    if not instr or not out:
        return None
    return {"instruction": instr, "input": inp, "output": out}


# Registry: dataset_name -> adapter function
_ADAPTER_REGISTRY: dict[str, Adapter] = {
    "opc_sft_stage1":    _adapt_opc_sft_stage1,
    "the_stack_v2":      _adapt_the_stack_v2,
    "code_search_net":   _adapt_code_search_net,
    "swe_bench_verified": _adapt_swe_bench_verified,
}


def get_adapter(dataset_name: str, adapter_hint: str | None = None) -> Adapter:
    """Return the adapter for *dataset_name*.

    Lookup order:
    1. *adapter_hint* — the ``adapter`` field from ``configs/datasets.yaml``
       (the registry is the single source of truth when populated).
    2. *dataset_name* in ``_ADAPTER_REGISTRY`` — legacy fallback for names
       not yet in the YAML.
    3. ``_adapt_generic`` — heuristic fallback for unknown schemas.
    """
    key = adapter_hint or dataset_name
    return _ADAPTER_REGISTRY.get(key, _adapt_generic)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

_TOKEN_BUCKETS = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _tok_bucket(n_tokens: int) -> str:
    """Return the histogram bucket label for *n_tokens*."""
    for i in range(len(_TOKEN_BUCKETS) - 1):
        if n_tokens <= _TOKEN_BUCKETS[i + 1]:
            return f"{_TOKEN_BUCKETS[i]}-{_TOKEN_BUCKETS[i + 1]}"
    return f"{_TOKEN_BUCKETS[-1]}+"


@dataclass
class SourceStats:
    """Per-source conversion statistics."""
    name: str
    raw_rows: int = 0
    converted: int = 0
    dropped: int = 0

    @property
    def drop_pct(self) -> float:
        return round(100.0 * self.dropped / max(self.raw_rows, 1), 2)


@dataclass
class BuildStats:
    """Global statistics for the full build run."""
    generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    seed: int = 42
    val_ratio: float = 0.05
    sources: list[SourceStats] = field(default_factory=list)
    total_merged: int = 0
    train_rows: int = 0
    val_rows: int = 0
    token_histogram: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0

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
            "val_ratio": self.val_ratio,
            "total_source_rows": self.total_source_rows,
            "total_merged": self.total_merged,
            "total_dropped": self.total_dropped,
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
            "elapsed_s": round(self.elapsed_s, 2),
            "sources": [
                {
                    "name": s.name,
                    "raw_rows": s.raw_rows,
                    "converted": s.converted,
                    "dropped": s.dropped,
                    "drop_pct": s.drop_pct,
                }
                for s in self.sources
            ],
            "token_histogram": self.token_histogram,
        }


# ---------------------------------------------------------------------------
# Token histogram builder
# ---------------------------------------------------------------------------


def _build_token_histogram(dataset: hf_datasets.Dataset) -> dict[str, int]:
    """Count rows by estimated token bucket (instruction + output combined)."""
    counts: dict[str, int] = {b: 0 for b in [
        f"{_TOKEN_BUCKETS[i]}-{_TOKEN_BUCKETS[i+1]}"
        for i in range(len(_TOKEN_BUCKETS) - 1)
    ] + [f"{_TOKEN_BUCKETS[-1]}+"]}

    for row in tqdm(dataset, desc="  Token histogram", leave=False, unit="row"):
        text = (row.get(_INSTRUCTION_COL) or "") + (row.get(_OUTPUT_COL) or "")
        n_tok = math.ceil(len(text) / _CHARS_PER_TOKEN)
        bucket = _tok_bucket(n_tok)
        counts[bucket] = counts.get(bucket, 0) + 1

    return {k: v for k, v in counts.items() if v > 0}


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _build_markdown(stats: BuildStats) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Training Dataset Build Report")
    a("")
    a(f"- **Generated:** {stats.generated_at}")
    a(f"- **Random seed:** {stats.seed}")
    a(f"- **Validation ratio:** {stats.val_ratio:.1%}")
    a(f"- **Elapsed:** {stats.elapsed_s:.1f}s")
    a("")

    # Overview
    a("## Overview")
    a("")
    a("| Metric | Count |")
    a("|--------|-------|")
    a(f"| Total source rows | {stats.total_source_rows:,} |")
    a(f"| Total merged (after conversion) | {stats.total_merged:,} |")
    a(f"| Total dropped (adapter mismatch) | {stats.total_dropped:,} |")
    a(f"| Train rows | {stats.train_rows:,} |")
    a(f"| Validation rows | {stats.val_rows:,} |")
    a("")

    # Per-source
    a("## Per-Source Breakdown")
    a("")
    a("| Source | Raw Rows | Converted | Dropped | Drop % |")
    a("|--------|----------|-----------|---------|--------|")
    for s in stats.sources:
        a(f"| `{s.name}` | {s.raw_rows:,} | {s.converted:,} | {s.dropped:,} | {s.drop_pct:.1f}% |")
    a("")

    # Token histogram
    if stats.token_histogram:
        a("## Token Length Distribution (train split)")
        a("")
        a("| Token bucket | Row count |")
        a("|-------------|-----------|")
        for bucket, cnt in sorted(stats.token_histogram.items()):
            a(f"| {bucket} | {cnt:,} |")
        a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _load_cleaned(cleaned_dir: Path, logger: logging.Logger) -> hf_datasets.Dataset:
    """Load a cleaned Arrow dataset, flattening DatasetDict to a single split."""
    raw = hf_datasets.load_from_disk(str(cleaned_dir))
    if isinstance(raw, hf_datasets.DatasetDict):
        split = list(raw.keys())[0]
        logger.debug("  DatasetDict — using split %r", split)
        return raw[split]  # type: ignore[return-value]
    return raw  # type: ignore[return-value]


def _convert_dataset(
    dataset: hf_datasets.Dataset,
    dataset_name: str,
    logger: logging.Logger,
    adapter_hint: str | None = None,
) -> tuple[hf_datasets.Dataset, SourceStats]:
    """Apply the per-dataset adapter, return canonical Dataset + SourceStats."""
    adapter = get_adapter(dataset_name, adapter_hint)
    adapter_name = adapter_hint or (dataset_name if dataset_name in _ADAPTER_REGISTRY else "generic")
    logger.info(
        "  Adapter: %r (%d rows)",
        adapter_name,
        len(dataset),
    )

    stats = SourceStats(name=dataset_name, raw_rows=len(dataset))

    canonical_rows: list[dict[str, str]] = []
    for row in tqdm(dataset, desc=f"  Converting [{dataset_name}]", leave=False, unit="row"):
        result = adapter(row)
        # Validate: adapter must return non-None with non-empty instruction AND output
        if (
            result is None
            or not result.get(_INSTRUCTION_COL)
            or not result.get(_OUTPUT_COL)
        ):
            stats.dropped += 1
        else:
            result[_SOURCE_COL] = dataset_name
            canonical_rows.append(result)
            stats.converted += 1

    if not canonical_rows:
        logger.warning("  No rows survived conversion for %s!", dataset_name)
        # Return an empty dataset with the correct schema
        empty = hf_datasets.Dataset.from_dict(
            {col: [] for col in _CANONICAL_COLUMNS}
        )
        return empty, stats

    converted = hf_datasets.Dataset.from_list(canonical_rows)
    logger.info(
        "  Converted %d / %d rows (dropped %d, %.1f%%)",
        stats.converted, stats.raw_rows, stats.dropped, stats.drop_pct,
    )
    return converted, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_training_dataset.py",
        description=(
            "Merge cleaned datasets into a single canonical Alpaca-format "
            "training corpus, shuffle, and split into train/validation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from all cleaned datasets with defaults
  python scripts/build_training_dataset.py

  # 10% validation split, fixed seed
  python scripts/build_training_dataset.py --val-ratio 0.10 --seed 123

  # Process only one source
  python scripts/build_training_dataset.py --name opc_sft_stage1

  # Dry run (merge + stats, no disk writes)
  python scripts/build_training_dataset.py --dry-run
""",
    )
    parser.add_argument("--cleaned-dir", type=Path, default=None, metavar="PATH",
                        help="Root of cleaned datasets (default: datasets/cleaned).")
    parser.add_argument("--out-dir", type=Path, default=None, metavar="PATH",
                        help="Output directory (default: datasets/final).")
    parser.add_argument("--name", type=str, default=None, metavar="NAME",
                        help="Process only the cleaned dataset with this name.")
    parser.add_argument("--val-ratio", type=float, default=0.05, metavar="RATIO",
                        help="Fraction of rows for the validation split (default: 0.05).")
    parser.add_argument("--seed", type=int, default=42, metavar="SEED",
                        help="Random seed for shuffling (default: 42).")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Apply pipeline but do not write files to disk.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level console output.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logger = _configure_logging(verbose=args.verbose)

    cleaned_root: Path = args.cleaned_dir or _CLEANED_BASE
    out_root: Path = args.out_dir or _FINAL_BASE
    val_ratio: float = max(0.0, min(args.val_ratio, 0.5))  # clamp [0, 0.5]
    seed: int = args.seed

    # Validate inputs
    if not cleaned_root.exists():
        logger.error(
            "Cleaned dataset directory not found: %s\n"
            "Run 'python scripts/clean_dataset.py' first.",
            cleaned_root,
        )
        return 1

    # Discover cleaned dataset sub-directories (skip hidden + stats files)
    candidates = sorted(
        d for d in cleaned_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not candidates:
        logger.warning("No sub-directories found under %s.", cleaned_root)
        return 0

    if args.name:
        candidates = [d for d in candidates if d.name == args.name]
        if not candidates:
            logger.error(
                "Dataset %r not found under %s. Available: %s",
                args.name, cleaned_root,
                [d.name for d in sorted(cleaned_root.iterdir()) if d.is_dir()],
            )
            return 1

    # ── 0. Load adapter hints from registry (non-fatal if registry unavailable) ─
    _adapter_hints: dict[str, str | None] = {}
    try:
        from src.data.registry import load_registry  # noqa: PLC0415
        registry_entries = load_registry()
        _adapter_hints = {e.name: e.adapter for e in registry_entries}
        logger.debug("Loaded adapter hints for %d registry entries.", len(_adapter_hints))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load adapter hints from registry: %s — using heuristic adapters.", exc
        )

    t0 = time.monotonic()
    logger.info(
        "Building training corpus from %d cleaned dataset(s): %s%s",
        len(candidates),
        [d.name for d in candidates],
        " [DRY-RUN]" if args.dry_run else "",
    )
    logger.info("val_ratio=%.3f  seed=%d", val_ratio, seed)

    # ── 1. Load, convert, and merge ───────────────────────────────────────────
    all_parts: list[hf_datasets.Dataset] = []
    source_stats: list[SourceStats] = []

    for dataset_dir in tqdm(candidates, desc="Loading & converting", unit="ds"):
        name = dataset_dir.name
        logger.info("[LOAD] %s", name)

        try:
            ds = _load_cleaned(dataset_dir, logger)
        except Exception as exc:  # noqa: BLE001
            logger.error("[FAIL] Could not load %s: %s", name, exc, exc_info=True)
            # Record as fully dropped
            source_stats.append(SourceStats(name=name, raw_rows=0, dropped=0))
            continue

        adapter_hint = _adapter_hints.get(name)
        converted, ss = _convert_dataset(ds, name, logger, adapter_hint=adapter_hint)
        source_stats.append(ss)

        if len(converted) > 0:
            all_parts.append(converted)


    if not all_parts:
        logger.error("No rows were converted from any dataset. Aborting.")
        return 1

    # ── 2. Concatenate ────────────────────────────────────────────────────────
    logger.info("Concatenating %d source(s) ...", len(all_parts))
    merged = hf_datasets.concatenate_datasets(all_parts)
    logger.info("Total merged rows: %d", len(merged))

    # ── 3. Shuffle ────────────────────────────────────────────────────────────
    logger.info("Shuffling with seed=%d ...", seed)
    merged = merged.shuffle(seed=seed)

    # ── 4. Train/validation split ─────────────────────────────────────────────
    n_total = len(merged)
    n_val = round(n_total * val_ratio) if val_ratio > 0 else 0
    n_train = n_total - n_val

    logger.info(
        "Splitting: train=%d (%.1f%%)  val=%d (%.1f%%)",
        n_train, 100.0 * n_train / n_total,
        n_val,   100.0 * n_val   / n_total,
    )

    train_ds = merged.select(range(n_train))
    val_ds = merged.select(range(n_train, n_total))

    # ── 5. Build statistics ───────────────────────────────────────────────────
    logger.info("Building token histogram ...")
    token_hist = _build_token_histogram(train_ds)

    build_stats = BuildStats(
        seed=seed,
        val_ratio=val_ratio,
        sources=source_stats,
        total_merged=n_total,
        train_rows=n_train,
        val_rows=n_val,
        token_histogram=token_hist,
        elapsed_s=time.monotonic() - t0,
    )

    # ── 6. Save ───────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info(
            "[DRY-RUN] Would save train=%d rows and val=%d rows to %s",
            n_train, n_val, out_root,
        )
    else:
        out_root.mkdir(parents=True, exist_ok=True)
        train_path = out_root / "train"
        val_path = out_root / "validation"

        logger.info("Saving train split (%d rows) -> %s ...", n_train, train_path)
        train_ds.save_to_disk(str(train_path))

        logger.info("Saving validation split (%d rows) -> %s ...", n_val, val_path)
        val_ds.save_to_disk(str(val_path))

        # JSON stats
        stats_path = out_root / "build_stats.json"
        stats_path.write_text(
            json.dumps(build_stats.as_dict(), indent=2),
            encoding="utf-8",
        )

        # Markdown report
        md_path = out_root / "build_report.md"
        md_path.write_text(_build_markdown(build_stats), encoding="utf-8")

        logger.info("Stats  -> %s", stats_path)
        logger.info("Report -> %s", md_path)

    # ── 7. Summary table ──────────────────────────────────────────────────────
    _W = 70
    sep = "  " + "-" * _W

    def _row(a: str, b: str, c: str, d: str, e: str) -> str:
        return f"  {a:<22} {b:>10} {c:>10} {d:>9} {e:>8}"

    lines = [
        "",
        "  " + "=" * _W,
        "  BUILD SUMMARY",
        "  " + "=" * _W,
        _row("Source", "Raw", "Converted", "Dropped", "Drop%"),
        sep,
    ]
    for s in source_stats:
        lines.append(
            _row(s.name[:22], f"{s.raw_rows:,}", f"{s.converted:,}",
                 f"{s.dropped:,}", f"{s.drop_pct:.1f}%")
        )
    lines += [
        sep,
        _row("TOTAL", f"{build_stats.total_source_rows:,}",
             f"{build_stats.total_merged:,}",
             f"{build_stats.total_dropped:,}", ""),
        sep,
        f"  Train : {n_train:>10,} rows",
        f"  Val   : {n_val:>10,} rows",
        "  " + "=" * _W,
        "",
    ]
    if build_stats.token_histogram:
        lines.insert(-1, "  Token histogram (train):")
        for bucket, cnt in sorted(token_hist.items()):
            lines.insert(-1, f"    {bucket:<16} {cnt:>8,} rows")
        lines.insert(-1, "")

    logger.info("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
