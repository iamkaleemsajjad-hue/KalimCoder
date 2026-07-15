"""
scripts/clean_dataset.py — Cleaning pipeline for the KalimCoder dataset pipeline.

Reads every Arrow dataset under ``datasets/raw/``, applies a configurable set
of cleaning rules via ``src.data.cleaner.apply_pipeline``, and saves the results
under ``datasets/cleaned/<dataset_name>/``.

The *raw* datasets are **never modified** — cleaned data is always written to a
separate directory.

Cleaning rules (all configurable via CLI flags)
-----------------------------------------------
1. strip_whitespace          — strip leading/trailing whitespace from text cells
2. normalize_line_endings    — normalise \\r\\n / \\r to \\n
3. remove_empty_outputs      — drop rows with empty output/response/code columns
4. remove_empty_instructions — drop rows with empty instruction/prompt columns
5. remove_long_samples       — drop rows whose total token estimate exceeds a limit
6. remove_duplicate_rows     — full-row hash deduplication
7. remove_duplicate_code     — text-column fingerprint deduplication

Output
------
* Cleaned dataset: ``datasets/cleaned/<name>/`` (Arrow format, same schema)
* JSON stats report: ``datasets/cleaned/<name>/_cleaning_stats.json``
* Markdown report:  ``datasets/cleaned/<name>/_cleaning_report.md``
* Session log:      ``logs/cleaning/cleaning_<timestamp>.log``

Usage
-----
    python scripts/clean_dataset.py
    python scripts/clean_dataset.py --name opc_sft_stage1
    python scripts/clean_dataset.py --max-tokens 4096
    python scripts/clean_dataset.py --no-dedup-code --no-dedup-rows
    python scripts/clean_dataset.py --raw-dir datasets/raw --out-dir datasets/cleaned
    python scripts/clean_dataset.py --dry-run
    python scripts/clean_dataset.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# HF datasets import — must happen BEFORE project root enters sys.path
# because the repo has a `datasets/` directory that shadows the library.
# ---------------------------------------------------------------------------
_cwd_entries = ["", ".", str(Path(".").resolve())]
_shadow_entries = _cwd_entries + [str(_PROJECT_ROOT)]
for _e in _shadow_entries:
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

from src.data.cleaner import CleaningConfig, CleaningStats, apply_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RAW_BASE: Path = _PROJECT_ROOT / "datasets" / "raw"
_CLEANED_BASE: Path = _PROJECT_ROOT / "datasets" / "cleaned"
_LOG_DIR: Path = _PROJECT_ROOT / "logs" / "cleaning"

_STATS_FILENAME = "_cleaning_stats.json"
_REPORT_FILENAME = "_cleaning_report.md"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


class Status(str, Enum):
    OK = "OK"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    DRY_RUN = "DRY-RUN"
    EMPTY_RAW = "EMPTY-RAW"


@dataclass
class CleanResult:
    """Outcome record for a single dataset cleaning run."""

    name: str
    status: Status
    stats: Optional[CleaningStats] = None
    elapsed_s: float = 0.0
    message: str = ""

    def summary_row(self) -> tuple[str, str, str, str, str, str]:
        if self.stats:
            orig = f"{self.stats.original_rows:,}"
            final = f"{self.stats.final_rows:,}"
            removed = f"{self.stats.total_removed:,}"
            ret = f"{self.stats.retention_pct:.1f}%"
        else:
            orig = final = removed = ret = "—"
        return (
            self.name[:22],
            self.status.value,
            orig,
            final,
            removed,
            ret,
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = _LOG_DIR / f"cleaning_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    return logging.getLogger("clean_dataset")


# ---------------------------------------------------------------------------
# Report serialisers
# ---------------------------------------------------------------------------


def _stats_to_markdown(name: str, stats: CleaningStats, config: CleaningConfig, elapsed: float) -> str:
    """Render a cleaning stats summary as a Markdown document."""
    lines: list[str] = []
    a = lines.append

    a(f"# Cleaning Report — `{name}`")
    a("")
    a(f"- **Generated:** {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    a(f"- **Elapsed:** {elapsed:.1f}s")
    a("")
    a("## Configuration")
    a("")
    a("| Rule | Enabled | Parameter |")
    a("|------|---------|-----------|")
    a(f"| strip_whitespace | {config.strip_whitespace} | — |")
    a(f"| normalize_line_endings | {config.normalize_line_endings} | — |")
    a(f"| remove_empty_outputs | {config.remove_empty_outputs} | — |")
    a(f"| remove_empty_instructions | {config.remove_empty_instructions} | — |")
    a(f"| remove_long_samples | {config.max_tokens > 0} | max_tokens={config.max_tokens} |")
    a(f"| remove_duplicate_rows | {config.remove_duplicate_rows} | — |")
    a(f"| remove_duplicate_code | {config.remove_duplicate_code} | — |")
    a("")
    a("## Results")
    a("")
    a("| Metric | Count |")
    a("|--------|-------|")
    a(f"| Original rows | {stats.original_rows:,} |")
    a(f"| Final rows | {stats.final_rows:,} |")
    a(f"| Total removed | {stats.total_removed:,} |")
    a(f"| Retention | {stats.retention_pct:.2f}% |")
    a("")
    a("## Rows Removed per Rule")
    a("")
    a("| Rule | Rows Removed |")
    a("|------|-------------|")
    a(f"| strip_whitespace | 0 (normalisation only) |")
    a(f"| normalize_line_endings | 0 (normalisation only) |")
    a(f"| remove_empty_outputs | {stats.removed_empty_outputs:,} |")
    a(f"| remove_empty_instructions | {stats.removed_empty_instructions:,} |")
    a(f"| remove_long_samples | {stats.removed_long_samples:,} |")
    a(f"| remove_duplicate_rows | {stats.removed_duplicate_rows:,} |")
    a(f"| remove_duplicate_code | {stats.removed_duplicate_code:,} |")
    a("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core per-dataset logic
# ---------------------------------------------------------------------------


def _dataset_already_cleaned(out_dir: Path) -> bool:
    """Return True if the cleaned output directory looks complete."""
    if not out_dir.exists():
        return False
    return (
        (out_dir / "dataset_dict.json").exists()
        or (out_dir / "dataset_info.json").exists()
        or any(out_dir.glob("*.arrow"))
        or any(out_dir.glob("*.parquet"))
    )


def _clean_one(
    dataset_dir: Path,
    out_dir: Path,
    config: CleaningConfig,
    force: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> CleanResult:
    """Load, clean, and save one dataset.

    Parameters
    ----------
    dataset_dir:
        Path to the raw Arrow dataset (``datasets/raw/<name>``).
    out_dir:
        Destination for the cleaned dataset (``datasets/cleaned/<name>``).
    config:
        :class:`CleaningConfig` governing which rules run.
    force:
        When ``True``, re-clean even if *out_dir* already exists.
    dry_run:
        When ``True``, load and clean but do *not* write to disk.
    logger:
        Logger instance.

    Returns
    -------
    CleanResult
    """
    name = dataset_dir.name
    t0 = time.monotonic()
    logger.info("[START] Cleaning: %s", name)

    # ── Skip-if-exists ────────────────────────────────────────────────────────
    if not force and _dataset_already_cleaned(out_dir):
        logger.info("[SKIP] %s — cleaned output already exists: %s", name, out_dir)
        return CleanResult(name=name, status=Status.SKIPPED,
                           message=f"output already exists: {out_dir}")

    # ── Load raw ──────────────────────────────────────────────────────────────
    try:
        raw = hf_datasets.load_from_disk(str(dataset_dir))
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not load raw dataset from {dataset_dir}: {exc}"
        logger.error("[FAIL] %s — %s", name, msg)
        return CleanResult(name=name, status=Status.FAILED, message=msg,
                           elapsed_s=time.monotonic() - t0)

    # Flatten DatasetDict to single split
    if isinstance(raw, hf_datasets.DatasetDict):
        split_name = list(raw.keys())[0]
        logger.info("  DatasetDict — using split %r", split_name)
        dataset: hf_datasets.Dataset = raw[split_name]
    else:
        dataset = raw  # type: ignore[assignment]

    if len(dataset) == 0:
        logger.warning("[WARN] %s — raw dataset is empty; skipping.", name)
        return CleanResult(name=name, status=Status.EMPTY_RAW,
                           message="raw dataset has 0 rows",
                           elapsed_s=time.monotonic() - t0)

    logger.info(
        "  Loaded %d rows, %d columns: %s",
        len(dataset), len(dataset.column_names), dataset.column_names,
    )

    # ── Apply cleaning pipeline ───────────────────────────────────────────────
    try:
        cleaned, stats = apply_pipeline(dataset, config)
    except Exception as exc:  # noqa: BLE001
        msg = f"Pipeline error: {exc}"
        logger.error("[FAIL] %s — %s", name, msg, exc_info=True)
        return CleanResult(name=name, status=Status.FAILED, message=msg,
                           elapsed_s=time.monotonic() - t0)

    elapsed = time.monotonic() - t0
    logger.info(
        "  Cleaned: %d -> %d rows (removed %d, retention %.1f%%) in %.1fs",
        stats.original_rows, stats.final_rows,
        stats.total_removed, stats.retention_pct, elapsed,
    )

    # ── Dry-run: skip saving ──────────────────────────────────────────────────
    if dry_run:
        logger.info("[DRY-RUN] %s — would save to %s", name, out_dir)
        return CleanResult(name=name, status=Status.DRY_RUN, stats=stats,
                           elapsed_s=elapsed,
                           message="dry-run mode, nothing saved")

    # ── Persist cleaned dataset + reports ────────────────────────────────────
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("  Saving cleaned dataset to %s ...", out_dir)
        cleaned.save_to_disk(str(out_dir))

        # JSON stats
        stats_path = out_dir / _STATS_FILENAME
        stats_dict = {
            "dataset": name,
            "config": {
                "strip_whitespace": config.strip_whitespace,
                "normalize_line_endings": config.normalize_line_endings,
                "remove_empty_outputs": config.remove_empty_outputs,
                "remove_empty_instructions": config.remove_empty_instructions,
                "max_tokens": config.max_tokens,
                "remove_duplicate_rows": config.remove_duplicate_rows,
                "remove_duplicate_code": config.remove_duplicate_code,
                "text_columns": config.text_columns,
            },
            "stats": stats.as_dict(),
            "elapsed_s": round(elapsed, 2),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        stats_path.write_text(json.dumps(stats_dict, indent=2), encoding="utf-8")

        # Markdown report
        md_path = out_dir / _REPORT_FILENAME
        md_path.write_text(
            _stats_to_markdown(name, stats, config, elapsed),
            encoding="utf-8",
        )

        logger.info("  Stats  -> %s", stats_path)
        logger.info("  Report -> %s", md_path)

    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to save cleaned dataset: {exc}"
        logger.error("[FAIL] %s — %s", name, msg, exc_info=True)
        return CleanResult(name=name, status=Status.FAILED, stats=stats,
                           message=msg, elapsed_s=elapsed)

    return CleanResult(name=name, status=Status.OK, stats=stats, elapsed_s=elapsed)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

_COL_WIDTHS = (23, 9, 10, 10, 10, 9)
_HDR = ("Dataset", "Status", "Original", "Cleaned", "Removed", "Retain%")
_DIVIDER = "  " + "-" * (sum(_COL_WIDTHS) + 2 * (len(_COL_WIDTHS) - 1))


def _print_summary(results: list[CleanResult], logger: logging.Logger) -> None:
    def _row(*cols: str) -> str:
        return "  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, _COL_WIDTHS))

    lines = [
        "",
        "  " + "=" * (len(_DIVIDER) - 2),
        "  CLEANING SUMMARY",
        "  " + "=" * (len(_DIVIDER) - 2),
        _row(*_HDR),
        _DIVIDER,
    ]
    counts: dict[Status, int] = {s: 0 for s in Status}
    for r in results:
        lines.append(_row(*r.summary_row()))
        counts[r.status] += 1
        if r.message and r.status == Status.FAILED:
            lines.append(f"    ERROR: {r.message[:110]}")
    lines += [
        _DIVIDER,
        f"  Total: {len(results)}  "
        f"OK: {counts[Status.OK]}  "
        f"Skipped: {counts[Status.SKIPPED]}  "
        f"Failed: {counts[Status.FAILED]}  "
        f"Dry-run: {counts[Status.DRY_RUN]}",
        "  " + "=" * (len(_DIVIDER) - 2),
        "",
    ]
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="clean_dataset.py",
        description=(
            "Apply a configurable cleaning pipeline to every raw Arrow dataset "
            "under datasets/raw/ and save cleaned outputs to datasets/cleaned/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean all raw datasets with default settings
  python scripts/clean_dataset.py

  # Clean a single dataset
  python scripts/clean_dataset.py --name opc_sft_stage1

  # Tighter token limit (e.g. for 4k context models)
  python scripts/clean_dataset.py --max-tokens 4096

  # Disable duplicate-code removal (useful for stack datasets)
  python scripts/clean_dataset.py --no-dedup-code

  # Dry-run: apply pipeline but do NOT write to disk
  python scripts/clean_dataset.py --dry-run

  # Force re-clean even if cleaned output already exists
  python scripts/clean_dataset.py --force
""",
    )

    # Paths
    parser.add_argument("--raw-dir", type=Path, default=None, metavar="PATH",
                        help="Root of raw datasets (default: datasets/raw).")
    parser.add_argument("--out-dir", type=Path, default=None, metavar="PATH",
                        help="Root for cleaned datasets (default: datasets/cleaned).")
    parser.add_argument("--name", type=str, default=None, metavar="NAME",
                        help="Clean only the dataset sub-folder with this name.")

    # Rule toggles
    parser.add_argument("--no-strip-whitespace", dest="strip_whitespace",
                        action="store_false", default=True,
                        help="Disable whitespace stripping.")
    parser.add_argument("--no-normalize-line-endings", dest="normalize_line_endings",
                        action="store_false", default=True,
                        help="Disable line-ending normalisation.")
    parser.add_argument("--no-remove-empty-outputs", dest="remove_empty_outputs",
                        action="store_false", default=True,
                        help="Keep rows with empty output columns.")
    parser.add_argument("--no-remove-empty-instructions", dest="remove_empty_instructions",
                        action="store_false", default=True,
                        help="Keep rows with empty instruction columns.")
    parser.add_argument("--no-dedup-rows", dest="remove_duplicate_rows",
                        action="store_false", default=True,
                        help="Disable full-row deduplication.")
    parser.add_argument("--no-dedup-code", dest="remove_duplicate_code",
                        action="store_false", default=True,
                        help="Disable code-fingerprint deduplication.")

    # Parameters
    parser.add_argument("--max-tokens", type=int, default=8_192, metavar="N",
                        help="Drop rows exceeding N estimated tokens (0 = disabled). "
                             "Default: 8192.")
    parser.add_argument("--text-columns", nargs="+", default=[], metavar="COL",
                        help="Explicit list of column names to treat as text/code. "
                             "If omitted, columns are auto-detected by name.")

    # Behaviour
    parser.add_argument("--force", action="store_true", default=False,
                        help="Re-clean even if cleaned output already exists.")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Apply pipeline but do not save results to disk.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level console output.")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the cleaning pipeline.

    Returns
    -------
    int
        0 if all datasets cleaned successfully (or skipped), 1 if any failed.
    """
    args = _parse_args(argv)
    logger = _configure_logging(verbose=args.verbose)

    raw_dir: Path = args.raw_dir or _RAW_BASE
    out_dir_root: Path = args.out_dir or _CLEANED_BASE

    if not raw_dir.exists():
        logger.error(
            "Raw dataset directory not found: %s\n"
            "Run 'python scripts/download_datasets.py' first.",
            raw_dir,
        )
        return 1

    # ── Discover raw datasets ─────────────────────────────────────────────────
    candidates = sorted(
        d for d in raw_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not candidates:
        logger.warning("No sub-directories found under %s.", raw_dir)
        return 0

    if args.name:
        candidates = [d for d in candidates if d.name == args.name]
        if not candidates:
            logger.error(
                "Dataset %r not found under %s. Available: %s",
                args.name, raw_dir,
                [d.name for d in sorted(raw_dir.iterdir()) if d.is_dir()],
            )
            return 1

    # ── Build CleaningConfig from parsed args ─────────────────────────────────
    config = CleaningConfig(
        text_columns=args.text_columns,
        strip_whitespace=args.strip_whitespace,
        normalize_line_endings=args.normalize_line_endings,
        remove_empty_outputs=args.remove_empty_outputs,
        remove_empty_instructions=args.remove_empty_instructions,
        remove_duplicate_rows=args.remove_duplicate_rows,
        remove_duplicate_code=args.remove_duplicate_code,
        max_tokens=args.max_tokens,
    )

    logger.info(
        "Cleaning pipeline starting: %d dataset(s) queued%s.",
        len(candidates),
        " [DRY-RUN]" if args.dry_run else "",
    )
    logger.info(
        "Config: strip_ws=%s | norm_endings=%s | rm_empty_out=%s | "
        "rm_empty_instr=%s | max_tokens=%d | dedup_rows=%s | dedup_code=%s",
        config.strip_whitespace,
        config.normalize_line_endings,
        config.remove_empty_outputs,
        config.remove_empty_instructions,
        config.max_tokens,
        config.remove_duplicate_rows,
        config.remove_duplicate_code,
    )

    # ── Main loop ─────────────────────────────────────────────────────────────
    results: list[CleanResult] = []

    with tqdm(
        total=len(candidates),
        desc="Cleaning",
        unit="ds",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as pbar:
        for dataset_dir in candidates:
            pbar.set_postfix_str(dataset_dir.name)
            out_dir = out_dir_root / dataset_dir.name
            result = _clean_one(
                dataset_dir=dataset_dir,
                out_dir=out_dir,
                config=config,
                force=args.force,
                dry_run=args.dry_run,
                logger=logger,
            )
            results.append(result)
            pbar.update(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(results, logger)

    any_failed = any(r.status == Status.FAILED for r in results)
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
