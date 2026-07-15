"""
scripts/download_datasets.py — Download manager for the KalimCoder dataset pipeline.

Reads ``configs/datasets.yaml``, iterates over every enabled dataset entry, and
downloads each one via the Hugging Face ``datasets`` library.  Downloaded data is
saved under ``datasets/raw/<dataset_name>/``.

Behaviour
---------
* **Skip-if-exists**: A dataset whose destination folder already contains a
  ``dataset_info.json`` (or any Arrow / Parquet shard) is considered complete and
  is skipped without re-downloading.
* **Fault-tolerant**: A failure in one download is logged as an error and the
  manager continues with the next dataset.
* **Progress**: Outer progress is shown via ``tqdm``; HF's own per-shard progress
  is surfaced on stdout.
* **Summary report**: A table is printed at the end listing the outcome
  (SKIPPED / OK / FAILED) for every dataset.

Usage
-----
    python scripts/download_datasets.py
    python scripts/download_datasets.py --config configs/datasets.yaml
    python scripts/download_datasets.py --dry-run
    python scripts/download_datasets.py --force      # re-download existing
    python scripts/download_datasets.py --name opc_sft_stage1  # single dataset
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from any CWD inside the repo
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Third-party imports (fail fast with a friendly message)
#
# IMPORTANT: import hf `datasets` BEFORE inserting the project root into
# sys.path.  The repo contains a `datasets/` data directory that acts as a
# namespace package and shadows the Hugging Face library when the project
# root (or CWD) sits at the front of sys.path.
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
    # Restore project root so that src.* imports work.
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.registry import DatasetEntry, get_enabled_datasets

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RAW_BASE: Path = _PROJECT_ROOT / "datasets" / "raw"
_LOG_DIR: Path = _PROJECT_ROOT / "logs" / "download"
_SENTINEL_FILE = "dataset_dict.json"          # written by save_to_disk
_SENTINEL_FILE_ALT = "dataset_info.json"      # written by single-split save

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(log_dir: Path, verbose: bool) -> logging.Logger:
    """Delegates to the shared pipeline logging utility."""
    from src.utils.logging import configure_pipeline_logging  # noqa: PLC0415
    return configure_pipeline_logging(
        log_dir=log_dir,
        log_prefix="download",
        logger_name="download_manager",
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


class Status(str, Enum):
    SKIPPED = "SKIPPED"
    OK      = "OK"
    FAILED  = "FAILED"
    DRY_RUN = "DRY-RUN"


@dataclass
class DownloadResult:
    """Outcome record for a single dataset download attempt."""

    entry: DatasetEntry
    status: Status
    message: str = ""
    elapsed_s: float = 0.0
    num_rows: Optional[int] = field(default=None)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.entry.name

    def summary_row(self) -> tuple[str, str, str, str]:
        """Return a 4-tuple for the final report table."""
        rows_str = f"{self.num_rows:,}" if self.num_rows is not None else "—"
        elapsed_str = f"{self.elapsed_s:.1f}s" if self.elapsed_s else "—"
        return (self.name, self.status.value, rows_str, elapsed_str)


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------


def _destination_for(entry: DatasetEntry) -> Path:
    """Return the raw download destination path for *entry*.

    Delegates to :attr:`~src.data.registry.DatasetEntry.destination_path`
    so the YAML ``destination`` field is the single source of truth.
    """
    return entry.destination_path


def _already_exists(dest: Path) -> bool:
    """Return ``True`` if *dest* looks like a completed download.

    Checks for the sentinel files that ``save_to_disk`` / ``save`` write on
    completion.
    """
    if not dest.exists():
        return False
    return (
        (dest / _SENTINEL_FILE).exists()
        or (dest / _SENTINEL_FILE_ALT).exists()
        or any(dest.glob("*.arrow"))
        or any(dest.glob("*.parquet"))
    )


def _count_rows(dataset: hf_datasets.Dataset | hf_datasets.DatasetDict) -> int:
    """Return the total number of rows across all splits."""
    if isinstance(dataset, hf_datasets.DatasetDict):
        return sum(len(split) for split in dataset.values())
    return len(dataset)


def _download_one(
    entry: DatasetEntry,
    dest: Path,
    force: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> DownloadResult:
    """Attempt to download a single dataset.

    Parameters
    ----------
    entry:
        The :class:`~src.data.registry.DatasetEntry` to download.
    dest:
        Absolute path to the raw destination directory.
    force:
        When ``True``, re-download even if *dest* already exists.
    dry_run:
        When ``True``, skip actual download and return :attr:`Status.DRY_RUN`.
    logger:
        Caller's logger for structured messages.

    Returns
    -------
    DownloadResult
        Outcome record for this entry.
    """
    cfg_display = f" (config={entry.config!r})" if entry.config else ""
    header = f"{entry.name}  [{entry.repo_id}{cfg_display}, split={entry.split}]"

    # ── Dry run ──────────────────────────────────────────────────────────────
    if dry_run:
        logger.info("[DRY-RUN] Would download: %s -> %s", header, dest)
        return DownloadResult(entry=entry, status=Status.DRY_RUN,
                              message="dry-run mode, no download performed")

    # ── Skip-if-exists ────────────────────────────────────────────────────────
    if not force and _already_exists(dest):
        logger.info("[SKIP] %s — destination already exists: %s", entry.name, dest)
        return DownloadResult(entry=entry, status=Status.SKIPPED,
                              message=f"destination already exists: {dest}")

    # ── Download ──────────────────────────────────────────────────────────────
    logger.info("[START] Downloading %s", header)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    try:
        load_kwargs: dict = dict(
            path=entry.repo_id,
            split=entry.split,
            trust_remote_code=True,
        )
        if entry.config is not None:
            load_kwargs["name"] = entry.config

        logger.debug("load_dataset kwargs: %s", load_kwargs)
        dataset = hf_datasets.load_dataset(**load_kwargs)

        # save_to_disk writes dataset_dict.json / dataset_info.json + Arrow shards
        logger.info("[SAVE] Saving %s to %s ...", entry.name, dest)
        dataset.save_to_disk(str(dest))

        elapsed = time.monotonic() - t0
        n_rows = _count_rows(dataset)
        logger.info(
            "[OK] %s — %d rows saved in %.1fs → %s",
            entry.name, n_rows, elapsed, dest,
        )
        return DownloadResult(
            entry=entry,
            status=Status.OK,
            elapsed_s=elapsed,
            num_rows=n_rows,
        )

    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        logger.error(
            "[FAIL] %s — %s (%.1fs elapsed)",
            entry.name, exc, elapsed, exc_info=True,
        )
        return DownloadResult(
            entry=entry,
            status=Status.FAILED,
            message=str(exc),
            elapsed_s=elapsed,
        )


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

_COL_WIDTHS = (25, 9, 14, 10)   # name, status, rows, elapsed
_DIVIDER = "-" * sum(_COL_WIDTHS + (3,) * len(_COL_WIDTHS))


def _print_report(results: list[DownloadResult], logger: logging.Logger) -> None:
    """Print a formatted summary table to stdout and the log."""

    def _row(*cols: str) -> str:
        parts = [f"{c:<{w}}" for c, w in zip(cols, _COL_WIDTHS)]
        return "  ".join(parts)

    lines = [
        "",
        "=" * len(_DIVIDER),
        "  DOWNLOAD SUMMARY",
        "=" * len(_DIVIDER),
        _row("Dataset", "Status", "Rows", "Elapsed"),
        _DIVIDER,
    ]

    counts: dict[Status, int] = {s: 0 for s in Status}
    for r in results:
        lines.append(_row(*r.summary_row()))
        counts[r.status] += 1
        if r.message and r.status == Status.FAILED:
            # Indent error message under the row
            lines.append(f"    ERROR: {r.message[:120]}")

    lines += [
        _DIVIDER,
        f"  Total: {len(results)}   "
        f"OK: {counts[Status.OK]}   "
        f"Skipped: {counts[Status.SKIPPED]}   "
        f"Failed: {counts[Status.FAILED]}   "
        f"Dry-run: {counts[Status.DRY_RUN]}",
        "=" * len(_DIVIDER),
        "",
    ]

    report = "\n".join(lines)
    # Print to console via logger so it also appears in the log file
    logger.info(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_datasets.py",
        description="Download every enabled dataset defined in configs/datasets.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all enabled datasets
  python scripts/download_datasets.py

  # Dry run — print what would happen without downloading
  python scripts/download_datasets.py --dry-run

  # Force re-download even if the destination exists
  python scripts/download_datasets.py --force

  # Download a specific dataset by name
  python scripts/download_datasets.py --name opc_sft_stage1

  # Use a custom registry file
  python scripts/download_datasets.py --config path/to/other.yaml
""",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the datasets YAML registry (default: configs/datasets.yaml).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="NAME",
        help="Download only the dataset with this name (from the registry).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download datasets even if the destination directory exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show DEBUG-level log messages on the console.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        metavar="TOKEN",
        help=(
            "Hugging Face access token for gated datasets. "
            "Overrides the HF_TOKEN environment variable."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the download manager.

    Returns
    -------
    int
        Exit code: 0 if all downloads succeeded (or were skipped), 1 if any
        download failed.
    """
    args = _parse_args(argv)
    logger = _configure_logging(_LOG_DIR, verbose=args.verbose)

    # ── HF token ─────────────────────────────────────────────────────────────
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from huggingface_hub import login  # type: ignore[import-untyped]
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Authenticated with Hugging Face Hub.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("HF login failed: %s", exc)
    else:
        logger.debug("No HF_TOKEN provided; proceeding without authentication.")

    # ── Load registry ─────────────────────────────────────────────────────────
    logger.info("Loading dataset registry …")
    try:
        datasets = get_enabled_datasets(config_path=args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Failed to load registry: %s", exc)
        return 1

    if not datasets:
        logger.warning("No enabled datasets found. Nothing to download.")
        return 0

    # ── Optional name filter ──────────────────────────────────────────────────
    if args.name:
        matched = [d for d in datasets if d.name == args.name]
        if not matched:
            available = [d.name for d in datasets]
            logger.error(
                "Dataset %r not found in enabled entries. Available: %s",
                args.name,
                available,
            )
            return 1
        datasets = matched
        logger.info("Filtered to single dataset: %s", args.name)

    logger.info(
        "Starting download manager — %d dataset(s) queued%s.",
        len(datasets),
        " [DRY-RUN]" if args.dry_run else "",
    )

    # ── Main download loop ────────────────────────────────────────────────────
    results: list[DownloadResult] = []

    with tqdm(
        total=len(datasets),
        desc="Datasets",
        unit="ds",
        dynamic_ncols=True,
        file=sys.stdout,
    ) as outer_bar:
        for entry in datasets:
            outer_bar.set_postfix_str(entry.name)
            dest = _destination_for(entry)
            result = _download_one(
                entry=entry,
                dest=dest,
                force=args.force,
                dry_run=args.dry_run,
                logger=logger,
            )
            results.append(result)
            outer_bar.update(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_report(results, logger)

    # Exit 1 if any download failed, 0 otherwise
    failed = any(r.status == Status.FAILED for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
