"""
scripts/validate_dataset.py — Dataset validation inspector for KalimCoder.

Scans every dataset saved under ``datasets/raw/`` (Arrow / Parquet shards written
by ``download_datasets.py``), computes a rich set of quality metrics for every
text-bearing column, and persists a JSON + Markdown report per dataset under
``datasets/evaluation/``.

Metrics reported per dataset
----------------------------
* Row count
* Column names and types
* Missing values per column  (count + %)
* Duplicate rows             (count + %)
* Per text-column statistics:
    - Average / max character length
    - Estimated token count  (whitespace split, BPE-approximated as chars / 4)
    - Top-3 detected programming languages (via heuristic keyword matching)
* Combined summary JSON + human-readable Markdown report

Usage
-----
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --name opc_sft_stage1
    python scripts/validate_dataset.py --raw-dir path/to/raw
    python scripts/validate_dataset.py --out-dir path/to/reports
    python scripts/validate_dataset.py --sample 5000   # analyse a random sample
    python scripts/validate_dataset.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Third-party imports
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RAW_BASE: Path = _PROJECT_ROOT / "datasets" / "raw"
_OUT_BASE: Path = _PROJECT_ROOT / "datasets" / "evaluation"
_LOG_DIR: Path = _PROJECT_ROOT / "logs" / "validation"

# Columns considered "text" for deep analysis (heuristic; extended at runtime)
_TEXT_COLUMN_HINTS: frozenset[str] = frozenset(
    {
        "text", "content", "code", "prompt", "response", "input", "output",
        "instruction", "answer", "question", "body", "solution", "func_code_string",
        "whole_func_string", "func_documentation_string", "docstring",
        "problem_statement", "patch", "hint",
    }
)

# Approximate chars-per-token for BPE estimation (GPT-2 / Qwen empirical average)
_CHARS_PER_TOKEN: float = 4.0

# Minimum fraction of non-null values in a column to treat it as a text column
_MIN_TEXT_FILL_RATE: float = 0.5

# Maximum string length to scan for language detection (avoid huge files)
_LANG_SCAN_CHARS: int = 2_000

# Sample size for duplicate detection (full scan if dataset is small)
_DUP_SAMPLE_LIMIT: int = 500_000


# ---------------------------------------------------------------------------
# Language detection (lightweight, regex-based — no external deps)
# ---------------------------------------------------------------------------

_LANG_PATTERNS: list[tuple[str, list[str]]] = [
    # (language_name, [regex_patterns])
    ("Python",     [r"\bdef\b", r"\bimport\b", r"\bclass\b.*:", r"if __name__"]),
    ("JavaScript", [r"\bfunction\b", r"\bconst\b", r"\blet\b", r"=>"]),
    ("TypeScript", [r"\binterface\b", r": string\b", r": number\b", r"\btype\b.*="]),
    ("Java",       [r"\bpublic\s+class\b", r"\bSystem\.out", r"@Override"]),
    ("C/C++",      [r"#include\s*[<\"]", r"\bstd::", r"\bvoid\b.*\(.*\)\s*\{"]),
    ("Rust",       [r"\bfn\b", r"\blet\s+mut\b", r"\bimpl\b", r"->.*Result"]),
    ("Go",         [r"\bfunc\b", r"\bpackage\b", r"\bfmt\.Print"]),
    ("SQL",        [r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b", r"\bJOIN\b"]),
    ("Shell/Bash", [r"#!/bin/(ba)?sh", r"\becho\b", r"\$\{.*\}"]),
    ("Ruby",       [r"\bdef\b.*\bend\b", r"\bputs\b", r"\brequire\b"]),
    ("PHP",        [r"<\?php", r"\$\w+\s*=", r"\becho\b"]),
]


def _detect_language(text: str) -> str:
    """Return the most-likely programming language name, or 'Unknown'."""
    sample = text[:_LANG_SCAN_CHARS]
    scores: dict[str, int] = {}
    for lang, patterns in _LANG_PATTERNS:
        hits = sum(1 for p in patterns if re.search(p, sample))
        if hits:
            scores[lang] = hits
    if not scores:
        return "Unknown"
    return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ColumnStats:
    """Statistics for a single column."""

    name: str
    dtype: str
    total: int
    missing: int
    missing_pct: float
    is_text: bool

    # Populated only when is_text=True
    avg_char_len: Optional[float] = field(default=None)
    max_char_len: Optional[int] = field(default=None)
    avg_token_est: Optional[float] = field(default=None)
    max_token_est: Optional[int] = field(default=None)
    lang_distribution: Optional[dict[str, int]] = field(default=None)


@dataclass
class DatasetReport:
    """Full validation report for one dataset."""

    name: str
    path: str
    generated_at: str
    elapsed_s: float

    num_rows: int
    num_columns: int
    columns: list[ColumnStats]

    duplicate_rows: int
    duplicate_pct: float

    # True when a sample was used instead of the full dataset
    sampled: bool = False
    sample_size: Optional[int] = field(default=None)

    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _is_text_column(name: str, dtype: str, sample_values: list[Any]) -> bool:
    """Heuristic: decide whether a column should receive deep text analysis."""
    # Name-based hint
    if name.lower() in _TEXT_COLUMN_HINTS:
        return True
    # Any name containing a hint word
    if any(hint in name.lower() for hint in _TEXT_COLUMN_HINTS):
        return True
    # dtype string
    if "string" in dtype.lower() or "str" in dtype.lower():
        non_null = [v for v in sample_values if v is not None]
        if not non_null:
            return False
        fill = len(non_null) / max(len(sample_values), 1)
        return fill >= _MIN_TEXT_FILL_RATE
    return False


def _analyse_text_column(
    values: list[Any],
    logger: logging.Logger,
    col_name: str,
    sample_size: Optional[int] = None,
) -> tuple[float, int, float, int, dict[str, int]]:
    """Compute text statistics for a list of raw column values.

    Returns
    -------
    avg_char_len, max_char_len, avg_token_est, max_token_est, lang_dist
    """
    texts: list[str] = [str(v) for v in values if v is not None]
    if not texts:
        return 0.0, 0, 0.0, 0, {}

    # Optionally sub-sample for speed on very large columns
    analysis_texts = texts
    if sample_size and len(texts) > sample_size:
        import random
        analysis_texts = random.sample(texts, sample_size)
        logger.debug(
            "  Column %r: sub-sampled %d of %d rows for text analysis.",
            col_name, sample_size, len(texts),
        )

    char_lens = [len(t) for t in analysis_texts]
    avg_char = sum(char_lens) / len(char_lens)
    max_char = max(char_lens)
    avg_tok = avg_char / _CHARS_PER_TOKEN
    max_tok = math.ceil(max_char / _CHARS_PER_TOKEN)

    # Language detection — count most-likely language per row
    lang_counts: Counter[str] = Counter(
        _detect_language(t) for t in analysis_texts
    )
    lang_dist = dict(lang_counts.most_common(5))

    return avg_char, max_char, avg_tok, max_tok, lang_dist


def _count_duplicates(
    dataset: hf_datasets.Dataset,
    logger: logging.Logger,
) -> int:
    """Return the number of duplicate rows via hashing.

    For datasets larger than _DUP_SAMPLE_LIMIT, only the first
    _DUP_SAMPLE_LIMIT rows are checked (annotated in the report).
    """
    n = len(dataset)
    limit = min(n, _DUP_SAMPLE_LIMIT)
    if limit < n:
        logger.info(
            "  Duplicate scan: checking first %d of %d rows (large dataset).",
            limit, n,
        )

    seen: set[int] = set()
    dupes = 0

    for i in tqdm(range(limit), desc="  Dup-scan", unit="row", leave=False):
        row = dataset[i]
        row_hash = hash(json.dumps(row, sort_keys=True, default=str))
        if row_hash in seen:
            dupes += 1
        else:
            seen.add(row_hash)

    return dupes


def _validate_one(
    dataset_dir: Path,
    sample_size: Optional[int],
    logger: logging.Logger,
) -> DatasetReport:
    """Validate a single Arrow dataset folder.

    Parameters
    ----------
    dataset_dir:
        Path to the folder containing Arrow / Parquet shards.
    sample_size:
        If set, analyse only this many randomly drawn rows. ``None`` = full scan.
    logger:
        Caller's logger instance.

    Returns
    -------
    DatasetReport
        Fully populated report (errors list non-empty if partial failures occurred).
    """
    name = dataset_dir.name
    errors: list[str] = []
    t0 = time.monotonic()
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    logger.info("[START] Validating: %s", name)

    # ── Load from disk ────────────────────────────────────────────────────────
    try:
        raw = hf_datasets.load_from_disk(str(dataset_dir))
    except Exception as exc:
        msg = f"Failed to load dataset from {dataset_dir}: {exc}"
        logger.error("  [FAIL] %s", msg)
        return DatasetReport(
            name=name,
            path=str(dataset_dir),
            generated_at=generated_at,
            elapsed_s=time.monotonic() - t0,
            num_rows=0,
            num_columns=0,
            columns=[],
            duplicate_rows=0,
            duplicate_pct=0.0,
            errors=[msg],
        )

    # Flatten DatasetDict to a single Dataset (use first split)
    if isinstance(raw, hf_datasets.DatasetDict):
        split_name = list(raw.keys())[0]
        logger.info("  DatasetDict detected; using split %r.", split_name)
        dataset: hf_datasets.Dataset = raw[split_name]
    else:
        dataset = raw  # type: ignore[assignment]

    total_rows = len(dataset)
    logger.info("  Rows: %d  |  Columns: %d", total_rows, len(dataset.column_names))

    # ── Optional row sampling ─────────────────────────────────────────────────
    sampled = False
    effective_sample: Optional[int] = None
    if sample_size and total_rows > sample_size:
        import random
        indices = random.sample(range(total_rows), sample_size)
        dataset = dataset.select(indices)
        sampled = True
        effective_sample = sample_size
        logger.info("  Sampled %d rows from %d for analysis.", sample_size, total_rows)

    n = len(dataset)

    # ── Per-column analysis ───────────────────────────────────────────────────
    col_stats: list[ColumnStats] = []

    for col_name in tqdm(dataset.column_names, desc=f"  Cols [{name}]", leave=False):
        dtype_str = str(dataset.features[col_name])
        values = dataset[col_name]

        # Missing values
        missing = sum(1 for v in values if v is None)
        missing_pct = round(100.0 * missing / n, 4) if n else 0.0

        # Sample for type decision (cheap)
        sample_vals = values[:200]
        is_text = _is_text_column(col_name, dtype_str, sample_vals)

        stat = ColumnStats(
            name=col_name,
            dtype=dtype_str,
            total=n,
            missing=missing,
            missing_pct=missing_pct,
            is_text=is_text,
        )

        if is_text:
            try:
                avg_c, max_c, avg_t, max_t, lang_dist = _analyse_text_column(
                    values, logger, col_name
                )
                stat.avg_char_len = round(avg_c, 2)
                stat.max_char_len = max_c
                stat.avg_token_est = round(avg_t, 2)
                stat.max_token_est = max_t
                stat.lang_distribution = lang_dist
                logger.debug(
                    "  Column %r: avg_chars=%.1f max_chars=%d avg_tokens=%.1f "
                    "top_lang=%s",
                    col_name, avg_c, max_c, avg_t,
                    next(iter(lang_dist), "N/A"),
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"Text analysis failed for column {col_name!r}: {exc}"
                logger.warning("  [WARN] %s", msg)
                errors.append(msg)

        col_stats.append(stat)

    # ── Duplicate detection ───────────────────────────────────────────────────
    logger.info("  Running duplicate scan ...")
    try:
        dupe_count = _count_duplicates(dataset, logger)
        dupe_pct = round(100.0 * dupe_count / n, 4) if n else 0.0
        logger.info("  Duplicates: %d (%.2f%%)", dupe_count, dupe_pct)
    except Exception as exc:  # noqa: BLE001
        msg = f"Duplicate scan failed: {exc}"
        logger.warning("  [WARN] %s", msg)
        errors.append(msg)
        dupe_count, dupe_pct = 0, 0.0

    elapsed = time.monotonic() - t0
    logger.info("[DONE] %s — validated in %.1fs", name, elapsed)

    return DatasetReport(
        name=name,
        path=str(dataset_dir),
        generated_at=generated_at,
        elapsed_s=round(elapsed, 2),
        num_rows=total_rows,
        num_columns=len(dataset.column_names),
        columns=col_stats,
        duplicate_rows=dupe_count,
        duplicate_pct=dupe_pct,
        sampled=sampled,
        sample_size=effective_sample,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Report serialisers
# ---------------------------------------------------------------------------

_DIVIDER_W = 72


def _report_to_dict(report: DatasetReport) -> dict[str, Any]:
    """Convert a DatasetReport to a plain dict suitable for JSON serialisation."""
    d = asdict(report)
    return d


def _report_to_markdown(report: DatasetReport) -> str:
    """Render a DatasetReport as a human-readable Markdown document."""
    lines: list[str] = []
    add = lines.append

    add(f"# Dataset Validation Report — `{report.name}`")
    add("")
    add(f"- **Generated:** {report.generated_at}")
    add(f"- **Source path:** `{report.path}`")
    add(f"- **Elapsed:** {report.elapsed_s:.1f}s")
    if report.sampled:
        add(
            f"- **Note:** Metrics computed on a random sample of "
            f"**{report.sample_size:,}** rows (total rows: {report.num_rows:,})."
        )
    add("")

    # ── Overview ──────────────────────────────────────────────────────────────
    add("## Overview")
    add("")
    add(f"| Metric | Value |")
    add(f"|--------|-------|")
    add(f"| Total rows | {report.num_rows:,} |")
    add(f"| Total columns | {report.num_columns} |")
    add(f"| Duplicate rows | {report.duplicate_rows:,} ({report.duplicate_pct:.2f}%) |")
    add("")

    # ── Column table ──────────────────────────────────────────────────────────
    add("## Column Summary")
    add("")
    add("| Column | Type | Missing | Missing % | Text? |")
    add("|--------|------|---------|-----------|-------|")
    for c in report.columns:
        text_flag = "Yes" if c.is_text else "—"
        add(
            f"| `{c.name}` | `{c.dtype}` | {c.missing:,} "
            f"| {c.missing_pct:.2f}% | {text_flag} |"
        )
    add("")

    # ── Text column deep stats ────────────────────────────────────────────────
    text_cols = [c for c in report.columns if c.is_text]
    if text_cols:
        add("## Text Column Statistics")
        add("")
        for c in text_cols:
            add(f"### `{c.name}`")
            add("")
            add("| Metric | Value |")
            add("|--------|-------|")
            add(f"| Avg character length | {c.avg_char_len:,.2f} |")
            add(f"| Max character length | {c.max_char_len:,} |")
            add(f"| Avg estimated tokens | {c.avg_token_est:,.2f} |")
            add(f"| Max estimated tokens | {c.max_token_est:,} |")
            add("")
            if c.lang_distribution:
                add("**Detected language distribution (top 5):**")
                add("")
                add("| Language | Row count |")
                add("|----------|-----------|")
                for lang, cnt in sorted(
                    c.lang_distribution.items(), key=lambda x: -x[1]
                ):
                    add(f"| {lang} | {cnt:,} |")
                add("")

    # ── Errors ────────────────────────────────────────────────────────────────
    if report.errors:
        add("## Warnings / Errors")
        add("")
        for e in report.errors:
            add(f"- {e}")
        add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _print_summary(results: list[tuple[DatasetReport, bool]], logger: logging.Logger) -> None:
    """Print a final summary table for all validated datasets."""
    _W = _DIVIDER_W
    sep = "-" * _W

    def _row(name: str, rows: str, cols: str, dupes: str, errors: str) -> str:
        return f"  {name:<22} {rows:>10} {cols:>6} {dupes:>10} {errors:>8}"

    lines = [
        "",
        "=" * _W,
        "  VALIDATION SUMMARY",
        "=" * _W,
        _row("Dataset", "Rows", "Cols", "Dup%", "Errors"),
        sep,
    ]
    for report, _saved in results:
        lines.append(
            _row(
                report.name[:22],
                f"{report.num_rows:,}",
                str(report.num_columns),
                f"{report.duplicate_pct:.2f}%",
                str(len(report.errors)),
            )
        )
    lines += [sep, "=" * _W, ""]
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_dataset.py",
        description="Inspect and report on datasets stored under datasets/raw/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all raw datasets
  python scripts/validate_dataset.py

  # Validate a single dataset
  python scripts/validate_dataset.py --name opc_sft_stage1

  # Analyse a random 10 000-row sample (much faster for large datasets)
  python scripts/validate_dataset.py --sample 10000

  # Custom raw and output directories
  python scripts/validate_dataset.py --raw-dir datasets/raw --out-dir datasets/evaluation
""",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Root directory containing raw dataset folders (default: datasets/raw).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Directory to write JSON + Markdown reports into (default: datasets/evaluation).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="NAME",
        help="Validate only the dataset sub-folder with this name.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Use a random sample of N rows instead of the full dataset for analysis.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level console output.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging bootstrap (reuses the project util pattern)
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = _LOG_DIR / f"validation_{timestamp}.log"

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

    return logging.getLogger("validate_dataset")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logger = _configure_logging(verbose=args.verbose)

    raw_dir: Path = args.raw_dir or _RAW_BASE
    out_dir: Path = args.out_dir or _OUT_BASE

    if not raw_dir.exists():
        logger.error(
            "Raw dataset directory not found: %s\n"
            "Run 'python scripts/download_datasets.py' first.",
            raw_dir,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover dataset sub-directories ─────────────────────────────────────
    candidate_dirs = sorted(
        d for d in raw_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not candidate_dirs:
        logger.warning(
            "No sub-directories found under %s.\n"
            "Have you run the download manager yet?",
            raw_dir,
        )
        return 0

    if args.name:
        candidate_dirs = [d for d in candidate_dirs if d.name == args.name]
        if not candidate_dirs:
            logger.error(
                "Dataset %r not found under %s. "
                "Available: %s",
                args.name,
                raw_dir,
                [d.name for d in sorted(raw_dir.iterdir()) if d.is_dir()],
            )
            return 1

    logger.info(
        "Validating %d dataset(s): %s",
        len(candidate_dirs),
        [d.name for d in candidate_dirs],
    )

    # ── Validate each dataset ─────────────────────────────────────────────────
    results: list[tuple[DatasetReport, bool]] = []
    any_failed = False

    for dataset_dir in candidate_dirs:
        try:
            report = _validate_one(
                dataset_dir=dataset_dir,
                sample_size=args.sample,
                logger=logger,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[UNHANDLED] %s: %s", dataset_dir.name, exc, exc_info=True)
            any_failed = True
            continue

        if report.errors:
            any_failed = True

        # ── Save JSON report ──────────────────────────────────────────────────
        saved = False
        try:
            json_path = out_dir / f"{report.name}_validation.json"
            json_path.write_text(
                json.dumps(_report_to_dict(report), indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("  JSON report -> %s", json_path)

            md_path = out_dir / f"{report.name}_validation.md"
            md_path.write_text(_report_to_markdown(report), encoding="utf-8")
            logger.info("  MD   report -> %s", md_path)

            saved = True
        except Exception as exc:  # noqa: BLE001
            logger.error("  Failed to save reports: %s", exc)
            any_failed = True

        results.append((report, saved))

    # ── Final summary table ───────────────────────────────────────────────────
    if results:
        _print_summary(results, logger)

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
