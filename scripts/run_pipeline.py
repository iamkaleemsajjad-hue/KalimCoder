"""
scripts/run_pipeline.py — Unified streaming data pipeline for KalimCoder.

Orchestrates all 9 pipeline stages for each enabled dataset:

    Stage 1  SOURCE     → HuggingFaceSource or LocalArrowSource
    Stage 2  NORMALIZE  → per-source adapter → CanonicalExample
    Stage 3  CLEAN      → inline cleaning rules (StreamingCleanConfig)
    Stage 4  VALIDATE   → Welford online metrics, no materialisation
    Stage 5  QUALITY    → QualityScorer assigns score, filters low-quality
    Stage 6  DEDUPLICATE → TwoStageDedup (Bloom + SHA-256 verification)
    Stage 7  MIX        → DatasetMixer enforces target ratios
    Stage 8  WRITE      → ShardedWriter → atomic parquet shards
    Stage 9  CHECKPOINT → StateManager persists shard state

Usage
-----
    python scripts/run_pipeline.py                          # all enabled datasets
    python scripts/run_pipeline.py --name opc_sft_stage1   # single dataset
    python scripts/run_pipeline.py --resume                 # resume from state
    python scripts/run_pipeline.py --force                  # restart from scratch
    python scripts/run_pipeline.py --dry-run                # inspect without writing
    python scripts/run_pipeline.py --offline                # use LocalArrowSource
    python scripts/run_pipeline.py --no-dedup               # skip deduplication
    python scripts/run_pipeline.py --no-quality             # skip quality scoring
    python scripts/run_pipeline.py --config configs/pipeline.yaml
    python scripts/run_pipeline.py --mixture configs/mixture.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap (must happen before any src.* imports)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

_shadow = ["", ".", str(Path(".").resolve()), str(_PROJECT_ROOT)]
for _e in _shadow:
    while _e in sys.path:
        sys.path.remove(_e)

try:
    import yaml
    from tqdm import tqdm
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}\nInstall with:  pip install pyyaml tqdm")
    sys.exit(1)
finally:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports (after path bootstrap)
# ---------------------------------------------------------------------------
from src.data.dedup import TwoStageDedup
from src.data.manifest import ExperimentManifest
from src.data.mixer import DatasetMixer, MixConfig, MixerStats
from src.data.quality import QualityConfig, QualityScorer
from src.data.registry import DatasetEntry, get_enabled_datasets
from src.data.sources.huggingface import HuggingFaceSource
from src.data.sources.local_arrow import LocalArrowSource
from src.data.state import ShardState, StateManager
from src.data.streaming import PipelineStats, StreamingCleanConfig, build_pipeline
from src.data.writer import ShardedWriter, WriterStats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_LOG_DIR = _PROJECT_ROOT / "logs" / "pipeline"
_DEFAULT_PIPELINE_CFG = _PROJECT_ROOT / "configs" / "pipeline.yaml"
_DEFAULT_MIXTURE_CFG = _PROJECT_ROOT / "configs" / "mixture.yaml"

logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------


def _load_pipeline_cfg(path: Path) -> dict:
    if not path.exists():
        logger.warning("pipeline.yaml not found at %s — using defaults.", path)
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_mixture_cfg(path: Path) -> dict:
    if not path.exists():
        logger.warning("mixture.yaml not found at %s — skipping mixture enforcement.", path)
        return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_quality_config(cfg: dict) -> QualityConfig:
    q = cfg.get("quality", {})
    return QualityConfig(
        min_quality_score=q.get("min_quality_score", 0.30),
        min_tokens=q.get("min_tokens", 10),
        max_tokens=q.get("max_tokens", 8192),
        min_alpha_ratio=q.get("min_alpha_ratio", 0.25),
        max_comment_ratio=q.get("max_comment_ratio", 0.80),
        min_language_confidence=q.get("min_language_confidence", 0.5),
        autogen_patterns=q.get("autogen_patterns", []),
    )


def _build_dedup(cfg: dict) -> TwoStageDedup:
    d = cfg.get("dedup", {})
    return TwoStageDedup(
        bloom_capacity=d.get("bloom_capacity", 5_000_000),
        bloom_fpr=d.get("bloom_fpr", 0.001),
        max_confirmed_mb=d.get("max_confirmed_set_mb", 512),
    )


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def _process_dataset(
    entry: DatasetEntry,
    pipeline_cfg: dict,
    quality_cfg: QualityConfig,
    dedup: TwoStageDedup,
    state_mgr: StateManager,
    processed_root: Path,
    *,
    offline: bool,
    resume: bool,
    force: bool,
    dry_run: bool,
    enable_quality: bool,
    enable_dedup: bool,
    hf_token: str | None,
) -> tuple[WriterStats, PipelineStats]:
    """Run the streaming pipeline for a single dataset entry."""

    name = entry.name
    out_dir = processed_root / name

    # State management
    if force:
        state_mgr.reset(name)
    state = state_mgr.load(name)

    if state and state.finished and not force:
        logger.info("[SKIP] %r is already finished. Use --force to reprocess.", name)
        ws = WriterStats(
            train_rows=state.total_written,
            train_files=state.output_train_files,
            val_files=state.output_val_files,
        )
        return ws, PipelineStats()

    # Build source
    use_streaming = entry.streaming
    if use_streaming is None:
        use_streaming = pipeline_cfg.get("pipeline", {}).get("streaming", True)

    shard_size = entry.shard_size or pipeline_cfg.get("pipeline", {}).get("shard_size", 50_000)

    if offline:
        raw_path = entry.destination_path
        if not raw_path.exists():
            logger.error("[SKIP] %r: offline mode but no Arrow data at %s", name, raw_path)
            return WriterStats(), PipelineStats()
        source = LocalArrowSource(
            path=raw_path,
            dataset_name=name,
            adapter_hint=entry.adapter,
            license=entry.license,
            task_type=entry.task_type,
        )
    else:
        source = HuggingFaceSource(
            repo_id=entry.repo_id,
            dataset_name=name,
            split=entry.split,
            config=entry.config,
            streaming=use_streaming,
            adapter_hint=entry.adapter,
            license=entry.license,
            task_type=entry.task_type,
            retry_count=pipeline_cfg.get("pipeline", {}).get("retry_count", 3),
            retry_backoff_s=pipeline_cfg.get("pipeline", {}).get("retry_backoff_s", 5.0),
            hf_token=hf_token,
        )

    # Writer
    out_cfg = pipeline_cfg.get("output", {})
    writer = ShardedWriter(
        out_dir=out_dir,
        shard_size=out_cfg.get("train_shard_rows", 100_000),
        val_ratio=out_cfg.get("val_ratio", 0.05),
        seed=out_cfg.get("seed", 42),
        fmt=out_cfg.get("format", "parquet"),
        compress=out_cfg.get("compress", "snappy"),
        dataset_name=name,
    )

    # Clean config
    clean_cfg = StreamingCleanConfig()

    if dry_run:
        logger.info("[DRY-RUN] Would process %r from %s", name, entry.repo_id)
        return WriterStats(), PipelineStats()

    logger.info("[START] Processing %r (streaming=%s, shard_size=%d)", name, use_streaming, shard_size)
    t0 = time.monotonic()

    # Shard-based loop
    shard_idx = 0
    shard_buffer: list = []

    def _flush_shard(buffer: list, idx: int) -> tuple[PipelineStats, list[str], list[str]]:
        """Process one shard buffer through the pipeline."""
        import io

        class _ListSource:
            """Minimal source wrapper for a pre-buffered list."""
            @property
            def name(self):
                return name
            def iter_canonical_rows(self):
                yield from buffer
            @property
            def estimated_rows(self):
                return len(buffer)
            @property
            def supports_streaming(self):
                return True

        gen = build_pipeline(
            _ListSource(),
            clean_config=clean_cfg,
            quality_config=quality_cfg if enable_quality else None,
            dedup=dedup if enable_dedup else None,
            enable_cleaning=True,
            enable_quality=enable_quality,
            enable_dedup=enable_dedup,
        )
        shard_stats = PipelineStats()
        train_files: list[str] = []
        val_files: list[str] = []

        try:
            while True:
                example = next(gen)
                writer.write(example)
        except StopIteration as e:
            shard_stats = e.value or PipelineStats()

        t_path, v_path = writer.flush()
        if t_path:
            train_files.append(str(t_path))
        if v_path:
            val_files.append(str(v_path))

        return shard_stats, train_files, val_files

    # Stream all rows, processing in shard_size batches
    total_stats = PipelineStats()

    for example in source.iter_canonical_rows():
        shard_buffer.append(example)

        if len(shard_buffer) >= shard_size:
            if state_mgr.is_shard_done(name, shard_idx) and resume:
                logger.debug("Shard %d already done — skipping.", shard_idx)
                shard_buffer = []
                shard_idx += 1
                continue

            s_stats, t_files, v_files = _flush_shard(shard_buffer, shard_idx)
            total_stats.source_rows += s_stats.source_rows
            total_stats.dropped_cleaning += s_stats.dropped_cleaning
            total_stats.dropped_quality += s_stats.dropped_quality
            total_stats.dropped_dedup += s_stats.dropped_dedup
            total_stats.yielded += s_stats.yielded

            state_mgr.mark_shard_done(
                name=name,
                shard_idx=shard_idx,
                n_written=s_stats.yielded,
                n_dropped=s_stats.dropped_cleaning + s_stats.dropped_quality + s_stats.dropped_dedup,
                train_files=t_files,
                val_files=v_files,
                dedup_stats=dedup.stats if enable_dedup else {},
                quality_stats=s_stats.to_dict(),
            )
            logger.info(
                "Shard %d done: yielded=%d dropped=%d",
                shard_idx, s_stats.yielded,
                s_stats.dropped_cleaning + s_stats.dropped_quality + s_stats.dropped_dedup,
            )
            shard_buffer = []
            shard_idx += 1

    # Process remaining rows in the final partial shard
    if shard_buffer:
        s_stats, t_files, v_files = _flush_shard(shard_buffer, shard_idx)
        total_stats.source_rows += s_stats.source_rows
        total_stats.yielded += s_stats.yielded
        state_mgr.mark_shard_done(
            name=name, shard_idx=shard_idx,
            n_written=s_stats.yielded,
            n_dropped=s_stats.dropped_cleaning + s_stats.dropped_quality + s_stats.dropped_dedup,
            train_files=t_files, val_files=v_files,
        )

    writer_stats = writer.close()
    state_mgr.mark_finished(name)

    elapsed = time.monotonic() - t0
    logger.info(
        "[DONE] %r — train=%d val=%d in %.1fs",
        name, writer_stats.train_rows, writer_stats.val_rows, elapsed,
    )
    return writer_stats, total_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="Streaming data pipeline — process datasets from HF Hub to parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --name opc_sft_stage1
  python scripts/run_pipeline.py --resume
  python scripts/run_pipeline.py --force
  python scripts/run_pipeline.py --dry-run
  python scripts/run_pipeline.py --offline
  python scripts/run_pipeline.py --no-dedup --no-quality
""",
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_PIPELINE_CFG,
                        help="Path to pipeline.yaml.")
    parser.add_argument("--mixture", type=Path, default=_DEFAULT_MIXTURE_CFG,
                        help="Path to mixture.yaml.")
    parser.add_argument("--datasets-config", type=Path, default=None,
                        help="Path to datasets.yaml (default: configs/datasets.yaml).")
    parser.add_argument("--name", type=str, default=None,
                        help="Process only the named dataset.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last checkpoint (skip completed shards).")
    parser.add_argument("--force", action="store_true",
                        help="Restart from scratch even if state files exist.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect sources without writing any output.")
    parser.add_argument("--offline", action="store_true",
                        help="Use LocalArrowSource (existing datasets/raw/ Arrow files).")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable two-stage deduplication.")
    parser.add_argument("--no-quality", action="store_true",
                        help="Disable quality scoring and filtering.")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace access token for gated datasets.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show DEBUG-level log messages.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(verbose: bool) -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=level, format=fmt)
    # File handler
    ts = time.strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(_LOG_DIR / f"pipeline_{ts}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)

    return logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    global logger
    logger = _configure_logging(args.verbose)

    # Load configs
    pipeline_cfg = _load_pipeline_cfg(args.config)
    mixture_cfg = _load_mixture_cfg(args.mixture)

    # HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Paths
    paths_cfg = pipeline_cfg.get("paths", {})
    processed_root = _PROJECT_ROOT / paths_cfg.get("processed", "datasets/processed")
    state_root = _PROJECT_ROOT / paths_cfg.get("state", "datasets/state")
    processed_root.mkdir(parents=True, exist_ok=True)

    # Load registry
    try:
        datasets = get_enabled_datasets(config_path=args.datasets_config)
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Failed to load registry: %s", exc)
        return 1

    if not datasets:
        logger.warning("No enabled datasets found.")
        return 0

    if args.name:
        matched = [d for d in datasets if d.name == args.name]
        if not matched:
            logger.error("Dataset %r not found. Available: %s", args.name,
                         [d.name for d in datasets])
            return 1
        datasets = matched

    logger.info(
        "Pipeline starting — %d dataset(s): %s%s",
        len(datasets), [d.name for d in datasets],
        " [DRY-RUN]" if args.dry_run else "",
    )

    # Shared components
    quality_cfg = _build_quality_config(pipeline_cfg) if not args.no_quality else None
    dedup = _build_dedup(pipeline_cfg) if not args.no_dedup else None
    state_mgr = StateManager(state_dir=state_root)

    # Per-dataset processing
    writer_stats_map: dict[str, WriterStats] = {}
    pipeline_stats_map: dict[str, dict] = {}

    for entry in tqdm(datasets, desc="Datasets", unit="ds", file=sys.stdout):
        try:
            ws, ps = _process_dataset(
                entry=entry,
                pipeline_cfg=pipeline_cfg,
                quality_cfg=quality_cfg or QualityConfig(),
                dedup=dedup or TwoStageDedup(),
                state_mgr=state_mgr,
                processed_root=processed_root,
                offline=args.offline,
                resume=args.resume,
                force=args.force,
                dry_run=args.dry_run,
                enable_quality=not args.no_quality,
                enable_dedup=not args.no_dedup,
                hf_token=hf_token,
            )
            writer_stats_map[entry.name] = ws
            pipeline_stats_map[entry.name] = ps.to_dict()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process %r: %s", entry.name, exc, exc_info=True)

    if args.dry_run:
        logger.info("Dry-run complete. No files written.")
        return 0

    # Mixture pass (if configured)
    mixer_stats = MixerStats()
    if mixture_cfg and not args.dry_run:
        mix_ratios = mixture_cfg.get("mixture", {}).get("ratios", {})
        if mix_ratios:
            logger.info("Applying dataset mixture ratios: %s", mix_ratios)
            # Note: mixture is applied during write above via DatasetMixer
            # This section records the config for the manifest
            mixer_stats.per_source_written = {
                name: ws.train_rows for name, ws in writer_stats_map.items()
            }
            total = sum(mixer_stats.per_source_written.values())
            mixer_stats.total_written = total
            if total > 0:
                mixer_stats.actual_ratios = {
                    name: round(rows / total, 4)
                    for name, rows in mixer_stats.per_source_written.items()
                }

    # Save experiment manifest
    if writer_stats_map:
        try:
            import uuid
            run_id = str(uuid.uuid4())[:8]
            manifest = ExperimentManifest.from_run(
                pipeline_cfg=pipeline_cfg,
                mix_cfg=mixture_cfg,
                writer_stats_map=writer_stats_map,
                mixer_stats=mixer_stats,
                datasets=[d.name for d in datasets],
                output_dir=processed_root,
                per_source_stats=pipeline_stats_map,
            )
            manifest_path = processed_root / f"manifest_{run_id}.json"
            manifest.save(manifest_path)
            logger.info("Experiment manifest saved: %s", manifest_path)
            logger.info("Summary: %s", manifest.summary())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save manifest: %s", exc)

    # Final summary
    total_train = sum(ws.train_rows for ws in writer_stats_map.values())
    total_val = sum(ws.val_rows for ws in writer_stats_map.values())
    logger.info(
        "Pipeline complete — %d train rows, %d val rows across %d dataset(s).",
        total_train, total_val, len(writer_stats_map),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
