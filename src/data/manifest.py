"""
src/data/manifest.py — Experiment manifest for reproducible training data builds.

:class:`ExperimentManifest` records everything needed to reproduce a training
data pipeline run: dataset list, mixture config, pipeline config, git commit,
tokenizer, file manifest, and per-source statistics.

Saved to::

    datasets/processed/<run_id>/manifest.json

Loading::

    manifest = ExperimentManifest.load(Path("datasets/processed/<run_id>/manifest.json"))

Usage in run_pipeline.py::

    manifest = ExperimentManifest.from_run(
        pipeline_cfg=cfg_dict,
        mix_cfg=mix_dict,
        writer_stats=writer.close(),
        mixer_stats=mixer_stats,
        tokenizer="Qwen/Qwen3-8B",
        model="Qwen/Qwen3-8B",
    )
    manifest.save(out_dir / "manifest.json")
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.data.schema import SCHEMA_VERSION
from src.data.mixer import MixerStats
from src.data.writer import WriterStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    """Return the current git commit hash, or ``"unknown"`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# ExperimentManifest
# ---------------------------------------------------------------------------


@dataclass
class ExperimentManifest:
    """Records all metadata for a training data build for full reproducibility.

    Attributes
    ----------
    run_id:
        UUID4 uniquely identifying this build.
    schema_version:
        :data:`~src.data.schema.SCHEMA_VERSION` at build time.
    datasets:
        Registry entry names processed in this run.
    mixture_config:
        Snapshot of ``configs/mixture.yaml`` at build time.
    pipeline_config:
        Snapshot of ``configs/pipeline.yaml`` at build time.
    tokenizer:
        HF tokenizer repo (e.g. ``"Qwen/Qwen3-8B"``), or ``None``.
    model:
        Target model for training, or ``None``.
    seed:
        Global random seed used across all pipeline stages.
    git_commit:
        Short git commit hash at build time.
    date:
        ISO-8601 timestamp of the build.
    output_dir:
        Absolute path of the processed output directory.
    train_files:
        Relative paths of all training parquet files.
    val_files:
        Relative paths of all validation parquet files.
    total_train_rows:
        Total training rows across all datasets.
    total_val_rows:
        Total validation rows across all datasets.
    per_source_stats:
        Per-dataset stats (pipeline drops, quality scores, dedup counts, etc.).
    mixer_stats:
        :class:`~src.data.mixer.MixerStats` from the mixing stage.
    """

    run_id: str
    schema_version: str
    datasets: list[str]
    mixture_config: dict[str, Any]
    pipeline_config: dict[str, Any]
    tokenizer: str | None
    model: str | None
    seed: int
    git_commit: str
    date: str
    output_dir: str
    train_files: list[str]
    val_files: list[str]
    total_train_rows: int
    total_val_rows: int
    per_source_stats: dict[str, Any] = field(default_factory=dict)
    mixer_stats: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Atomically write this manifest to *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        data = asdict(self)
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "ExperimentManifest":
        """Load an :class:`ExperimentManifest` from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_run(
        cls,
        pipeline_cfg: dict[str, Any],
        mix_cfg: dict[str, Any],
        writer_stats_map: dict[str, WriterStats],   # {dataset_name: WriterStats}
        mixer_stats: MixerStats,
        datasets: list[str],
        output_dir: Path,
        per_source_stats: dict[str, Any] | None = None,
        tokenizer: str | None = None,
        model: str | None = None,
        seed: int = 42,
    ) -> "ExperimentManifest":
        """Construct a manifest from pipeline run outputs."""
        all_train: list[str] = []
        all_val: list[str] = []
        total_train = 0
        total_val = 0

        for ws in writer_stats_map.values():
            all_train.extend(ws.train_files)
            all_val.extend(ws.val_files)
            total_train += ws.train_rows
            total_val += ws.val_rows

        return cls(
            run_id=str(uuid.uuid4()),
            schema_version=SCHEMA_VERSION,
            datasets=datasets,
            mixture_config=mix_cfg,
            pipeline_config=pipeline_cfg,
            tokenizer=tokenizer,
            model=model,
            seed=seed,
            git_commit=_git_commit(),
            date=time.strftime("%Y-%m-%dT%H:%M:%S"),
            output_dir=str(output_dir),
            train_files=sorted(set(all_train)),
            val_files=sorted(set(all_val)),
            total_train_rows=total_train,
            total_val_rows=total_val,
            per_source_stats=per_source_stats or {},
            mixer_stats=asdict(mixer_stats),
        )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        return (
            f"[{self.run_id[:8]}] "
            f"{self.total_train_rows:,} train + {self.total_val_rows:,} val rows | "
            f"{len(self.datasets)} datasets | "
            f"commit {self.git_commit} | "
            f"{self.date}"
        )
