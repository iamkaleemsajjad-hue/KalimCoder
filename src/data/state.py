"""
src/data/state.py — Resumable pipeline state management.

Stores shard-level checkpoint state under ``datasets/state/<name>/state.json``
using atomic writes (write to ``.json.tmp``, then :func:`os.replace`) so a
crash never leaves a corrupt state file.

Resumability contract
---------------------
* Each shard is either **complete** (in ``completed_shard_indices``) or
  **not started**.  There is no partial-shard state.
* On resume, every completed shard is skipped; processing starts from the
  first non-completed shard index.
* ``--force`` calls :meth:`StateManager.reset` which deletes the state file,
  causing the pipeline to start from shard 0.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_DEFAULT_STATE_DIR: Path = _PROJECT_ROOT / "datasets" / "state"


# ---------------------------------------------------------------------------
# ShardState
# ---------------------------------------------------------------------------


@dataclass
class ShardState:
    """Persistent checkpoint for one dataset's processing run.

    Attributes
    ----------
    dataset_name:
        Registry entry name (matches ``configs/datasets.yaml``).
    total_written:
        Cumulative number of examples written to parquet across all shards.
    total_dropped:
        Cumulative number of examples dropped by any pipeline stage.
    completed_shard_indices:
        List of shard indices (0-based) that have been fully processed.
    output_train_files:
        Relative paths of train parquet files written so far.
    output_val_files:
        Relative paths of validation parquet files written so far.
    dedup_stats:
        Latest snapshot from :attr:`TwoStageDedup.stats`.
    quality_stats:
        Aggregated quality score distribution summary.
    started_at:
        ISO-8601 timestamp when the run was first started.
    updated_at:
        ISO-8601 timestamp of the last successful shard checkpoint.
    finished:
        ``True`` once all shards have been processed.
    """

    dataset_name: str
    total_written: int = 0
    total_dropped: int = 0
    completed_shard_indices: list[int] = field(default_factory=list)
    output_train_files: list[str] = field(default_factory=list)
    output_val_files: list[str] = field(default_factory=list)
    dedup_stats: dict[str, Any] = field(default_factory=dict)
    quality_stats: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    finished: bool = False

    def mark_updated(self) -> None:
        """Refresh ``updated_at`` to the current time."""
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------


class StateManager:
    """Atomic JSON persistence for per-dataset shard state.

    Parameters
    ----------
    state_dir:
        Root directory for state files.  Defaults to
        ``<project_root>/datasets/state/``.
    """

    def __init__(self, state_dir: Path | None = None) -> None:
        self._root = state_dir or _DEFAULT_STATE_DIR

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def state_path(self, name: str) -> Path:
        """Return the absolute path to *name*'s state JSON file."""
        return self._root / name / "state.json"

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def load(self, name: str) -> ShardState | None:
        """Load existing :class:`ShardState` for *name*, or ``None``."""
        path = self.state_path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ShardState(**data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load state for %r (%s) — starting fresh.", name, exc
            )
            return None

    def save(self, state: ShardState) -> None:
        """Atomically persist *state* to disk.

        Writes to a ``.json.tmp`` sibling first, then renames to avoid
        partial writes corrupting an existing good state file.
        """
        state.mark_updated()
        path = self.state_path(state.dataset_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
        logger.debug(
            "State saved: %r — %d shards done, %d written.",
            state.dataset_name,
            len(state.completed_shard_indices),
            state.total_written,
        )

    def reset(self, name: str) -> None:
        """Delete the state file for *name* (used by ``--force``).

        The next pipeline run will start from shard 0.
        """
        path = self.state_path(name)
        if path.exists():
            path.unlink()
            logger.info("State reset for %r.", name)
        else:
            logger.debug("No state to reset for %r.", name)

    # ------------------------------------------------------------------
    # Checkpoint helpers (called by run_pipeline.py)
    # ------------------------------------------------------------------

    def is_shard_done(self, name: str, shard_idx: int) -> bool:
        """Return ``True`` if *shard_idx* is already marked complete."""
        state = self.load(name)
        return state is not None and shard_idx in state.completed_shard_indices

    def mark_shard_done(
        self,
        name: str,
        shard_idx: int,
        n_written: int,
        n_dropped: int,
        train_files: list[str] | None = None,
        val_files: list[str] | None = None,
        dedup_stats: dict | None = None,
        quality_stats: dict | None = None,
    ) -> ShardState:
        """Record that *shard_idx* completed successfully and persist."""
        state = self.load(name) or ShardState(dataset_name=name)
        if shard_idx not in state.completed_shard_indices:
            state.completed_shard_indices.append(shard_idx)
        state.total_written += n_written
        state.total_dropped += n_dropped
        if train_files:
            state.output_train_files.extend(
                f for f in train_files if f not in state.output_train_files
            )
        if val_files:
            state.output_val_files.extend(
                f for f in val_files if f not in state.output_val_files
            )
        if dedup_stats is not None:
            state.dedup_stats = dedup_stats
        if quality_stats is not None:
            state.quality_stats = quality_stats
        self.save(state)
        return state

    def mark_finished(self, name: str) -> None:
        """Mark dataset processing as fully complete."""
        state = self.load(name) or ShardState(dataset_name=name)
        state.finished = True
        self.save(state)
        logger.info("Dataset %r marked as finished.", name)

    def all_states(self) -> list[ShardState]:
        """Return all states found under the state root directory."""
        if not self._root.exists():
            return []
        states: list[ShardState] = []
        for json_path in sorted(self._root.glob("*/state.json")):
            name = json_path.parent.name
            state = self.load(name)
            if state:
                states.append(state)
        return states
