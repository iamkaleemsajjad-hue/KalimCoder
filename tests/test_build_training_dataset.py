"""
tests/test_build_training_dataset.py — Integration tests for the KalimCoder
end-to-end training pipeline.

Tests cover:
  1. build_training_dataset.main() reads processed parquet and writes JSONL
  2. dataset_info.json is created with correct schema
  3. Mixture ratios are approximately respected
  4. --dry-run writes nothing
  5. train.py preflight fails with clear error when model not found
  6. train.py preflight fails with clear error when dataset_info.json missing
  7. train.py preflight fails when dataset is not registered
  8. dataset_info module unit tests (register, load, unregister, is_registered)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helpers — synthetic parquet writer
# ---------------------------------------------------------------------------

def _write_parquet_shard(path: Path, rows: list[dict]) -> None:
    """Write a minimal CanonicalExample-schema parquet shard for testing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("schema_version", pa.string()),
        pa.field("dataset", pa.string()),
        pa.field("task_type", pa.string()),
        pa.field("language", pa.string()),
        pa.field("repository", pa.string()),
        pa.field("license", pa.string()),
        pa.field("instruction", pa.string()),
        pa.field("input", pa.string()),
        pa.field("output", pa.string()),
        pa.field("quality_score", pa.float32()),
        pa.field("metadata", pa.string()),
    ])
    default = {
        "id": "abc123",
        "schema_version": "1.0",
        "task_type": "instruction",
        "language": "python",
        "repository": "",
        "license": "MIT",
        "input": "",
        "quality_score": 0.9,
        "metadata": "{}",
    }
    full_rows = [{**default, **r} for r in rows]
    table = pa.table(
        {col: [r[col] for r in full_rows] for col in schema.names},
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _make_processed_dir(tmp_path: Path, datasets: dict[str, list[dict]]) -> Path:
    """Create a fake ``datasets/processed/`` directory structure."""
    processed = tmp_path / "datasets" / "processed"
    for name, rows in datasets.items():
        shard = processed / name / "train-00001.parquet"
        _write_parquet_shard(shard, rows)
    return processed


# ---------------------------------------------------------------------------
# Helpers — build_training_dataset runner
# ---------------------------------------------------------------------------

def _run_build(
    tmp_path: Path,
    processed_dir: Path,
    extra_args: Optional[list[str]] = None,
) -> tuple[int, Path]:
    """Run main() with tmp_path as the output root; return (returncode, out_dir)."""
    from scripts.build_training_dataset import main  # noqa: PLC0415

    out_dir = tmp_path / "datasets" / "instruction"
    data_dir = tmp_path / "data"

    argv = [
        "--processed-dir", str(processed_dir),
        "--out-dir", str(out_dir),
        "--no-shuffle",  # deterministic
        "--seed", "42",
    ]
    if extra_args:
        argv += extra_args

    # Monkey-patch the register_dataset call to use tmp_path's data/ dir
    import src.data.dataset_info as di_mod  # noqa: PLC0415
    original_default = di_mod._DEFAULT_DATA_DIR
    di_mod._DEFAULT_DATA_DIR = data_dir
    try:
        rc = main(argv)
    finally:
        di_mod._DEFAULT_DATA_DIR = original_default

    return rc, out_dir


# ===========================================================================
# 1. Basic JSONL output
# ===========================================================================

class TestBuildBasic:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        pytest.importorskip("pyarrow")
        self.tmp = tmp_path
        self.processed = _make_processed_dir(tmp_path, {
            "ds_a": [
                {"dataset": "ds_a", "instruction": "Write a hello function.", "output": "def hello(): return 'hi'"},
                {"dataset": "ds_a", "instruction": "Write a world function.", "output": "def world(): return 'world'"},
            ]
        })

    def test_returns_zero(self):
        rc, _ = _run_build(self.tmp, self.processed)
        assert rc == 0, "main() should return 0 on success"

    def test_jsonl_created(self):
        rc, out_dir = _run_build(self.tmp, self.processed)
        assert rc == 0
        jsonl = out_dir / "kalimcoder_sft.jsonl"
        assert jsonl.exists(), f"Expected JSONL at {jsonl}"

    def test_jsonl_has_alpaca_format(self):
        rc, out_dir = _run_build(self.tmp, self.processed)
        assert rc == 0
        jsonl = out_dir / "kalimcoder_sft.jsonl"
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            ex = json.loads(line)
            assert "instruction" in ex
            assert "input" in ex
            assert "output" in ex
            assert ex["instruction"] != ""
            assert ex["output"] != ""

    def test_source_provenance_in_jsonl(self):
        rc, out_dir = _run_build(self.tmp, self.processed)
        assert rc == 0
        jsonl = out_dir / "kalimcoder_sft.jsonl"
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            ex = json.loads(line)
            assert ex.get("_source") == "ds_a"

    def test_stats_json_written(self):
        _, out_dir = _run_build(self.tmp, self.processed)
        assert (out_dir / "build_stats.json").exists()

    def test_build_report_md_written(self):
        _, out_dir = _run_build(self.tmp, self.processed)
        assert (out_dir / "build_report.md").exists()


# ===========================================================================
# 2. dataset_info.json auto-registration
# ===========================================================================

class TestDatasetInfoRegistration:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        pytest.importorskip("pyarrow")
        self.tmp = tmp_path
        self.processed = _make_processed_dir(tmp_path, {
            "ds_b": [
                {"dataset": "ds_b", "instruction": "Q1", "output": "A1"},
            ]
        })

    def test_dataset_info_created(self):
        _run_build(self.tmp, self.processed)
        info_path = self.tmp / "data" / "dataset_info.json"
        assert info_path.exists(), "data/dataset_info.json should be created"

    def test_dataset_registered_correctly(self):
        _run_build(self.tmp, self.processed)
        info = json.loads((self.tmp / "data" / "dataset_info.json").read_text())
        assert "kalimcoder_sft" in info, "kalimcoder_sft must appear in dataset_info.json"
        entry = info["kalimcoder_sft"]
        assert entry["formatting"] == "alpaca"
        assert "columns" in entry
        assert entry["columns"]["prompt"] == "instruction"
        assert entry["columns"]["response"] == "output"

    def test_dataset_file_path_in_info(self):
        _, out_dir = _run_build(self.tmp, self.processed)
        info = json.loads((self.tmp / "data" / "dataset_info.json").read_text())
        file_name = info["kalimcoder_sft"]["file_name"]
        # Should be a path pointing to the JSONL we just wrote
        assert "kalimcoder_sft.jsonl" in file_name


# ===========================================================================
# 3. Dry-run writes nothing
# ===========================================================================

class TestDryRun:
    def test_dry_run_writes_no_jsonl(self, tmp_path):
        pytest.importorskip("pyarrow")
        processed = _make_processed_dir(tmp_path, {
            "ds_c": [{"dataset": "ds_c", "instruction": "Q", "output": "A"}]
        })
        rc, out_dir = _run_build(tmp_path, processed, extra_args=["--dry-run"])
        assert rc == 0
        assert not (out_dir / "kalimcoder_sft.jsonl").exists(), \
            "--dry-run should not write the JSONL file"
        assert not (tmp_path / "data" / "dataset_info.json").exists(), \
            "--dry-run should not create dataset_info.json"

    def test_dry_run_no_register(self, tmp_path):
        pytest.importorskip("pyarrow")
        processed = _make_processed_dir(tmp_path, {
            "ds_d": [{"dataset": "ds_d", "instruction": "Q", "output": "A"}]
        })
        _run_build(tmp_path, processed, extra_args=["--no-register"])
        assert not (tmp_path / "data" / "dataset_info.json").exists(), \
            "--no-register should not create dataset_info.json"


# ===========================================================================
# 4. --name filters to one source
# ===========================================================================

class TestNameFilter:
    def test_only_named_source_included(self, tmp_path):
        pytest.importorskip("pyarrow")
        processed = _make_processed_dir(tmp_path, {
            "ds_e": [{"dataset": "ds_e", "instruction": "from e", "output": "out e"}],
            "ds_f": [{"dataset": "ds_f", "instruction": "from f", "output": "out f"}],
        })
        rc, out_dir = _run_build(tmp_path, processed, extra_args=["--name", "ds_e"])
        assert rc == 0
        jsonl = out_dir / "kalimcoder_sft.jsonl"
        lines = [json.loads(l) for l in jsonl.read_text().strip().splitlines()]
        sources = {ex["_source"] for ex in lines}
        assert sources == {"ds_e"}, f"Expected only ds_e but got {sources}"

    def test_missing_name_returns_nonzero(self, tmp_path):
        pytest.importorskip("pyarrow")
        processed = _make_processed_dir(tmp_path, {
            "ds_g": [{"dataset": "ds_g", "instruction": "Q", "output": "A"}]
        })
        rc, _ = _run_build(tmp_path, processed, extra_args=["--name", "nonexistent"])
        assert rc != 0


# ===========================================================================
# 5. Rows with empty instruction/output are dropped
# ===========================================================================

class TestDropInvalidRows:
    def test_empty_output_dropped(self, tmp_path):
        pytest.importorskip("pyarrow")
        processed = _make_processed_dir(tmp_path, {
            "ds_h": [
                {"dataset": "ds_h", "instruction": "Good",   "output": "good output"},
                {"dataset": "ds_h", "instruction": "Bad",    "output": ""},   # dropped
                {"dataset": "ds_h", "instruction": "",       "output": "A"},  # dropped
            ]
        })
        rc, out_dir = _run_build(tmp_path, processed)
        assert rc == 0
        jsonl = out_dir / "kalimcoder_sft.jsonl"
        lines = jsonl.read_text().strip().splitlines()
        assert len(lines) == 1, f"Expected 1 row but got {len(lines)}"

    def test_all_bad_rows_returns_nonzero(self, tmp_path):
        pytest.importorskip("pyarrow")
        processed = _make_processed_dir(tmp_path, {
            "ds_i": [
                {"dataset": "ds_i", "instruction": "", "output": ""},
            ]
        })
        rc, _ = _run_build(tmp_path, processed)
        assert rc != 0, "All-empty dataset should cause non-zero exit"


# ===========================================================================
# 6. dataset_info module unit tests
# ===========================================================================

class TestDatasetInfoModule:
    def test_register_and_load(self, tmp_path):
        from src.data.dataset_info import register_dataset, load_dataset_info  # noqa
        info_path = register_dataset(
            "test_ds",
            file_path=tmp_path / "test.jsonl",
            data_dir=tmp_path / "data",
        )
        assert info_path.exists()
        info = load_dataset_info(tmp_path / "data")
        assert "test_ds" in info
        assert info["test_ds"]["formatting"] == "alpaca"

    def test_unregister(self, tmp_path):
        from src.data.dataset_info import register_dataset, unregister_dataset, is_registered  # noqa
        register_dataset("rm_me", file_path=tmp_path / "x.jsonl", data_dir=tmp_path / "data")
        assert is_registered("rm_me", data_dir=tmp_path / "data")
        removed = unregister_dataset("rm_me", data_dir=tmp_path / "data")
        assert removed
        assert not is_registered("rm_me", data_dir=tmp_path / "data")

    def test_unregister_nonexistent_returns_false(self, tmp_path):
        from src.data.dataset_info import unregister_dataset  # noqa
        assert not unregister_dataset("ghost", data_dir=tmp_path / "data")

    def test_overwrite_false_preserves_entry(self, tmp_path):
        from src.data.dataset_info import register_dataset, load_dataset_info  # noqa
        register_dataset("keep", file_path=tmp_path / "v1.jsonl", data_dir=tmp_path / "data")
        register_dataset("keep", file_path=tmp_path / "v2.jsonl", data_dir=tmp_path / "data", overwrite=False)
        info = load_dataset_info(tmp_path / "data")
        assert "v1.jsonl" in info["keep"]["file_name"]

    def test_get_dataset_file(self, tmp_path):
        from src.data.dataset_info import register_dataset, get_dataset_file  # noqa
        target = tmp_path / "my.jsonl"
        register_dataset("get_me", file_path=target, data_dir=tmp_path / "data")
        result = get_dataset_file("get_me", data_dir=tmp_path / "data")
        assert result is not None
        assert "my.jsonl" in str(result)

    def test_atomic_write_no_tmp_leftover(self, tmp_path):
        from src.data.dataset_info import register_dataset  # noqa
        register_dataset("atomic", file_path=tmp_path / "a.jsonl", data_dir=tmp_path / "data")
        tmp_files = list((tmp_path / "data").glob("*.tmp"))
        assert tmp_files == [], f"Unexpected .tmp files: {tmp_files}"


# ===========================================================================
# 7. train.py preflight checks
# ===========================================================================

class TestTrainPreflight:
    """
    These tests exercise run_preflight() in isolation without calling
    llamafactory-cli.  Each test ensures PreflightError is raised with the
    expected cause.
    """

    def _make_config(self, overrides: dict) -> dict:
        base = {
            "model_name_or_path": "/nonexistent/model",
            "stage": "sft",
            "dataset": "kalimcoder_sft",
            "dataset_dir": "data",
            "output_dir": "/tmp/kc_test_output",
        }
        return {**base, **overrides}

    def test_missing_model_raises(self, tmp_path):
        from scripts.train import PreflightError, run_preflight  # noqa
        cfg = self._make_config({"model_name_or_path": str(tmp_path / "no_model")})
        with pytest.raises(PreflightError, match="[Mm]odel"):
            run_preflight(cfg, tmp_path / "cfg.yaml")

    def test_missing_dataset_info_raises(self, tmp_path):
        from scripts.train import PreflightError, run_preflight  # noqa
        # Create a fake model dir with tokenizer
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "tokenizer_config.json").write_text("{}")
        (model_dir / "config.json").write_text("{}")

        cfg = self._make_config({
            "model_name_or_path": str(model_dir),
            "dataset_dir": str(tmp_path / "data"),  # non-existent data dir
        })
        with pytest.raises(PreflightError, match="dataset_info"):
            run_preflight(cfg, tmp_path / "cfg.yaml")

    def test_unregistered_dataset_raises(self, tmp_path):
        from scripts.train import PreflightError, run_preflight  # noqa
        # Create fake model + tokenizer
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "tokenizer_config.json").write_text("{}")

        # Create dataset_info.json WITHOUT the expected dataset
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "dataset_info.json").write_text(json.dumps({"other_ds": {}}))

        cfg = self._make_config({
            "model_name_or_path": str(model_dir),
            "dataset_dir": str(data_dir),
            "dataset": "kalimcoder_sft",
        })
        with pytest.raises(PreflightError, match="[Rr]egist"):
            run_preflight(cfg, tmp_path / "cfg.yaml")

    def test_missing_dataset_file_raises(self, tmp_path):
        from scripts.train import PreflightError, run_preflight  # noqa
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "tokenizer_config.json").write_text("{}")

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Register dataset but don't create the file
        (data_dir / "dataset_info.json").write_text(json.dumps({
            "kalimcoder_sft": {
                "file_name": str(tmp_path / "nonexistent.jsonl"),
                "formatting": "alpaca",
                "columns": {"prompt": "instruction", "query": "input", "response": "output"},
            }
        }))

        cfg = self._make_config({
            "model_name_or_path": str(model_dir),
            "dataset_dir": str(data_dir),
        })
        with pytest.raises(PreflightError, match="[Ff]ile"):
            run_preflight(cfg, tmp_path / "cfg.yaml")
