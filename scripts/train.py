"""
scripts/train.py — Preflight-checked LLaMA Factory training launcher.

Performs full validation before handing off to ``llamafactory-cli``:

  ✓ Config file exists and is valid YAML
  ✓ Model directory exists and has model weights
  ✓ Tokenizer present in model directory
  ✓ data/dataset_info.json exists
  ✓ Dataset name is registered in dataset_info.json
  ✓ Dataset file exists on disk
  ✓ Output directory is writable
  ✓ llamafactory-cli is on PATH

On success the script hands off to ``llamafactory-cli train <config>`` and
exits with the same return code.  On failure it prints a human-readable error
and exits with code 1.

Checkpoint resume
-----------------
If ``output_dir`` already contains a ``trainer_state.json``, the script
automatically appends ``--resume_from_checkpoint last`` to the call.

Usage
-----
    python scripts/train.py --config configs/qwen3_sft.yaml
    python scripts/train.py --config configs/qwen3_sft.yaml --dry-run
    python scripts/train.py --config configs/dpo.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _tick(msg: str) -> None:
    print(f"  {_green('✓')} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_red('✗')} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_yellow('!')} {msg}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_model_path(raw: str) -> Path:
    """Resolve model_name_or_path to an absolute Path.

    Tries (in order):
    1. Absolute path as-is
    2. Relative to project root
    3. Relative to CWD
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    candidate_root = _PROJECT_ROOT / p
    if candidate_root.exists():
        return candidate_root
    candidate_cwd = Path.cwd() / p
    if candidate_cwd.exists():
        return candidate_cwd
    # Return root-relative even if it doesn't exist (for clear error message)
    return candidate_root


def _count_model_size(model_dir: Path) -> str:
    """Return a human-readable total size of model files in *model_dir*."""
    try:
        total = sum(
            f.stat().st_size
            for f in model_dir.rglob("*.safetensors")
        )
        if total == 0:
            total = sum(f.stat().st_size for f in model_dir.rglob("*.bin"))
        if total == 0:
            return "size unknown"
        gb = total / (1024 ** 3)
        return f"{gb:.2f} GB"
    except Exception:  # noqa: BLE001
        return "size unknown"


def _count_jsonl_rows(path: Path) -> int:
    """Count lines in a JSONL file without reading all into memory."""
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    except Exception:  # noqa: BLE001
        return -1


def _count_json_rows(path: Path) -> int:
    """Count rows in a JSON array file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return len(data)
        return -1
    except Exception:  # noqa: BLE001
        return -1


def _find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the path to the latest checkpoint if resumable, else None."""
    if not output_dir.exists():
        return None
    trainer_state = output_dir / "trainer_state.json"
    if trainer_state.exists():
        return output_dir
    # Also check for checkpoint-XXXX subdirectories
    checkpoints = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
    )
    return checkpoints[-1] if checkpoints else None


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------


class PreflightError(Exception):
    """Raised when a preflight check fails."""


def run_preflight(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Run all preflight checks. Returns a context dict with resolved values.

    Raises :class:`PreflightError` with a human-readable message on failure.
    """
    print(_bold("\nKalimCoder training preflight checks"))
    print("─" * 50)

    ctx: dict[str, Any] = {}

    # ── 1. Config ─────────────────────────────────────────────────────────────
    _tick(f"Config          {config_path.name}")
    ctx["config_path"] = config_path

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model_raw = config.get("model_name_or_path", "")
    if not model_raw:
        _fail("model_name_or_path is missing from the config")
        raise PreflightError("model_name_or_path not set in config")

    model_path = _resolve_model_path(str(model_raw))
    if not model_path.exists():
        _fail(f"Model not found: {model_path}")
        print(f"\n  {_yellow('Hint:')} Run 'python scripts/download_model.py' first.")
        raise PreflightError(f"Model not found: {model_path}")

    size = _count_model_size(model_path)
    _tick(f"Model           {model_path}  ({size})")
    ctx["model_path"] = model_path

    # ── 3. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer_cfg = model_path / "tokenizer_config.json"
    if not tokenizer_cfg.exists():
        _fail(f"Tokenizer config not found: {tokenizer_cfg}")
        raise PreflightError(f"Tokenizer missing in {model_path}")
    _tick(f"Tokenizer       {tokenizer_cfg.name}")
    ctx["tokenizer_path"] = tokenizer_cfg

    # ── 4. dataset_info.json ─────────────────────────────────────────────────
    # LLaMA Factory looks in dataset_dir (default: data/) relative to CWD
    dataset_dir_raw = config.get("dataset_dir", "data")
    dataset_dir = _PROJECT_ROOT / dataset_dir_raw
    info_path = dataset_dir / "dataset_info.json"

    if not info_path.exists():
        _fail(f"dataset_info.json not found: {info_path}")
        print(
            f"\n  {_yellow('Hint:')} Run 'python scripts/build_training_dataset.py' to create it."
        )
        raise PreflightError(f"dataset_info.json missing: {info_path}")
    _tick(f"Dataset info    {info_path}")
    ctx["info_path"] = info_path

    # ── 5. Dataset registration ───────────────────────────────────────────────
    dataset_name = config.get("dataset", "")
    if not dataset_name:
        _fail("'dataset' key is missing from the training config")
        raise PreflightError("'dataset' not set in config")

    with info_path.open(encoding="utf-8") as fh:
        info: dict = json.load(fh)

    if dataset_name not in info:
        _fail(f"Dataset {dataset_name!r} not in dataset_info.json")
        registered = list(info.keys())
        print(f"\n  Registered datasets: {registered}")
        print(f"  Hint: Run 'python scripts/build_training_dataset.py' to register it.")
        raise PreflightError(f"Dataset {dataset_name!r} not registered")

    _tick(f"Dataset info    {dataset_name!r} registered")
    ctx["dataset_name"] = dataset_name
    ctx["dataset_entry"] = info[dataset_name]

    # ── 6. Dataset file on disk ───────────────────────────────────────────────
    file_name = info[dataset_name].get("file_name", "")
    if not file_name:
        _fail(f"Dataset entry {dataset_name!r} has no file_name")
        raise PreflightError("Dataset entry has no file_name")

    ds_file = Path(file_name)
    if not ds_file.is_absolute():
        ds_file = _PROJECT_ROOT / ds_file

    if not ds_file.exists():
        _fail(f"Dataset file not found: {ds_file}")
        print(f"\n  Hint: Run 'python scripts/build_training_dataset.py' to generate it.")
        raise PreflightError(f"Dataset file missing: {ds_file}")

    # Count rows
    if ds_file.suffix == ".jsonl":
        n_rows = _count_jsonl_rows(ds_file)
    elif ds_file.suffix == ".json":
        n_rows = _count_json_rows(ds_file)
    else:
        n_rows = -1
    row_str = f"{n_rows:,} rows" if n_rows >= 0 else "size unknown"
    _tick(f"Dataset file    {ds_file.name}  ({row_str})")
    ctx["dataset_file"] = ds_file

    # ── 7. Output directory ───────────────────────────────────────────────────
    output_dir_raw = config.get("output_dir", "")
    if not output_dir_raw:
        _warn("output_dir not set — LLaMA Factory will use its default")
    else:
        output_dir = Path(output_dir_raw)
        if not output_dir.is_absolute():
            output_dir = _PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        # Writability check
        test_file = output_dir / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
            _tick(f"Output dir      {output_dir}  (writable)")
        except OSError as exc:
            _fail(f"Output directory not writable: {output_dir} — {exc}")
            raise PreflightError(f"Output dir not writable: {exc}") from exc
        ctx["output_dir"] = output_dir

    # ── 8. llamafactory-cli ────────────────────────────────────────────────────
    cli = shutil.which("llamafactory-cli")
    if cli is None:
        _fail("llamafactory-cli not found on PATH")
        print("\n  Hint: pip install llamafactory")
        raise PreflightError("llamafactory-cli not found on PATH")
    _tick(f"llamafactory-cli  {cli}")
    ctx["cli"] = cli

    print("─" * 50)
    return ctx


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def launch_training(ctx: dict[str, Any], config_path: Path, dry_run: bool) -> int:
    """Build and run the llamafactory-cli command. Returns the exit code."""
    output_dir: Path | None = ctx.get("output_dir")

    # Check for existing checkpoint (auto-resume)
    checkpoint = None
    if output_dir:
        checkpoint = _find_latest_checkpoint(output_dir)
        if checkpoint:
            print(
                f"\n  {_yellow('[RESUME]')} Checkpoint found: {checkpoint}\n"
                "              Training will resume automatically."
            )

    # Build command
    cmd = [ctx["cli"], "train", str(config_path)]
    if checkpoint:
        cmd += ["--resume_from_checkpoint", str(checkpoint)]

    print(f"\n  {_bold('Command:')} {' '.join(cmd)}")
    print(f"\n{_bold('  Starting training …')}\n")
    print("=" * 60)

    if dry_run:
        print(f"  {_yellow('[DRY-RUN] Skipping execution.')}")
        return 0

    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    return result.returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Preflight-checked LLaMA Factory training launcher.",
    )
    parser.add_argument(
        "--config", type=Path, required=True, metavar="PATH",
        help="Path to the LLaMA Factory training YAML config.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Run all preflight checks and print the command, but do not execute.",
    )
    parser.add_argument(
        "--skip-checks", action="store_true", default=False,
        help="Skip preflight checks and launch directly (not recommended).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    config_path = args.config
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    if not config_path.is_absolute():
        config_path = Path.cwd() / args.config

    if not config_path.exists():
        print(_red(f"[ERROR] Config not found: {config_path}"))
        return 1

    try:
        config = _load_yaml(config_path)
    except Exception as exc:  # noqa: BLE001
        print(_red(f"[ERROR] Cannot parse config YAML: {exc}"))
        return 1

    if args.skip_checks:
        print(_yellow("[WARN] Preflight checks skipped."))
        cli = shutil.which("llamafactory-cli") or "llamafactory-cli"
        cmd = [cli, "train", str(config_path)]
        if args.dry_run:
            print(f"Command: {' '.join(cmd)}")
            return 0
        return subprocess.run(cmd, cwd=str(_PROJECT_ROOT)).returncode

    try:
        ctx = run_preflight(config, config_path)
    except PreflightError as exc:
        print(f"\n{_red('[FAIL]')} {exc}\n")
        return 1

    return launch_training(ctx, config_path, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
