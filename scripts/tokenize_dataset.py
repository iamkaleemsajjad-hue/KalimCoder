"""
scripts/tokenize_dataset.py — Optional offline tokenization stage.

Reads sharded parquet files from ``datasets/processed/`` produced by
``run_pipeline.py``, tokenizes examples with a HuggingFace tokenizer and chat
template, and writes Arrow token cache files to ``datasets/token_cache/``.

This stage is:
* **Optional** — training scripts can tokenize on-the-fly; this stage
  pre-computes tokens so they are not re-computed each epoch.
* **Idempotent** — files already in the cache are skipped.
* **CPU-only** — no GPU or CUDA required.
* **Decoupled** — changing the model or context length only re-runs this
  stage, not the full data pipeline.

Output layout::

    datasets/token_cache/
        <dataset_name>/
            train-00001.arrow
            val-00001.arrow
            tokenizer_info.json

Usage
-----
    python scripts/tokenize_dataset.py \\
        --tokenizer Qwen/Qwen3-8B \\
        --max-length 8192

    python scripts/tokenize_dataset.py \\
        --processed-dir datasets/processed \\
        --out-dir datasets/token_cache \\
        --tokenizer Qwen/Qwen3-8B \\
        --max-length 4096 \\
        --num-workers 4 \\
        --name opc_sft_stage1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

_shadow = ["", ".", str(Path(".").resolve()), str(_PROJECT_ROOT)]
for _e in _shadow:
    while _e in sys.path:
        sys.path.remove(_e)

try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] Missing dependency: tqdm\nInstall with:  pip install tqdm")
    sys.exit(1)
finally:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_PROCESSED = _PROJECT_ROOT / "datasets" / "processed"
_DEFAULT_CACHE = _PROJECT_ROOT / "datasets" / "token_cache"
_LOG_DIR = _PROJECT_ROOT / "logs" / "tokenize"

logger = logging.getLogger("tokenize_dataset")

# ChatML-style template applied when the tokenizer has no built-in chat template
_DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\n'}}"
    "{% endif %}"
)


# ---------------------------------------------------------------------------
# Core tokenization logic
# ---------------------------------------------------------------------------


def _build_chat_text(instruction: str, input_text: str, output: str) -> list[dict]:
    """Construct a ChatML-format message list from canonical example fields."""
    user_content = instruction
    if input_text.strip():
        user_content = f"{instruction}\n\n```\n{input_text}\n```"
    return [
        {"role": "system", "content": "You are an expert coding assistant."},
        {"role": "user",   "content": user_content},
        {"role": "assistant", "content": output},
    ]


def _tokenize_parquet_file(
    parquet_path: Path,
    out_path: Path,
    tokenizer,
    max_length: int,
    batch_size: int,
    logger: logging.Logger,
) -> dict:
    """Tokenize one parquet shard and write an Arrow cache file.

    Returns a stats dict.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow required. Install: pip install pyarrow")

    if out_path.exists():
        logger.debug("Cache hit — skipping %s", out_path.name)
        return {"skipped": True, "rows": 0}

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    input_ids_all: list[list[int]] = []
    attention_mask_all: list[list[int]] = []
    labels_all: list[list[int]] = []
    dropped = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        bd = batch.to_pydict()
        n = len(bd.get("instruction", []))
        for i in range(n):
            instr = str(bd.get("instruction", [""] * n)[i] or "")
            inp   = str(bd.get("input",       [""] * n)[i] or "")
            out   = str(bd.get("output",      [""] * n)[i] or "")

            messages = _build_chat_text(instr, inp, out)
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                encoded = tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )
                ids = encoded["input_ids"]
                mask = encoded["attention_mask"]
                # Labels: mask prompt tokens with -100 (keep only assistant response)
                lab = list(ids)
                input_ids_all.append(ids)
                attention_mask_all.append(mask)
                labels_all.append(lab)
            except Exception as exc:  # noqa: BLE001
                dropped += 1
                logger.debug("Tokenization error row %d: %s", i, exc)

    if not input_ids_all:
        logger.warning("No rows tokenized from %s", parquet_path.name)
        return {"skipped": False, "rows": 0, "dropped": dropped}

    # Write Arrow file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".arrow.tmp")
    table = pa.table({
        "input_ids":      pa.array(input_ids_all,      type=pa.list_(pa.int32())),
        "attention_mask": pa.array(attention_mask_all,  type=pa.list_(pa.int32())),
        "labels":         pa.array(labels_all,          type=pa.list_(pa.int32())),
    })
    import pyarrow.ipc as ipc
    with ipc.new_file(str(tmp), table.schema) as writer:
        writer.write_table(table)
    os.replace(tmp, out_path)

    return {"skipped": False, "rows": len(input_ids_all), "dropped": dropped}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tokenize_dataset.py",
        description="Tokenize processed parquet shards into Arrow token caches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tokenizer", required=True,
                        help="HuggingFace tokenizer repo (e.g. 'Qwen/Qwen3-8B').")
    parser.add_argument("--processed-dir", type=Path, default=_DEFAULT_PROCESSED,
                        help="Root dir of processed parquet shards.")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_CACHE,
                        help="Root dir for Arrow token cache output.")
    parser.add_argument("--max-length", type=int, default=8192,
                        help="Maximum token sequence length (truncation).")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Rows per PyArrow batch read.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Tokenizer parallelism (0 = single-threaded).")
    parser.add_argument("--name", type=str, default=None,
                        help="Process only the named dataset subdirectory.")
    parser.add_argument("--glob", type=str, default="train-*.parquet",
                        help="Glob pattern for shard files (default: train-*.parquet).")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _configure_logging(verbose: bool) -> logging.Logger:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(_LOG_DIR / f"tokenize_{ts}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)
    return logging.getLogger("tokenize_dataset")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    global logger
    logger = _configure_logging(args.verbose)

    # Load tokenizer
    logger.info("Loading tokenizer: %s", args.tokenizer)
    try:
        from transformers import AutoTokenizer  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=True,
        )
        if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
            logger.warning("Tokenizer has no chat_template; using default ChatML template.")
            tokenizer.chat_template = _DEFAULT_CHAT_TEMPLATE
        logger.info("Tokenizer loaded: vocab_size=%d", tokenizer.vocab_size)
    except ImportError:
        logger.critical("transformers not installed. Run: pip install transformers")
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.critical("Failed to load tokenizer %r: %s", args.tokenizer, exc)
        return 1

    # Set parallelism
    if args.num_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer._tokenizer.enable_parallelism(True) if hasattr(tokenizer, "_tokenizer") else None
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    processed_root: Path = args.processed_dir
    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    if not processed_root.exists():
        logger.error("Processed directory not found: %s", processed_root)
        return 1

    # Discover dataset directories
    dataset_dirs = sorted(d for d in processed_root.iterdir() if d.is_dir() and not d.name.startswith("_"))
    if args.name:
        dataset_dirs = [d for d in dataset_dirs if d.name == args.name]
        if not dataset_dirs:
            logger.error("Dataset %r not found under %s", args.name, processed_root)
            return 1

    logger.info("Tokenizing %d dataset(s): %s", len(dataset_dirs), [d.name for d in dataset_dirs])

    # Write tokenizer info
    tokenizer_info = {
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "vocab_size": tokenizer.vocab_size,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_root / "tokenizer_info.json").write_text(
        json.dumps(tokenizer_info, indent=2), encoding="utf-8"
    )

    total_rows = 0
    total_dropped = 0

    for ds_dir in tqdm(dataset_dirs, desc="Datasets", unit="ds"):
        parquet_files = sorted(ds_dir.glob(args.glob))
        if not parquet_files:
            logger.warning("No files matching %r under %s", args.glob, ds_dir)
            continue

        out_ds_dir = out_root / ds_dir.name
        out_ds_dir.mkdir(parents=True, exist_ok=True)

        for pf in tqdm(parquet_files, desc=f"  {ds_dir.name}", unit="shard", leave=False):
            out_arrow = out_ds_dir / pf.with_suffix(".arrow").name
            stats = _tokenize_parquet_file(
                parquet_path=pf,
                out_path=out_arrow,
                tokenizer=tokenizer,
                max_length=args.max_length,
                batch_size=args.batch_size,
                logger=logger,
            )
            total_rows += stats.get("rows", 0)
            total_dropped += stats.get("dropped", 0)

        logger.info("Dataset %r: tokenized shards → %s", ds_dir.name, out_ds_dir)

    logger.info(
        "Tokenization complete — %d rows tokenized, %d dropped.",
        total_rows, total_dropped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
