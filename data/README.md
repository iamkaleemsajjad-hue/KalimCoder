# data/ — LLaMA Factory dataset registry

This directory is managed automatically by `scripts/build_training_dataset.py`.

**Do not edit `dataset_info.json` by hand.**

## Contents

| File | Created by | Purpose |
|------|-----------|---------|
| `dataset_info.json` | `scripts/build_training_dataset.py` | LLaMA Factory dataset registry |

## How it works

Running:

```bash
python scripts/build_training_dataset.py
```

will:
1. Read every `datasets/processed/<name>/train-*.parquet` shard
2. Convert rows to Alpaca format
3. Write `datasets/instruction/kalimcoder_sft.jsonl`
4. Write/update `data/dataset_info.json` with the correct entry

LLaMA Factory reads this file when `dataset_dir: data` is set in the training YAML.
