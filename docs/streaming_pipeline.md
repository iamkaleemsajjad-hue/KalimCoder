# Streaming Data Pipeline

> Production-quality, zero-copy dataset pipeline for KalimCoder — processes terabytes of code without ever materialising the full corpus in RAM or temporary disk files.

## Overview

The streaming pipeline replaces the old **download → clean → build** three-stage chain with a single iterator-based pass that processes each example once and writes atomic parquet shards directly to `datasets/processed/`.

```
HuggingFace Hub   (or local Arrow)
        │
        ▼
 [1] DatasetSource          — streams raw rows, no materialisation
        │
        ▼
 [2] Adapter (per-source)   — raw row → CanonicalExample
        │
        ▼
 [3] clean_stage()          — drops too-short, HTML-heavy, blocked content
        │
        ▼
 [4] quality_stage()        — QualityScorer → drops low-scoring examples
        │
        ▼
 [5] dedup_stage()          — TwoStageDedup → drops output duplicates
        │
        ▼
 [6] ShardedWriter           — buffers → atomic parquet shards
        │
        ▼
 [7] StateManager            — checkpoints every shard
```

**Memory bound:** < 2 GB RAM at all times.  
**Disk bound:** Only output parquet files; no intermediate Arrow copies.  
**Resumable:** Crashes restart from the last completed shard.

---

## Quick Start

```bash
# Full streaming pipeline (all enabled datasets from configs/datasets.yaml)
make stream-pipeline

# Single dataset
python scripts/run_pipeline.py --name opc_sft_stage1

# Resume after interruption
make stream-resume

# Dry-run (inspect sources only, no writes)
make stream-pipeline-dry-run

# Use pre-downloaded Arrow files (no network)
make stream-pipeline-offline

# Check processing status of all datasets
make pipeline-status
```

---

## Configuration

### `configs/pipeline.yaml`
Global pipeline settings — shard size, streaming mode, quality thresholds, dedup capacity, output format.

```yaml
pipeline:
  shard_size: 50000       # examples per shard buffer
  streaming: true         # HF streaming mode
  batch_size: 1000        # rows per processing batch
  num_workers: 0          # 0 = single thread (safe on Kaggle)
  delete_after_processing: true

dedup:
  enabled: true
  bloom_capacity: 5000000  # ~8.6 MB RAM
  bloom_fpr: 0.001
  max_confirmed_set_mb: 512

quality:
  min_quality_score: 0.30
  min_tokens: 10
  max_tokens: 8192

output:
  format: parquet
  train_shard_rows: 100000
  val_ratio: 0.05
  compress: snappy
```

### `configs/mixture.yaml`
Dataset blend ratios for the final training corpus.

```yaml
mixture:
  strategy: approximate   # approximate | exact | oversample
  ratios:
    the_stack_v2:       0.40
    opc_sft_stage1:     0.20
    swe_bench_verified: 0.15
    code_search_net:    0.15
```

### `configs/datasets.yaml`
Registry of all datasets with streaming, license, and task-type metadata.

```yaml
datasets:
  - name:        opc_sft_stage1
    repo_id:     OpenCoder-LLM/opc-sft-stage1
    split:       train
    destination: datasets/raw/opc_sft_stage1
    adapter:     opc_sft_stage1
    enabled:     true
    streaming:   true
    license:     "Apache-2.0"
    task_type:   instruction
```

---

## Canonical Schema

Every pipeline stage operates on `CanonicalExample` objects. No source-specific format leaks past the adapter layer.

```python
@dataclass
class CanonicalExample:
    id:            str    # SHA-256(instruction + output) — deterministic
    schema_version: str   # bumped on breaking schema changes
    dataset:       str    # registry entry name
    task_type:     str    # instruction | completion | qa | debugging | documentation
    language:      str    # python | javascript | unknown | ...
    repository:    str    # source repo URL (when available)
    license:       str    # SPDX identifier
    instruction:   str    # the human turn / prompt
    input:         str    # optional context
    output:        str    # the target response
    quality_score: float  # [0.0, 1.0] assigned by QualityScorer
    metadata:      dict   # source-specific extras (JSON-encoded in parquet)
```

---

## Dataset Sources

| Source | Class | When to use |
|--------|-------|-------------|
| HuggingFace Hub (streaming) | `HuggingFaceSource` | New datasets; no local copy needed |
| Local Arrow (downloaded) | `LocalArrowSource` | Offline; re-processing existing Arrow files |
| JSONL files | `JSONLSource` | Local synthetic data; custom datasets |
| Processed parquet shards | `ParquetSource` | Re-mixing already processed data |
| GitHub repository (stub) | `GitRepositorySource` | Future: raw repo ingestion |

---

## Quality Scoring

`QualityScorer` assigns a score in `[0.0, 1.0]` to every example based on five weighted sub-scores:

| Sub-score | Weight | Signal |
|-----------|--------|--------|
| `token_score` | 0.25 | Linear ramp within `[min_tokens, max_tokens]` |
| `alpha_score` | 0.20 | Fraction of alphanumeric chars |
| `comment_ratio_score` | 0.20 | Penalises files that are mostly comments |
| `autogen_score` | 0.20 | 0.0 if an autogenerated-file pattern is detected |
| `language_score` | 0.15 | 0.5 for unknown languages, 1.0 for known |

Examples with `quality_score < min_quality_score` (default 0.30) are dropped.

---

## Two-Stage Deduplication

`TwoStageDedup` eliminates duplicate outputs with guaranteed zero false positives:

1. **BloomFilter** (Stage 1): O(1) probabilistic membership test; ~8.6 MB for 5M items at FPR=0.001.
2. **SHA-256 set** (Stage 2): Exact verification triggered only on Bloom hits. Capped at `max_confirmed_set_mb` (default 512 MB).

| Call sequence | Result |
|---------------|--------|
| 1st call with text T | Not dup — add to Bloom |
| 2nd call with text T | Bloom hit — add to SHA confirmed set |
| 3rd+ call with text T | SHA hit — **confirmed duplicate, drop** |

On Bloom-only mode (SHA cap exceeded): all Bloom hits are treated as duplicates with a logged warning.

---

## Resumability

`StateManager` writes an atomic JSON checkpoint after every shard. On restart with `--resume`, completed shards are skipped automatically.

```
datasets/state/
    opc_sft_stage1/
        state.json         ← {completed_shard_indices, total_written, ...}
    the_stack_v2/
        state.json
```

To restart from scratch:
```bash
python scripts/run_pipeline.py --force
```

---

## Output Layout

```
datasets/
    processed/
        opc_sft_stage1/
            train-00001.parquet
            train-00002.parquet
            val-00001.parquet
            _metadata.json
        the_stack_v2/
            train-00001.parquet
            ...
        manifest_<uuid>.json     ← experiment manifest
    token_cache/                  ← written by scripts/tokenize_dataset.py
        opc_sft_stage1/
            train-00001.arrow
        tokenizer_info.json
    state/                        ← shard checkpoints
```

---

## Experiment Manifest

Every pipeline run writes a `manifest_<uuid>.json` to `datasets/processed/` containing:

- Git commit hash
- Pipeline + mixture config snapshots
- Per-source statistics (rows written, dropped, quality distribution)
- Deduplication statistics
- List of all output files (train + val)

This makes every training run fully reproducible.

---

## Adding a New Dataset

1. **Add an adapter** in `src/data/adapters.py`:
   ```python
   def _adapt_my_dataset(row, dataset, license, task_type):
       instr = row.get("question", "")
       out   = row.get("answer",   "")
       if not instr or not out:
           return None
       return make_example(dataset=dataset, instruction=instr, output=out, ...)
   
   ADAPTER_REGISTRY["my_dataset"] = _adapt_my_dataset
   ```

2. **Register in `configs/datasets.yaml`**:
   ```yaml
   - name:     my_dataset
     repo_id:  org/my-dataset
     split:    train
     adapter:  my_dataset
     enabled:  true
     streaming: true
     license:  "MIT"
     task_type: instruction
   ```

3. **Add to `configs/mixture.yaml`**:
   ```yaml
   mixture:
     ratios:
       my_dataset: 0.10
   ```

4. **Run the pipeline**:
   ```bash
   python scripts/run_pipeline.py --name my_dataset
   ```

---

## CLI Reference

### `scripts/run_pipeline.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--name NAME` | all | Process only the named dataset |
| `--resume` | off | Skip completed shards |
| `--force` | off | Reset all state and restart |
| `--dry-run` | off | Inspect sources without writing |
| `--offline` | off | Use local Arrow files (no HF download) |
| `--no-dedup` | off | Disable deduplication |
| `--no-quality` | off | Disable quality scoring |
| `--hf-token TOKEN` | env | HuggingFace token for gated datasets |
| `--config PATH` | `configs/pipeline.yaml` | Pipeline config path |
| `--mixture PATH` | `configs/mixture.yaml` | Mixture config path |

### `scripts/tokenize_dataset.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--tokenizer REPO` | required | HuggingFace tokenizer repo |
| `--max-length N` | 8192 | Max sequence length (truncation) |
| `--processed-dir PATH` | `datasets/processed` | Input parquet root |
| `--out-dir PATH` | `datasets/token_cache` | Output Arrow cache root |
| `--name NAME` | all | Process only named dataset |
