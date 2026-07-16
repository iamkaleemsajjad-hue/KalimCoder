# KalimCoder Architecture

## Overview

```
Qwen3-8B (Base)
     │
     ▼
Data Pipeline (Streaming)
     │
     ▼
QLoRA Fine-tuning (SFT)
     │
     ▼
DPO Alignment
     │
     ▼
KalimCoder (Coding LLM)
     │
     ▼
Agent Layer (Tool Use)
```

## Base Model

| Property | Value |
|----------|-------|
| Model | Qwen3-8B |
| Parameters | 8 Billion |
| Context Length | 32,768 tokens |
| Architecture | Transformer (GQA) |

## Training Setup

### Hardware
- **Platform**: Kaggle
- **GPUs**: Tesla T4 x2 (2 × 16 GB VRAM)
- **Strategy**: Multi-GPU with DDP / DeepSpeed ZeRO

### Fine-tuning Method
- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit NF4 via bitsandbytes
- **Rank**: 64
- **Alpha**: 128
- **Target modules**: All linear layers

### Training Framework
- **LLaMA Factory** for SFT and DPO
- **PEFT** for LoRA adapter management
- **TRL** for DPO training

## Inference

- **Backend**: Hugging Face Transformers / vLLM
- **Quantization**: 4-bit (inference-time)
- **Context**: Up to 4,096 tokens

## Data Pipeline (Streaming)

The pipeline is fully streaming — no intermediate files are written to disk.  
Peak disk usage stays below **2 GB** even for 500 GB+ corpora.

```
HuggingFace Hub / Local Arrow
           │
           ▼
   [1] DatasetSource          — per-source streaming iterator
           │
           ▼
   [2] Schema Adapter          — raw row → CanonicalExample
           │
           ▼
   [3] clean_stage()           — drops malformed / blocked content
           │
           ▼
   [4] quality_stage()         — QualityScorer assigns [0,1] score; drops low-quality
           │
           ▼
   [5] dedup_stage()           — TwoStageDedup (Bloom + SHA-256); no false positives
           │
           ▼
   [6] ShardedWriter            — atomic parquet shards (100k rows, snappy-compressed)
           │
           ▼
   [7] StateManager             — JSON checkpoint after every shard
           │
           ▼
  datasets/processed/<name>/
    train-*.parquet + val-*.parquet + _metadata.json
```

### Dataset Sources (configured in `configs/datasets.yaml`)

| Dataset | Type | Task | Ratio |
|---------|------|------|-------|
| OpenCoder SFT Stage-1 | Instruction tuning | instruction | 20% |
| The Stack v2 | Code completion | completion | 40% |
| CodeSearchNet | Code + docstrings | documentation | 15% |
| SWE-bench Verified | Bug fixes / patches | debugging | 15% |

### Key Properties

| Property | Value |
|----------|-------|
| Max RAM | < 2 GB |
| Max temp disk | 0 (no intermediate files) |
| Output format | Parquet (snappy) |
| Deduplication | Two-stage Bloom + SHA-256 |
| Quality scoring | 5-signal weighted score |
| Resumable | Yes (shard-level checkpoints) |
| Reproducible | Yes (ExperimentManifest with git SHA) |

See [`docs/streaming_pipeline.md`](streaming_pipeline.md) for full details.

## Module Map

```
src/
  data/
    schema.py        — CanonicalExample dataclass (canonical interchange format)
    sources/
      base.py        — DatasetSource ABC
      huggingface.py — HuggingFaceSource (streaming + fallback)
      local_arrow.py — LocalArrowSource  (existing Arrow datasets)
      jsonl.py       — JSONLSource        (local JSONL files)
      parquet.py     — ParquetSource      (processed parquet shards)
      git_repo.py    — GitRepositorySource (stub, future)
    adapters.py      — Per-dataset row → CanonicalExample converters
    quality.py       — QualityScorer (5-signal weighted scoring)
    dedup.py         — BloomFilter + TwoStageDedup
    streaming.py     — build_pipeline() lazy generator chain
    writer.py        — ShardedWriter (atomic parquet output)
    mixer.py         — DatasetMixer (mixture ratio enforcement)
    manifest.py      — ExperimentManifest (reproducibility metadata)
    state.py         — StateManager (crash-safe shard checkpointing)
    registry.py      — Load configs/datasets.yaml
    cleaner.py       — Legacy batch cleaner (preserved)
    formatter.py     — Legacy formatter (preserved)
  agent/             — Tool-use agent layer
  inference/         — Inference utilities
  utils/             — Logging, I/O helpers
scripts/
  run_pipeline.py         — Unified streaming pipeline entry point (NEW)
  tokenize_dataset.py     — Optional offline tokenization stage (NEW)
  download_datasets.py    — HF dataset downloader (legacy / offline prep)
  clean_dataset.py        — Legacy batch cleaner
  build_training_dataset.py — Legacy dataset builder
  validate_dataset.py     — Dataset quality inspector
configs/
  pipeline.yaml    — Streaming pipeline settings (NEW)
  mixture.yaml     — Dataset blend ratios (NEW)
  datasets.yaml    — Dataset registry
  qlora.yaml       — QLoRA training config
```

