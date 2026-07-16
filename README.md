<div align="center">

# 🤖 KaleemCoder

**A coding-focused LLM fine-tuned from Qwen3-8B using QLoRA + DPO**

*Repository understanding · Bug fixing · PR review · Test generation · Agentic tool use*

[![CI](https://github.com/iamkaleemsajjad-hue/KalimCoder/actions/workflows/ci.yml/badge.svg)](https://github.com/iamkaleemsajjad-hue/KalimCoder/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://pre-commit.com/)

</div>

---

## 📖 Project Overview

KaleemCoder is a personal AI research project to build a **production-grade coding LLM** starting from Qwen3-8B.  
The goal is to train a model that can operate as a real software engineering assistant — not just autocomplete, but understand repositories, reason about bugs, review code, and act as an agent.

**Why?** Most open-source coding models are fine-tuned on generic instruction datasets. KaleemCoder targets *software engineering tasks specifically* — with curated data, reproducible experiments, and systematic benchmarking.

| Capability | Status |
|---|---|
| Code generation (Python, C++, JS …) | 🔄 In progress |
| Bug detection & fixing | 🔄 In progress |
| Pull request review | 📋 Planned |
| Code explanation | 📋 Planned |
| Test generation | 📋 Planned |
| Repository-level reasoning | 📋 Planned |
| Agentic tool use (ReAct loop) | 📋 Planned |

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│        Qwen3-8B  (Base)         │  8B params · 32k context
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   QLoRA SFT  (LLaMA Factory)   │  4-bit NF4 · rank 64 · Kaggle T4×2
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   DPO Alignment  (TRL)          │  chosen / rejected pairs
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│        KaleemCoder  ✨           │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Agent Layer  (ReAct + Tools)   │  run_code · read_file · write_file
└─────────────────────────────────┘
```

| Property | Value |
|---|---|
| **Base model** | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| **Fine-tuning** | QLoRA — 4-bit NF4, rank 64, all linear layers |
| **Framework** | [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) + [TRL](https://github.com/huggingface/trl) |
| **Alignment** | DPO (Direct Preference Optimization) |
| **Hardware** | Kaggle Tesla T4 × 2 (2 × 16 GB VRAM) |
| **Context** | 4,096 tokens |

---

## 📍 Roadmap

```
Phase 1 — Foundation       ✅ Repository structure & tooling
Phase 2 — Data             🔄 Dataset collection and pipeline
Phase 3 — SFT Training     📋 QLoRA fine-tuning runs
Phase 4 — Alignment        📋 DPO on preference pairs
Phase 5 — Agent            📋 ReAct loop with code execution
Phase 6 — Release          📋 HF Hub + model card + demo
```

See [`docs/roadmap.md`](docs/roadmap.md) for detailed milestones.

---

## 📦 Repository Structure

```
KalimCoder/
│
├── src/                        ← Core Python package
│   ├── data/                   ← Loader, cleaner, formatter
│   ├── models/                 ← QLoRA model loading
│   ├── training/               ← Trainer callbacks (metrics, GPU mem)
│   ├── evaluation/             ← Metrics (pass@k, exact match…)
│   ├── inference/              ← KaleemCoderGenerator class
│   ├── utils/                  ← Logging, I/O helpers
│   └── agent/                  ← ReAct agent + tool definitions
│
├── configs/                    ← All YAML configuration
│   ├── model/                  ← Model architecture (qwen3_8b.yaml)
│   ├── training/               ← SFT & DPO hyperparameters
│   ├── dataset/                ← Dataset preprocessing settings
│   ├── evaluation/             ← Benchmark runner configs
│   └── agent/                  ← Agent loop settings
│
├── datasets/                   ← Data pipeline (large files gitignored)
│   ├── raw/                    ← Original downloads
│   ├── cleaned/                ← Filtered & deduplicated
│   ├── instruction/            ← SFT prompt-response pairs
│   ├── preference/             ← DPO chosen/rejected pairs
│   ├── evaluation/             ← Eval & benchmark prompts
│   └── synthetic/              ← Synthetically generated examples
│
├── experiments/                ← One directory per training run
│   ├── 001_qwen_base/          ← config · metrics · notes · plots
│   ├── 002_python_dataset/
│   ├── 003_cpp_dataset/
│   ├── 004_bug_fix/
│   ├── 005_dpo/
│   └── 006_repo_agent/
│
├── benchmarks/                 ← Benchmark runners & result tracking
│   ├── HumanEval/
│   ├── MBPP/
│   ├── LiveCodeBench/
│   ├── SWE-bench/
│   ├── RepoBench/
│   └── BigCodeBench/
│
├── scripts/                    ← CLI entry points
│   ├── download_model.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── merge_lora.py
│
├── notebooks/                  ← Step-by-step Jupyter notebooks (01–08)
├── docs/                       ← Architecture · Roadmap · Training notes
├── tests/                      ← pytest unit tests
├── logs/                       ← Training & eval logs (gitignored)
├── checkpoints/                ← Base model weights (gitignored)
├── adapters/                   ← LoRA adapters (gitignored)
│
├── .github/workflows/ci.yml    ← GitHub Actions (lint · test · pre-commit)
├── .pre-commit-config.yaml     ← black · isort · ruff · nbstripout
├── pyproject.toml              ← Project metadata & tool configs
├── Makefile                    ← Task runner (make train, make test …)
├── requirements.txt            ← Direct pip install list
├── CONTRIBUTING.md             ← Dev setup & contribution guide
└── LICENSE                     ← MIT
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU for training (inference works on CPU with quantization)
- Git

### Quick Start

```bash
git clone https://github.com/iamkaleemsajjad-hue/KalimCoder.git
cd KalimCoder

# Dev environment (no heavy ML packages — for exploring/contributing)
make setup-dev

# Full environment (includes torch, transformers, etc.)
make setup

# Install pre-commit hooks (auto-runs black, isort, ruff on commit)
make install-hooks
```

### Manual install with optional groups

```bash
pip install -e ".[train]"          # training only
pip install -e ".[eval]"           # evaluation only
pip install -e ".[dev]"            # dev tools only
pip install -e ".[train,eval,dev]" # everything
```

---

## 🗄️ Streaming Dataset Pipeline

The pipeline is **fully streaming** — no intermediate files are written to disk.
Peak disk usage stays below **2 GB** even for 500 GB+ raw corpora.

```
HuggingFace Hub
        │
        ▼  scripts/download_datasets.py
datasets/raw/   (optional offline cache)
        │
        ▼  scripts/run_pipeline.py
           clean → quality score → dedup → shard
datasets/processed/<name>/train-*.parquet
        │
        ▼  scripts/build_training_dataset.py
           convert → mixture ratios → shuffle → register
datasets/instruction/kalimcoder_sft.jsonl
data/dataset_info.json                       ← auto-created
        │
        ▼  scripts/train.py
           preflight checks → llamafactory-cli train
adapters/kaleemcoder-sft/
        │
        ▼  scripts/merge_lora.py
           LoRA merge → full model
adapters/kaleemcoder-sft-merged/
```

**Datasets:**

| Dataset | Task | Ratio |
|---------|------|-------|
| [OpenCoder SFT Stage-1](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage1) | Instruction tuning | 20% |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | Code completion | 40% |
| [CodeSearchNet](https://huggingface.co/datasets/code_search_net) | Code + docstrings | 15% |
| [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) | Bug fixes | 15% |

Ratios are configured in [`configs/mixture.yaml`](configs/mixture.yaml).

---

## 🏋️ End-to-End Training Workflow

> After setup, a brand-new user runs these commands in order.
> Everything else (dataset conversion, registration, preflight checks) happens automatically.

### Step 1 — Download the base model

```bash
python scripts/download_model.py
# Downloads Qwen/Qwen3-8B → checkpoints/qwen3-8b-base/
```

### Step 2 — Download datasets

```bash
python scripts/download_datasets.py
# Streams from HuggingFace Hub → datasets/raw/  (optional offline cache)
```

### Step 3 — Run streaming pipeline

```bash
python scripts/run_pipeline.py
# Cleans · scores · deduplicates → datasets/processed/<name>/train-*.parquet
# Peak disk: < 2 GB  |  Resume: --resume  |  Dry-run: --dry-run
```

### Step 4 — Build training dataset

```bash
python scripts/build_training_dataset.py
# Reads parquet shards → Alpaca JSONL → registers in data/dataset_info.json
# Output: datasets/instruction/kalimcoder_sft.jsonl
# Auto-creates: data/dataset_info.json
```

### Step 5 — Train (QLoRA SFT)

```bash
python scripts/train.py --config configs/qwen3_sft.yaml
# Runs preflight checks (model · tokenizer · dataset · registration)
# Launches: llamafactory-cli train configs/qwen3_sft.yaml
# Output:   adapters/kaleemcoder-sft/
# Resume:   automatic if checkpoint exists
```

### Step 6 — DPO alignment (optional)

```bash
python scripts/train.py --config configs/dpo.yaml
```

### Step 7 — Merge LoRA → full model

```bash
python scripts/merge_lora.py
# Output: adapters/kaleemcoder-sft-merged/
```

### Step 8 — Evaluate

```bash
python scripts/evaluate.py \
    --model_path adapters/kaleemcoder-sft-merged \
    --eval_data  datasets/evaluation/eval_prompts.jsonl
```

---

### Makefile shortcuts

```bash
make download-model     # Step 1
make download-data      # Step 2
make stream-pipeline    # Step 3
make build-dataset      # Step 4
make train-sft          # Step 5
make merge-lora         # Step 7
make eval               # Step 8
```

---

### Adding a new dataset

1. Add an entry to [`configs/datasets.yaml`](configs/datasets.yaml)
2. Add an adapter to [`src/data/adapters.py`](src/data/adapters.py) (if the schema is new)
3. Add a ratio to [`configs/mixture.yaml`](configs/mixture.yaml)
4. Re-run Steps 2–4

No other code changes required.

---

## 📊 Evaluation

```bash
make eval
# or: python scripts/evaluate.py \
#       --model_path adapters/kaleemcoder-sft-merged \
#       --eval_data  datasets/evaluation/eval_prompts.jsonl
```

---

## 📈 Benchmark Results

> Results will be updated after each training run. All benchmarks tracked in `benchmarks/`.

| Benchmark | Metric | Qwen3-8B Base | KaleemCoder SFT | KaleemCoder DPO |
|---|---|---|---|---|
| [HumanEval](https://github.com/openai/human-eval) | pass@1 | TBD | TBD | TBD |
| [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) | pass@1 | TBD | TBD | TBD |
| [LiveCodeBench](https://livecodebench.github.io) | pass@1 | TBD | TBD | TBD |
| [SWE-bench Lite](https://www.swebench.com) | resolve% | TBD | TBD | TBD |
| [BigCodeBench](https://bigcode-bench.github.io) | pass@1 | TBD | TBD | TBD |

---

## 💡 Example Usage

```python
from src.models.loader import load_with_adapter
from src.inference.generator import KaleemCoderGenerator

# Load model + adapter
model, tokenizer = load_with_adapter(
    base_model_path="checkpoints/qwen3-8b-base",
    adapter_path="adapters/kaleemcoder-sft",
)

# Generate code
gen = KaleemCoderGenerator(model, tokenizer)
response = gen.generate("Write a Python binary search function with tests.")
print(response)
```

**Agent mode:**

```python
from src.agent.agent import KaleemCoderAgent

agent = KaleemCoderAgent(generator=gen)
result = agent.run("Find the bug in this function and fix it: def fib(n): return fib(n-1) + fib(n-2)")
print(result)
```

---

## 🧪 Running Tests

```bash
make test           # pytest with coverage report
make test-fast      # pytest without coverage (quicker)
make lint           # ruff lint check
make format         # black + isort formatting
make check          # full CI check: lint + format + tests
```

---

## 🔮 Future Work

- [ ] Scale to Qwen3-14B / 72B
- [ ] Synthetic data generation (self-instruct / OSS-Instruct)
- [ ] Full repo-level agent (multi-file reasoning, PR creation)
- [ ] Hugging Face Spaces demo
- [ ] RLHF pipeline (PPO / GRPO)
- [ ] Leaderboard submissions (HumanEval, SWE-bench)
- [ ] CI/CD for auto-eval after each training run

---

## 🤝 Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) first.

```bash
make setup-dev      # install dev tools
make install-hooks  # set up pre-commit
make check          # make sure all checks pass before opening a PR
```

---

## 📄 License

MIT © [iamkaleemsajjad-hue](https://github.com/iamkaleemsajjad-hue)

---

<div align="center">

*Built with 🔥 on Kaggle · Powered by Qwen3-8B · Trained with LLaMA Factory*

</div>
