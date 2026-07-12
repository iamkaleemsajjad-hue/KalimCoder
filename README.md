# KaleemCoder 🤖

> A coding-focused LLM fine-tuned from **Qwen3-8B** using QLoRA + DPO.  
> Built to understand repositories, fix bugs, review pull requests, generate tests, and act as a software engineering assistant.

---

## 🎯 Project Goal

Train a state-of-the-art coding LLM from scratch (fine-tuning) that can:

| Capability | Status |
|---|---|
| Code generation (Python, C++, JS, …) | 🔄 In progress |
| Bug detection & fixing | 🔄 In progress |
| Pull request review | 📋 Planned |
| Code explanation | 📋 Planned |
| Test generation | 📋 Planned |
| Repository-level reasoning | 📋 Planned |
| Agentic tool use | 📋 Planned |

---

## 🏗️ Architecture

```
Qwen3-8B (Base)
      │
      ▼
QLoRA SFT (Supervised Fine-Tuning)
      │
      ▼
DPO Alignment (Direct Preference Optimization)
      │
      ▼
KaleemCoder ✨
      │
      ▼
Agent Layer (ReAct loop + tool use)
```

| Property | Value |
|---|---|
| **Base Model** | Qwen3-8B |
| **Training** | QLoRA (4-bit NF4) + LLaMA Factory |
| **Alignment** | DPO via TRL |
| **Hardware** | Kaggle Tesla T4 × 2 |
| **Context** | 4,096 tokens |

---

## 📍 Roadmap

- [x] Repository structure & tooling
- [ ] Download & verify Qwen3-8B base model
- [ ] Prepare coding datasets (Python, C++, bug-fix pairs)
- [ ] SFT training run #1 — CodeAlpaca
- [ ] Evaluate on HumanEval & MBPP
- [ ] DPO alignment pass
- [ ] Agentic loop with code execution
- [ ] Push to Hugging Face Hub

See [`docs/roadmap.md`](docs/roadmap.md) for the full roadmap.

---

## 🚀 Installation

```bash
git clone https://github.com/iamkaleemsajjad-hue/KalimCoder.git
cd KalimCoder
pip install -r requirements.txt
```

> **Hardware note**: Training requires a CUDA-capable GPU.  
> Inference can run on CPU with quantization (slow).

---

## 📦 Repository Structure

```
KalimCoder/
│
├── src/                    ← Core Python package
│   ├── data/               ← Dataset loading, cleaning, formatting
│   ├── models/             ← Model loading + QLoRA helpers
│   ├── training/           ← Trainer callbacks & utilities
│   ├── evaluation/         ← Metric computation
│   ├── inference/          ← Generation pipeline
│   ├── utils/              ← Shared logging, I/O helpers
│   └── agent/              ← Agentic loop + tool use
│
├── configs/                ← All YAML configs
│   ├── model/              ← Model architecture configs
│   ├── training/           ← SFT & DPO training configs
│   ├── dataset/            ← Dataset preprocessing configs
│   ├── evaluation/         ← Benchmark eval configs
│   └── agent/              ← Agent configs
│
├── datasets/               ← Training data (gitignored for large files)
│   ├── raw/                ← Raw downloaded data
│   ├── cleaned/            ← Filtered & deduplicated
│   ├── instruction/        ← SFT instruction-response pairs
│   ├── preference/         ← DPO chosen/rejected pairs
│   ├── evaluation/         ← Eval & benchmark prompts
│   └── synthetic/          ← Synthetically generated examples
│
├── experiments/            ← One folder per experiment
│   ├── 001_qwen_base/      ← config, metrics, notes, plots
│   └── …
│
├── benchmarks/             ← Benchmark runners & results
│   ├── HumanEval/
│   ├── MBPP/
│   ├── LiveCodeBench/
│   ├── SWE-bench/
│   ├── RepoBench/
│   └── BigCodeBench/
│
├── scripts/                ← CLI entry points
├── notebooks/              ← Step-by-step Jupyter notebooks
├── docs/                   ← Architecture, roadmap, experiment notes
├── tests/                  ← Unit tests (pytest)
├── logs/                   ← Training & evaluation logs (gitignored)
├── checkpoints/            ← Base model weights (gitignored)
├── adapters/               ← LoRA adapters (gitignored)
└── .github/                ← CI workflows & issue templates
```

---

## 🏋️ Training

### 1. Download the base model

```bash
python scripts/download_model.py \
  --model_id Qwen/Qwen3-8B \
  --output_dir checkpoints/qwen3-8b-base
```

### 2. Prepare datasets

```bash
python scripts/prepare_dataset.py \
  --dataset codealpaca \
  --output_dir datasets/instruction
```

### 3. Run SFT

```bash
python scripts/train.py --config configs/training/sft_qlora.yaml
```

### 4. Run DPO

```bash
python scripts/train.py --config configs/training/dpo.yaml
```

### 5. Merge LoRA adapter

```bash
python scripts/merge_lora.py \
  --base_model checkpoints/qwen3-8b-base \
  --adapter_path adapters/kaleemcoder-sft \
  --output_dir adapters/kaleemcoder-sft-merged
```

---

## 📊 Evaluation

```bash
python scripts/evaluate.py \
  --model_path adapters/kaleemcoder-sft-merged \
  --eval_data datasets/evaluation/eval_prompts.jsonl \
  --output_file logs/evaluation/results.json
```

---

## 📈 Benchmarks & Results

> Results will be updated after each training run.

| Benchmark | Qwen3-8B Base | KaleemCoder SFT | KaleemCoder DPO |
|---|---|---|---|
| HumanEval (pass@1) | TBD | TBD | TBD |
| MBPP (pass@1) | TBD | TBD | TBD |
| LiveCodeBench | TBD | TBD | TBD |
| SWE-bench Lite | TBD | TBD | TBD |

---

## 💡 Examples

```python
from src.models.loader import load_with_adapter
from src.inference.generator import KaleemCoderGenerator

model, tokenizer = load_with_adapter(
    base_model_path="checkpoints/qwen3-8b-base",
    adapter_path="adapters/kaleemcoder-sft",
)
gen = KaleemCoderGenerator(model, tokenizer)
print(gen.generate("Write a Python binary search function."))
```

---

## 🔬 Running Tests

```bash
pytest tests/ -v
```

---

## 🔮 Future Work

- [ ] Scale to Qwen3-14B / 72B with larger hardware
- [ ] Synthetic data generation (self-instruct)
- [ ] Full repository-level agent (multi-file reasoning)
- [ ] Hugging Face Spaces demo
- [ ] RLHF pipeline (PPO)
- [ ] Leaderboard submission (HumanEval, SWE-bench)

---

## 📄 License

MIT © [iamkaleemsajjad-hue](https://github.com/iamkaleemsajjad-hue)
