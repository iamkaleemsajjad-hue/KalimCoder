---
language:
  - en
license: mit
library_name: transformers
base_model: Qwen/Qwen3-8B
tags:
  - code
  - coding
  - qwen
  - qlora
  - fine-tuned
  - software-engineering
datasets:
  - sahil2801/CodeAlpaca-20k
pipeline_tag: text-generation
---

# KaleemCoder-Qwen3-8B-v1

> **Status**: 🚧 In training — this card will be updated when the model is released.

A coding-focused LLM fine-tuned from **Qwen3-8B** using QLoRA + DPO on curated software engineering datasets.

## Model Details

| Property | Value |
|---|---|
| **Base model** | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| **Fine-tuning** | QLoRA (4-bit NF4, rank 64, all linear layers) |
| **Alignment** | DPO (Direct Preference Optimization) |
| **Training hardware** | Kaggle Tesla T4 × 2 |
| **Context length** | 4,096 tokens |
| **Language** | Python · C++ · JavaScript · (more) |

## Intended Use

KaleemCoder is designed for software engineering tasks:

- ✅ Code generation from natural language
- ✅ Bug detection and fixing
- ✅ Code explanation and documentation
- ✅ Test case generation
- ✅ Pull request review assistance
- ✅ Repository-level reasoning

## Training Details

### Datasets

| Dataset | Split | Examples | Purpose |
|---|---|---|---|
| CodeAlpaca-20k | train | 20,000 | SFT baseline |
| Custom curated | train | TBD | Domain-specific coding |
| DPO pairs | train | TBD | Alignment |

### Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Learning rate | 2e-4 |
| Batch size (effective) | 32 |
| Epochs | 3 |
| Quantization | 4-bit NF4 |
| Precision | bfloat16 |

See full config: [`configs/training/sft_qlora.yaml`](../../configs/training/sft_qlora.yaml)

## Evaluation Results

> Results pending — will be filled after training.

| Benchmark | Score | Date |
|---|---|---|
| HumanEval (pass@1) | TBD | — |
| MBPP (pass@1) | TBD | — |
| LiveCodeBench | TBD | — |
| SWE-bench Lite | TBD | — |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "iamkaleemsajjad-hue/KaleemCoder-Qwen3-8B-v1"  # HF Hub path (when released)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are KaleemCoder, an expert software engineering AI assistant."},
    {"role": "user",   "content": "Write a Python function to merge two sorted lists."},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Limitations

- May hallucinate APIs or function signatures for uncommon libraries
- Context window limited to 4,096 tokens (may struggle with very large files)
- DPO alignment is an approximation — always review generated code before use
- Not intended for safety-critical applications

## Citation

```bibtex
@misc{kaleemcoder2026,
  author    = {iamkaleemsajjad-hue},
  title     = {KaleemCoder: A Coding-Focused LLM Fine-tuned from Qwen3-8B},
  year      = {2026},
  url       = {https://github.com/iamkaleemsajjad-hue/KalimCoder},
}
```

## License

MIT — see [LICENSE](../../LICENSE)
