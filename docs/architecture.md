# KaleemCoder Architecture

## Overview

```
Qwen3-8B (Base)
     │
     ▼
QLoRA Fine-tuning (SFT)
     │
     ▼
DPO Alignment
     │
     ▼
KaleemCoder (Coding LLM)
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

## Data Pipeline

```
Raw Datasets → Cleaning → Formatting → JSONL → Training
     ↑
CodeAlpaca, GitHub Code, Custom Prompts
```
