# Training Notes

## Run Log

### Run 001 — Baseline (Planned)
- **Date**: TBD
- **Config**: `configs/qwen3_sft.yaml`
- **Dataset**: CodeAlpaca (20k examples)
- **Hardware**: Kaggle T4 x2
- **Status**: Not started

---

## Hyperparameter Notes

### Learning Rate
- Start with `2e-4` for QLoRA SFT
- Reduce to `5e-5` for DPO

### Batch Size
- Per-device batch: 2
- Gradient accumulation: 8
- Effective batch size: 32

### LoRA
- Rank 64 with alpha 128 provides good capacity without overfitting
- Target all linear layers for coding tasks

## Known Issues / Gotchas

- bitsandbytes may require CUDA 11.8+ on Kaggle
- LLaMA Factory template must match model's chat format (`qwen`)
- Use `bf16=true` on T4; avoid `fp16` for stability

## Useful Commands

```bash
# Check GPU memory usage during training
nvidia-smi --loop=1

# Resume from checkpoint
llamafactory-cli train configs/qwen3_sft.yaml --resume_from_checkpoint checkpoints/run001/checkpoint-500
```
