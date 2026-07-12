# KaleemCoder Roadmap

## Phase 1 — Foundation (Current)
- [x] Repository setup
- [x] Environment configuration
- [ ] Download Qwen3-8B base model
- [ ] Verify base model inference on Kaggle T4 x2

## Phase 2 — Data
- [ ] Collect and clean coding datasets (CodeAlpaca, GitHub Code, etc.)
- [ ] Create custom KaleemCoder prompt-response pairs
- [ ] Build evaluation benchmark (50–100 hand-crafted coding problems)

## Phase 3 — SFT Training
- [ ] Configure QLoRA + LLaMA Factory
- [ ] First training run on Kaggle
- [ ] Evaluate on coding benchmarks
- [ ] Iterate on data and hyperparameters

## Phase 4 — Alignment
- [ ] Collect preference pairs (chosen vs. rejected)
- [ ] DPO training
- [ ] Re-evaluate

## Phase 5 — Agent
- [ ] Build agent loop with tool use
- [ ] Integrate with code execution sandbox
- [ ] Test on real repository tasks (bug fixes, PR reviews)

## Phase 6 — Release
- [ ] Merge LoRA → full model
- [ ] Push to Hugging Face Hub
- [ ] Write model card
- [ ] Demo notebook
