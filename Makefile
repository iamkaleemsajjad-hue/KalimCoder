# KaleemCoder — Makefile
# Run `make help` to see all available commands.

.DEFAULT_GOAL := help
PYTHON        := python
PIP           := pip

# ── Setup ────────────────────────────────────────────────────────────────────

.PHONY: setup
setup: ## Install all dependencies (train + eval + dev extras)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[train,eval,dev]"

.PHONY: setup-dev
setup-dev: ## Install dev tools only (no heavy ML packages)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

.PHONY: install-hooks
install-hooks: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

# ── Data ─────────────────────────────────────────────────────────────────────

.PHONY: download-model
download-model: ## Download Qwen3-8B base model from HF Hub
	$(PYTHON) scripts/download_model.py \
		--model_id Qwen/Qwen3-8B \
		--output_dir checkpoints/qwen3-8b-base

.PHONY: prepare-data
prepare-data: ## Download and format CodeAlpaca dataset
	$(PYTHON) scripts/prepare_dataset.py \
		--dataset codealpaca \
		--output_dir datasets/instruction

.PHONY: download-data
download-data: ## Download all enabled datasets from configs/datasets.yaml
	$(PYTHON) scripts/download_datasets.py

.PHONY: download-data-force
download-data-force: ## Re-download all datasets even if they already exist
	$(PYTHON) scripts/download_datasets.py --force

.PHONY: validate-data
validate-data: ## Validate all raw datasets and write reports to datasets/evaluation/
	$(PYTHON) scripts/validate_dataset.py

.PHONY: validate-data-sample
validate-data-sample: ## Validate using a 10 000-row sample (fast preview)
	$(PYTHON) scripts/validate_dataset.py --sample 10000

.PHONY: clean-data
clean-data: ## Apply cleaning pipeline to all raw datasets
	$(PYTHON) scripts/clean_dataset.py

.PHONY: clean-data-force
clean-data-force: ## Force re-clean all datasets even if cleaned output exists
	$(PYTHON) scripts/clean_dataset.py --force






# ── Training ─────────────────────────────────────────────────────────────────

.PHONY: train-sft
train-sft: ## Run SFT training with QLoRA
	$(PYTHON) scripts/train.py --config configs/training/sft_qlora.yaml

.PHONY: train-dpo
train-dpo: ## Run DPO alignment training
	$(PYTHON) scripts/train.py --config configs/training/dpo.yaml

.PHONY: merge-lora
merge-lora: ## Merge LoRA adapter into base model
	$(PYTHON) scripts/merge_lora.py \
		--base_model checkpoints/qwen3-8b-base \
		--adapter_path adapters/kaleemcoder-sft \
		--output_dir adapters/kaleemcoder-sft-merged

# ── Evaluation ───────────────────────────────────────────────────────────────

.PHONY: eval
eval: ## Run evaluation on test prompts
	$(PYTHON) scripts/evaluate.py \
		--model_path adapters/kaleemcoder-sft-merged \
		--eval_data datasets/evaluation/eval_prompts.jsonl \
		--output_file logs/evaluation/results.json

# ── Quality ──────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run unit tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

.PHONY: test-fast
test-fast: ## Run tests without coverage (fast)
	pytest tests/ -v --tb=short

.PHONY: lint
lint: ## Run ruff linter
	ruff check src/ scripts/ tests/

.PHONY: lint-fix
lint-fix: ## Auto-fix ruff lint issues
	ruff check --fix src/ scripts/ tests/

.PHONY: format
format: ## Format code with black + isort
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

.PHONY: format-check
format-check: ## Check formatting without modifying files
	black --check src/ scripts/ tests/
	isort --check-only src/ scripts/ tests/

.PHONY: pre-commit
pre-commit: ## Run all pre-commit hooks on all files
	pre-commit run --all-files

.PHONY: check
check: lint format-check test ## Run lint + format check + tests (full CI locally)

# ── Utilities ────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove Python cache files and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true

.PHONY: gpu-info
gpu-info: ## Show GPU information
	$(PYTHON) -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "KaleemCoder — Available Commands"
	@echo "================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
