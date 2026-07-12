# Contributing to KaleemCoder

Thank you for your interest in contributing! This document explains how to get started.

---

## Development Setup

```bash
git clone https://github.com/iamkaleemsajjad-hue/KalimCoder.git
cd KalimCoder

# Install dev dependencies (no heavy ML packages needed for dev)
make setup-dev

# Install pre-commit hooks (runs black, isort, ruff on every commit)
make install-hooks
```

---

## Code Style

We use three tools, all configured in `pyproject.toml`:

| Tool | Purpose | Command |
|---|---|---|
| **black** | Formatting | `make format` |
| **isort** | Import ordering | `make format` |
| **ruff** | Linting | `make lint` |

Run `make check` to run all checks at once (same as CI).

Pre-commit hooks run automatically on `git commit` — you only need to set them up once with `make install-hooks`.

---

## Running Tests

```bash
make test          # with coverage report
make test-fast     # without coverage (quicker)
```

All tests live in `tests/`. Add tests alongside any new `src/` code you write.

---

## Experiment Workflow

1. Create a new experiment directory under `experiments/`:
   ```
   experiments/007_your_experiment/
   ├── config.yaml
   ├── metrics.json
   ├── notes.md
   ├── checkpoint_link.txt
   └── plots/
   ```
2. Copy and adapt the template from `experiments/001_qwen_base/`.
3. Record your results in `metrics.json` and observations in `notes.md`.

---

## Pull Request Process

1. Fork the repository and create a branch: `git checkout -b feat/my-feature`
2. Make your changes with tests
3. Run `make check` — all checks must pass
4. Open a PR against `main` with a clear description of the change
5. Link relevant experiment or issue numbers

---

## Reporting Issues

Use the GitHub issue templates:
- 🐛 **Bug Report** — something is broken
- ✨ **Feature Request** — something you'd like to see
