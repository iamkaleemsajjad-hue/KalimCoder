"""
callbacks.py — Custom Hugging Face Trainer callbacks for KaleemCoder training.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class MetricsLoggerCallback(TrainerCallback):
    """Log training metrics to a JSON file after each epoch."""

    def __init__(self, log_path: str = "logs/training/metrics.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step, "epoch": state.epoch, **logs, "timestamp": time.time()}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


class GPUMemoryCallback(TrainerCallback):
    """Log GPU memory usage at each logging step."""

    def __init__(self, log_path: str = "logs/gpu/memory.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        try:
            import torch
            if torch.cuda.is_available() and state.global_step % args.logging_steps == 0:
                mem = {f"gpu_{i}": torch.cuda.memory_reserved(i) / 1e9 for i in range(torch.cuda.device_count())}
                entry = {"step": state.global_step, **mem, "timestamp": time.time()}
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
