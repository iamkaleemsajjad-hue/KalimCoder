"""
metrics.py — Compute evaluation metrics for generated code.
"""

from __future__ import annotations

import re
from typing import Callable


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if prediction == reference (stripped), else 0.0."""
    return float(prediction.strip() == reference.strip())


def contains_code_block(text: str) -> bool:
    """Check if the text contains a fenced code block."""
    return bool(re.search(r"```[\w]*\n[\s\S]+?```", text))


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimate pass@k from n total samples and c correct ones.
    Formula from Chen et al. (HumanEval paper).
    """
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)


def batch_evaluate(
    predictions: list[str],
    references: list[str],
    metric_fn: Callable[[str, str], float],
) -> dict:
    """Run a metric function over lists and return aggregate stats."""
    scores = [metric_fn(p, r) for p, r in zip(predictions, references)]
    return {
        "mean": sum(scores) / len(scores) if scores else 0.0,
        "scores": scores,
        "n": len(scores),
    }
