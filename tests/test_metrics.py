"""
tests/test_metrics.py — Unit tests for src/evaluation/metrics.py
"""

from src.evaluation.metrics import exact_match, pass_at_k, batch_evaluate, contains_code_block


def test_exact_match_identical():
    assert exact_match("def foo(): pass", "def foo(): pass") == 1.0


def test_exact_match_different():
    assert exact_match("def foo(): pass", "def bar(): pass") == 0.0


def test_exact_match_strips_whitespace():
    assert exact_match("  hello  ", "hello") == 1.0


def test_pass_at_k_all_correct():
    # If all n samples are correct, pass@k should be 1.0
    assert pass_at_k(n=10, c=10, k=1) == 1.0


def test_pass_at_k_none_correct():
    assert pass_at_k(n=10, c=0, k=1) == 0.0


def test_pass_at_k_partial():
    score = pass_at_k(n=10, c=5, k=1)
    assert 0.0 < score < 1.0


def test_contains_code_block_true():
    text = "Here is code:\n```python\nprint('hi')\n```"
    assert contains_code_block(text)


def test_contains_code_block_false():
    assert not contains_code_block("just plain text")


def test_batch_evaluate():
    preds = ["hello", "world"]
    refs  = ["hello", "earth"]
    result = batch_evaluate(preds, refs, exact_match)
    assert result["n"] == 2
    assert result["mean"] == 0.5
