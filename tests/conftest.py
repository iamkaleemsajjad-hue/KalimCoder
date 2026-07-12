"""
tests/conftest.py — Shared pytest fixtures for KaleemCoder test suite.
"""

import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_jsonl(tmp_path) -> Path:
    """Write a small JSONL file and return its path."""
    records = [
        {"prompt": f"prompt_{i}", "response": f"response_{i}" * 20}
        for i in range(10)
    ]
    p = tmp_path / "sample.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records))
    return p


@pytest.fixture
def sample_codealpaca_record() -> dict:
    return {
        "instruction": "Write a Python function to reverse a string.",
        "input": "",
        "output": "def reverse_string(s: str) -> str:\n    return s[::-1]",
    }
