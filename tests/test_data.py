"""
tests/test_data.py — Unit tests for src/data utilities.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.cleaner import deduplicate, filter_by_length
from src.data.formatter import format_codealpaca, to_chatml
from src.data.loader import load_jsonl, iter_jsonl


# ─── loader ────────────────────────────────────────────────────────────────

def test_load_jsonl_basic(tmp_path):
    data = [{"prompt": "hello", "response": "world"}, {"prompt": "foo", "response": "bar"}]
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in data))
    result = load_jsonl(p)
    assert len(result) == 2
    assert result[0]["prompt"] == "hello"


def test_iter_jsonl_yields_dicts(tmp_path):
    data = [{"a": i} for i in range(5)]
    p = tmp_path / "iter.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in data))
    result = list(iter_jsonl(p))
    assert len(result) == 5


# ─── cleaner ───────────────────────────────────────────────────────────────

def test_deduplicate_removes_duplicates():
    records = [
        {"prompt": "same prompt", "response": "a"},
        {"prompt": "same prompt", "response": "b"},
        {"prompt": "different", "response": "c"},
    ]
    result = deduplicate(records, key="prompt")
    assert len(result) == 2


def test_filter_by_length():
    records = [
        {"response": "x" * 10},   # too short
        {"response": "x" * 200},  # ok
        {"response": "x" * 9000}, # too long
    ]
    result = filter_by_length(records, min_chars=50, max_chars=8000)
    assert len(result) == 1


# ─── formatter ─────────────────────────────────────────────────────────────

def test_format_codealpaca_no_input():
    rec = {"instruction": "Write a hello world", "input": "", "output": "print('hello')"}
    result = format_codealpaca(rec)
    assert result["prompt"] == "Write a hello world"
    assert "print" in result["response"]


def test_format_codealpaca_with_input():
    rec = {"instruction": "Fix this", "input": "def foo(): pass", "output": "def foo(): return 1"}
    result = format_codealpaca(rec)
    assert "Input:" in result["prompt"]


def test_to_chatml_structure():
    result = to_chatml("Write a sort", "def sort(arr): return sorted(arr)")
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][2]["role"] == "assistant"
