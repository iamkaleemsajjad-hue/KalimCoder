"""
formatter.py — Convert raw dataset records into instruction-response format.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are KaleemCoder, an expert software engineering AI assistant. "
    "You help with code generation, debugging, code review, and test writing. "
    "Always provide clean, well-commented, production-ready code."
)


def to_chatml(prompt: str, response: str, system: str = SYSTEM_PROMPT) -> dict:
    """Format as ChatML messages list (for chat models)."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def to_alpaca(instruction: str, input_text: str, output: str) -> dict:
    """Format as Alpaca-style instruction dict."""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def format_codealpaca(record: dict) -> dict:
    """Convert a CodeAlpaca record to prompt/response pair."""
    instruction = record.get("instruction", "")
    inp = record.get("input", "")
    output = record.get("output", "")
    prompt = f"{instruction}\n\nInput:\n{inp}" if inp else instruction
    return {"prompt": prompt, "response": output}
