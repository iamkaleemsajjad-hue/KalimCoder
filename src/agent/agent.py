"""
agent.py — KaleemCoder coding agent with tool use.

Tools available:
  - run_code(code: str) → str        Execute Python in a sandbox
  - read_file(path: str) → str       Read a file from disk
  - write_file(path, content) → None Write content to a file
  - search_web(query: str) → str     (stub) Web search
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class Tool:
    def __init__(self, name: str, description: str, fn):
        self.name = name
        self.description = description
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def run_code(code: str, timeout: int = 10) -> str:
    """Execute Python code in a subprocess and return stdout/stderr."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out."
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    Path(path).write_text(content, encoding="utf-8")
    return f"Written to {path}"


TOOLS = [
    Tool("run_code", "Execute Python code and return the output", run_code),
    Tool("read_file", "Read a file from disk", read_file),
    Tool("write_file", "Write content to a file", write_file),
]


class KaleemCoderAgent:
    """Simple ReAct-style coding agent backed by KaleemCoder."""

    def __init__(self, generator):
        self.generator = generator
        self.tools = {t.name: t for t in TOOLS}

    def run(self, task: str, max_steps: int = 5) -> str:
        """Run the agent on a task and return the final answer."""
        history = []
        for step in range(max_steps):
            prompt = self._build_prompt(task, history)
            response = self.generator.generate(prompt)
            history.append({"role": "assistant", "content": response})

            if "<FINAL_ANSWER>" in response:
                return response.split("<FINAL_ANSWER>")[-1].strip()

        return history[-1]["content"] if history else ""

    def _build_prompt(self, task: str, history: list[dict]) -> str:
        tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in TOOLS)
        base = f"Task: {task}\n\nAvailable tools:\n{tool_descs}\n\nThink step by step."
        for h in history:
            base += f"\n\n{h['role'].capitalize()}: {h['content']}"
        return base
