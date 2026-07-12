"""
prepare_dataset.py — Download and format coding datasets for SFT training.

Usage:
    python scripts/prepare_dataset.py --dataset codealpaca --output_dir datasets/processed
"""

import argparse
import json
import os
from datasets import load_dataset


SUPPORTED_DATASETS = {
    "codealpaca": "sahil2801/CodeAlpaca-20k",
}


def format_codealpaca(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    prompt = f"{instruction}\n\nInput:\n{input_text}" if input_text else instruction
    return {"prompt": prompt, "response": output}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for SFT training")
    parser.add_argument("--dataset", type=str, default="codealpaca", choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--output_dir", type=str, default="datasets/processed")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    hf_name = SUPPORTED_DATASETS[args.dataset]
    print(f"Loading: {hf_name}")
    ds = load_dataset(hf_name, split="train")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    if args.dataset == "codealpaca":
        ds = ds.map(format_codealpaca, remove_columns=ds.column_names)

    out_path = os.path.join(args.output_dir, f"{args.dataset}_formatted.jsonl")
    ds.to_json(out_path)
    print(f"Saved {len(ds)} examples → {out_path}")


if __name__ == "__main__":
    main()
