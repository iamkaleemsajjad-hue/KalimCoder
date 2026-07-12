"""
evaluate.py — Evaluate KaleemCoder on coding prompts.

Usage:
    python scripts/evaluate.py --model_path adapters/kaleemcoder-sft \
                               --eval_data datasets/evaluation/eval_prompts.jsonl \
                               --output_file eval_results.json
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on coding prompts")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_data", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def load_eval_data(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def main():
    args = parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    examples = load_eval_data(args.eval_data)
    results = []

    for ex in tqdm(examples, desc="Evaluating"):
        prompt = ex["prompt"]
        expected = ex.get("expected", "")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        results.append({"prompt": prompt, "expected": expected, "generated": response})

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
