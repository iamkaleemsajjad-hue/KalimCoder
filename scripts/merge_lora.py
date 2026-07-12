"""
merge_lora.py — Merge LoRA adapter weights into the base model.

Usage:
    python scripts/merge_lora.py \
        --base_model checkpoints/qwen3-8b-base \
        --adapter_path adapters/kaleemcoder-sft \
        --output_dir adapters/kaleemcoder-sft-merged
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA adapter path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for merged model")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model → {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
