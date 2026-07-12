"""
download_model.py — Download Qwen3-8B from Hugging Face Hub.

Usage:
    python scripts/download_model.py --model_id Qwen/Qwen3-8B --output_dir checkpoints/qwen3-8b-base
"""

import argparse
import os
from huggingface_hub import snapshot_download, login


def parse_args():
    parser = argparse.ArgumentParser(description="Download model from Hugging Face Hub")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B", help="HF model ID")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen3-8b-base", help="Local save path")
    parser.add_argument("--hf_token", type=str, default=None, help="HF token (or set HF_TOKEN env var)")
    return parser.parse_args()


def main():
    args = parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if token:
        login(token=token)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Downloading {args.model_id} → {args.output_dir}")

    snapshot_download(
        repo_id=args.model_id,
        local_dir=args.output_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
