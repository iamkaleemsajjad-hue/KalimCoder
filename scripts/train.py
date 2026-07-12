"""
train.py — Launch SFT or DPO training via LLaMA Factory CLI.

Usage:
    python scripts/train.py --config configs/qwen3_sft.yaml
    python scripts/train.py --config configs/dpo.yaml
"""

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Launch training with LLaMA Factory")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--dry_run", action="store_true", help="Print command without running")
    return parser.parse_args()


def main():
    args = parse_args()

    cmd = ["llamafactory-cli", "train", args.config]
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        print("[dry-run] Skipping execution.")
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
