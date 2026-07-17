import os
import subprocess
from pathlib import Path

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, snapshot_download

# --------------------------------------------------
# Configuration
# --------------------------------------------------

HF_REPO = "kalim133173/KalimCoder"

CHECKPOINT_DIR = "/kaggle/working/KalimCoder/checkpoints/qwen3-lora"

CONFIG = "/kaggle/working/KalimCoder/configs/qwen3_sft.yaml"

LLAMA_FACTORY = "/kaggle/working/LLaMA-Factory"

# --------------------------------------------------
# Login
# --------------------------------------------------

print("=" * 60)
print("Logging into Hugging Face...")
print("=" * 60)

token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=token)

# --------------------------------------------------
# Download previous checkpoint if one exists
# --------------------------------------------------

print("=" * 60)
print("Downloading latest checkpoint (if available)...")
print("=" * 60)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

try:
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=CHECKPOINT_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Repository synchronized.")
except Exception as e:
    print("No previous checkpoint found.")
    print(e)

# --------------------------------------------------
# Show existing checkpoints
# --------------------------------------------------

print("=" * 60)
print("Local checkpoints")
print("=" * 60)

for p in sorted(Path(CHECKPOINT_DIR).glob("checkpoint-*")):
    print(p)

print("=" * 60)
print("Starting training...")
print("=" * 60)

os.chdir(LLAMA_FACTORY)

command = [
    "torchrun",
    "--nproc_per_node=2",
    "src/llamafactory/launcher.py",
    CONFIG,
]

subprocess.run(command, check=True)
