"""
loader.py — Load base models with optional quantization and LoRA adapters.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def get_bnb_config(bits: int = 4) -> BitsAndBytesConfig:
    """Return a BitsAndBytesConfig for QLoRA."""
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported bits: {bits}")


def load_base_model(
    model_path: str,
    quantization_bits: int | None = 4,
    device_map: str = "auto",
) -> tuple:
    """Load a causal LM and its tokenizer, optionally with quantization."""
    kwargs = {"device_map": device_map, "trust_remote_code": True}
    if quantization_bits:
        kwargs["quantization_config"] = get_bnb_config(quantization_bits)
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    return model, tokenizer


def load_with_adapter(
    base_model_path: str,
    adapter_path: str,
    quantization_bits: int | None = 4,
) -> tuple:
    """Load base model and attach a LoRA adapter."""
    model, tokenizer = load_base_model(base_model_path, quantization_bits)
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer
