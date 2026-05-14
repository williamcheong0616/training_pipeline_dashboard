from __future__ import annotations
import os
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def _get_bnb_config(quantization: str) -> Optional[BitsAndBytesConfig]:
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model(
    model_name_or_path: str,
    quantization: Optional[str] = None,  # "4bit" | "8bit" | None
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_flash_attention: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    bnb_config = _get_bnb_config(quantization)
    dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    kwargs: dict = {
        "pretrained_model_name_or_path": model_name_or_path,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
    }
    if bnb_config:
        kwargs["quantization_config"] = bnb_config
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(**kwargs)
    model.config.use_cache = False
    return model


def load_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    padding_side: str = "right",
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
