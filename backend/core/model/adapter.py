from __future__ import annotations
from typing import List, Optional
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from transformers import PreTrainedModel


def apply_lora(
    model: PreTrainedModel,
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    lora_dropout: float = 0.05,
    use_dora: bool = False,
) -> PreTrainedModel:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules or ["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        use_dora=use_dora,
    )
    return get_peft_model(model, config)


def merge_and_save(model: PreTrainedModel, save_path: str) -> None:
    """Merge LoRA weights into base model and save."""
    merged = model.merge_and_unload()
    merged.save_pretrained(save_path)
