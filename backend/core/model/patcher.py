"""Optional model patches applied before training."""
from __future__ import annotations
from transformers import PreTrainedModel


def patch_model(
    model: PreTrainedModel,
    gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    if gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    return model
