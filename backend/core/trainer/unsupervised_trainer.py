"""Unsupervised / Continued Pre-Training (CPT) trainer.

Trains on raw text using causal language modeling loss (next-token prediction)
without any instruction template — suitable for domain adaptation.
"""
from __future__ import annotations

from transformers import Trainer, TrainingArguments

from backend.core.model.loader import load_model, load_tokenizer
from backend.core.model.patcher import patch_model
from backend.core.model.adapter import apply_lora
from backend.core.data.dataset import build_plain_text_dataset
from backend.core.data.collator import get_clm_collator
from .base_trainer import BasePipelineTrainer


class UnsupervisedPipelineTrainer(BasePipelineTrainer):

    def train(self) -> None:
        cfg = self.config
        model_path = cfg["model_path"]
        peft_method = cfg.get("peft_method", "lora")
        output_dir = cfg.get("output_dir", f"outputs/job_{self.job_id}")

        tokenizer = load_tokenizer(model_path)
        model = load_model(
            model_path,
            quantization=cfg.get("quantization"),
            use_flash_attention=cfg.get("use_flash_attention", False),
            device_map=self._device_map(),
        )
        model = patch_model(model, gradient_checkpointing=True)

        if peft_method in ("lora", "qlora", "dora"):
            model = apply_lora(
                model,
                r=cfg.get("lora_r", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                target_modules=cfg.get("target_modules"),
                lora_dropout=cfg.get("lora_dropout", 0.05),
                use_dora=(peft_method == "dora"),
            )

        dataset = build_plain_text_dataset(
            path_or_repo=cfg["dataset_path"],
            tokenizer=tokenizer,
            max_length=cfg.get("max_seq_length", 2048),
        )
        collator = get_clm_collator(tokenizer)

        callbacks = [self.callback] if self.callback else []
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**self._training_args(output_dir)),
            train_dataset=dataset,
            data_collator=collator,
            callbacks=callbacks,
        )
        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
