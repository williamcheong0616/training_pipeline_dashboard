from __future__ import annotations

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from backend.core.model.loader import load_model, load_tokenizer
from backend.core.model.adapter import apply_lora
from backend.core.data.dataset import build_dataset
from .base_trainer import BasePipelineTrainer


class SFTPipelineTrainer(BasePipelineTrainer):

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

        model = self._prepare_model(model, gradient_checkpointing=True)

        if peft_method in ("lora", "qlora", "dora"):
            model = apply_lora(
                model,
                r=int(cfg.get("lora_r", 16)),
                lora_alpha=int(cfg.get("lora_alpha", 32)),
                target_modules=cfg.get("target_modules"),
                lora_dropout=float(cfg.get("lora_dropout", 0.05)),
                use_dora=(peft_method == "dora"),
            )

        dataset = build_dataset(
            path_or_repo=cfg["dataset_path"],
            format=cfg.get("dataset_format", "alpaca"),
            template_name=cfg.get("template", "alpaca"),
            tokenizer=tokenizer,
            max_length=int(cfg.get("max_seq_length", 2048)),
        )

        # Use plain transformers.Trainer instead of trl.SFTTrainer to avoid trl API
        # instability (dataset_text_field, processing_class, max_seq_length have all
        # moved between trl versions). DataCollatorForLanguageModeling(mlm=False)
        # derives labels = input_ids with padding positions set to -100.
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        self.offload_model(model)
