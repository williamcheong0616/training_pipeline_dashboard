from __future__ import annotations

from datasets import Dataset
from trl import DPOTrainer, DPOConfig

from backend.core.model.loader import load_model, load_tokenizer
from backend.core.model.patcher import patch_model
from backend.core.model.adapter import apply_lora
from .base_trainer import BasePipelineTrainer


class DPOPipelineTrainer(BasePipelineTrainer):

    def train(self) -> None:
        cfg = self.config
        model_path = cfg["model_path"]
        peft_method = cfg.get("peft_method", "lora")
        output_dir = cfg.get("output_dir", f"outputs/job_{self.job_id}")

        tokenizer = load_tokenizer(model_path)
        model = load_model(model_path, quantization=cfg.get("quantization"), device_map=self._device_map())
        model = patch_model(model)

        if peft_method in ("lora", "qlora", "dora"):
            model = apply_lora(
                model,
                r=cfg.get("lora_r", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                use_dora=(peft_method == "dora"),
            )

        # DPO dataset must have: prompt, chosen, rejected
        import json
        from pathlib import Path
        raw = json.loads(Path(cfg["dataset_path"]).read_text())
        dataset = Dataset.from_list(raw)

        dpo_config = DPOConfig(
            **vars(self._training_args(output_dir)),
            beta=cfg.get("beta", 0.1),
            loss_type=cfg.get("loss_type", "sigmoid"),
            max_length=cfg.get("max_seq_length", 2048),
        )

        callbacks = [self.callback] if self.callback else []
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        trainer.train()
        trainer.save_model(output_dir)
