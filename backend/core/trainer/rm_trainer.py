from __future__ import annotations

from datasets import Dataset
from trl import RewardTrainer, RewardConfig

from backend.core.model.loader import load_tokenizer
from transformers import AutoModelForSequenceClassification
from backend.core.model.patcher import patch_model
from .base_trainer import BasePipelineTrainer


class RMPipelineTrainer(BasePipelineTrainer):

    def train(self) -> None:
        cfg = self.config
        model_path = cfg["model_path"]
        output_dir = cfg.get("output_dir", f"outputs/job_{self.job_id}")

        tokenizer = load_tokenizer(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1, trust_remote_code=True
        )
        model = patch_model(model)

        import json
        from pathlib import Path
        raw = json.loads(Path(cfg["dataset_path"]).read_text())
        dataset = Dataset.from_list(raw)

        reward_config = RewardConfig(
            **vars(self._training_args(output_dir)),
            max_length=cfg.get("max_seq_length", 2048),
        )

        callbacks = [self.callback] if self.callback else []
        trainer = RewardTrainer(
            model=model,
            args=reward_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        trainer.train()
        trainer.save_model(output_dir)
