from __future__ import annotations
import abc
import gc
from datetime import datetime
from typing import Any, Dict

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from backend.utils.time import now_utc


class MetricLoggingCallback(TrainerCallback):
    """Writes training metrics to the DB after each logging step."""

    def __init__(self, job_id: int, db_session_factory):
        self.job_id = job_id
        self.db_session_factory = db_session_factory

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        from backend.db.models import TrainingMetric
        db = self.db_session_factory()
        try:
            metric = TrainingMetric(
                job_id=self.job_id,
                step=state.global_step,
                epoch=state.epoch,
                loss=logs.get("loss") or logs.get("train_loss"),
                eval_loss=logs.get("eval_loss"),
                learning_rate=logs.get("learning_rate"),
                reward=logs.get("reward") or logs.get("rewards/chosen"),
                grad_norm=logs.get("grad_norm"),
                timestamp=now_utc(),
            )
            db.add(metric)
            db.commit()
        finally:
            db.close()


class BasePipelineTrainer(abc.ABC):
    """Abstract base for all training method wrappers."""

    def __init__(self, job_id: int, config: Dict[str, Any], db_session_factory=None):
        self.job_id = job_id
        self.config = config
        self.db_session_factory = db_session_factory
        self.callback = MetricLoggingCallback(job_id, db_session_factory) if db_session_factory else None

    @abc.abstractmethod
    def train(self) -> None: ...

    def offload_model(self, model) -> None:
        """Delete model and free GPU memory after training."""
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _device_map(self) -> str:
        gpu_id = self.config.get("gpu_id", "auto")
        if gpu_id and str(gpu_id) != "auto":
            try:
                return f"cuda:{int(gpu_id)}"
            except (ValueError, TypeError):
                pass
        return "auto"

    def _prepare_model(self, model, gradient_checkpointing: bool = True):
        """Apply gradient checkpointing correctly for both normal and quantized models.

        Quantized models (4bit/8bit) must use peft's prepare_model_for_kbit_training
        instead of the plain gradient_checkpointing_enable() path, or gradients
        won't flow through the frozen quantized layers.
        """
        if self.config.get("quantization") in ("4bit", "8bit"):
            from peft import prepare_model_for_kbit_training
            return prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=gradient_checkpointing,
            )
        from backend.core.model.patcher import patch_model
        return patch_model(model, gradient_checkpointing=gradient_checkpointing)

    def _training_args(self, output_dir: str, **overrides) -> Dict[str, Any]:
        """Return a plain dict of constructor kwargs for any XxxConfig/TrainingArguments.

        Returns a dict (not a TrainingArguments instance) so no computed attributes
        like mixed_precision can leak into trl Config constructors that don't accept them.
        All values are explicitly cast to the expected type so string values from the
        frontend form don't cause TypeErrors.
        """
        import torch
        cfg = self.config

        # Auto-select bf16/fp16: prefer bf16 on supporting hardware, never set both.
        bf16 = bool(cfg.get("bf16", True)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16 = bool(cfg.get("fp16", False)) and not bf16

        args: Dict[str, Any] = dict(
            output_dir=output_dir,
            num_train_epochs=int(cfg.get("num_epochs", 3)),
            per_device_train_batch_size=int(cfg.get("batch_size", 4)),
            per_device_eval_batch_size=int(cfg.get("eval_batch_size", 4)),
            gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
            lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_strategy="epoch",
            max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
            seed=int(cfg.get("seed", 42)),
            fp16=fp16,
            bf16=bf16,
            report_to="none",
            dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
        )
        args.update(overrides)
        return args
