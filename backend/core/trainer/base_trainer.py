from __future__ import annotations
import abc
from datetime import datetime
from typing import Any, Dict

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments  # TrainingArguments kept for type hints in on_log


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
                timestamp=datetime.utcnow(),
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

    def _device_map(self) -> str:
        gpu_id = self.config.get("gpu_id", "auto")
        if gpu_id and str(gpu_id) != "auto":
            try:
                return f"cuda:{int(gpu_id)}"
            except (ValueError, TypeError):
                pass
        return "auto"

    def _training_args(self, output_dir: str, **overrides) -> Dict[str, Any]:
        """Return a plain dict of constructor kwargs for any XxxConfig/TrainingArguments.

        Returning a dict (not a TrainingArguments instance) prevents computed
        attributes like `mixed_precision` from leaking into subclass Config
        constructors that don't accept them.
        """
        cfg = self.config
        args: Dict[str, Any] = dict(
            output_dir=output_dir,
            num_train_epochs=cfg.get("num_epochs", 3),
            per_device_train_batch_size=cfg.get("batch_size", 4),
            per_device_eval_batch_size=cfg.get("eval_batch_size", 4),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            learning_rate=cfg.get("learning_rate", 2e-4),
            warmup_ratio=cfg.get("warmup_ratio", 0.05),
            lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
            logging_steps=cfg.get("logging_steps", 10),
            save_strategy="epoch",
            fp16=cfg.get("fp16", False),
            bf16=cfg.get("bf16", True),
            report_to="none",
        )
        args.update(overrides)
        return args
