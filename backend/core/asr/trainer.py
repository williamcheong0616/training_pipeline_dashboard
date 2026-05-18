from __future__ import annotations
import warnings
from typing import Any, Dict

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from backend.core.trainer.base_trainer import BasePipelineTrainer
from backend.core.asr.loader import load_whisper_model, load_whisper_processor
from backend.core.asr.dataset import build_asr_dataset
from backend.core.asr.collator import DataCollatorSpeechSeq2SeqWithPadding
from backend.core.asr.metrics import make_compute_metrics

warnings.filterwarnings("ignore", message=".*`use_cache = True` is incompatible.*")


class ASRPipelineTrainer(BasePipelineTrainer):

    def train(self) -> None:
        cfg = self.config
        model_path = cfg["model_path"]
        training_method = cfg.get("training_method", cfg.get("peft_method", "lora"))
        output_dir = cfg.get("output_dir", f"outputs/asr_job_{self.job_id}")
        use_max_steps = cfg.get("use_max_steps", True)

        # Normalise language: "auto" or missing → None (multilingual/code-mixed mode)
        language = cfg.get("language") or None
        if language == "auto":
            language = None
        task = cfg.get("task", "transcribe")

        # ── Processor & dataset ──────────────────────────────────────────────
        processor = load_whisper_processor(model_path, language=language, task=task)
        encoded = build_asr_dataset(
            train_csv=cfg["train_csv"],
            processor=processor,
            val_csv=cfg.get("val_csv"),
            val_split=cfg.get("val_split", 0.1),
            sample_rate=cfg.get("sample_rate", 16000),
            audio_col=cfg.get("audio_col", "audio_path"),
            text_col=cfg.get("text_col", "text"),
            language=language,
        )

        # ── Model ────────────────────────────────────────────────────────────
        model = load_whisper_model(
            model_path,
            quantization=cfg.get("quantization"),
            language=language,
            task=task,
            processor=processor,
        )

        if training_method in ("lora", "qlora"):
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                inference_mode=False,
                r=cfg.get("lora_r", 32),
                lora_alpha=cfg.get("lora_alpha", 64),
                lora_dropout=cfg.get("lora_dropout", 0.1),
                target_modules=cfg.get("target_modules", ["q_proj", "v_proj"]),
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        # ── Training args ────────────────────────────────────────────────────
        steps_kwargs: Dict[str, Any] = {}
        if use_max_steps:
            steps_kwargs["max_steps"] = cfg.get("max_steps", 3000)
            steps_kwargs["warmup_steps"] = cfg.get("warmup_steps", 500)
        else:
            steps_kwargs["num_train_epochs"] = cfg.get("num_epochs", 3)
            steps_kwargs["warmup_ratio"] = cfg.get("warmup_ratio", 0.05)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=cfg.get("batch_size", 2),
            per_device_eval_batch_size=cfg.get("eval_batch_size", 1),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
            learning_rate=cfg.get("learning_rate", 1e-4),
            fp16=cfg.get("fp16", True),
            bf16=cfg.get("bf16", False),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
            eval_strategy="steps",
            eval_steps=cfg.get("eval_steps", 500),
            save_steps=cfg.get("save_steps", 500),
            save_total_limit=cfg.get("save_total_limit", 2),
            load_best_model_at_end=cfg.get("load_best_model_at_end", True),
            predict_with_generate=cfg.get("predict_with_generate", True),
            generation_max_length=cfg.get("generation_max_length", 225),
            logging_steps=cfg.get("logging_steps", 50),
            report_to="none",
            metric_for_best_model="wer",
            greater_is_better=False,
            remove_unused_columns=False,
            label_names=["labels"],
            **steps_kwargs,
        )

        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        compute_metrics = make_compute_metrics(processor)
        callbacks = [self.callback] if self.callback else []

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=encoded["train"],
            eval_dataset=encoded["validation"],
            data_collator=collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            callbacks=callbacks,
        )

        trainer.train()

        # SFT saves full model weights; LoRA/QLoRA saves adapter only
        final_dir = f"{output_dir}/final_model" if training_method == "sft" else f"{output_dir}/final_adapter"
        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
