"""
Seed two completed demo jobs (one LLM, one ASR) with realistic metric curves.

Usage:
    python -m backend.scripts.seed_demo
"""
from __future__ import annotations
import math
import random
from datetime import datetime, timedelta

from backend.db.session import SessionLocal, init_db
from backend.db.models import Job, TrainingMetric

random.seed(42)


def _loss_curve(steps: int, start: float, end: float, noise: float = 0.04):
    """Exponential decay with small noise."""
    for i in range(steps):
        t = i / max(steps - 1, 1)
        val = start * math.exp(-3 * t) + end
        val += random.gauss(0, noise)
        yield max(val, 0.01)


def _lr_warmup_cosine(steps: int, peak: float, warmup: int):
    for i in range(steps):
        if i < warmup:
            yield peak * (i + 1) / warmup
        else:
            t = (i - warmup) / max(steps - warmup - 1, 1)
            yield peak * 0.5 * (1 + math.cos(math.pi * t))


def seed_llm(db) -> Job:
    started = datetime.utcnow() - timedelta(hours=3, minutes=22)
    finished = started + timedelta(hours=3, minutes=10)

    job = Job(
        name="Llama-3.2-3B · SFT demo",
        status="completed",
        training_method="sft",
        peft_method="lora",
        model_id=None,
        dataset_id=None,
        config_json={
            "model_path": "meta-llama/Llama-3.2-3B-Instruct",
            "learning_rate": 2e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "max_steps": 500,
            "warmup_steps": 50,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "quantization": "4bit",
            "fp16": True,
            "max_seq_length": 2048,
            "output_dir": "outputs/demo_llm_job",
            "template": "llama3",
        },
        output_dir="outputs/demo_llm_job",
        remarks="Demo seed — Llama-3.2-3B LoRA on alpaca subset. Loss converged well around step 300.",
        created_at=started - timedelta(minutes=1),
        started_at=started,
        finished_at=finished,
    )
    db.add(job)
    db.flush()

    steps = 500
    train_losses = list(_loss_curve(steps, start=2.6, end=0.55, noise=0.05))
    lrs = list(_lr_warmup_cosine(steps, peak=2e-4, warmup=50))
    grad_norms = [random.gauss(1.2, 0.3) + max(0, 0.8 * math.exp(-i / 120)) for i in range(steps)]

    eval_at = {99: None, 199: None, 299: None, 399: None, 499: None}
    eval_losses = list(_loss_curve(len(eval_at), start=2.4, end=0.62, noise=0.02))

    ts = started
    step_dur = (finished - started) / steps
    eval_iter = iter(eval_losses)

    for i, (tl, lr, gn) in enumerate(zip(train_losses, lrs, grad_norms)):
        if i % 10 != 0:
            continue
        el = None
        if i in eval_at:
            el = next(eval_iter, None)
        db.add(TrainingMetric(
            job_id=job.id,
            step=i + 1,
            epoch=round((i + 1) / steps * 3, 3),
            loss=round(tl, 5),
            eval_loss=round(el, 5) if el else None,
            learning_rate=round(lr, 8),
            grad_norm=round(max(gn, 0.05), 4),
            timestamp=ts + step_dur * i,
        ))

    print(f"  ✓ LLM job #{job.id}: {job.name}")
    return job


def seed_asr(db) -> Job:
    started = datetime.utcnow() - timedelta(hours=1, minutes=45)
    finished = started + timedelta(hours=1, minutes=38)

    job = Job(
        name="Whisper-large-v3 · LoRA ASR demo",
        status="completed",
        training_method="asr_whisper",
        peft_method="lora",
        model_id=None,
        dataset_id=None,
        config_json={
            "model_path": "openai/whisper-large-v3",
            "learning_rate": 1e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_steps": 3000,
            "warmup_steps": 500,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "quantization": "4bit",
            "fp16": True,
            "language": "ms",
            "task": "transcribe",
            "output_dir": "outputs/demo_asr_job",
            "train_csv": "data/asr_demo/train.csv",
        },
        output_dir="outputs/demo_asr_job",
        remarks="Demo seed — Whisper large-v3 LoRA on Malay speech. WER dropped from ~38% to ~14% on eval set.",
        created_at=started - timedelta(minutes=2),
        started_at=started,
        finished_at=finished,
    )
    db.add(job)
    db.flush()

    steps = 3000
    log_every = 50
    n = steps // log_every

    train_losses = list(_loss_curve(n, start=1.8, end=0.28, noise=0.03))
    lrs = list(_lr_warmup_cosine(n, peak=1e-4, warmup=500 // log_every))
    grad_norms = [random.gauss(0.9, 0.2) + max(0, 0.6 * math.exp(-i / 30)) for i in range(n)]

    eval_steps_set = {9, 19, 29, 39, 49, 59}  # every 500 real steps → every 10 logged
    eval_losses = list(_loss_curve(len(eval_steps_set), start=1.6, end=0.31, noise=0.015))

    ts = started
    step_dur = (finished - started) / n
    eval_iter = iter(eval_losses)

    for i, (tl, lr, gn) in enumerate(zip(train_losses, lrs, grad_norms)):
        el = next(eval_iter, None) if i in eval_steps_set else None
        db.add(TrainingMetric(
            job_id=job.id,
            step=(i + 1) * log_every,
            epoch=round((i + 1) / n * 10, 3),
            loss=round(tl, 5),
            eval_loss=round(el, 5) if el else None,
            learning_rate=round(lr, 8),
            grad_norm=round(max(gn, 0.05), 4),
            timestamp=ts + step_dur * i,
        ))

    print(f"  ✓ ASR job #{job.id}: {job.name}")
    return job


def main():
    init_db()
    db = SessionLocal()
    try:
        print("Seeding demo jobs…")
        seed_llm(db)
        seed_asr(db)
        db.commit()
        print("Done.")
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    main()
