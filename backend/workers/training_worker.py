from __future__ import annotations
from datetime import datetime

from backend.workers.celery_app import celery_app
from backend.db.session import SessionLocal
from backend.db.models import Job


def _get_trainer(method: str, job_id: int, config: dict, db_factory):
    if method == "sft":
        from backend.core.trainer.sft_trainer import SFTPipelineTrainer
        return SFTPipelineTrainer(job_id, config, db_factory)
    if method == "unsupervised":
        from backend.core.trainer.unsupervised_trainer import UnsupervisedPipelineTrainer
        return UnsupervisedPipelineTrainer(job_id, config, db_factory)
    if method == "dpo":
        from backend.core.trainer.dpo_trainer import DPOPipelineTrainer
        return DPOPipelineTrainer(job_id, config, db_factory)
    if method == "rm":
        from backend.core.trainer.rm_trainer import RMPipelineTrainer
        return RMPipelineTrainer(job_id, config, db_factory)
    if method == "kto":
        from backend.core.trainer.kto_trainer import KTOPipelineTrainer
        return KTOPipelineTrainer(job_id, config, db_factory)
    if method == "orpo":
        from backend.core.trainer.orpo_trainer import ORPOPipelineTrainer
        return ORPOPipelineTrainer(job_id, config, db_factory)
    if method == "asr_whisper":
        from backend.core.asr.trainer import ASRPipelineTrainer
        return ASRPipelineTrainer(job_id, config, db_factory)
    raise ValueError(f"Unknown training method: {method}")


@celery_app.task(bind=True, name="training_pipeline.run_job")
def run_training_job(self, job_id: int):
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}

        job.status = "running"
        job.started_at = datetime.utcnow()
        job.celery_task_id = self.request.id
        db.commit()

        trainer = _get_trainer(job.training_method, job_id, job.config_json, SessionLocal)
        trainer.train()

        job = db.get(Job, job_id)
        job.status = "completed"
        job.finished_at = datetime.utcnow()
        db.commit()
        return {"status": "completed", "job_id": job_id}

    except Exception as exc:
        db.rollback()
        job = db.get(Job, job_id)
        if job:
            job.status = "failed"
            job.finished_at = datetime.utcnow()
            job.error_msg = str(exc)
            db.commit()
        raise
    finally:
        db.close()
