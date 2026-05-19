from __future__ import annotations
import asyncio
import json
from backend.utils.time import now_utc
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from backend.api.deps import get_db
from backend.db.models import Job, TrainingMetric
from backend.db.session import SessionLocal
from backend.workers.training_worker import run_training_job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobCreate(BaseModel):
    name: str
    training_method: str
    peft_method: str = "lora"
    model_id: Optional[int] = None
    dataset_id: Optional[int] = None
    config: dict = {}


class JobResponse(BaseModel):
    id: int
    name: str
    status: str
    training_method: str
    peft_method: str
    model_id: Optional[int]
    dataset_id: Optional[int]
    config_json: Optional[dict] = None
    output_dir: Optional[str]
    error_msg: Optional[str]
    remarks: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


class RemarksUpdate(BaseModel):
    remarks: str


@router.post("", response_model=JobResponse, status_code=201)
def create_job(body: JobCreate, db: Session = Depends(get_db)):
    job = Job(
        name=body.name,
        training_method=body.training_method,
        peft_method=body.peft_method,
        model_id=body.model_id,
        dataset_id=body.dataset_id,
        config_json=body.config,
        output_dir=body.config.get("output_dir"),
    )
    db.add(job)
    db.flush()  # get ID without committing so we can roll back if Celery fails
    try:
        task = run_training_job.delay(job.id)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Task queue unavailable: {exc}")
    job.celery_task_id = task.id
    db.commit()
    db.refresh(job)
    return job


@router.get("", response_model=List[JobResponse])
def list_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
):
    return db.query(Job).order_by(Job.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/{job_id}", status_code=204)
def cancel_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.celery_task_id:
        from backend.workers.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True)
    job.status = "cancelled"
    job.finished_at = now_utc()
    db.commit()


@router.get("/{job_id}/metrics")
async def stream_metrics(job_id: int):
    async def event_generator():
        last_id = 0
        db = SessionLocal()
        try:
            job = db.get(Job, job_id)
            if not job:
                yield {"event": "done", "data": json.dumps({"status": "not_found"})}
                return
            while True:
                rows = (
                    db.query(TrainingMetric)
                    .filter(TrainingMetric.job_id == job_id, TrainingMetric.id > last_id)
                    .order_by(TrainingMetric.id)
                    .all()
                )
                for row in rows:
                    last_id = row.id
                    yield {
                        "data": json.dumps({
                            "id": row.id,
                            "step": row.step,
                            "epoch": row.epoch,
                            "loss": row.loss,
                            "eval_loss": row.eval_loss,
                            "learning_rate": row.learning_rate,
                            "reward": row.reward,
                            "grad_norm": row.grad_norm,
                            "timestamp": row.timestamp.isoformat(),
                        })
                    }
                # Use query() not get() to bypass identity map and get fresh status
                current_job = db.query(Job).filter(Job.id == job_id).first()
                if current_job and current_job.status in ("completed", "failed", "cancelled"):
                    yield {"event": "done", "data": json.dumps({"status": current_job.status})}
                    break
                await asyncio.sleep(2)
        finally:
            db.close()

    return EventSourceResponse(event_generator())


@router.get("/{job_id}/metrics/all")
async def get_all_metrics(job_id: int):
    def _query():
        db = SessionLocal()
        try:
            job = db.get(Job, job_id)
            if not job:
                return None
            rows = (
                db.query(TrainingMetric)
                .filter(TrainingMetric.job_id == job_id)
                .order_by(TrainingMetric.step)
                .all()
            )
            return [
                {
                    "id": r.id, "step": r.step, "epoch": r.epoch,
                    "loss": r.loss, "eval_loss": r.eval_loss,
                    "learning_rate": r.learning_rate, "reward": r.reward,
                    "grad_norm": r.grad_norm,
                }
                for r in rows
            ]
        finally:
            db.close()

    result = await asyncio.to_thread(_query)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


@router.patch("/{job_id}/remarks", response_model=JobResponse)
def update_remarks(job_id: int, body: RemarksUpdate, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.remarks = body.remarks
    db.commit()
    db.refresh(job)
    return job


@router.delete("/{job_id}/purge", status_code=204)
def purge_job(job_id: int, db: Session = Depends(get_db)):
    """Permanently remove a job and all its metrics from the database."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail="Only terminal jobs (completed/failed/cancelled) can be purged")
    db.delete(job)
    db.commit()
