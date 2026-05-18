from __future__ import annotations
import asyncio
import csv
import io
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from backend.api.deps import get_db
from backend.db.models import Job, Dataset, TrainingMetric
from backend.db.session import SessionLocal
from backend.workers.training_worker import run_training_job

router = APIRouter(prefix="/api/asr", tags=["asr"])

DATASETS_DIR = os.getenv("DATASETS_DIR", "./datasets")
AUDIO_DIR = os.path.join(DATASETS_DIR, "audio")
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

WHISPER_MODELS = [
    {"id": "openai/whisper-tiny",     "params": "39M"},
    {"id": "openai/whisper-base",     "params": "74M"},
    {"id": "openai/whisper-small",    "params": "244M"},
    {"id": "openai/whisper-medium",   "params": "769M"},
    {"id": "openai/whisper-large-v2", "params": "1.5B"},
    {"id": "openai/whisper-large-v3", "params": "1.5B"},
]


# ── Schemas ──────────────────────────────────────────────────────────────────

class ASRJobCreate(BaseModel):
    name: str
    peft_method: str = "lora"
    dataset_id: Optional[int] = None
    val_dataset_id: Optional[int] = None
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
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


class DatasetResponse(BaseModel):
    id: int
    name: str
    path: str
    format: str
    num_samples: Optional[int]
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Whisper model list ────────────────────────────────────────────────────────

@router.get("/models")
def list_whisper_models():
    return WHISPER_MODELS


# ── ASR Datasets ─────────────────────────────────────────────────────────────

@router.get("/datasets", response_model=List[DatasetResponse])
def list_asr_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).filter(Dataset.format == "asr_csv").order_by(Dataset.created_at.desc()).all()


@router.post("/datasets", response_model=DatasetResponse, status_code=201)
async def upload_asr_dataset(
    name: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    dest = os.path.join(DATASETS_DIR, f"asr_{file.filename}")
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    # Count rows (minus header)
    num_samples = max(0, len(content.decode(errors="replace").splitlines()) - 1)

    entry = Dataset(
        name=name,
        path=dest,
        format="asr_csv",
        num_samples=num_samples,
        description=description or None,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.post("/datasets/zip", response_model=DatasetResponse, status_code=201)
async def upload_asr_zip(
    name: str = Form(...),
    description: str = Form(""),
    audio_col: str = Form("audio_path"),
    text_col: str = Form("text"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    _MAX_ZIP_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

    import re as _re
    safe_name = _re.sub(r'[^\w\-]', '_', name.strip())[:100] or "dataset"
    extract_dir = os.path.join(AUDIO_DIR, safe_name)

    zip_bytes = await file.read()
    if len(zip_bytes) > _MAX_ZIP_BYTES:
        raise HTTPException(status_code=413, detail="ZIP too large — maximum 5 GB")

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            # Zip-slip guard: reject any member that escapes the target directory
            for member in zf.namelist():
                member_path = os.path.realpath(os.path.join(extract_dir, member))
                if not member_path.startswith(os.path.realpath(extract_dir) + os.sep):
                    raise HTTPException(status_code=400, detail=f"Unsafe path in ZIP: {member}")
            # Validate CSV exists in the manifest BEFORE extracting anything
            csv_names = [
                n for n in zf.namelist()
                if n.lower().endswith(".csv") and not n.startswith("__MACOSX")
            ]
            if not csv_names:
                raise HTTPException(status_code=400, detail="No CSV file found inside ZIP")
            os.makedirs(extract_dir, exist_ok=True)
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive")
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to process ZIP: {e}")

    # Find the CSV inside the extracted dir
    csv_files = list(Path(extract_dir).rglob("*.csv"))
    if not csv_files:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No CSV file found inside ZIP")
    src_csv = csv_files[0]

    # Build filename → absolute path lookup for all audio files in the ZIP
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_lookup: dict[str, str] = {}
    for p in Path(extract_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in audio_exts:
            audio_lookup[p.name.lower()] = str(p.resolve())

    # Rewrite audio_col paths: match by filename, keep original if not found
    rows: list[dict] = []
    missing = 0
    with open(src_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            orig = row.get(audio_col, "")
            fname = Path(orig).name.lower()
            resolved = audio_lookup.get(fname)
            if resolved:
                row[audio_col] = resolved
            else:
                missing += 1
            rows.append(row)

    out_csv = os.path.join(DATASETS_DIR, f"asr_{safe_name}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    desc_parts = [description] if description else []
    if missing:
        desc_parts.append(f"{missing} audio path(s) unresolved — trainer will skip them")

    entry = Dataset(
        name=name,
        path=out_csv,
        format="asr_csv",
        num_samples=len(rows),
        description="; ".join(desc_parts) or None,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.get("/datasets/{dataset_id}/preview")
def preview_asr_dataset(dataset_id: int, db: Session = Depends(get_db)):
    import csv as csv_mod
    entry = db.get(Dataset, dataset_id)
    if not entry or entry.format != "asr_csv":
        raise HTTPException(status_code=404, detail="ASR dataset not found")
    if not os.path.exists(entry.path):
        raise HTTPException(status_code=404, detail="CSV file not found on disk")
    try:
        with open(entry.path, newline="", encoding="utf-8-sig") as f:
            reader = csv_mod.DictReader(f)
            rows = [dict(r) for r in reader][:5]
        columns = list(rows[0].keys()) if rows else []
        return {"total": entry.num_samples, "columns": columns, "samples": rows}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_asr_dataset(dataset_id: int, db: Session = Depends(get_db)):
    entry = db.get(Dataset, dataset_id)
    if not entry or entry.format != "asr_csv":
        raise HTTPException(status_code=404, detail="ASR dataset not found")
    csv_path = entry.path
    db.delete(entry)
    db.commit()
    # Delete the rewritten CSV
    if csv_path and os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except OSError:
            pass
    # Delete the extracted audio directory (named after the dataset stem minus "asr_" prefix)
    if csv_path:
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        if stem.startswith("asr_"):
            audio_dir = os.path.join(AUDIO_DIR, stem[4:])
            if os.path.isdir(audio_dir):
                shutil.rmtree(audio_dir, ignore_errors=True)


# ── ASR Jobs ─────────────────────────────────────────────────────────────────

@router.post("/jobs", response_model=JobResponse, status_code=201)
def create_asr_job(body: ASRJobCreate, db: Session = Depends(get_db)):
    cfg = body.config.copy()

    # Resolve dataset paths from DB
    if body.dataset_id:
        ds = db.get(Dataset, body.dataset_id)
        if ds:
            cfg.setdefault("train_csv", ds.path)

    if body.val_dataset_id:
        val_ds = db.get(Dataset, body.val_dataset_id)
        if val_ds:
            cfg["val_csv"] = val_ds.path

    job = Job(
        name=body.name,
        training_method="asr_whisper",
        peft_method=body.peft_method,
        dataset_id=body.dataset_id,
        config_json=cfg,
        output_dir=cfg.get("output_dir"),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    task = run_training_job.delay(job.id)
    job.celery_task_id = task.id
    db.commit()
    db.refresh(job)
    return job


@router.get("/jobs", response_model=List[JobResponse])
def list_asr_jobs(db: Session = Depends(get_db)):
    return (
        db.query(Job)
        .filter(Job.training_method == "asr_whisper")
        .order_by(Job.created_at.desc())
        .all()
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_asr_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job or job.training_method != "asr_whisper":
        raise HTTPException(status_code=404, detail="ASR job not found")
    return job


@router.delete("/jobs/{job_id}", status_code=204)
def cancel_asr_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.celery_task_id:
        from backend.workers.celery_app import celery_app
        celery_app.control.revoke(job.celery_task_id, terminate=True)
    job.status = "cancelled"
    job.finished_at = datetime.utcnow()
    db.commit()


@router.get("/jobs/{job_id}/metrics")
async def stream_asr_metrics(job_id: int):
    async def event_generator():
        last_id = 0
        db = SessionLocal()
        try:
            job = db.get(Job, job_id)
            if not job:
                yield {"event": "done", "data": json.dumps({"status": "not_found"})}
                return
            while True:
                db.expire_all()
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
                current = db.get(Job, job_id)
                if current and current.status in ("completed", "failed", "cancelled"):
                    yield {"event": "done", "data": json.dumps({"status": current.status})}
                    break
                await asyncio.sleep(2)
        finally:
            db.close()

    return EventSourceResponse(event_generator())
