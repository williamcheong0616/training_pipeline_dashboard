from __future__ import annotations
import json
import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.db.models import Dataset

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATASETS_DIR = os.getenv("DATASETS_DIR", "./datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
_ALLOWED_EXTS = {".json", ".jsonl"}


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


@router.get("", response_model=List[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).filter(Dataset.format != "asr_csv").order_by(Dataset.created_at.desc()).all()


@router.post("", response_model=DatasetResponse, status_code=201)
async def upload_dataset(
    name: str = Form(...),
    format: str = Form("alpaca"),
    description: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail="Only .json and .jsonl files are accepted")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large — maximum 200 MB")

    safe_filename = os.path.basename(file.filename or "upload")
    dest = os.path.join(DATASETS_DIR, safe_filename)
    with open(dest, "wb") as f:
        f.write(content)

    num_samples = None
    try:
        data = json.loads(content)
        num_samples = len(data) if isinstance(data, list) else None
    except Exception:
        num_samples = sum(1 for line in content.decode().splitlines() if line.strip())

    entry = Dataset(
        name=name,
        path=dest,
        format=format,
        num_samples=num_samples,
        description=description or None,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: int, db: Session = Depends(get_db)):
    entry = db.get(Dataset, dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not os.path.exists(entry.path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    try:
        with open(entry.path, encoding="utf-8") as f:
            content = f.read()
        if entry.path.endswith(".jsonl"):
            records = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                if len(records) >= 5:
                    break
        else:
            data = json.loads(content)
            records = (data[:5] if isinstance(data, list) else [data])
        return {"format": entry.format, "total": entry.num_samples, "samples": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    entry = db.get(Dataset, dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    path = entry.path
    db.delete(entry)
    db.commit()
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
