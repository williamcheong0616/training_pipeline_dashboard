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
from backend.core.data.detector import detect_format
from backend.core.data.converter import VALID_TARGETS, CONVERSION_NOTES, convert_dataset
from backend.core.data.dataset import _load_raw

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
    template: Optional[str]
    num_samples: Optional[int]
    description: Optional[str]
    created_at: datetime
    detected_format: Optional[str] = None
    detection_confidence: Optional[str] = None

    class Config:
        from_attributes = True


class ConvertRequest(BaseModel):
    target_format: str
    template_name: str = "alpaca"
    output_name: Optional[str] = None


@router.get("", response_model=List[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).filter(Dataset.format != "asr_csv").order_by(Dataset.created_at.desc()).all()


@router.post("", response_model=DatasetResponse, status_code=201)
async def upload_dataset(
    name: str = Form(...),
    format: str = Form("auto"),
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

    # Parse records for detection + count
    records: list[dict] = []
    num_samples: int | None = None
    try:
        text = content.decode("utf-8", errors="replace")
        if safe_filename.endswith(".jsonl"):
            records = [json.loads(l) for l in text.splitlines() if l.strip()]
        else:
            parsed = json.loads(text)
            records = parsed if isinstance(parsed, list) else [parsed]
        num_samples = len(records)
    except Exception:
        num_samples = None

    # Server-side format detection
    detection = detect_format(records)
    resolved_format = detection["format"] if format == "auto" else format

    entry = Dataset(
        name=name,
        path=dest,
        format=resolved_format,
        num_samples=num_samples,
        description=description or None,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    # Attach detection metadata to response without persisting it
    resp = DatasetResponse.model_validate(entry)
    resp.detected_format = detection["format"]
    resp.detection_confidence = detection["confidence"]
    return resp


@router.post("/{dataset_id}/convert", response_model=DatasetResponse, status_code=201)
def convert_dataset_endpoint(
    dataset_id: int,
    body: ConvertRequest,
    db: Session = Depends(get_db),
):
    entry = db.get(Dataset, dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not os.path.exists(entry.path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    source_fmt = entry.format
    target_fmt = body.target_format

    if target_fmt not in VALID_TARGETS.get(source_fmt, []):
        valid = VALID_TARGETS.get(source_fmt, [])
        raise HTTPException(
            status_code=400,
            detail=f"Cannot convert {source_fmt!r} → {target_fmt!r}. Valid targets: {valid}",
        )

    # Load + convert
    try:
        records = _load_raw(entry.path, source_fmt)
        converted = convert_dataset(records, source_fmt, target_fmt, body.template_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {exc}")

    if not converted:
        raise HTTPException(status_code=422, detail="Conversion produced 0 records — check source data and format.")

    # Write output
    output_name = body.output_name or f"{entry.name}_as_{target_fmt}"
    safe_out = os.path.basename(output_name.replace(" ", "_")) + ".jsonl"
    dest = os.path.join(DATASETS_DIR, safe_out)
    with open(dest, "w", encoding="utf-8") as f:
        for r in converted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    new_entry = Dataset(
        name=output_name,
        path=dest,
        format=target_fmt,
        template=body.template_name if target_fmt == "plain_text" else entry.template,
        num_samples=len(converted),
        description=f"Converted from {entry.name!r} ({source_fmt} → {target_fmt})",
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return new_entry


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
        return {
            "format": entry.format,
            "total": entry.num_samples,
            "samples": records,
            "valid_targets": VALID_TARGETS.get(entry.format, []),
            "conversion_notes": CONVERSION_NOTES.get(entry.format, {}),
        }
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
