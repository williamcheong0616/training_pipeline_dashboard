from __future__ import annotations
import asyncio
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
    template: Optional[str] = None
    num_samples: Optional[int] = None
    description: Optional[str] = None
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

    # Move blocking file-write + parse + detect off the event loop
    def _write_and_parse() -> tuple[list[dict], int | None, dict]:
        with open(dest, "wb") as fh:
            fh.write(content)
        records: list[dict] = []
        num: int | None = None
        try:
            text = content.decode("utf-8", errors="replace")
            if safe_filename.endswith(".jsonl"):
                records = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
            else:
                parsed = json.loads(text)
                records = parsed if isinstance(parsed, list) else [parsed]
            num = len(records)
        except Exception:
            pass
        return records, num, detect_format(records)

    records, num_samples, detection = await asyncio.to_thread(_write_and_parse)
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
async def convert_dataset_endpoint(
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

    # Capture everything needed inside the thread (avoid touching db/entry from a thread)
    source_path = entry.path
    source_name = entry.name
    source_template = entry.template
    template_name = body.template_name
    output_name = body.output_name or f"{source_name}_as_{target_fmt}"
    safe_out = os.path.basename(output_name.replace(" ", "_")) + ".jsonl"
    dest = os.path.join(DATASETS_DIR, safe_out)

    # Move all blocking file I/O + CPU conversion off the event loop
    def _load_convert_write() -> int:
        records = _load_raw(source_path, source_fmt)
        converted = convert_dataset(records, source_fmt, target_fmt, template_name)
        if not converted:
            raise ValueError("Conversion produced 0 records — check source data and format.")
        with open(dest, "w", encoding="utf-8") as fh:
            for r in converted:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(converted)

    try:
        num_samples = await asyncio.to_thread(_load_convert_write)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {exc}")

    new_entry = Dataset(
        name=output_name,
        path=dest,
        format=target_fmt,
        template=template_name if target_fmt == "plain_text" else source_template,
        num_samples=num_samples,
        description=f"Converted from {source_name!r} ({source_fmt} → {target_fmt})",
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)
    return new_entry


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: int, db: Session = Depends(get_db)):
    entry = db.get(Dataset, dataset_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not os.path.exists(entry.path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    path = entry.path
    fmt = entry.format
    total = entry.num_samples

    def _read_samples() -> list:
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                records = []
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
                    if len(records) >= 5:
                        break
                return records
            else:
                data = json.loads(f.read())
                return data[:5] if isinstance(data, list) else [data]

    try:
        samples = await asyncio.to_thread(_read_samples)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "format": fmt,
        "total": total,
        "samples": samples,
        "valid_targets": VALID_TARGETS.get(fmt, []),
        "conversion_notes": CONVERSION_NOTES.get(fmt, {}),
    }


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
