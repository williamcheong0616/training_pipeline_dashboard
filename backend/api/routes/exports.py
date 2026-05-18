from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.db.models import Job

router = APIRouter(prefix="/api/exports", tags=["exports"])

EXPORTS_DIR = os.getenv("EXPORTS_DIR", "./exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)


class ExportRequest(BaseModel):
    output_name: str = ""


class PathExportRequest(BaseModel):
    adapter_path: str
    output_name: str = ""


@router.get("")
def list_exports():
    entries = []
    for p in sorted(Path(EXPORTS_DIR).iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_dir():
            size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 2)
            entries.append({
                "name": p.name,
                "path": str(p),
                "size_mb": round(size_mb, 1),
                "created_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            })
    return entries


def _merge_adapter(adapter_path: str, save_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = open(os.path.join(adapter_path, "adapter_config.json")).read()
    import json
    cfg = json.loads(base_model_name)
    base = cfg.get("base_model_name_or_path", "")

    model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(save_path)


@router.post("/from-path")
def export_from_path(body: PathExportRequest, background_tasks: BackgroundTasks):
    if not os.path.isdir(body.adapter_path):
        raise HTTPException(status_code=400, detail="Adapter path does not exist")
    name = body.output_name or f"merged_custom_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(EXPORTS_DIR, name)
    background_tasks.add_task(_merge_adapter, body.adapter_path, save_path)
    return {"message": "Merge started", "save_path": save_path}


@router.post("/{job_id}")
def export_job(job_id: int, body: ExportRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job must be completed before exporting")
    if not job.output_dir or not os.path.isdir(job.output_dir):
        raise HTTPException(status_code=400, detail="Output directory not found")

    name = body.output_name or f"merged_job_{job_id}"
    save_path = os.path.join(EXPORTS_DIR, name)
    background_tasks.add_task(_merge_adapter, job.output_dir, save_path)
    return {"message": "Merge started", "save_path": save_path}
