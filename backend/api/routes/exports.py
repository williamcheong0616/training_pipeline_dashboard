from __future__ import annotations
import os
import re
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


def _safe_output_name(raw: str, fallback: str) -> str:
    name = re.sub(r'[/\\:*?"<>|]', '_', (raw or "").strip())[:128]
    return name or fallback


def _validated_save_path(name: str) -> str:
    exports_real = os.path.realpath(EXPORTS_DIR)
    candidate = os.path.realpath(os.path.join(EXPORTS_DIR, name))
    if not candidate.startswith(exports_real + os.sep) and candidate != exports_real:
        raise ValueError(f"Invalid output name — path escapes exports directory")
    return candidate


def _merge_adapter(adapter_path: str, save_path: str):
    import json
    from peft import PeftModel

    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    try:
        cfg = json.loads(open(cfg_path).read())
    except FileNotFoundError:
        raise RuntimeError("adapter_config.json not found — is this a valid adapter directory?")
    except json.JSONDecodeError:
        raise RuntimeError("adapter_config.json is malformed")
    base = cfg.get("base_model_name_or_path", "")

    if "whisper" in base.lower():
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        model = WhisperForConditionalGeneration.from_pretrained(base, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        merged = model.merge_and_unload()
        merged.save_pretrained(save_path)
        WhisperProcessor.from_pretrained(adapter_path).save_pretrained(save_path)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        merged = model.merge_and_unload()
        merged.save_pretrained(save_path)
        AutoTokenizer.from_pretrained(adapter_path).save_pretrained(save_path)


@router.post("/from-path")
def export_from_path(body: PathExportRequest, background_tasks: BackgroundTasks):
    if not os.path.isdir(body.adapter_path):
        raise HTTPException(status_code=400, detail="Adapter path does not exist")
    name = _safe_output_name(body.output_name, f"merged_custom_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    try:
        save_path = _validated_save_path(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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

    name = _safe_output_name(body.output_name, f"merged_job_{job_id}")
    try:
        save_path = _validated_save_path(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    background_tasks.add_task(_merge_adapter, job.output_dir, save_path)
    return {"message": "Merge started", "save_path": save_path}
