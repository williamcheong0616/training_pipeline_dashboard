from __future__ import annotations
import os

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
