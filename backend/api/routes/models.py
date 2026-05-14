from __future__ import annotations
import os
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.db.models import ModelEntry

router = APIRouter(prefix="/api/models", tags=["models"])

MODELS_DIR = os.getenv("MODELS_DIR", "./models")


class ModelCreate(BaseModel):
    name: str
    hf_repo: str
    architecture: Optional[str] = None
    template: str = "alpaca"


class ModelResponse(BaseModel):
    id: int
    name: str
    hf_repo: str
    local_path: Optional[str]
    architecture: Optional[str]
    template: str
    is_downloaded: str
    downloaded_at: Optional[datetime]

    class Config:
        from_attributes = True


class HFSearchResult(BaseModel):
    model_id: str
    pipeline_tag: Optional[str]
    downloads: Optional[int]
    likes: Optional[int]


def _download_model(hf_repo: str, local_path: str, model_id: int, db_factory):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=hf_repo, local_dir=local_path)
    db = db_factory()
    try:
        entry = db.get(ModelEntry, model_id)
        if entry:
            entry.is_downloaded = "true"
            entry.downloaded_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


@router.get("", response_model=List[ModelResponse])
def list_models(db: Session = Depends(get_db)):
    return db.query(ModelEntry).all()


@router.post("", response_model=ModelResponse, status_code=201)
def register_model(body: ModelCreate, db: Session = Depends(get_db)):
    existing = db.query(ModelEntry).filter(ModelEntry.hf_repo == body.hf_repo).first()
    if existing:
        return existing
    entry = ModelEntry(
        name=body.name,
        hf_repo=body.hf_repo,
        architecture=body.architecture,
        template=body.template,
        local_path=os.path.join(MODELS_DIR, body.hf_repo.replace("/", "--")),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.post("/{model_id}/download")
def download_model(model_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    entry = db.get(ModelEntry, model_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Model not found")
    if entry.is_downloaded == "true":
        return {"message": "Already downloaded", "local_path": entry.local_path}
    from backend.db.session import SessionLocal
    background_tasks.add_task(_download_model, entry.hf_repo, entry.local_path, model_id, SessionLocal)
    return {"message": f"Download started for {entry.hf_repo}"}


@router.get("/search/hub")
def search_hub(q: str, limit: int = 10) -> List[HFSearchResult]:
    from huggingface_hub import HfApi
    api = HfApi()
    results = api.list_models(search=q, limit=limit, sort="downloads", direction=-1)
    return [
        HFSearchResult(
            model_id=m.id,
            pipeline_tag=getattr(m, "pipeline_tag", None),
            downloads=getattr(m, "downloads", None),
            likes=getattr(m, "likes", None),
        )
        for m in results
    ]


@router.delete("/{model_id}", status_code=204)
def delete_model(model_id: int, db: Session = Depends(get_db)):
    entry = db.get(ModelEntry, model_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Model not found")
    db.delete(entry)
    db.commit()
