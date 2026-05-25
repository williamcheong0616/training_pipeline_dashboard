from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.db.models import PromptProfile

router = APIRouter(prefix="/api/prompt-profiles", tags=["prompt-profiles"])


class ProfileCreate(BaseModel):
    name: str
    system_prompt: Optional[str] = None
    gen_params: Optional[dict] = None


class ProfileOut(BaseModel):
    id: int
    name: str
    system_prompt: Optional[str]
    gen_params: Optional[dict]
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("", response_model=List[ProfileOut])
def list_profiles(db: Session = Depends(get_db)):
    return db.query(PromptProfile).order_by(PromptProfile.created_at.desc()).all()


@router.post("", response_model=ProfileOut, status_code=201)
def create_profile(body: ProfileCreate, db: Session = Depends(get_db)):
    p = PromptProfile(name=body.name, system_prompt=body.system_prompt, gen_params=body.gen_params)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.delete("/{profile_id}", status_code=204)
def delete_profile(profile_id: int, db: Session = Depends(get_db)):
    p = db.get(PromptProfile, profile_id)
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found")
    db.delete(p)
    db.commit()
