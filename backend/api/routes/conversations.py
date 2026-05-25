from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.db.models import Conversation, ChatMessage
from backend.utils.time import now_utc

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    title: str = "New Chat"
    model_path: Optional[str] = None
    adapter_path: Optional[str] = None
    system_prompt: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    system_prompt: Optional[str] = None


class MessageCreate(BaseModel):
    role: str
    content: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationSummary(BaseModel):
    id: int
    title: str
    model_path: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int


class ConversationDetail(BaseModel):
    id: int
    title: str
    model_path: Optional[str]
    adapter_path: Optional[str]
    system_prompt: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: List[MessageOut]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=List[ConversationSummary])
def list_conversations(db: Session = Depends(get_db)):
    convs = db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
    return [
        ConversationSummary(
            id=c.id,
            title=c.title,
            model_path=c.model_path,
            created_at=c.created_at,
            updated_at=c.updated_at,
            message_count=len(c.messages),
        )
        for c in convs
    ]


@router.post("", response_model=ConversationSummary, status_code=201)
def create_conversation(body: ConversationCreate, db: Session = Depends(get_db)):
    c = Conversation(
        title=body.title,
        model_path=body.model_path,
        adapter_path=body.adapter_path,
        system_prompt=body.system_prompt,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return ConversationSummary(
        id=c.id, title=c.title, model_path=c.model_path,
        created_at=c.created_at, updated_at=c.updated_at, message_count=0,
    )


@router.get("/{conv_id}", response_model=ConversationDetail)
def get_conversation(conv_id: int, db: Session = Depends(get_db)):
    c = db.get(Conversation, conv_id)
    if not c:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail(
        id=c.id, title=c.title, model_path=c.model_path, adapter_path=c.adapter_path,
        system_prompt=c.system_prompt, created_at=c.created_at, updated_at=c.updated_at,
        messages=[MessageOut(id=m.id, role=m.role, content=m.content, created_at=m.created_at) for m in c.messages],
    )


@router.patch("/{conv_id}", response_model=ConversationSummary)
def update_conversation(conv_id: int, body: ConversationUpdate, db: Session = Depends(get_db)):
    c = db.get(Conversation, conv_id)
    if not c:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if body.title is not None:
        c.title = body.title
    if body.system_prompt is not None:
        c.system_prompt = body.system_prompt
    c.updated_at = now_utc()
    db.commit()
    db.refresh(c)
    return ConversationSummary(
        id=c.id, title=c.title, model_path=c.model_path,
        created_at=c.created_at, updated_at=c.updated_at, message_count=len(c.messages),
    )


@router.delete("/{conv_id}", status_code=204)
def delete_conversation(conv_id: int, db: Session = Depends(get_db)):
    c = db.get(Conversation, conv_id)
    if not c:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(c)
    db.commit()


@router.post("/{conv_id}/messages", response_model=MessageOut, status_code=201)
def add_message(conv_id: int, body: MessageCreate, db: Session = Depends(get_db)):
    c = db.get(Conversation, conv_id)
    if not c:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msg = ChatMessage(conversation_id=conv_id, role=body.role, content=body.content)
    db.add(msg)
    c.updated_at = now_utc()
    db.commit()
    db.refresh(msg)
    return MessageOut(id=msg.id, role=msg.role, content=msg.content, created_at=msg.created_at)
