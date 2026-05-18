from __future__ import annotations
import asyncio
import gc
import json
from threading import Thread
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Single model singleton — one model loaded at a time
_state: dict = {
    "model": None,
    "tokenizer": None,
    "model_path": None,
    "adapter_path": None,
    "status": "unloaded",   # unloaded | loading | ready | error
    "error": None,
}


class LoadRequest(BaseModel):
    model_path: str
    adapter_path: Optional[str] = None
    quantization: Optional[str] = None


class Message(BaseModel):
    role: str   # "system" | "user" | "assistant"
    content: str


class GenerateRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


def _load_model(model_path: str, adapter_path: Optional[str], quantization: Optional[str]):
    from backend.core.model.loader import load_model, load_tokenizer

    _state["status"] = "loading"
    _state["error"] = None
    try:
        _state["tokenizer"] = load_tokenizer(model_path)
        _state["model"] = load_model(model_path, quantization=quantization)
        if adapter_path:
            from peft import PeftModel
            _state["model"] = PeftModel.from_pretrained(_state["model"], adapter_path)
        _state["model_path"] = model_path
        _state["adapter_path"] = adapter_path
        _state["status"] = "ready"
    except Exception as e:
        _state["status"] = "error"
        _state["error"] = str(e)
        _state["model"] = None
        _state["tokenizer"] = None


@router.post("/load")
def load_model(req: LoadRequest, background_tasks: BackgroundTasks):
    if _state["status"] == "loading":
        raise HTTPException(status_code=409, detail="A model is already loading")
    background_tasks.add_task(_load_model, req.model_path, req.adapter_path, req.quantization)
    return {"message": "Loading started"}


@router.get("/status")
def get_status():
    return {
        "status": _state["status"],
        "model_path": _state["model_path"],
        "adapter_path": _state["adapter_path"],
        "error": _state["error"],
    }


@router.post("/unload")
def unload_model():
    import torch
    _state["model"] = None
    _state["tokenizer"] = None
    _state["model_path"] = None
    _state["adapter_path"] = None
    _state["status"] = "unloaded"
    _state["error"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"message": "Model unloaded"}


def _build_prompt(messages: List[Message], tokenizer) -> str:
    # Use apply_chat_template if available, else fall back to simple format
    try:
        chat = [{"role": m.role, "content": m.content} for m in messages]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"System: {m.content}")
            elif m.role == "user":
                parts.append(f"User: {m.content}")
            elif m.role == "assistant":
                parts.append(f"Assistant: {m.content}")
        parts.append("Assistant:")
        return "\n".join(parts)


@router.post("/generate")
async def generate(req: GenerateRequest):
    if _state["status"] != "ready":
        raise HTTPException(status_code=400, detail=f"Model not ready (status: {_state['status']})")

    model = _state["model"]
    tokenizer = _state["tokenizer"]

    prompt = _build_prompt(req.messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        do_sample=req.temperature > 0,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    async def token_generator():
        loop = asyncio.get_event_loop()
        for token in streamer:
            yield {"data": json.dumps({"token": token})}
            await asyncio.sleep(0)
        yield {"data": json.dumps({"token": "__done__"})}

    return EventSourceResponse(token_generator())
