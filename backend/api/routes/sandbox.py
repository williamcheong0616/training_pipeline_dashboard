from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])

SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".ogg"}


def _run_asr(audio_path: str, model_path: str, language: Optional[str], task: str) -> dict:
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    generate_kwargs: dict = {"task": task}
    if language and language.lower() not in ("auto", ""):
        generate_kwargs["language"] = language

    result = pipe(audio_path, generate_kwargs=generate_kwargs, return_timestamps=False)
    return {"transcript": result["text"].strip(), "model_path": model_path}


@router.post("/asr/transcribe")
async def sandbox_asr_transcribe(
    audio: UploadFile = File(...),
    model_path: str = Form("openai/whisper-base"),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
):
    suffix = Path(audio.filename or "audio.wav").suffix.lower()
    if suffix not in SUPPORTED_AUDIO:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {suffix}")

    contents = await audio.read()
    if len(contents) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 100 MB)")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(contents)
        tmp_path = f.name

    try:
        result = await asyncio.to_thread(_run_asr, tmp_path, model_path, language or None, task)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
