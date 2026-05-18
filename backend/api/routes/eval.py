from __future__ import annotations
import asyncio
import json
import math
import os
import uuid
from datetime import datetime, timezone
from threading import Thread
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from backend.api.deps import get_db
from backend.db.models import Dataset

router = APIRouter(prefix="/api/eval", tags=["eval"])

# In-memory run store: run_id → { status, logs, result, started_at }
_runs: dict[str, dict] = {}

_RUN_TTL_SECONDS = 86400  # 24 h


def _cleanup_old_runs() -> None:
    now = datetime.now(timezone.utc).timestamp()
    stale = [
        rid for rid, run in list(_runs.items())
        if run["status"] in ("completed", "failed")
        and (now - run["ts"]) > _RUN_TTL_SECONDS
    ]
    for rid in stale:
        _runs.pop(rid, None)


class EvalRequest(BaseModel):
    model_path: str
    adapter_path: Optional[str] = None
    dataset_id: Optional[int] = None
    dataset_path: Optional[str] = None
    mode: str = "evaluate"          # "evaluate" | "predict"
    batch_size: int = 4
    max_seq_len: int = 2048
    predict_output: Optional[str] = None
    quantization: Optional[str] = None
    # ASR-specific
    is_asr: bool = False
    audio_col: str = "audio_path"
    text_col: str = "text"
    language: Optional[str] = None
    task: str = "transcribe"


def _push(run_id: str, line: str):
    _runs[run_id]["logs"].append(line)


def _run_eval(run_id: str, req: EvalRequest, csv_path: Optional[str]):
    import torch
    from backend.core.model.loader import load_model, load_tokenizer

    _runs[run_id]["status"] = "running"
    try:
        _push(run_id, f"[eval] Loading tokenizer from {req.model_path}…")
        tokenizer = load_tokenizer(req.model_path)

        _push(run_id, f"[eval] Loading model…")
        model = load_model(req.model_path, quantization=req.quantization)

        if req.adapter_path:
            from peft import PeftModel
            _push(run_id, f"[eval] Applying adapter from {req.adapter_path}…")
            model = PeftModel.from_pretrained(model, req.adapter_path)

        model.eval()

        # Resolve dataset path
        data_path = req.dataset_path or csv_path
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        if req.mode == "evaluate":
            _push(run_id, "[eval] Mode: evaluate (loss + perplexity)")
            from backend.core.data.dataset import build_plain_text_dataset
            from torch.utils.data import DataLoader
            from transformers import DataCollatorForLanguageModeling

            ds = build_plain_text_dataset(data_path, tokenizer, max_length=req.max_seq_len)
            collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            loader = DataLoader(ds, batch_size=req.batch_size, collate_fn=collator)

            total_loss, total_steps = 0.0, 0
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    total_loss += outputs.loss.item()
                    total_steps += 1
                    if (i + 1) % 10 == 0:
                        _push(run_id, f"[eval] step {i+1}/{len(loader)}  loss={outputs.loss.item():.4f}")

            avg_loss = total_loss / max(total_steps, 1)
            perplexity = math.exp(min(avg_loss, 20))
            _push(run_id, f"[eval] Done — loss={avg_loss:.4f}  perplexity={perplexity:.2f}")
            _runs[run_id]["result"] = {"loss": avg_loss, "perplexity": perplexity}

        else:
            # Predict mode
            out_path = req.predict_output or f"./outputs/predict_{run_id[:8]}.jsonl"
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            _push(run_id, f"[predict] Output → {out_path}")

            import csv as csv_mod
            rows: list[dict] = []
            with open(data_path, newline="", encoding="utf-8-sig") as f:
                reader = csv_mod.DictReader(f) if data_path.endswith(".csv") else None
                if reader:
                    rows = list(reader)
                else:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                rows.append(json.loads(line))
                            except Exception:
                                rows.append({"text": line})

            with open(out_path, "w", encoding="utf-8") as out_f:
                for i, row in enumerate(rows):
                    prompt = row.get("instruction") or row.get("text") or row.get("input") or ""
                    if not prompt:
                        continue
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=req.max_seq_len).to(model.device)
                    with torch.no_grad():
                        ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                    output = tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    out_f.write(json.dumps({"prompt": prompt, "output": output}, ensure_ascii=False) + "\n")
                    if (i + 1) % 10 == 0:
                        _push(run_id, f"[predict] {i+1}/{len(rows)} done")

            _push(run_id, f"[predict] Saved {len(rows)} predictions → {out_path}")
            _runs[run_id]["result"] = {"output_file": out_path, "num_samples": len(rows)}

        _runs[run_id]["status"] = "completed"

    except Exception as e:
        _push(run_id, f"[error] {e}")
        _runs[run_id]["status"] = "failed"
        _runs[run_id]["result"] = {"error": str(e)}
    finally:
        _runs[run_id]["ts"] = datetime.now(timezone.utc).timestamp()


def _run_asr_eval(run_id: str, req: EvalRequest, csv_path: Optional[str]):
    import csv as csv_mod
    import torch
    from backend.core.asr.loader import load_whisper_processor, load_whisper_model

    _runs[run_id]["status"] = "running"
    try:
        language = req.language if req.language and req.language != "auto" else None

        _push(run_id, f"[eval] Loading Whisper processor from {req.model_path}…")
        processor = load_whisper_processor(req.model_path, language=language, task=req.task)

        _push(run_id, f"[eval] Loading Whisper model…")
        model = load_whisper_model(
            req.model_path, quantization=req.quantization,
            language=language, task=req.task, processor=processor,
        )

        if req.adapter_path:
            from peft import PeftModel
            _push(run_id, f"[eval] Applying LoRA adapter from {req.adapter_path}…")
            model = PeftModel.from_pretrained(model, req.adapter_path)
            model = model.merge_and_unload()

        model.eval()

        data_path = req.dataset_path or csv_path
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset CSV not found: {data_path}")

        with open(data_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv_mod.DictReader(f))
        _push(run_id, f"[eval] Loaded {len(rows)} rows from {data_path}")

        import librosa

        if req.mode == "evaluate":
            import evaluate as hf_eval
            wer_metric = hf_eval.load("wer")
            preds, refs = [], []
            for i, row in enumerate(rows):
                audio_path = row.get(req.audio_col, "")
                ref = row.get(req.text_col, "")
                if not os.path.exists(audio_path):
                    _push(run_id, f"[warn] Audio not found: {audio_path}"); continue
                audio, _ = librosa.load(audio_path, sr=16000)
                feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
                with torch.no_grad():
                    ids = model.generate(feats)
                pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
                preds.append(pred); refs.append(ref)
                if (i + 1) % 10 == 0:
                    _push(run_id, f"[eval] {i+1}/{len(rows)} transcribed…")
            wer = wer_metric.compute(predictions=preds, references=refs) * 100
            _push(run_id, f"[eval] Done — WER={wer:.2f}%  ({len(preds)} samples)")
            _runs[run_id]["result"] = {"wer": round(wer, 4), "n_samples": len(preds)}

        else:
            out_path = req.predict_output or f"./outputs/asr_predict_{run_id[:8]}.jsonl"
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            _push(run_id, f"[predict] Output → {out_path}")
            with open(out_path, "w", encoding="utf-8") as fout:
                for i, row in enumerate(rows):
                    audio_path = row.get(req.audio_col, "")
                    ref = row.get(req.text_col, "")
                    if not os.path.exists(audio_path):
                        _push(run_id, f"[warn] Audio not found: {audio_path}"); continue
                    audio, _ = librosa.load(audio_path, sr=16000)
                    feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
                    with torch.no_grad():
                        ids = model.generate(feats)
                    pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
                    fout.write(json.dumps({"audio_path": audio_path, "reference": ref, "prediction": pred}, ensure_ascii=False) + "\n")
                    if (i + 1) % 10 == 0:
                        _push(run_id, f"[predict] {i+1}/{len(rows)} done")
            _push(run_id, f"[predict] Saved {len(rows)} predictions → {out_path}")
            _runs[run_id]["result"] = {"output_file": out_path, "n_samples": len(rows)}

        _runs[run_id]["status"] = "completed"

    except Exception as e:
        _push(run_id, f"[error] {e}")
        _runs[run_id]["status"] = "failed"
        _runs[run_id]["result"] = {"error": str(e)}
    finally:
        _runs[run_id]["ts"] = datetime.now(timezone.utc).timestamp()


@router.post("/run")
def start_eval(req: EvalRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    _cleanup_old_runs()
    run_id = str(uuid.uuid4())
    _runs[run_id] = {
        "status": "pending", "logs": [], "result": {},
        "started_at": datetime.utcnow().isoformat(),
        "ts": datetime.now(timezone.utc).timestamp(),
    }

    # Resolve dataset_id → path
    csv_path: Optional[str] = None
    if req.dataset_id:
        ds = db.get(Dataset, req.dataset_id)
        if ds:
            csv_path = ds.path

    fn = _run_asr_eval if req.is_asr else _run_eval
    background_tasks.add_task(fn, run_id, req, csv_path)
    return {"run_id": run_id}


@router.get("/{run_id}/result")
def get_result(run_id: str):
    run = _runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": run["status"], **run["result"]}


@router.get("/{run_id}/stream")
async def stream_logs(run_id: str):
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    sent = 0

    async def generator():
        nonlocal sent
        while True:
            logs = _runs[run_id]["logs"]
            while sent < len(logs):
                yield {"data": json.dumps({"line": logs[sent], "status": _runs[run_id]["status"]})}
                sent += 1
            if _runs[run_id]["status"] in ("completed", "failed"):
                yield {"data": json.dumps({"line": "__done__", "status": _runs[run_id]["status"], "result": _runs[run_id]["result"]})}
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(generator())
