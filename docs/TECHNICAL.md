# Forge — Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Backend](#backend)
   - [API Layer](#api-layer)
   - [Training Workers](#training-workers)
   - [Core Modules](#core-modules)
   - [Database Schema](#database-schema)
3. [Frontend](#frontend)
   - [Pages & Routing](#pages--routing)
   - [Data Fetching](#data-fetching)
   - [Real-Time Streaming](#real-time-streaming)
   - [Design System](#design-system)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [CLI Reference](#cli-reference)

---

## Architecture Overview

```
Browser
  │  HTTP / SSE
  ▼
Next.js (port 3000)
  │  /api/* proxy rewrite
  ▼
FastAPI (port 8000)
  │               │
  ├─ sync routes  └─ BackgroundTasks / SSE
  │
  ▼
Celery Worker ──── Redis (broker)
  │
  ▼
GPU Training (PyTorch + HuggingFace)
  │
  ▼
SQLite / PostgreSQL (metrics, jobs, models, datasets)
```

**Key design decisions:**
- Long-running training jobs are dispatched to a Celery worker via Redis. The API never blocks.
- Short-lived tasks (eval, export, chat model load) use FastAPI `BackgroundTasks`.
- All real-time output (training metrics, eval logs, chat tokens) is delivered via SSE (`text/event-stream`). No WebSockets.
- ASR and LLM training are completely isolated: separate routes, separate trainer classes, same DB tables discriminated by `training_method` and `format` columns.

---

## Backend

### API Layer

**Entry point:** `backend/api/main.py`

```python
app = FastAPI(title="Training Pipeline API", version="1.0.0")
# CORS: allow all origins (restrict in production)
# Routers: jobs, models, datasets, exports, asr, eval, chat
# Lifespan: runs init_db() on startup
```

**Health & system:**
- `GET /api/health` → `{ "status": "ok" }`
- `GET /api/system` → CPU %, RAM, disk, GPU memory (per device)

---

### Training Workers

**`backend/workers/celery_app.py`**

```python
celery_app = Celery("forge", broker=REDIS_URL, backend=REDIS_URL)
```

**`backend/workers/training_worker.py`**

```python
@celery_app.task(bind=True)
def run_training_job(self, job_id: int):
    # 1. Load job from DB
    # 2. Dispatch to correct trainer via _get_trainer(method, config)
    # 3. Update job.status: pending → running → completed/failed
    # 4. Write error_msg on failure
```

Trainer dispatch table (`_get_trainer`):

| `training_method` | Trainer class |
|-------------------|---------------|
| `sft` | `SFTPipelineTrainer` |
| `unsupervised` | `UnsupervisedTrainer` |
| `dpo` | `DPOPipelineTrainer` |
| `rm` | `RMPipelineTrainer` |
| `kto` | `KTOPipelineTrainer` |
| `orpo` | `ORPOPipelineTrainer` |
| `asr_whisper` | `ASRPipelineTrainer` |

---

### Core Modules

#### `backend/core/model/loader.py`

```python
load_model(model_name, quantization=None, device_map="auto") → AutoModelForCausalLM
# quantization: None (fp16/bf16), "4bit" (QLoRA via BitsAndBytes), "8bit"

load_tokenizer(model_name) → AutoTokenizer
# Sets padding_side="right", adds pad_token if missing
```

#### `backend/core/model/adapter.py`

```python
apply_lora(model, config: dict) → PeftModel
# Wraps model with LoraConfig (r, lora_alpha, dropout, target_modules, use_dora)

merge_and_save(peft_model, save_path)
# Calls model.merge_and_unload() then saves
```

#### `backend/core/trainer/base_trainer.py`

All trainers inherit from `BasePipelineTrainer`:

```python
class BasePipelineTrainer:
    def __init__(self, job_id, config, db_factory):
        self.callback = MetricLoggingCallback(job_id, db_factory)

    def train(self): ...  # implemented by subclass
```

`MetricLoggingCallback` writes a `TrainingMetric` row to the database after every logging step. The SSE endpoint polls this table and streams new rows to the browser.

#### `backend/core/data/template.py`

Supported prompt templates:

| Template | Format |
|----------|--------|
| `alpaca` | `### Instruction:\n{inst}\n\n### Response:\n{output}` |
| `sharegpt` | Alternating human/gpt conversation turns |
| `llama3` | `<|start_header_id|>user<|end_header_id|>...` |
| `mistral` | `[INST] {instruction} [/INST]` |
| `qwen` | `<|im_start|>user\n...` |
| `phi3` | `<|user|>\n...<|end|>` |
| `chatml` | `<|im_start|>system\n...<|im_end|>` |

#### `backend/core/asr/`

| File | Purpose |
|------|---------|
| `loader.py` | `load_whisper_processor()`, `load_whisper_model()` with language/task/forced_decoder_ids |
| `dataset.py` | `build_asr_dataset()` — CSV → librosa audio features + tokenized labels |
| `collator.py` | `DataCollatorSpeechSeq2SeqWithPadding` — pads audio + labels, strips leading BOS token |
| `metrics.py` | `make_compute_metrics()` — WER via `evaluate` + `jiwer` |
| `trainer.py` | `ASRPipelineTrainer` — orchestrates Seq2SeqTrainer with above components |

**Multilingual / code-mixed mode:**

When `language = None` or `language = "auto"`:
- `load_whisper_processor()` loads without language forcing
- `load_whisper_model()` sets `forced_decoder_ids = None` and `generation_config.language = None`
- Whisper auto-detects language per audio segment at inference time
- Text normalization in `dataset.py` preserves original casing (not lowercased)

---

### Database Schema

**`jobs`**

| Column | Type | Notes |
|--------|------|-------|
| id | Integer PK | |
| name | String | User-defined run name |
| status | String | `pending` / `running` / `completed` / `failed` / `cancelled` |
| training_method | String | `sft` / `dpo` / `kto` / `orpo` / `rm` / `unsupervised` / `asr_whisper` |
| peft_method | String | `lora` / `qlora` / `dora` / `full` / `sft` |
| model_id | FK → model_entries | Optional — null if model_path given directly |
| dataset_id | FK → datasets | Optional |
| config_json | JSON | Full training config dict |
| celery_task_id | String | For task revocation |
| output_dir | String | Path to training output |
| error_msg | Text | Set on failure |
| created_at / started_at / finished_at | DateTime | |

**`training_metrics`**

| Column | Type | Notes |
|--------|------|-------|
| job_id | FK → jobs | |
| step | Integer | Global training step |
| epoch | Float | |
| loss | Float | Training loss |
| eval_loss | Float | Validation loss |
| learning_rate | Float | |
| reward | Float | DPO reward OR WER (ASR) |
| grad_norm | Float | |
| timestamp | DateTime | |

**`model_entries`**

| Column | Type | Notes |
|--------|------|-------|
| name | String | Display name |
| hf_repo | String UNIQUE | HuggingFace repo ID |
| local_path | String | Absolute path after download |
| architecture | String | llama / mistral / qwen / phi3 / gemma / deepseek |
| template | String | Prompt template to use |
| is_downloaded | String | `"true"` / `"false"` |

**`datasets`**

| Column | Type | Notes |
|--------|------|-------|
| name | String | Display name |
| path | String | Absolute file path (CSV or JSONL) |
| format | String | `alpaca` / `sharegpt` / `plain_text` / `custom` / `asr_csv` |
| num_samples | Integer | Row count |
| description | Text | Optional |

The `format` column is the discriminator between LLM datasets and ASR datasets. `GET /api/datasets` excludes `asr_csv`; `GET /api/asr/datasets` filters to `asr_csv` only.

---

## Frontend

### Pages & Routing

All pages use Next.js 14 App Router with `"use client"`.

| Route | File | Description |
|-------|------|-------------|
| `/` | `app/page.tsx` | LLM training — full config form + metrics + log console |
| `/asr` | `app/asr/page.tsx` | Whisper ASR training |
| `/asr/datasets` | `app/asr/datasets/page.tsx` | ASR dataset upload (ZIP or CSV) |
| `/evaluate` | `app/evaluate/page.tsx` | Evaluate (loss/perplexity) + Predict |
| `/chat` | `app/chat/page.tsx` | Interactive chat / inference |
| `/export` | `app/export/page.tsx` | Merge adapters + list exported models |
| `/jobs` | `app/jobs/page.tsx` | All training jobs table |
| `/jobs/[id]` | `app/jobs/[id]/page.tsx` | Job detail with live Recharts metrics |
| `/models` | `app/models/page.tsx` | Model registry + HF Hub search + download |
| `/datasets` | `app/datasets/page.tsx` | Text dataset upload + format guide |

### Data Fetching

All API calls go through `frontend/lib/api.ts` (axios instance with `baseURL: "/api"`).

The Next.js proxy in `next.config.mjs` rewrites `/api/*` → `http://localhost:8000/api/*`.

React Query is used for all server state:

```typescript
// Polling example
useQuery({ queryKey: ["jobs"], queryFn: getJobs, refetchInterval: 5000 })

// Mutation example
useMutation({ mutationFn: createJob, onSuccess: () => qc.invalidateQueries(...) })
```

### Real-Time Streaming

Two SSE patterns are used:

**1. Training metrics** (`frontend/lib/sse.ts`):

```typescript
export function useMetricsStream(jobId: number | null): Metric[] {
  // Opens EventSource to /api/jobs/{id}/metrics
  // Each event is a JSON Metric row
  // Accumulates into state, returns array
}
```

**2. Eval logs & chat tokens** (raw `fetch` + `ReadableStream`):

```typescript
// Used in /evaluate and /chat pages
const resp = await fetch("/api/eval/{run_id}/stream");
const reader = resp.body.getReader();
// Read chunks, parse SSE lines, update state
```

**3. Chat token streaming** uses `fetch` POST with `TextIteratorStreamer` on the backend:

```python
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
Thread(target=model.generate, kwargs={..., "streamer": streamer}).start()
# SSE generator yields each token from streamer
```

### Design System

`frontend/app/globals.css` defines a complete `lf-*` CSS class system:

```css
/* Core variables */
--bg: #0d0f14          /* page background */
--bg-panel: #11141b    /* panel/card background */
--bg-input: #0a0c11    /* input field background */
--accent: #4a9eff      /* primary blue */
--green: #3ddc84       /* success / running */
--amber: #f59e0b       /* warning / pending */
--red: #ef4444         /* error / danger */
--mono: 'JetBrains Mono', monospace
```

Key utility classes:

| Class | Usage |
|-------|-------|
| `.lf-panel` | Dark bordered container |
| `.lf-input` | Input field / select |
| `.lf-btn` | Button base |
| `.lf-btn-primary` | Blue accent button |
| `.lf-btn-danger` | Red destructive button |
| `.lf-btn-ghost` | Transparent outline button |
| `.lf-chip` | Toggle chip (method/model selector) |
| `.lf-chip-active` | Selected chip state |
| `.lf-table` | Dense data table |
| `.lf-console` | Monospace log output panel |
| `.lf-section` | Section header label |
| `.lf-label` | Form field label |
| `.lf-toggle` | Checkbox toggle switch |
| `.lf-tab` | Navigation tab link |
| `.lf-tab-active` | Active tab state |
| `.lf-spin` | Loading spinner animation |
| `.lf-badge` | Status/format badge |
| `.lf-row lf-row-2/3/4` | Responsive grid rows |

---

## API Reference

### Jobs

```
POST   /api/jobs
GET    /api/jobs
GET    /api/jobs/{id}
DELETE /api/jobs/{id}
GET    /api/jobs/{id}/metrics   (SSE)
```

**POST /api/jobs body:**
```json
{
  "name": "my-sft-run",
  "training_method": "sft",
  "peft_method": "lora",
  "model_id": 1,
  "dataset_id": 2,
  "config": {
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "batch_size": 4,
    "lora_r": 16,
    "lora_alpha": 32,
    "output_dir": "./outputs/my-sft-run"
  }
}
```

**SSE event format** (`/metrics`):
```json
{ "step": 100, "epoch": 1.2, "loss": 1.43, "eval_loss": 1.51, "learning_rate": 1.8e-4, "reward": null, "grad_norm": 0.82 }
```

### Models

```
GET    /api/models
POST   /api/models
POST   /api/models/{id}/download
GET    /api/models/search/hub?q=llama
DELETE /api/models/{id}
```

### Datasets

```
GET    /api/datasets              (excludes asr_csv)
POST   /api/datasets              (multipart: name, format, description, file)
DELETE /api/datasets/{id}
```

### ASR

```
GET    /api/asr/models
GET    /api/asr/datasets
POST   /api/asr/datasets          (CSV manifest)
POST   /api/asr/datasets/zip      (ZIP with audio + CSV; rewrites audio paths)
DELETE /api/asr/datasets/{id}
POST   /api/asr/jobs
GET    /api/asr/jobs
GET    /api/asr/jobs/{id}
DELETE /api/asr/jobs/{id}
GET    /api/asr/jobs/{id}/metrics (SSE)
```

**POST /api/asr/datasets/zip** (multipart):
- `name` — dataset name
- `audio_col` — CSV column name for audio paths (default: `audio_path`)
- `text_col` — CSV column name for transcripts (default: `text`)
- `file` — `.zip` file

The ZIP must contain at least one `.csv` file and audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`). Audio paths in the CSV are matched by filename — the original machine paths do not need to match.

### Evaluate

```
POST /api/eval/run
GET  /api/eval/{run_id}/result
GET  /api/eval/{run_id}/stream    (SSE log lines)
```

**POST /api/eval/run body:**
```json
{
  "model_path": "./models/llama3-8b",
  "adapter_path": "./outputs/run1/final_adapter",
  "dataset_id": 3,
  "mode": "evaluate",
  "batch_size": 4,
  "max_seq_len": 2048
}
```

**Result (evaluate mode):**
```json
{ "status": "completed", "loss": 1.23, "perplexity": 3.42 }
```

**Result (predict mode):**
```json
{ "status": "completed", "output_file": "./outputs/predict_abc123.jsonl", "num_samples": 500 }
```

### Chat

```
POST /api/chat/load
GET  /api/chat/status
POST /api/chat/unload
POST /api/chat/generate           (SSE token stream)
```

**POST /api/chat/load body:**
```json
{ "model_path": "meta-llama/Meta-Llama-3-8B-Instruct", "quantization": "4bit" }
```

**GET /api/chat/status response:**
```json
{ "status": "ready", "model_path": "...", "adapter_path": null, "error": null }
```

**POST /api/chat/generate body:**
```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello!" }
  ],
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1
}
```

**SSE token stream:** Each event is `{"token": "Hello"}`. Stream ends with `{"token": "__done__"}`.

### Exports

```
GET  /api/exports
POST /api/exports/{job_id}        (body: { "output_name": "..." })
POST /api/exports/from-path       (body: { "adapter_path": "...", "output_name": "..." })
```

---

## Configuration

### Training Config Keys (LLM)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `learning_rate` | float | 2e-4 | AdamW learning rate |
| `num_epochs` | int | 3 | Training epochs |
| `batch_size` | int | 4 | Per-device train batch |
| `gradient_accumulation_steps` | int | 4 | Grad accumulation |
| `lr_scheduler` | string | `cosine` | cosine / linear / constant |
| `warmup_ratio` | float | 0.05 | Warmup fraction |
| `max_grad_norm` | float | 1.0 | Gradient clipping |
| `lora_r` | int | 16 | LoRA rank |
| `lora_alpha` | int | 32 | LoRA alpha (scaling) |
| `lora_dropout` | float | 0.05 | LoRA dropout |
| `target_modules` | list | model-specific | Which linear layers to adapt |
| `max_seq_length` | int | 2048 | Maximum token length |
| `bf16` | bool | true | bfloat16 precision |
| `fp16` | bool | false | float16 precision |
| `gradient_checkpointing` | bool | true | Memory saving |
| `output_dir` | string | `./outputs/...` | Checkpoint save path |

### Training Config Keys (ASR)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model_path` | string | — | Whisper model ID or local path |
| `language` | string | `null` | `null`/`"auto"` for multilingual, or `"malay"`, `"english"`, etc. |
| `task` | string | `transcribe` | `transcribe` or `translate` |
| `training_method` | string | `lora` | `sft` / `lora` / `qlora` |
| `max_steps` | int | 3000 | Training steps (use_max_steps=true) |
| `warmup_steps` | int | 500 | Warmup steps |
| `predict_with_generate` | bool | true | WER evaluation using generation |
| `generation_max_length` | int | 225 | Max decode length |
| `eval_steps` | int | 500 | Eval every N steps |

---

## CLI Reference

```bash
# Train a model from the command line (no UI required)
python -m forge train \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --method sft \
  --dataset ./data/alpaca.json \
  --lora-rank 16 \
  --epochs 3 \
  --output-dir ./outputs/run1

# Evaluate a model
python -m forge eval \
  --model ./outputs/run1 \
  --dataset ./data/eval.json \
  --batch-size 4

# Export / merge a LoRA adapter
python -m forge export \
  --adapter ./outputs/run1/final_adapter \
  --output ./exports/merged_model
```

---

## Adding a New Training Method

1. Create `backend/core/trainer/my_trainer.py` extending `BasePipelineTrainer`
2. Implement `train()` using a TRL trainer; the `MetricLoggingCallback` (from `self.callback`) will handle DB writes automatically
3. Register in `backend/workers/training_worker.py` inside `_get_trainer()`
4. Add the method name to the `METHODS` constant in `frontend/app/page.tsx`
