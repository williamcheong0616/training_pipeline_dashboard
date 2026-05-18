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
- SSE generators own their own `SessionLocal()` DB session (separate from the request-scoped session) and use `db.query()` instead of `db.get()` to bypass the SQLAlchemy identity map cache — ensuring fresh reads of job status updated by a remote Celery worker.
- CORS allowed origins are read from `FRONTEND_URL` env var and support comma-separated values for multi-origin deployments.

---

## Backend

### API Layer

**Entry point:** `backend/api/main.py`

```python
app = FastAPI(title="Training Pipeline API", version="1.0.0")
# CORS: origins from FRONTEND_URL env var (comma-separated)
# Routers: jobs, models, datasets, exports, asr, eval, chat
# Lifespan: runs init_db() on startup (creates tables + additive indexes)
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

**Job creation race-condition fix:**
`create_job` uses `db.flush()` to obtain the primary key before the Celery task is enqueued. If `run_training_job.delay()` raises (e.g. Redis unavailable), the transaction is rolled back. Only after a successful `.delay()` call is the job committed.

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

`MetricLoggingCallback` writes a `TrainingMetric` row to the database after every logging step. The SSE endpoint polls this table (using `db.query()` for cache-bypassing reads) and streams new rows to the browser.

#### `backend/core/data/template.py`

Supported prompt templates (used at tokenisation time, and optionally baked in at format conversion):

| Template | Model family | Human prefix |
|----------|-------------|--------------|
| `alpaca` | General | `### Instruction:\n` |
| `chatml` | Qwen, InternLM | `<\|im_start\|>user\n` |
| `llama3` | Meta-Llama-3 | `<\|start_header_id\|>user<\|end_header_id\|>\n\n` |
| `mistral` | Mistral, Mixtral | `[INST] ` |
| `qwen` | Qwen2 | `<\|im_start\|>user\n` |
| `phi3` | Phi-3 | `<\|user\|>\n` |
| `gemma` | Gemma | `<start_of_turn>user\n` |

#### `backend/core/data/detector.py` *(new)*

Server-side format detection by scoring up to 20 sampled records:

```python
def detect_format(records: list[dict]) -> dict:
    # Returns { "format": str, "confidence": "high"|"medium"|"low", "scores": dict }
```

Scoring rules (cumulative per record):
- `conversations`/`messages` key with list-of-dicts → +3 **sharegpt**
- `prompt` + `chosen` + `rejected` → +5 **dpo**
- `prompt` + `completion` + `label` → +5 **kto**
- `instruction` or `output` → +2 **alpaca**; also has `input` → +1 **alpaca**
- Only `text` key (≤ 3 total keys) → +2 **plain_text**

Confidence: `high` ≥ 70% share of total score, `medium` ≥ 40%, else `low`.

#### `backend/core/data/converter.py` *(new)*

Format-to-format conversion with a defined compatibility matrix:

| Source | Valid targets |
|--------|--------------|
| `alpaca` | `sharegpt`, `plain_text` |
| `sharegpt` | `alpaca` (first turn), `plain_text` |
| `dpo` | `sharegpt` (chosen only), `alpaca` (chosen only) |
| `kto` | `sharegpt` (label=True only), `alpaca` (label=True only) |
| `plain_text` | `alpaca` (text → instruction) |

When converting to `plain_text`, a `PromptTemplate` is applied and the formatted text is baked into the `{"text": "…"}` field. This is the only case where the template choice matters at conversion time.

#### `backend/core/data/dataset.py`

```python
_load_raw(path_or_repo, format) → list[dict]   # handles .json, .jsonl, HF Hub
_alpaca_to_text(row, template) → str
_sharegpt_to_text(row, template) → str
_plain_text(row) → str
build_dataset(path, format, template_name, tokenizer, max_length) → Dataset
build_plain_text_dataset(path, tokenizer, max_length) → Dataset
```

#### `backend/core/asr/`

| File | Purpose |
|------|---------|
| `loader.py` | `load_whisper_processor()`, `load_whisper_model()` with language/task/forced_decoder_ids |
| `dataset.py` | `build_asr_dataset()` — CSV → librosa audio features + tokenized labels |
| `collator.py` | `DataCollatorSpeechSeq2SeqWithPadding` — pads audio + labels, strips leading BOS token |
| `metrics.py` | `make_compute_metrics()` — WER via `evaluate` + `jiwer` |
| `trainer.py` | `ASRPipelineTrainer` — orchestrates Seq2SeqTrainer with above components |

**Multilingual / code-mixed mode:** When `language = None` or `"auto"`, Whisper auto-detects language per audio segment. `forced_decoder_ids` is set to `None` and text normalization preserves original casing.

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
| model_id | FK → model_entries | Optional |
| dataset_id | FK → datasets | Optional |
| config_json | JSON | Full training config dict |
| celery_task_id | String | For task revocation |
| output_dir | String | Path to training output |
| error_msg | Text | Set on failure |
| created_at / started_at / finished_at | DateTime | |

**`training_metrics`**

| Column | Type | Notes |
|--------|------|-------|
| job_id | FK → jobs | Indexed (`ix_training_metrics_job_id`) |
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
| path | String | Absolute file path (CSV, JSON, or JSONL) |
| format | String | `alpaca` / `sharegpt` / `dpo` / `kto` / `plain_text` / `asr_csv` |
| template | String | Chat template the data was created with (used for auto-fill in training form) |
| num_samples | Integer | Row count |
| description | Text | Optional — auto-filled with conversion provenance for converted datasets |

The `format` column discriminates LLM vs ASR datasets. `GET /api/datasets` excludes `asr_csv`; `GET /api/asr/datasets` filters to `asr_csv` only.

Selecting a dataset in the LLM training form auto-maps `dataset_format` from `d.format` and `template` from `d.template`, with an "auto" badge on both fields.

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
| `/chat` | `app/chat/page.tsx` | Interactive chat / inference with Markdown output |
| `/export` | `app/export/page.tsx` | Merge adapters + list exported models |
| `/jobs` | `app/jobs/page.tsx` | All training jobs table |
| `/jobs/[id]` | `app/jobs/[id]/page.tsx` | Job detail with live Recharts metrics |
| `/models` | `app/models/page.tsx` | Model registry + HF Hub search + download |
| `/datasets` | `app/datasets/page.tsx` | Dataset upload (auto-detect), preview, and format conversion |

### Data Fetching

All API calls go through `frontend/lib/api.ts` (axios instance with `baseURL: "/api"`). The Next.js proxy in `next.config.mjs` rewrites `/api/*` → `http://localhost:8000/api/*`.

React Query manages all server state:

```typescript
// Polling example
useQuery({ queryKey: ["jobs"], queryFn: getJobs, refetchInterval: 5000 })

// Mutation example
useMutation({ mutationFn: createJob, onSuccess: () => qc.invalidateQueries(...) })
```

### Real-Time Streaming

**1. Training metrics** (`frontend/lib/sse.ts`):
```typescript
export function useMetricsStream(jobId: number | null): Metric[]
// Opens EventSource to /api/jobs/{id}/metrics
// Accumulates JSON Metric rows into state
```

**2. Eval logs** — raw `fetch` + `ReadableStream`, SSE lines parsed manually. EventSource stored in a `useRef` and closed on component unmount to prevent leaks.

**3. Chat token streaming** — `fetch` POST with `TextIteratorStreamer` on the backend:
```python
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
Thread(target=model.generate, kwargs={..., "streamer": streamer}).start()
# SSE generator yields each token from streamer
```
The frontend streams tokens into a `MarkdownMessage` component rendered with `react-markdown` + `remark-gfm` + `rehype-highlight` (github-dark theme), so code blocks, tables, and bold/italic in assistant responses render correctly.

### Design System

`frontend/app/globals.css` defines a complete `lf-*` CSS class system with light/dark theme via `data-theme` on `<html>`.

**CSS variables (dark defaults):**
```css
--bg: #0d0f14          /* page background */
--bg-panel: #13161e    /* panel/card background */
--bg-input: #1a1d27    /* input field background */
--bg-hover: #1f2333    /* hover state */
--border: #252a3a      /* default border */
--border-hi: #353c52   /* highlighted border */
--text: #c8ccd8        /* body text */
--text-dim: #5e6478    /* muted text */
--text-hi: #e8eaf0     /* high-emphasis text */
--accent: #4a9eff      /* primary blue */
--accent-dim: #1e3a5c  /* accent background tint */
--green: #3dd68c       /* success */
--amber: #e8a820       /* warning */
--red: #f05050         /* error */
--mono: 'JetBrains Mono', monospace
--sans: 'IBM Plex Sans', sans-serif
```

**Key utility classes:**

| Class | Usage |
|-------|-------|
| `.lf-panel` | Dark bordered container |
| `.lf-input` / `.lf-select` / `.lf-textarea` | Form controls |
| `.lf-label` | Form field label (monospace, dimmed) |
| `.lf-btn` | Button base |
| `.lf-btn-primary` | Blue accent button |
| `.lf-btn-danger` | Red destructive button |
| `.lf-btn-ghost` | Transparent outline button |
| `.lf-btn-success` | Green confirm button |
| `.lf-chip` | Toggle chip (method/module selector) |
| `.lf-chip-active` | Selected chip state |
| `.lf-table` | Dense data table |
| `.lf-console` | Monospace log output panel |
| `.lf-section` | Section header label |
| `.lf-toggle` | Checkbox toggle switch |
| `.lf-tab` / `.lf-tab-active` | Navigation tab links |
| `.lf-spin` | Loading spinner animation |
| `.lf-badge` | Status/format badge |
| `.lf-row lf-row-2/3/4` | CSS grid form rows |
| `.lf-progress-track` / `.lf-progress-fill` | Progress bar |
| `.lf-tt-wrap` / `.lf-tt-icon` / `.lf-tt-box` | Tooltip — CSS-only hover popover |
| `.lf-train-layout` / `.lf-train-config` / `.lf-train-output` | Two-column training layout |

**Tooltip system** (`frontend/components/Tooltip.tsx`):

The `Tooltip` component renders a `?` icon with a CSS-only hover popover (no JS positioning). It is used on every training parameter label, section header, and chip button across the LLM and ASR pages. To attach a tooltip to a chip button, wrap the button in `.lf-tt-wrap` with `style={{ marginLeft: 0 }}` and add a `.lf-tt-box` sibling:

```tsx
<span className="lf-tt-wrap" style={{ marginLeft: 0 }}>
  <button className="lf-chip">q_proj</button>
  <span className="lf-tt-box">Query projection — …</span>
</span>
```

**Training page layout** (`.lf-train-layout`):

Two-column CSS grid: left config panel scrolls naturally with the page; right output panel is `position: sticky` pinned below the 40px TopNav. This avoids fragmented independent scroll zones. Responsive breakpoints at 1100px (narrow side-by-side) and 820px (stacked).

---

## API Reference

### Jobs

```
POST   /api/jobs
GET    /api/jobs
GET    /api/jobs/{id}
DELETE /api/jobs/{id}              (cancel — only running jobs)
DELETE /api/jobs/{id}/purge        (delete record — only terminal jobs)
PATCH  /api/jobs/{id}/remarks      (body: { remarks })
GET    /api/jobs/{id}/metrics      (SSE)
GET    /api/jobs/{id}/metrics/all  (snapshot of all metric rows)
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
GET    /api/models/{id}/download-status
GET    /api/models/search/hub?q=llama
DELETE /api/models/{id}
```

### Datasets

```
GET    /api/datasets                     (excludes asr_csv)
POST   /api/datasets                     (multipart: name, format="auto", description, file)
GET    /api/datasets/{id}/preview        → { format, total, samples, valid_targets, conversion_notes }
POST   /api/datasets/{id}/convert        (body: { target_format, template_name, output_name })
DELETE /api/datasets/{id}
```

**POST /api/datasets** — format detection:
- Pass `format=auto` (the default) to let the server detect the format from file content.
- The response includes `detected_format` and `detection_confidence` ("high" / "medium" / "low").
- Pass an explicit format to override detection.

**POST /api/datasets/{id}/convert** — format conversion:
```json
{
  "target_format": "sharegpt",
  "template_name": "alpaca",
  "output_name": "my_dataset_as_sharegpt"
}
```
- `template_name` is only relevant when `target_format = "plain_text"` — it controls which chat template is baked into the `text` field.
- Returns a new `DatasetResponse` for the converted dataset (saved as a new JSONL file).
- Returns 400 if the source→target conversion is not supported.

**GET /api/datasets/{id}/preview** response:
```json
{
  "format": "alpaca",
  "total": 5000,
  "samples": [...],
  "valid_targets": ["sharegpt", "plain_text"],
  "conversion_notes": {
    "sharegpt": "Wraps each sample as a two-turn conversation (human/gpt).",
    "plain_text": "Applies the chosen chat template and bakes it into a text field."
  }
}
```

### ASR

```
GET    /api/asr/models
GET    /api/asr/datasets
POST   /api/asr/datasets          (CSV manifest)
POST   /api/asr/datasets/zip      (ZIP with audio + CSV; rewrites audio paths)
DELETE /api/asr/datasets/{id}
POST   /api/asr/jobs
GET    /api/asr/jobs?skip=0&limit=50
GET    /api/asr/jobs/{id}
DELETE /api/asr/jobs/{id}
GET    /api/asr/jobs/{id}/metrics (SSE)
```

**POST /api/asr/datasets/zip** (multipart):
- `name` — dataset name
- `audio_col` — CSV column name for audio paths (default: `audio_path`)
- `text_col` — CSV column name for transcripts (default: `text`)
- `file` — `.zip` file

The ZIP must contain at least one `.csv` and audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`). Audio paths in the CSV are matched by filename — original machine paths are rewritten automatically.

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

Predict output files default to `OUTPUTS_DIR/predict_<run_id[:8]>.jsonl` (configurable via `OUTPUTS_DIR` env var).

### Chat

```
POST /api/chat/load
GET  /api/chat/status
POST /api/chat/unload
POST /api/chat/generate           (SSE token stream)
```

**POST /api/chat/generate** SSE: each event is `{"token": "Hello"}`. Stream ends with `{"token": "__done__"}`. The frontend renders streamed tokens through `react-markdown` so Markdown in model output is formatted.

### Exports

```
GET  /api/exports
POST /api/exports/{job_id}        (body: { "output_name": "..." })
POST /api/exports/from-path       (body: { "adapter_path": "...", "output_name": "..." })
```

Both export endpoints validate the output name for path traversal (no `../` or absolute paths).

---

## Configuration

### Training Config Keys (LLM)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `learning_rate` | float | 2e-4 | AdamW learning rate |
| `num_epochs` | int | 3 | Training epochs |
| `batch_size` | int | 4 | Per-device train batch |
| `gradient_accumulation_steps` | int | 4 | Grad accumulation |
| `lr_scheduler` | string | `cosine` | cosine / linear / constant / cosine_with_restarts / polynomial |
| `warmup_ratio` | float | 0.05 | Warmup fraction of total steps |
| `max_grad_norm` | float | 1.0 | Gradient clipping |
| `lora_r` | int | 16 | LoRA rank |
| `lora_alpha` | int | 32 | LoRA alpha (scaling = alpha/r) |
| `lora_dropout` | float | 0.05 | LoRA dropout |
| `target_modules` | list | model-specific | Which linear layers to adapt |
| `max_seq_length` | int | 2048 | Maximum token length (truncation) |
| `bf16` | bool | true | bfloat16 precision |
| `fp16` | bool | false | float16 precision |
| `gradient_checkpointing` | bool | true | Trade speed for VRAM |
| `output_dir` | string | `./outputs/...` | Checkpoint save path |

### Training Config Keys (ASR)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model_path` | string | — | Whisper model ID or local path |
| `language` | string | `null` | `null`/`"auto"` = multilingual auto-detect; or `"malay"`, `"english"`, etc. |
| `task` | string | `transcribe` | `transcribe` or `translate` |
| `training_method` | string | `lora` | `sft` / `lora` / `qlora` |
| `max_steps` | int | 3000 | Training steps (when use_max_steps=true) |
| `warmup_steps` | int | 500 | Warmup steps |
| `predict_with_generate` | bool | true | WER evaluation using generation |
| `generation_max_length` | int | 225 | Max decode length |
| `eval_steps` | int | 500 | Eval every N steps |
| `load_best_model_at_end` | bool | true | Save checkpoint with best WER |

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
2. Implement `train()` using a TRL trainer; `self.callback` (`MetricLoggingCallback`) handles DB writes automatically
3. Register in `backend/workers/training_worker.py` inside `_get_trainer()`
4. Add the method name to the `METHODS` constant in `frontend/app/page.tsx`
5. Add a tooltip description to `METHOD_TIPS` in `frontend/app/page.tsx`

## Adding a New Dataset Format

1. Add detection rules in `backend/core/data/detector.py` (`_score_records`)
2. Add conversion functions in `backend/core/data/converter.py` and register in `VALID_TARGETS`
3. Add the format string to `FORMATS` in `frontend/app/datasets/page.tsx` upload panel
