# Forge — System Flow Documentation

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (Next.js)                    │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │  LLM /   │   ASR /  │ Evaluate │   Chat   │  Export  │  │
│  │ Training │ Training │  /eval   │  /chat   │ /export  │  │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘  │
│       │ REST/SSE │          │          │          │         │
└───────┼──────────┼──────────┼──────────┼──────────┼─────────┘
        │          │          │          │          │
        ▼          ▼          ▼          ▼          ▼
┌───────────────────────────────────────────────────────────┐
│              FastAPI Backend  (uvicorn)                   │
│  /api/jobs  /api/asr  /api/eval  /api/chat  /api/exports  │
│  /api/models  /api/datasets  /api/system                  │
└──────────┬───────────────────────────┬────────────────────┘
           │                           │
     ┌─────▼─────┐              ┌──────▼──────┐
     │  SQLite / │              │  Celery     │
     │  Postgres │              │  Worker     │
     │  (ORM)    │              │  (Redis)    │
     └─────┬─────┘              └──────┬──────┘
           │                           │
           └──────────┬────────────────┘
                      │
               ┌──────▼──────┐
               │  HuggingFace│
               │  Trainer    │
               │  (GPU)      │
               └─────────────┘
```

### Component Responsibilities

| Component | Role |
|-----------|------|
| **Next.js frontend** | SPA with React Query for data fetching; SSE for real-time metrics/tokens |
| **FastAPI** | REST API + SSE endpoints; request validation; background task dispatch |
| **SQLite/Postgres** | Persistent store for jobs, models, datasets, metrics |
| **Celery + Redis** | Async task queue for long-running training jobs |
| **HuggingFace Trainer** | Actual model training (runs inside the Celery worker process) |

---

## 2. Training Job Lifecycle

```mermaid
flowchart TD
    A([User fills config form]) --> B[POST /api/jobs]
    B --> C{"Validate request"}
    C -- invalid --> D[400 Bad Request]
    C -- valid --> E["Create Job row — status=pending"]
    E --> F["run_training_job.delay(job_id) — Celery enqueue"]
    F --> G[Return Job JSON to frontend]
    G --> H["Frontend opens SSE stream\nGET /api/jobs/id/metrics"]

    subgraph CeleryWorker["Celery Worker"]
        I[Pick up task] --> J["Set status=running, started_at=now"]
        J --> K{"training_method"}
        K -- "sft / dpo / lora" --> L["LLM Trainer\nSFTTrainer / DPOTrainer"]
        K -- asr_whisper --> M["Whisper Trainer\nSeq2SeqTrainer"]
        L --> N["INSERT TrainingMetric rows\nevery logging_steps"]
        M --> N
        N --> O{"Training done"}
        O -- success --> P["status=completed, output_dir saved"]
        O -- error --> Q["status=failed, error_msg saved"]
        O -- revoked --> R[status=cancelled]
    end

    H --> S["SSE polls DB every 2s\nyields metric events"]
    S --> T{"Job terminal"}
    T -- yes --> U["yield done event, close stream"]
    T -- no --> S

    P --> V([User can Export or Evaluate])
```

---

## 3. LLM Training Flow (detailed)

```mermaid
sequenceDiagram
    participant UI as Frontend
    participant API as FastAPI
    participant DB as SQLite/PG
    participant Q as Redis/Celery
    participant W as Worker

    UI->>API: POST /api/jobs
    API->>DB: INSERT Job status=pending
    API->>Q: run_training_job.delay(job_id)
    API-->>UI: 201 Job created

    UI->>API: GET /api/jobs/id/metrics (SSE)
    API-->>UI: SSE stream opened

    Q->>W: execute task
    W->>DB: UPDATE status=running, started_at
    W->>W: load_tokenizer(model_path)
    W->>W: load_model(model_path, quant)
    W->>W: wrap with PeftModel LoRA/QLoRA
    W->>W: load dataset CSV/JSONL
    W->>W: SFTTrainer.train()

    loop Every logging_steps
        W->>DB: INSERT TrainingMetric (loss, lr, grad_norm)
    end

    W->>DB: UPDATE status=completed, output_dir

    API->>DB: SELECT new metrics every 2s
    API-->>UI: SSE metric events
    API-->>UI: SSE event done
```

---

## 4. ASR Training Flow

```mermaid
flowchart LR
    A(["Upload ZIP or CSV"]) --> B{"Upload type"}

    subgraph ZipUpload["ZIP Upload"]
        B -- ZIP --> C["Extract to datasets/audio/name/"]
        C --> D["Zip-slip guard:\nrealpath prefix check"]
        D --> E[Find CSV in extracted dir]
        E --> F["Rewrite audio paths\nby filename match"]
        F --> G["Save canonical CSV\nto datasets/"]
    end

    subgraph CsvUpload["CSV Upload"]
        B -- CSV --> H["Save CSV to datasets/"]
    end

    G --> I[("Dataset DB row\nformat=asr_csv")]
    H --> I

    I --> J(["Create ASR Job\nPOST /api/asr/jobs"])
    J --> K["Celery: run_training_job"]

    subgraph ASRWorker["ASR Worker"]
        K --> L[load WhisperProcessor]
        L --> M[load WhisperForConditionalGeneration]
        M --> N[wrap PeftModel LoRA]
        N --> O[Read CSV rows]
        O --> P["librosa.load each audio"]
        P --> Q[Seq2SeqTrainer.train]
        Q --> R[TrainingMetric rows]
    end

    R --> S(["Completed adapter\nin output_dir/"])
```

---

## 5. Dataset Upload Flow

```mermaid
flowchart TD
    subgraph TextDataset["Text Dataset — JSON / JSONL"]
        A([Select file]) --> B[Client reads first 3 records]
        B --> C{"Auto-detect format"}
        C -- "conversations / messages" --> D[sharegpt]
        C -- "prompt + chosen + rejected" --> E[dpo]
        C -- "prompt + completion + label" --> F[kto]
        C -- "instruction / output" --> G[alpaca]
        C -- other --> H[plain_text]
        D --> I["POST /api/datasets\nmultipart form"]
        E --> I
        F --> I
        G --> I
        H --> I
        I --> J[("Dataset row\nformat detected")]
    end

    subgraph ASRZip["ASR Dataset — ZIP"]
        K([Select ZIP]) --> L["POST /api/asr/datasets/zip"]
        L --> M[Read all bytes into memory]
        M --> N["Validate each member path\nzip-slip guard"]
        N --> O["Validate CSV exists\nbefore extracting"]
        O --> P["extractall to datasets/audio/name/"]
        P --> Q[Build audio filename lookup]
        Q --> R["Rewrite CSV audio_col\nwith absolute paths"]
        R --> S[("Dataset row\nformat=asr_csv")]
    end

    subgraph ASRCsv["ASR Dataset — CSV"]
        T(["Select CSV\nwith server paths"]) --> U["POST /api/asr/datasets"]
        U --> V[("Dataset row\nformat=asr_csv")]
    end
```

---

## 6. Evaluate & Predict Flow

```mermaid
flowchart TD
    A(["User selects model + dataset + mode"]) --> B{"is_asr"}

    B -- "No, LLM" --> C["POST /api/eval/run\nis_asr=false"]
    C --> D["BackgroundTask: _run_eval"]
    D --> E["load_tokenizer + load_model"]
    E --> F{"mode"}
    F -- evaluate --> G["DataLoader loop\navg loss + perplexity"]
    F -- predict --> H["model.generate per sample\nwrite JSONL output"]
    G --> I[("result store\nloss, perplexity")]
    H --> I

    B -- "Yes, ASR" --> J["POST /api/eval/run\nis_asr=true"]
    J --> K["BackgroundTask: _run_asr_eval"]
    K --> L["load WhisperProcessor + Model"]
    L --> M["apply PeftModel if adapter"]
    M --> N{"mode"}
    N -- evaluate --> O["librosa.load each audio\nmodel.generate, decode\nevaluate.load WER"]
    N -- predict --> P["generate predictions\nwrite JSONL"]
    O --> Q[("result store\nWER score")]
    P --> Q

    I --> R["GET /api/eval/run_id/result"]
    Q --> R
    R --> S([Frontend shows metrics])
```

---

## 7. Chat Flow

```mermaid
sequenceDiagram
    participant UI as Frontend
    participant API as FastAPI
    participant State as chat_state
    participant GPU as Model/GPU

    UI->>API: POST /api/chat/load
    API->>State: status = loading
    API->>GPU: load_tokenizer + load_model (BackgroundTask)
    GPU-->>State: model stored, status = ready

    loop Poll every 2s
        UI->>API: GET /api/chat/status
        API-->>UI: status loading or ready
    end

    UI->>API: POST /api/chat/generate
    API->>GPU: model.generate with TextIteratorStreamer

    loop Token streaming
        GPU-->>API: next token via streamer
        API-->>UI: SSE data token
    end

    API-->>UI: SSE data __done__

    UI->>API: POST /api/chat/unload
    API->>State: model = None, tokenizer = None
    API->>GPU: gc.collect + cuda.empty_cache
    API-->>UI: unloaded
```

---

## 8. Export (Merge) Flow

```mermaid
flowchart TD
    A([Choose source]) --> B{"Source type"}

    B -- "From completed Job" --> C["POST /api/exports/job_id"]
    C --> D["Validate job.status == completed"]
    D --> E["Validate job.output_dir exists"]

    B -- "From adapter path" --> F["POST /api/exports/from-path"]
    F --> G["Validate adapter_path is dir"]

    E --> H["_safe_output_name\nstrip dangerous chars, cap 128"]
    G --> H
    H --> I["_validated_save_path\nos.path.realpath prefix check"]
    I --> J{"Path escapes EXPORTS_DIR"}
    J -- Yes --> K[400 Bad Request]
    J -- No --> L["BackgroundTask: _merge_adapter"]

    subgraph MergeAdapter["_merge_adapter"]
        L --> M["Read adapter_config.json"]
        M --> N{"base_model contains whisper"}
        N -- Yes --> O["WhisperForConditionalGeneration\n+ WhisperProcessor"]
        N -- No --> P["AutoModelForCausalLM\n+ AutoTokenizer"]
        O --> Q["PeftModel.from_pretrained(adapter_path)"]
        P --> Q
        Q --> R[model.merge_and_unload]
        R --> S["save_pretrained to EXPORTS_DIR/name"]
    end

    S --> T(["GET /api/exports\nlists merged dirs"])
```

---

## 9. Security Controls Summary

| Attack Vector | Location | Mitigation |
|---------------|----------|------------|
| **Path traversal** in `output_name` | `exports.py` | `_safe_output_name()` strips dangerous chars; `_validated_save_path()` uses `realpath()` prefix check |
| **Zip-slip** in ZIP upload | `asr.py` | Every member path checked with `realpath` before `extractall` |
| **SSE DB session leak** | `jobs.py`, `asr.py` | Generator opens its own `SessionLocal()` with `finally: db.close()` |
| **Stale localStorage value** | `ThemeProvider.tsx` | Explicit equality check: `stored === "light" \|\| stored === "dark"` |
| **Chat fetch not cleaned up** on unmount | `chat/page.tsx` | `useEffect` cleanup aborts in-flight request via `AbortController` |
| **Silent chat errors** | `chat/page.tsx` | Non-2xx response body shown as `[Error: …]` assistant message |

---

## 10. Database Schema

```
jobs
├── id              INTEGER PK
├── name            TEXT
├── status          TEXT  (pending / running / completed / failed / cancelled)
├── training_method TEXT  (sft / dpo / kto / asr_whisper / …)
├── peft_method     TEXT  (lora / qlora / dora / full)
├── model_id        FK → models.id
├── dataset_id      FK → datasets.id
├── config_json     JSON
├── output_dir      TEXT
├── error_msg       TEXT
├── celery_task_id  TEXT
├── remarks         TEXT  (nullable — user notes)
├── created_at      DATETIME
├── started_at      DATETIME
└── finished_at     DATETIME

training_metrics
├── id              INTEGER PK
├── job_id          FK → jobs.id (CASCADE DELETE)
├── step            INTEGER
├── epoch           FLOAT
├── loss            FLOAT
├── eval_loss       FLOAT
├── learning_rate   FLOAT
├── reward          FLOAT
├── grad_norm       FLOAT
└── timestamp       DATETIME

models
├── id              INTEGER PK
├── name            TEXT
├── hf_repo         TEXT
├── architecture    TEXT
├── template        TEXT
├── is_downloaded   BOOL
├── local_path      TEXT
├── downloaded_at   DATETIME
└── created_at      DATETIME

datasets
├── id              INTEGER PK
├── name            TEXT
├── path            TEXT
├── format          TEXT  (alpaca / sharegpt / dpo / kto / plain_text / asr_csv)
├── num_samples     INTEGER
├── description     TEXT
└── created_at      DATETIME
```

---

## 11. API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/jobs` | List all LLM jobs |
| POST | `/api/jobs` | Create LLM training job |
| DELETE | `/api/jobs/{id}` | Cancel job (Celery revoke) |
| DELETE | `/api/jobs/{id}/purge` | Hard-delete job + metrics |
| PATCH | `/api/jobs/{id}/remarks` | Update job notes |
| GET | `/api/jobs/{id}/metrics` | SSE stream of training metrics |
| GET | `/api/jobs/{id}/metrics/all` | All metrics as JSON |
| GET | `/api/asr/jobs` | List ASR jobs |
| POST | `/api/asr/jobs` | Create ASR training job |
| GET | `/api/asr/datasets` | List ASR datasets |
| POST | `/api/asr/datasets` | Upload CSV dataset |
| POST | `/api/asr/datasets/zip` | Upload ZIP (audio + CSV) |
| GET | `/api/asr/datasets/{id}/preview` | First 5 CSV rows |
| GET | `/api/datasets` | List text datasets |
| POST | `/api/datasets` | Upload JSON/JSONL dataset |
| GET | `/api/datasets/{id}/preview` | First 5 records |
| GET | `/api/models` | List registered models |
| POST | `/api/models` | Register HuggingFace model |
| POST | `/api/models/{id}/download` | Start model download |
| POST | `/api/eval/run` | Start eval/predict run |
| GET | `/api/eval/{run_id}/result` | Poll eval result |
| POST | `/api/chat/load` | Load model for chat |
| GET | `/api/chat/status` | Chat model status |
| POST | `/api/chat/generate` | Generate (SSE token stream) |
| POST | `/api/chat/unload` | Unload chat model |
| GET | `/api/exports` | List merged exports |
| POST | `/api/exports/{job_id}` | Export (merge) from job |
| POST | `/api/exports/from-path` | Export from arbitrary adapter path |
| GET | `/api/system` | GPU/CPU/RAM stats |
