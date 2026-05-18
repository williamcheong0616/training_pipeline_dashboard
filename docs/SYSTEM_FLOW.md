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
    C -- valid --> E["Create Job row — status=pending\ndb.flush to get PK"]
    E --> F["run_training_job.delay(job_id)\nCelery enqueue"]
    F --> G{"Celery available"}
    G -- No --> H["db.rollback — 503 returned"]
    G -- Yes --> I["db.commit — job persisted\nReturn Job JSON to frontend"]
    I --> J["Frontend opens SSE stream\nGET /api/jobs/id/metrics"]

    subgraph CeleryWorker["Celery Worker"]
        K[Pick up task] --> L["Set status=running, started_at=now"]
        L --> M{"training_method"}
        M -- "sft / dpo / lora" --> N["LLM Trainer\nSFTTrainer / DPOTrainer"]
        M -- asr_whisper --> O["Whisper Trainer\nSeq2SeqTrainer"]
        N --> P["INSERT TrainingMetric rows\nevery logging_steps"]
        O --> P
        P --> Q{"Training done"}
        Q -- success --> R["status=completed, output_dir saved"]
        Q -- error --> S["status=failed, error_msg saved"]
        Q -- revoked --> T[status=cancelled]
    end

    J --> U["SSE generator owns its own DB session\npolls db.query(Job) every 2s\nyields metric events"]
    U --> V{"Job terminal"}
    V -- yes --> W["yield done event, close stream"]
    V -- no --> U

    R --> X([User can Export or Evaluate])
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
    API->>DB: INSERT Job status=pending (flush, no commit yet)
    API->>Q: run_training_job.delay(job_id)
    API->>DB: commit
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

    loop Every 2s
        API->>DB: db.query(Job) + db.query(TrainingMetric)
        API-->>UI: SSE metric events
    end
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

## 5. Dataset Upload & Format Detection Flow

```mermaid
flowchart TD
    subgraph TextDataset["Text Dataset — JSON / JSONL"]
        A([Select file + name]) --> B["POST /api/datasets\nformat=auto default"]
        B --> C["Server parses up to 20 records\ndetector.detect_format()"]
        C --> D{"Scoring per record"}
        D -- "conversations/messages list" --> E[+3 sharegpt]
        D -- "prompt+chosen+rejected" --> F[+5 dpo]
        D -- "prompt+completion+label" --> G[+5 kto]
        D -- "instruction or output" --> H[+2 alpaca]
        D -- "text only key" --> I[+2 plain_text]
        E --> J{"Winner >= 70% total score"}
        F --> J
        G --> J
        H --> J
        I --> J
        J -- yes --> K["confidence = high"]
        J -- ">= 40%" --> L["confidence = medium"]
        J -- else --> M["confidence = low"]
        K --> N["Save Dataset row\nformat=detected"]
        L --> N
        M --> N
        N --> O["Response includes\ndetected_format + detection_confidence"]
        O --> P(["Frontend shows\ndetected: alpaca · high confidence"])
    end

    subgraph ASRZip["ASR Dataset — ZIP"]
        Q([Select ZIP]) --> R["POST /api/asr/datasets/zip"]
        R --> S["Validate each member path\nzip-slip guard"]
        S --> T["extractall to datasets/audio/name/"]
        T --> U["Rewrite CSV audio_col\nwith absolute paths"]
        U --> V[("Dataset row\nformat=asr_csv")]
    end
```

---

## 6. Dataset Format Conversion Flow

```mermaid
flowchart TD
    A([User opens dataset preview]) --> B["GET /api/datasets/id/preview"]
    B --> C["Response includes\nvalid_targets + conversion_notes"]
    C --> D(["User picks target format\nin Convert tab"])
    D --> E["POST /api/datasets/id/convert\nbody: target_format, template_name, output_name"]
    E --> F{"target_format in\nVALID_TARGETS[source_format]"}
    F -- No --> G[400 Bad Request]
    F -- Yes --> H["_load_raw(source.path)"]
    H --> I["converter.convert_dataset(records, source_fmt, target_fmt)"]

    subgraph Conversions["Per-record conversion"]
        I --> J{"source_format"}
        J -- dpo --> K["Use chosen as output\ndiscard rejected"]
        J -- kto --> L["Drop label=false rows\nuse completion as output"]
        K --> M[Normalise to alpaca]
        L --> M
        J -- alpaca --> N[Direct convert]
        J -- sharegpt --> N
        J -- plain_text --> N
        M --> N

        N --> O{"target_format"}
        O -- sharegpt --> P["alpaca_to_sharegpt\nhuman turn = instruction+input\ngpt turn = output"]
        O -- alpaca --> Q["sharegpt_to_alpaca\nfirst human/gpt pair"]
        O -- plain_text --> R["Apply PromptTemplate\nbake into text field"]
    end

    P --> S["Write JSONL to\nDATASETS_DIR/output_name.jsonl"]
    Q --> S
    R --> S
    S --> T["INSERT new Dataset row\nformat=target, description=provenance"]
    T --> U(["New dataset appears\nin table"])
```

---

## 7. Evaluate & Predict Flow

```mermaid
flowchart TD
    A(["User selects model + dataset + mode"]) --> B{"is_asr"}

    B -- "No, LLM" --> C["POST /api/eval/run\nis_asr=false"]
    C --> D["BackgroundTask: _run_eval"]
    D --> E["load_tokenizer + load_model"]
    E --> F{"mode"}
    F -- evaluate --> G["DataLoader loop\navg loss + perplexity"]
    F -- predict --> H["model.generate per sample\nwrite JSONL to OUTPUTS_DIR"]
    G --> I[("result store\nloss, perplexity")]
    H --> I

    B -- "Yes, ASR" --> J["POST /api/eval/run\nis_asr=true"]
    J --> K["BackgroundTask: _run_asr_eval"]
    K --> L["load WhisperProcessor + Model"]
    L --> M["apply PeftModel if adapter"]
    M --> N{"mode"}
    N -- evaluate --> O["librosa.load each audio\nmodel.generate, decode\nevaluate.load WER"]
    N -- predict --> P["generate predictions\nwrite JSONL to OUTPUTS_DIR"]
    O --> Q[("result store\nWER score")]
    P --> Q

    I --> R["GET /api/eval/run_id/result"]
    Q --> R
    R --> S([Frontend shows metrics])
```

---

## 8. Chat Flow

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
    Note over UI: Tokens rendered through react-markdown\ncode blocks, tables, bold/italic formatted

    UI->>API: POST /api/chat/unload
    API->>State: model = None, tokenizer = None
    API->>GPU: gc.collect + cuda.empty_cache
    API-->>UI: unloaded
```

---

## 9. Export (Merge) Flow

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

## 10. Security Controls Summary

| Attack Vector | Location | Mitigation |
|---------------|----------|------------|
| **Path traversal** in `output_name` | `exports.py` | `_safe_output_name()` strips dangerous chars; `_validated_save_path()` uses `realpath()` prefix check |
| **Zip-slip** in ZIP upload | `asr.py` | Every member path checked with `realpath` before `extractall` |
| **SSE DB session leak** | `jobs.py`, `asr.py` | Generator opens its own `SessionLocal()` with `finally: db.close()` |
| **SQLAlchemy identity map stale reads** | `jobs.py`, `asr.py` | SSE uses `db.query(Job).filter(...)` — bypasses identity map cache; sees Celery worker updates |
| **Celery enqueue race** | `jobs.py` | `db.flush()` before `.delay()`; `db.rollback()` if enqueue fails; commit only on success |
| **Stale localStorage theme** | `ThemeProvider.tsx` | Explicit equality check: `stored === "light" \|\| stored === "dark"` |
| **Chat fetch not cleaned up** | `chat/page.tsx` | `useEffect` cleanup aborts in-flight request via `AbortController` |
| **Silent chat errors** | `chat/page.tsx` | Non-2xx response body shown as `[Error: …]` assistant message |
| **Multi-origin CORS misconfiguration** | `main.py` | `FRONTEND_URL` env var supports comma-separated origins; each trimmed and validated |

---

## 11. Database Schema

```
jobs
├── id              INTEGER PK
├── name            TEXT
├── status          TEXT  (pending / running / completed / failed / cancelled)
├── training_method TEXT  (sft / dpo / kto / orpo / rm / unsupervised / asr_whisper)
├── peft_method     TEXT  (lora / qlora / dora / full)
├── model_id        FK → models.id
├── dataset_id      FK → datasets.id
├── config_json     JSON
├── output_dir      TEXT
├── error_msg       TEXT
├── celery_task_id  TEXT
├── remarks         TEXT  (nullable — user notes on job detail page)
├── created_at      DATETIME
├── started_at      DATETIME
└── finished_at     DATETIME

training_metrics
├── id              INTEGER PK
├── job_id          FK → jobs.id (indexed: ix_training_metrics_job_id)
├── step            INTEGER
├── epoch           FLOAT
├── loss            FLOAT
├── eval_loss       FLOAT
├── learning_rate   FLOAT
├── reward          FLOAT  (DPO reward or WER for ASR)
├── grad_norm       FLOAT
└── timestamp       DATETIME

models
├── id              INTEGER PK
├── name            TEXT
├── hf_repo         TEXT UNIQUE
├── architecture    TEXT  (llama / mistral / qwen / phi3 / gemma / deepseek)
├── template        TEXT  (alpaca / chatml / llama3 / mistral / qwen / phi3 / gemma)
├── is_downloaded   BOOL
├── local_path      TEXT
├── downloaded_at   DATETIME
└── created_at      DATETIME

datasets
├── id              INTEGER PK
├── name            TEXT
├── path            TEXT  (absolute path to .json, .jsonl, or .csv)
├── format          TEXT  (alpaca / sharegpt / dpo / kto / plain_text / asr_csv)
├── template        TEXT  (chat template associated with data — used for auto-fill in training form)
├── num_samples     INTEGER
├── description     TEXT  (auto-set to conversion provenance for converted datasets)
└── created_at      DATETIME
```

---

## 12. API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/jobs` | List all LLM jobs |
| POST | `/api/jobs` | Create LLM training job |
| DELETE | `/api/jobs/{id}` | Cancel job (Celery revoke) |
| DELETE | `/api/jobs/{id}/purge` | Hard-delete job + metrics (terminal jobs only) |
| PATCH | `/api/jobs/{id}/remarks` | Update job notes |
| GET | `/api/jobs/{id}/metrics` | SSE stream of training metrics |
| GET | `/api/jobs/{id}/metrics/all` | All metrics as JSON snapshot |
| GET | `/api/asr/jobs?skip=0&limit=50` | List ASR jobs (paginated) |
| POST | `/api/asr/jobs` | Create ASR training job |
| DELETE | `/api/asr/jobs/{id}` | Cancel ASR job |
| GET | `/api/asr/jobs/{id}/metrics` | SSE stream of ASR metrics |
| GET | `/api/asr/datasets` | List ASR datasets |
| POST | `/api/asr/datasets` | Upload CSV dataset |
| POST | `/api/asr/datasets/zip` | Upload ZIP (audio + CSV) |
| GET | `/api/asr/datasets/{id}/preview` | First 5 CSV rows |
| GET | `/api/datasets` | List text datasets (excludes asr_csv) |
| POST | `/api/datasets` | Upload JSON/JSONL — format=auto by default |
| GET | `/api/datasets/{id}/preview` | First 5 records + valid_targets + conversion_notes |
| POST | `/api/datasets/{id}/convert` | Convert to another format — creates new dataset |
| DELETE | `/api/datasets/{id}` | Delete dataset record and file |
| GET | `/api/models` | List registered models |
| POST | `/api/models` | Register HuggingFace model |
| POST | `/api/models/{id}/download` | Start model download |
| GET | `/api/models/{id}/download-status` | Poll download progress |
| GET | `/api/models/search/hub?q=` | Search HuggingFace Hub |
| DELETE | `/api/models/{id}` | Unregister model |
| POST | `/api/eval/run` | Start eval/predict run |
| GET | `/api/eval/{run_id}/result` | Poll eval result |
| GET | `/api/eval/{run_id}/stream` | SSE log stream |
| POST | `/api/chat/load` | Load model for chat |
| GET | `/api/chat/status` | Chat model status |
| POST | `/api/chat/generate` | Generate response (SSE token stream) |
| POST | `/api/chat/unload` | Unload chat model and free GPU memory |
| GET | `/api/exports` | List merged exports |
| POST | `/api/exports/{job_id}` | Export (merge) adapter from completed job |
| POST | `/api/exports/from-path` | Export from arbitrary adapter path |
| GET | `/api/system` | GPU/CPU/RAM/disk stats |
| GET | `/api/health` | Health check |
