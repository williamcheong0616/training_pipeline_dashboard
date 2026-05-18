# Forge — LLM & ASR Fine-Tuning Dashboard

Forge is a self-hosted, browser-based training platform for fine-tuning large language models (LLMs) and Whisper ASR models. It mirrors the workflow of LlamaFactory with a dense, utilitarian UI, full real-time metric streaming, and first-class support for code-mixed (Bahasa Rojak / multilingual) datasets.

---

## Features

| Category | Capabilities |
|----------|-------------|
| **LLM Training** | SFT, DPO, KTO, ORPO, Reward Model, Unsupervised/CPT |
| **PEFT** | LoRA, QLoRA (4-bit / 8-bit), DoRA, full fine-tuning |
| **ASR Training** | Whisper fine-tuning (SFT, LoRA, QLoRA) with WER tracking |
| **Multilingual** | Auto-detect language mode for code-mixed / Bahasa Rojak data |
| **Evaluate** | Perplexity eval + batch prediction on any dataset |
| **Chat** | Interactive inference with streaming token output |
| **Export** | Merge LoRA adapters into standalone full models |
| **Job Queue** | Celery + Redis async training with SSE real-time logs |
| **Model Registry** | HuggingFace Hub search, download, and local model tracking |
| **Dataset Upload** | JSON / JSONL for LLM; CSV manifest or ZIP-with-audio for ASR |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Redis (for job queue)
- CUDA-capable GPU recommended (CPU works for small models)

### 1. Backend

```bash
cd backend
pip install -r requirements.txt

# Start Redis (or use Docker)
redis-server &

# Start API server
uvicorn backend.api.main:app --reload --port 8000

# Start Celery worker (separate terminal)
celery -A backend.workers.celery_app worker --loglevel=info --concurrency=1
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

### 3. Docker (recommended for production)

```bash
docker compose up --build
```

Services:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Flower (job monitor): http://localhost:5555

---

## Project Structure

```
training_pipeline_dashboard/
├── backend/
│   ├── api/
│   │   ├── main.py              # FastAPI app, CORS, router registration
│   │   └── routes/
│   │       ├── jobs.py          # Training job CRUD + SSE metrics
│   │       ├── models.py        # Model registry + HF Hub download
│   │       ├── datasets.py      # Text dataset upload/list
│   │       ├── exports.py       # Adapter merge + export listing
│   │       ├── asr.py           # ASR datasets, jobs, Whisper models
│   │       ├── eval.py          # Evaluate (perplexity) + Predict
│   │       └── chat.py          # Model load/unload/generate (SSE)
│   ├── core/
│   │   ├── trainer/             # SFT, DPO, KTO, ORPO, RM, Unsupervised
│   │   ├── asr/                 # Whisper trainer, dataset, collator, metrics
│   │   ├── model/               # Model/tokenizer loader, LoRA adapter
│   │   └── data/                # Dataset builders, prompt templates
│   ├── workers/                 # Celery app + training task dispatcher
│   ├── db/                      # SQLAlchemy models + session
│   ├── cli/                     # Typer CLI (train / eval / export)
│   └── config/
│       └── model_registry.yaml  # Pre-registered model definitions
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # LLM training tab
│   │   ├── asr/                 # ASR training tab + dataset upload
│   │   ├── evaluate/            # Evaluate & Predict tab
│   │   ├── chat/                # Chat / inference tab
│   │   ├── export/              # Export / merge tab
│   │   ├── jobs/                # Job list + job detail with live charts
│   │   ├── models/              # Model registry
│   │   └── datasets/            # Text dataset management
│   ├── components/
│   │   ├── TopNav.tsx           # Tab navigation + running job badge
│   │   └── MetricsPanel.tsx     # Recharts loss/lr/WER curves
│   └── lib/
│       ├── api.ts               # Typed API client (axios)
│       └── sse.ts               # SSE hook for live metrics
├── docs/
│   ├── TECHNICAL.md             # Architecture & API reference
│   └── USER_GUIDE.md            # End-to-end usage guide
└── docker-compose.yml
```

---

## Supported Models

Pre-registered in `config/model_registry.yaml`:

| Model | Params | Template |
|-------|--------|----------|
| LLaMA-3-8B-Instruct | 8B | llama3 |
| LLaMA-3-70B-Instruct | 70B | llama3 |
| Mistral-7B-Instruct | 7B | mistral |
| Mixtral-8x7B-Instruct | 47B | mistral |
| Qwen2-7B-Instruct | 7B | qwen |
| Qwen2-72B-Instruct | 72B | qwen |
| Phi-3-Mini | 3.8B | phi3 |
| Gemma-7B-IT | 7B | gemma |
| DeepSeek-R1-7B | 7B | deepseek |
| Whisper (tiny → large-v3) | 39M–1.5B | — |

---

## Documentation

- [Technical Documentation](docs/TECHNICAL.md) — architecture, API reference, database schema, configuration
- [User Guide](docs/USER_GUIDE.md) — step-by-step usage for all tabs

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker URL |
| `DATABASE_URL` | `sqlite:///./forge.db` | SQLAlchemy DB URL |
| `DATASETS_DIR` | `./datasets` | Where uploaded datasets are stored |
| `MODELS_DIR` | `./models` | Where downloaded models are stored |
| `EXPORTS_DIR` | `./exports` | Where merged models are saved |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend → backend base URL |

---

## License

MIT
