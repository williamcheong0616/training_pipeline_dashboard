# Forge — User Guide

This guide walks through every tab in the Forge dashboard with step-by-step instructions.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [LLM Tab — Fine-Tuning Language Models](#llm-tab)
3. [ASR Tab — Whisper Fine-Tuning](#asr-tab)
4. [Evaluate Tab — Evaluation & Prediction](#evaluate-tab)
5. [Chat Tab — Interactive Testing](#chat-tab)
6. [Export Tab — Merging Adapters](#export-tab)
7. [Jobs Tab — Monitoring Training Runs](#jobs-tab)
8. [Models Tab — Managing the Model Registry](#models-tab)
9. [Datasets Tab — Managing Text Datasets](#datasets-tab)
10. [Tips & Common Workflows](#tips--common-workflows)

---

## Getting Started

Open **http://localhost:3000** in your browser. You will see the top navigation bar:

```
⚡ Forge  |  LLM  |  ASR  |  Evaluate  |  Chat  |  Export  |  Jobs  |  Models  |  Datasets
```

The **TRAINING / IDLE** dot in the top-right shows whether any training job is currently running.

**Recommended first steps:**
1. Go to **Models** → register or download a model
2. Go to **Datasets** → upload your training data
3. Go to **LLM** (or **ASR**) → configure and start training
4. Go to **Jobs** → monitor progress with live charts
5. Go to **Evaluate** → check loss and perplexity
6. Go to **Chat** → test your fine-tuned model interactively
7. Go to **Export** → merge the adapter into a standalone model

---

## LLM Tab

The main fine-tuning page for text/language models (SFT, DPO, ORPO, KTO, RM, Unsupervised).

The page uses a two-column layout: the left panel holds all configuration and scrolls naturally; the right panel (metrics + log) stays pinned to the viewport as you scroll.

**Hover tooltips** — every parameter label and chip button has a tooltip explaining what it does and how it affects training. Hover over the `?` icon next to any label, or hover directly over a chip (e.g. `q_proj`, `sft`, `lora`) to see its description.

### Left Column — Configuration

#### Model Section
- **Base model**: Select from the dropdown (registered models) or type a HuggingFace repo ID or local path.
- **Template**: Prompt format that matches the model. Llama-3 uses `llama3`, Mistral uses `mistral`, Qwen uses `qwen`, etc. **Auto-filled when you select a dataset** (see below).
- **Quantization**: `4bit` for QLoRA (least VRAM), `8bit` for moderate savings, `none` for full precision.
- **Flash Attention**: Enable if your GPU supports it (Ampere+). Reduces VRAM and speeds up training.

#### Method Section
- **Training stage** — the learning algorithm. Hover each chip for a description.
  - `sft` — Supervised fine-tuning on instruction/response pairs. Most common.
  - `unsupervised` — Continued pre-training on raw text. Use for domain adaptation.
  - `dpo` — Trains on preferred vs rejected pairs. Requires `chosen`/`rejected` columns.
  - `rm` — Trains a reward model for RLHF.
  - `kto` — Alignment with per-sample binary labels (good/bad).
  - `orpo` — Combined SFT + preference in one pass; no reference model needed.
- **Finetuning type** — hover each chip for VRAM trade-off details.
  - `lora` — Low-rank adapters, low VRAM, recommended for most tasks.
  - `qlora` — LoRA + 4-bit base weights; enables 70B on 24GB VRAM.
  - `dora` — Decomposed LoRA; marginally better quality at same rank.
  - `full` — All weights updated; highest quality, requires full model VRAM.
- **LoRA rank (r)**: Higher = more parameters adapted = better quality but more VRAM. Start with 16–32.
- **LoRA alpha**: Usually `2 × r`. Controls update scaling.
- **Dropout**: Regularisation for LoRA layers. 0.05 is fine for most cases.
- **Target modules**: Which attention/FFN layers to adapt. Hover each chip for its architectural role.
  - Minimal: `q_proj`, `v_proj`
  - Extended: add `k_proj`, `o_proj` for better attention coverage
  - Full: add `gate_proj`, `up_proj`, `down_proj` for FFN adaptation (more VRAM)
  - `lm_head`: rarely needed; targets output token distribution directly

#### Dataset Section
- **Dataset**: Select from the dropdown. **When you pick a dataset, the format and template fields are automatically set** from the dataset's stored metadata — both show a green "auto" badge. You can still override them manually.
- **Format**: Data structure of the dataset (`alpaca`, `sharegpt`, `dpo`, `kto`, `plain_text`).
- **Cutoff / Max seq length**: Maximum tokens per sample. Longer = more VRAM. 2048 is a safe start.
- **Packing**: Packs multiple short samples into one sequence. Improves GPU utilisation for short-text datasets; disable if samples are already near the length limit.

#### Training Parameters

| Parameter | Guidance |
|-----------|----------|
| Learning rate | 1e-4 to 2e-4 for LoRA; 1e-5 to 5e-5 for full fine-tuning |
| Epochs | 2–5 for instruction tuning; 1 is often enough for DPO |
| Batch size | As large as VRAM allows; start with 4 |
| Gradient accumulation | Simulates larger batch without more VRAM |
| LR scheduler | `cosine` for most cases; `constant` for short runs |
| Warmup ratio | 0.05 (5% of steps as warmup) |
| Max grad norm | 1.0 — prevents exploding gradients |
| Logging steps | How often metrics appear in the right panel |
| Save steps | How often checkpoints are saved |

#### Advanced Parameters
- **bf16 / fp16**: `bf16` preferred on Ampere+ (A100, RTX 3090+). `fp16` for older GPUs.
- **Gradient checkpointing**: Saves VRAM at the cost of ~20% slower training.
- **Output dir**: Where checkpoints and the final adapter are saved.

### Right Column — Output

- **Status pill**: idle → running → completed / failed
- **Metrics charts**: Live loss, eval_loss, learning rate, reward, and grad norm curves.
- **Log console**: Raw training output appended in real time.
- **Step counter / elapsed time**: Visible in the status bar.

### Starting Training

1. Click **▶ Start Training**.
2. The status pill turns to `running`. Metrics appear after the first logging step.
3. Click **■ Abort Training** at any time to cancel.
4. After completion, a **View Job →** link appears.

---

## ASR Tab

Fine-tunes Whisper models for automatic speech recognition.

**Hover tooltips** are available on every parameter, section header, and method chip (SFT full, lora, qlora) — hover to see VRAM requirements and behaviour.

### Model Section
- **Whisper model**: Click a preset chip (tiny → large-v3) or type a custom path. Hover each chip to see parameter count and VRAM estimate.
- **Task**: `transcribe` (speech → same language text) or `translate` (speech → English text).
- **Language**:
  - **Auto-detect** (recommended for Bahasa Rojak / code-mixed audio): Whisper detects the language of each segment independently.
  - **Malay / English / Chinese / Tamil**: Forces a single language.
  - Custom: type any language name Whisper supports.

### Training Method Section
- **SFT (full)**: All Whisper weights updated. Best quality, most VRAM (40GB+ for large-v3).
- **LoRA**: Adapter-based fine-tuning. Trains in ~8GB VRAM, good balance of quality and speed.
- **QLoRA**: 4-bit quantised LoRA. Enables large-v3 on 12–16GB; automatically sets quantization to 4-bit.

**Target modules** (shown for LoRA/QLoRA) — hover each chip:
- `q_proj` / `v_proj`: cross-attention query/value — minimal effective pair for ASR adaptation
- `k_proj`: cross-attention key — add for accented or low-resource language tasks
- `o_proj`: cross-attention output — improves audio-decoder integration

### Dataset Section
- **Train CSV**: Select from registered ASR datasets. Upload via **Datasets ↗**.
- **Val CSV**: Optional separate validation set; train set is auto-split if not provided.
- **Audio column** / **Text column**: Column names in the CSV.
- **Val split %**: Fraction used as validation if no val CSV.

### Training Parameters
- **Step control**: Toggle between `max_steps` (recommended for ASR) and `epochs`.
- **Warmup steps**: 10–20% of max_steps.
- **Eval steps**: Run WER evaluation every N steps — don't set too low (expensive).

### Generation Section
- **predict_with_generate**: Leave enabled — required for WER computation.
- **Generation max length**: 225 tokens covers most transcripts.
- **load_best_model_at_end**: Keeps the checkpoint with the best WER.

---

## ASR Datasets

Access from **ASR → Datasets ↗** or navigate to `/asr/datasets`.

### Uploading with ZIP (recommended)

1. Create a ZIP containing audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`) and a CSV with `audio_path,text` columns.
2. The paths in the CSV can be your original machine paths — Forge matches by filename and rewrites them.
3. Select **ZIP (with audio)**, set column names if non-default, and upload.

**Example ZIP structure:**
```
malay_asr.zip
├── manifest.csv
├── recording_001.wav
└── recording_002.mp3
```

**Example manifest.csv:**
```csv
audio_path,text
/Users/me/recording_001.wav,Terima kasih kerana datang
/Users/me/recording_002.mp3,Hello how are you today
```

### Uploading CSV only (audio already on server)

Select **CSV (server path)** and upload a CSV where `audio_path` contains absolute paths accessible by the backend.

---

## Evaluate Tab

Evaluate a trained model's quality against a dataset.

### Evaluate Mode

Computes **loss** and **perplexity** — lower is better.

1. Select the base model and optionally an adapter path.
2. Select a dataset and set batch size / max seq length.
3. Click **▶ Start Evaluate**.
4. Results appear in the card: **Loss** and **Perplexity**.

**Interpreting perplexity:**
- < 5: excellent — model understands the domain well
- 5–20: reasonable
- > 20: model is struggling — check template, dataset format, or training duration

### Predict Mode

Generates outputs for every dataset sample and saves to a JSONL file.

1. Switch to **Predict** mode.
2. Set the save path (defaults to `OUTPUTS_DIR/predict_<id>.jsonl`).
3. Click **▶ Start Predict**.

**Output format:**
```jsonl
{"prompt": "What is the capital of Malaysia?", "output": "Kuala Lumpur."}
```

---

## Chat Tab

Interactively test a model by having a conversation with it.

### Loading a Model

1. Select a model and optionally an adapter path.
2. Choose quantization (`4bit` for large models on limited VRAM).
3. Click **Load Model**. The status dot turns amber then green (ready).
4. Click **Unload** to free GPU memory when done.

> Only one model can be loaded at a time.

### Generation Parameters

| Parameter | Effect |
|-----------|--------|
| Max new tokens | Maximum response length |
| Temperature | Higher = more creative; lower = more focused. 0.7 is a good default |
| Top-p | Nucleus sampling — 0.9 keeps 90% probability mass |
| Top-k | Limits vocabulary to top-k tokens per step |
| Repetition penalty | > 1.0 discourages token repetition |

### System Prompt

Set the model's persona here. Example:
```
Kamu adalah pembantu yang membantu dalam Bahasa Malaysia dan Bahasa Inggeris.
```

### Chatting

1. Type a message and press **Enter** (or **Shift+Enter** for a newline).
2. Tokens stream in real time as the model generates.
3. **Model responses render as Markdown** — code blocks, tables, bold/italic, and links are all formatted. Syntax highlighting is applied to code blocks.
4. Click **Clear** to start a fresh conversation.

---

## Export Tab

Merges a LoRA adapter into the base model weights, producing a standalone model that does not require the PEFT library.

### From Completed Job

1. Select **From Completed Job**.
2. Choose a completed job from the dropdown.
3. Give the merged model a name.
4. Click **⇓ Export / Merge Adapter**.

### From Adapter Path

Use this for adapters trained outside Forge (e.g. via CLI).

1. Select **From Adapter Path**.
2. Enter the full adapter directory path (must contain `adapter_config.json`).
3. Give it a name and click **⇓ Export / Merge Adapter**.

### Where are merged models saved?

In `./exports/` (configurable via `EXPORTS_DIR` env var). Loadable with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./exports/llama3-malay-v1")
tokenizer = AutoTokenizer.from_pretrained("./exports/llama3-malay-v1")
```

---

## Jobs Tab

View all training jobs (LLM and ASR) in one place.

- **Status badges**: `pending` → `running` → `completed` / `failed` / `cancelled`
- **View job**: Click any row to open the job detail page with live metric charts.
- **Cancel**: Click ✕ to abort a running job.
- **Purge**: Removes the DB record — only available for completed/failed/cancelled jobs.

### Job Detail Page (`/jobs/{id}`)

- Full training configuration (collapsible).
- Live charts: train loss, eval loss, learning rate, reward/WER, grad norm.
- Log console with SSE stream.
- Remarks field for notes (auto-saved).

---

## Models Tab

Manage the base model registry.

### Registering a Model

1. Click **Register Model**.
2. Enter the HuggingFace repo ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`).
3. Set the template and architecture.
4. Click **Register**.

### Searching HuggingFace Hub

1. Type a query (e.g. "whisper", "llama", "qwen").
2. Click **Register** next to any result to add it to the registry.

### Downloading a Model

Click **Download** on any registered model to trigger a background download to `./models/`. Training can start before a model is downloaded — the trainer will download on demand from HuggingFace Hub.

---

## Datasets Tab

Manage text datasets for LLM fine-tuning.

### Uploading a Dataset

1. Enter a dataset name.
2. Leave **format** as `auto-detect (recommended)` — the server analyses the file content and infers the format automatically, returning `detected: alpaca · high confidence` (or similar) after upload.
3. Override the format manually if needed.
4. Select the file (`.json` or `.jsonl`) and click **⇑ Upload**.

### Format Auto-Detection

The server scores up to 20 records from the uploaded file against each format and picks the highest-scoring match:

| Format | Detected by |
|--------|------------|
| `alpaca` | `instruction` or `output` keys present |
| `sharegpt` | `conversations` or `messages` key with list-of-dicts value |
| `dpo` | `prompt` + `chosen` + `rejected` all present |
| `kto` | `prompt` + `completion` + `label` all present |
| `plain_text` | Only a `text` key (≤ 3 total keys) |

Confidence (`high` / `medium` / `low`) reflects how dominant the winning format's score was.

### Supported Formats

**Alpaca** (instruction tuning):
```json
[{"instruction": "Translate to Malay", "input": "Good morning", "output": "Selamat pagi"}]
```

**ShareGPT** (multi-turn conversation):
```json
[{"conversations": [{"from": "human", "value": "Apa itu AI?"}, {"from": "gpt", "value": "AI bermaksud..."}]}]
```

**DPO** (preference):
```json
[{"prompt": "Write a poem", "chosen": "Roses are red...", "rejected": "bad response"}]
```

**KTO** (binary feedback):
```json
[{"prompt": "Summarise this", "completion": "The text says...", "label": true}]
```

**Plain text** (continued pre-training):
```json
[{"text": "Bahasa Melayu ialah bahasa rasmi Malaysia..."}]
```

### Previewing a Dataset

Click any row in the table to open the **Preview** panel on the right, showing the first 5 samples with field-by-field rendering.

### Converting Between Formats

1. Click a dataset row to open its preview.
2. Switch to the **⇄ convert** tab.
3. Select the **target format** from the chips (only valid conversions are shown).
4. If converting to `plain_text`, select the **template** to bake in (e.g. `llama3` to embed Llama-3 special tokens into the text field).
5. Edit the **output name** if needed.
6. Click **⇄ Convert & Save as New Dataset**.

A new dataset record is created — the original is unchanged. The description records the conversion provenance.

**Conversion matrix:**

| From | Can convert to |
|------|---------------|
| alpaca | sharegpt, plain_text |
| sharegpt | alpaca (first turn), plain_text |
| dpo | sharegpt (chosen only), alpaca (chosen only) |
| kto | sharegpt (label=True only), alpaca (label=True only) |
| plain_text | alpaca (text → instruction) |

### Auto-fill in Training Form

When you select a dataset in the **LLM** training form, the **format** and **template** dropdowns are automatically set from the dataset's stored metadata. Both fields show a green "auto" badge to indicate the auto-fill. You can override either field manually at any time.

---

## Tips & Common Workflows

### Fine-tuning Llama-3 on Malay Instructions

1. **Models**: Register `meta-llama/Meta-Llama-3-8B-Instruct`, template = `llama3`
2. **Datasets**: Upload your Alpaca-format JSON — the server auto-detects format
3. **LLM**: Select the dataset (template auto-fills to `llama3`), method = `sft`, PEFT = `qlora`, quantization = `4bit`, lora_r = 32, lr = 2e-4, epochs = 3
4. **Start** → monitor loss in the right panel
5. **Evaluate**: check perplexity drops vs. base model
6. **Chat**: load with same adapter path, test with Malay prompts (output renders as Markdown)
7. **Export**: merge for deployment

### Whisper LoRA for Bahasa Rojak

1. **ASR Datasets**: Upload ZIP with audio + CSV
2. **ASR**: model = `whisper-large-v3`, language = **Auto-detect**, method = `lora`, lora_r = 32, max_steps = 3000
3. Monitor **WER** — should decrease over time
4. Best checkpoint is saved automatically

### Repurposing a Dataset for a Different Training Objective

Say you have an Alpaca dataset and want to use it for DPO — but you only have chosen responses. You can convert to ShareGPT and then structure the conversations manually, or convert alpaca → sharegpt and use the sharegpt format directly with SFT. Use the **Convert** tab in the Datasets preview.

### Testing Before Exporting

Always test in **Chat** before exporting. Use the same adapter path as the training output. The Chat page renders Markdown in model responses so you can verify code generation, formatting, and structured output without leaving the browser.

### Running Without a GPU

Use `quantization = none` and a small model (`openai/whisper-tiny`, `Phi-3-Mini`). Training will be extremely slow but functional. Eval and Chat work fine on CPU.

### VRAM Estimates (approximate)

| Setup | VRAM needed |
|-------|-------------|
| Whisper large-v3 LoRA (fp16) | ~8 GB |
| Whisper large-v3 QLoRA (4bit) | ~5 GB |
| LLaMA-3-8B QLoRA (4bit) | ~6 GB |
| LLaMA-3-8B LoRA (bf16) | ~20 GB |
| LLaMA-3-8B full fine-tune | ~80 GB+ |
| LLaMA-3-70B QLoRA (4bit) | ~40 GB |
