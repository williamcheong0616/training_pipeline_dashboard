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

### Left Column — Configuration

#### Model Section
- **Model name/path**: Select from the dropdown (registered models) or type a HuggingFace repo ID or local path directly.
- **Template**: Prompt format that matches the model. Llama-3 models use `llama3`, Mistral uses `mistral`, etc. When in doubt, use `alpaca`.
- **Quantization**: Choose `4bit` for QLoRA (least VRAM), `8bit` for moderate savings, or `none` for full precision.
- **Flash Attention**: Enable if your GPU supports it (Ampere+). Reduces VRAM and speeds up training.

#### Method Section
- **Training stage**: The learning algorithm.
  - `sft` — Supervised fine-tuning on instruction/response pairs. Most common.
  - `unsupervised` — Continued pre-training on raw text. Use for domain adaptation.
  - `dpo` — Trains on preferred vs rejected response pairs. Requires `chosen`/`rejected` columns.
  - `rm` — Trains a reward model to score responses.
  - `kto` — Alignment with binary feedback (good/bad labels).
  - `orpo` — Combined SFT + preference optimization in one pass.
- **Finetuning type**: `lora` or `qlora` (adapter-based, low VRAM), `dora` (enhanced LoRA), or `full` (all weights trained, high VRAM).
- **LoRA rank (r)**: Higher = more parameters adapted = better quality but more VRAM. Start with 16 or 32.
- **LoRA alpha**: Usually set to `2 × r`. Controls the scaling of LoRA updates.
- **Dropout**: Regularisation for LoRA layers. Default 0.05 is fine for most cases.
- **Target modules**: Which attention/FFN layers to adapt. For instruction tuning: `q_proj`, `v_proj`. For more capacity: add `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

#### Dataset Section
- **Dataset**: Select a registered dataset from the dropdown or type a file path directly.
- **Format**: Must match how the dataset is structured (`alpaca`, `sharegpt`, `plain_text`, `custom`).
- **Cutoff len / Max seq length**: Maximum token length per sample. Longer = more VRAM. 2048 is a safe start.
- **Packing**: Packs multiple short samples into one sequence for efficiency. Only use with `plain_text` or if samples are much shorter than the cutoff.

#### Training Parameters
| Parameter | Guidance |
|-----------|----------|
| Learning rate | 1e-4 to 2e-4 for LoRA; 1e-5 for full fine-tuning |
| Epochs | 2–5 for instruction tuning; 1 is often enough for DPO |
| Batch size | As large as VRAM allows; start with 4 |
| Gradient accumulation | Use to simulate larger batch without more VRAM |
| LR scheduler | `cosine` for most cases; `constant` for short runs |
| Warmup ratio | 0.05 (5% of steps as warmup) |
| Max grad norm | 1.0 — prevents exploding gradients |
| Logging steps | How often metrics appear in the right panel |
| Save steps | How often checkpoints are saved |

#### Advanced Parameters
- **bf16 / fp16**: `bf16` preferred on Ampere+ (A100, RTX 3090+). `fp16` for older GPUs.
- **Gradient checkpointing**: Saves VRAM at the cost of ~20% slower training. Enable when VRAM is tight.
- **Output dir**: Where checkpoints and the final adapter are saved.

### Right Column — Output

- **Status pill**: idle → running → completed / failed
- **Metrics charts**: Live loss, eval_loss, learning rate, reward, and grad norm curves (via SSE stream).
- **Log console**: Raw training output appended in real time.
- **Step counter / elapsed time**: Visible in the status bar.

### Starting Training

1. Click **▶ Start Training**.
2. The status pill turns to `running`. Metrics will begin appearing after the first logging step.
3. Click **■ Abort Training** at any time to cancel (the Celery task is revoked).
4. After completion, the status turns to `completed` and a **View Job →** link appears.

---

## ASR Tab

Fine-tunes Whisper models for automatic speech recognition. Fully isolated from the LLM tab.

### Model Section
- **Whisper model**: Click a preset chip (tiny → large-v3) or type a custom path. `whisper-large-v3` gives the best quality for Malay/multilingual data.
- **Task**: `transcribe` (speech to same language text) or `translate` (speech to English text).
- **Language**: 
  - **Auto-detect** (recommended for Bahasa Rojak / code-mixed audio): Whisper will detect the language of each audio segment independently.
  - **Malay / English / Chinese / Tamil**: Forces a single language. Useful for monolingual datasets.
  - Custom: type any language name supported by Whisper.

### Training Method Section
- **SFT (full)**: Trains all Whisper weights. Best quality but requires most VRAM.
- **LoRA**: Adapter-based fine-tuning. Good balance of quality and efficiency.
- **QLoRA**: 4-bit quantised LoRA. Minimum VRAM, slightly lower quality.

### Dataset Section
- **Train CSV**: Select from registered ASR datasets. Upload via **Datasets ↗** button.
- **Val CSV**: Optional separate validation set. If not provided, the train set is split automatically.
- **Audio column**: Column name in the CSV that contains audio file paths. Default: `audio_path`.
- **Text column**: Column name for transcripts. Default: `text`.
- **Sample rate**: Always 16000 Hz for Whisper.
- **Val split %**: Fraction of training data to use as validation (if no val CSV).

### Training Parameters
- **Step control**: Whisper training typically uses `max_steps` (e.g. 3000) rather than epochs.
- **Warmup steps**: Usually 10–20% of max_steps.
- **Eval steps**: Run WER evaluation every N steps. Expensive — don't set too small.

### Generation Section
- **predict_with_generate**: Leave enabled. Required for WER computation.
- **Generation max length**: 225 tokens covers most transcripts. Increase for longer audio clips.
- **load_best_model_at_end**: Saves the checkpoint with the best WER.

### Right Panel
- **WER metric**: Displayed in the reward channel. Lower WER = better.
- Live loss and WER curves update in real time.

---

## ASR Datasets

Access from **ASR → Datasets ↗** or navigate to `/asr/datasets`.

### Uploading with ZIP (recommended)

Best for when you have audio files on your local machine.

1. Create a ZIP file containing:
   - Your audio files (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`) — any folder structure
   - A CSV file with at minimum two columns (e.g. `audio_path,text`)
2. The audio paths in the CSV can be the original paths from your machine — Forge matches by filename, not full path.
3. Select **ZIP (with audio)** mode in the upload panel.
4. Set the audio column and text column names if different from defaults.
5. Upload. Forge extracts the audio to `./datasets/audio/<name>/` and rewrites the CSV with the correct paths.

**Example ZIP structure:**
```
malay_asr.zip
├── manifest.csv        ← audio_path,text
├── recording_001.wav
├── recording_002.mp3
└── subfolder/
    └── recording_003.wav
```

**Example manifest.csv:**
```csv
audio_path,text
/Users/me/recordings/recording_001.wav,Terima kasih kerana datang
/Users/me/recordings/recording_002.mp3,Hello how are you today
/Users/me/subfolder/recording_003.wav,Apa khabar semua
```

### Uploading CSV only (audio already on server)

Select **CSV (server path)** mode and upload a CSV where `audio_path` contains absolute paths accessible by the backend server.

---

## Evaluate Tab

Evaluate a trained model's quality against a dataset.

### Evaluate Mode

Computes **loss** and **perplexity** — lower is better.

1. Select the base model (dropdown or path).
2. If you fine-tuned with LoRA, enter the adapter path (e.g. `./outputs/run1/final_adapter`).
3. Select a dataset (registry or file path).
4. Set batch size and max sequence length.
5. Click **▶ Start Evaluate**.
6. Results appear in the card above the log: **Loss** and **Perplexity**.

**Interpreting results:**
- Perplexity < 5 on a held-out domain dataset: excellent
- Perplexity 5–20: reasonable, model understands the domain
- Perplexity > 20: model is struggling — likely underfitting or wrong template

### Predict Mode

Generates model outputs for every sample in a dataset and saves them to a JSONL file.

1. Switch mode to **Predict**.
2. Configure the same model + dataset settings.
3. Set the **save path** for the output file (e.g. `./outputs/predictions.jsonl`).
4. Click **▶ Start Predict**.
5. The log shows progress (`N/total done`). When complete, the result card shows the output file path.

**Output format (`predictions.jsonl`):**
```jsonl
{"prompt": "What is the capital of Malaysia?", "output": "Kuala Lumpur."}
{"prompt": "Translate: Saya suka makan nasi lemak", "output": "I like eating nasi lemak."}
```

---

## Chat Tab

Interactively test a model by having a conversation with it.

### Loading a Model

1. Select a model from the dropdown or type a path.
2. Optionally enter an adapter path to test a fine-tuned LoRA.
3. Choose quantization (`4bit` for large models on limited VRAM).
4. Click **Load Model**. The status dot turns amber (loading) then green (ready).
   - Loading a 7B model in 4-bit takes 30–90 seconds.
5. To free GPU memory after testing, click **Unload**.

> **Note:** Only one model can be loaded at a time. Loading a new model does not automatically unload the previous one — click Unload first.

### Generation Parameters

| Parameter | Effect |
|-----------|--------|
| **Max new tokens** | Maximum length of the response |
| **Temperature** | Higher = more creative/random; lower = more focused. 0.7 is a good default |
| **Top-p** | Nucleus sampling. 0.9 keeps 90% probability mass |
| **Top-k** | Limits vocabulary to top-k tokens at each step |
| **Repetition penalty** | > 1.0 discourages repeating tokens |

### System Prompt

Set the model's persona here. Example:
```
Kamu adalah pembantu yang membantu dalam Bahasa Malaysia dan Bahasa Inggeris.
```

### Chatting

1. Type a message in the input box at the bottom.
2. Press **Enter** to send (or **Shift+Enter** for a new line within the input).
3. Tokens stream in real time as the model generates.
4. Click **Clear** to start a fresh conversation.

---

## Export Tab

Merges a LoRA adapter into the base model weights, producing a standalone full model that can be used without the PEFT library.

### From Completed Job

1. Select **From Completed Job**.
2. Choose a job from the dropdown (only `completed` jobs are listed).
3. Give the merged model a name (e.g. `llama3-malay-v1`).
4. Click **⇓ Export / Merge Adapter**.
5. The merge runs in the background (may take several minutes for large models).
6. The right panel shows previously exported models with their size.

### From Adapter Path

Use this if you have an adapter directory from outside Forge (e.g. trained via CLI).

1. Select **From Adapter Path**.
2. Enter the full path to the adapter directory (must contain `adapter_config.json` and `adapter_model.safetensors`).
3. Give it a name and click **⇓ Export / Merge Adapter**.

### Where are merged models saved?

In the `./exports/` directory (configurable via `EXPORTS_DIR` env var). Each merged model is a directory containing the full model weights and tokenizer, loadable with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("./exports/llama3-malay-v1")
tokenizer = AutoTokenizer.from_pretrained("./exports/llama3-malay-v1")
```

---

## Jobs Tab

View all training jobs (both LLM and ASR) in one place.

- **Status badges**: `pending` (queued) → `running` → `completed` / `failed` / `cancelled`
- **Running job counter**: Shown as a badge on the Jobs tab in the nav.
- **View job**: Click any row to open the job detail page with live Recharts metric curves.
- **Cancel**: Click the ✕ button to abort a running job.

### Job Detail Page (`/jobs/{id}`)

- Full training configuration accordion (collapsed by default).
- Live charts: train loss, eval loss, learning rate, reward/WER, grad norm.
- Log console with SSE stream.
- Elapsed time and step counter.

---

## Models Tab

Manage the base model registry.

### Registering a Model

1. Click **Register Model**.
2. Enter the HuggingFace repo ID (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`).
3. Set the template (prompt format) and architecture.
4. Click **Register**.

### Searching HuggingFace Hub

1. Type a search query in the **Search HF Hub** box (e.g. "whisper", "llama", "mistral").
2. Results appear with download counts and model sizes.
3. Click **Register** next to any result to add it to the registry.

### Downloading a Model

Click **Download** on any registered model to trigger a snapshot download from HuggingFace Hub to `./models/`. The download runs in the background. The model status changes to `downloaded` when complete.

> Models do not need to be downloaded to start training — you can use the HF repo ID directly in the training form and the trainer will download on demand. Registering + downloading is useful for offline/air-gapped environments.

---

## Datasets Tab

Manage text datasets for LLM fine-tuning.

### Uploading a Dataset

1. Enter a name and choose the format.
2. Select the file (`.json` or `.jsonl`) and click **⇑ Upload**.

### Supported Formats

**Alpaca** (instruction tuning):
```json
[
  {"instruction": "Translate to Malay", "input": "Good morning", "output": "Selamat pagi"},
  {"instruction": "Summarize this text", "input": "...", "output": "..."}
]
```

**ShareGPT** (multi-turn conversation):
```json
[
  {"conversations": [
    {"from": "human", "value": "Apa itu AI?"},
    {"from": "gpt", "value": "AI bermaksud kecerdasan buatan..."}
  ]}
]
```

**Plain text** (continued pre-training / unsupervised):
```json
[{"text": "Bahasa Melayu ialah bahasa rasmi Malaysia..."}, ...]
```
Or JSONL (one JSON object per line).

---

## Tips & Common Workflows

### Fine-tuning Llama-3 on Malay Instructions

1. **Models**: Register `meta-llama/Meta-Llama-3-8B-Instruct`, template = `llama3`
2. **Datasets**: Upload your Alpaca-format JSON
3. **LLM**: method = `sft`, PEFT = `qlora`, quantization = `4bit`, lora_r = 32, lr = 2e-4, epochs = 3
4. **Start** → monitor loss in the right panel
5. **Evaluate**: check perplexity drops compared to base model
6. **Chat**: load with same adapter path, test with Malay prompts
7. **Export**: merge for deployment

### Whisper LoRA for Bahasa Rojak

1. **ASR Datasets**: Upload ZIP with audio files + CSV
2. **ASR**: model = `whisper-large-v3`, language = **Auto-detect**, method = `lora`, lora_r = 32, max_steps = 3000
3. Monitor **WER** in the right panel — should decrease over time
4. Best checkpoint is saved automatically (`load_best_model_at_end = true`)

### Testing Before Exporting

Always test in **Chat** before exporting. Use the same adapter path as the training output. If the model behaves unexpectedly, adjust hyperparameters and run another training job without wasting time on the merge step.

### Running Without a GPU

For CPU-only testing use `quantization = none` and a small model like `openai/whisper-tiny` or `Phi-3-Mini`. Training will be extremely slow but functional. Eval and Chat work fine on CPU with small models.

### VRAM Estimates (approximate)

| Setup | VRAM needed |
|-------|-------------|
| Whisper large-v3 LoRA (fp16) | ~8 GB |
| Whisper large-v3 QLoRA (4bit) | ~5 GB |
| LLaMA-3-8B QLoRA (4bit) | ~6 GB |
| LLaMA-3-8B LoRA (bf16) | ~20 GB |
| LLaMA-3-8B full fine-tune | ~80 GB+ |
| LLaMA-3-70B QLoRA (4bit) | ~40 GB |
