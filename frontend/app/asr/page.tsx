"use client";
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { createASRJob, cancelASRJob, getASRModels, getASRDatasets, getSystemStats } from "@/lib/api";
import { useMetricsStream } from "@/lib/sse";
import MetricsPanel from "@/components/MetricsPanel";
import Tooltip from "@/components/Tooltip";
import type { Job } from "@/types";
import Link from "next/link";

const TRAINING_METHODS = ["sft", "lora", "qlora"] as const;
const QUANT_OPTIONS    = ["none", "4bit", "8bit"] as const;
const TASKS            = ["transcribe", "translate"] as const;
const TARGET_MODS      = ["q_proj", "k_proj", "v_proj", "o_proj"] as const;

const TRAINING_METHOD_TIPS: Record<string, string> = {
  sft:   "Full fine-tuning — all Whisper encoder and decoder weights are updated. Most expressive but requires the most VRAM (40GB+ for large-v3). Best for large domain shifts or low-resource languages.",
  lora:  "Low-Rank Adaptation — attaches small trainable matrices to attention layers while the base Whisper weights stay frozen. Trains in ~8GB VRAM; fast and effective for accent or domain adaptation.",
  qlora: "Quantized LoRA — runs the frozen Whisper base in 4-bit NormalFloat while LoRA adapters stay in bfloat16. Enables large-v3 on 12–16GB; automatically sets quantization to 4-bit.",
};

const MOD_TIPS: Record<string, string> = {
  q_proj: "Query projection in cross-attention — shapes how each decoder token queries the audio encoder outputs. Most influential for alignment between transcript and audio; always include for domain adaptation.",
  k_proj: "Key projection in cross-attention — controls how encoder frames are exposed to decoder queries. Tuning improves acoustic feature selection; add when adapting to accented speech or noisy audio.",
  v_proj: "Value projection in cross-attention — determines the audio content retrieved per decode step. Together with q_proj, the minimal effective LoRA pair for ASR fine-tuning.",
  o_proj: "Output projection — merges cross-attention heads back into the decoder residual stream. Adding LoRA here improves how audio features integrate into decoding; useful for low-resource languages.",
};
const LANG_PRESETS     = [
  { label: "Auto-detect", value: "auto" },
  { label: "Malay",       value: "malay" },
  { label: "English",     value: "english" },
  { label: "Chinese",     value: "chinese" },
  { label: "Tamil",       value: "tamil" },
] as const;

function Field({ label, tooltip, children }: { label: string; tooltip?: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="lf-label" style={{ display: "flex", alignItems: "center" }}>
        {label}{tooltip && <Tooltip text={tooltip} />}
      </label>
      {children}
    </div>
  );
}

function Section({ title, tooltip }: { title: string; tooltip?: string }) {
  return (
    <div className="lf-section" style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 0 }}>
      {title}{tooltip && <Tooltip text={tooltip} />}
    </div>
  );
}

function Toggle({ label, tooltip, checked, onChange }: { label: string; tooltip?: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="lf-toggle" style={{ display: "inline-flex", alignItems: "center" }}>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="lf-toggle-track" />
      {label}{tooltip && <Tooltip text={tooltip} />}
    </label>
  );
}

type FormState = {
  name: string;
  model_path: string;
  task: string;
  language: string;
  quantization: string;
  gpu_id: string;
  training_method: string;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[];
  dataset_id: string;
  val_dataset_id: string;
  audio_col: string;
  text_col: string;
  sample_rate: number;
  val_split: number;
  use_max_steps: boolean;
  max_steps: number;
  warmup_steps: number;
  num_epochs: number;
  warmup_ratio: number;
  batch_size: number;
  eval_batch_size: number;
  gradient_accumulation_steps: number;
  learning_rate: number;
  eval_steps: number;
  save_steps: number;
  save_total_limit: number;
  logging_steps: number;
  predict_with_generate: boolean;
  generation_max_length: number;
  load_best_model_at_end: boolean;
  fp16: boolean;
  bf16: boolean;
  gradient_checkpointing: boolean;
  output_dir: string;
};

const DEFAULT: FormState = {
  name: "", model_path: "openai/whisper-large-v3",
  task: "transcribe", language: "auto", quantization: "none", gpu_id: "auto",
  training_method: "lora",
  lora_r: 32, lora_alpha: 64, lora_dropout: 0.1,
  target_modules: ["q_proj", "v_proj"],
  dataset_id: "", val_dataset_id: "",
  audio_col: "audio_path", text_col: "text",
  sample_rate: 16000, val_split: 0.1,
  use_max_steps: true, max_steps: 3000, warmup_steps: 500,
  num_epochs: 3, warmup_ratio: 0.05,
  batch_size: 2, eval_batch_size: 1,
  gradient_accumulation_steps: 8,
  learning_rate: 1e-4,
  eval_steps: 500, save_steps: 500, save_total_limit: 2, logging_steps: 50,
  predict_with_generate: true, generation_max_length: 225,
  load_best_model_at_end: true,
  fp16: true, bf16: false, gradient_checkpointing: true,
  output_dir: (() => {
    const now = new Date();
    const dd = String(now.getDate()).padStart(2, "0");
    const mm = String(now.getMonth() + 1).padStart(2, "0");
    const yy = String(now.getFullYear()).slice(2);
    const hh = String(now.getHours() % 12 || 12);
    const min = String(now.getMinutes()).padStart(2, "0");
    const ap = now.getHours() < 12 ? "am" : "pm";
    return `./outputs/asr_run_${dd}${mm}${yy}_${hh}${min}${ap}`;
  })(),
};

function StatusPill({ status }: { status: string }) {
  const map: Record<string, { color: string; bg: string }> = {
    idle:      { color: "var(--text-dim)", bg: "transparent" },
    pending:   { color: "var(--amber)",    bg: "var(--amber-dim)" },
    running:   { color: "var(--accent)",   bg: "var(--accent-dim)" },
    completed: { color: "var(--green)",    bg: "var(--green-dim)" },
    failed:    { color: "var(--red)",      bg: "var(--red-dim)" },
    cancelled: { color: "var(--text-dim)", bg: "transparent" },
  };
  const s = map[status] ?? map.idle;
  return (
    <span style={{
      fontFamily: "var(--mono)", fontSize: 10, fontWeight: 600,
      textTransform: "uppercase", letterSpacing: "0.06em",
      color: s.color, background: s.bg, padding: "2px 7px", borderRadius: 2,
    }}>{status}</span>
  );
}

export default function ASRPage() {
  const qc = useQueryClient();
  const [form, setForm] = useState<FormState>(DEFAULT);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState<string[]>(["[system] ASR module ready. Select a Whisper model and dataset."]);
  const logRef = useRef<HTMLDivElement>(null);
  const metrics = useMetricsStream(activeJob?.status === "running" ? activeJob.id : null, "/api/asr/jobs");

  const { data: whisperModels = [] } = useQuery({ queryKey: ["asr-models"],  queryFn: getASRModels });
  const { data: datasets = [] }      = useQuery({ queryKey: ["asr-datasets"], queryFn: getASRDatasets });
  const { data: sysStats }           = useQuery({ queryKey: ["system"],       queryFn: getSystemStats, refetchInterval: 2000 });

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((p) => ({ ...p, [k]: v }));

  const dirTouchedRef = useRef(false);
  const handleNameChange = (v: string) => {
    set("name", v);
    if (!dirTouchedRef.current) {
      const now = new Date();
      const dd = String(now.getDate()).padStart(2, "0");
      const mm = String(now.getMonth() + 1).padStart(2, "0");
      const yy = String(now.getFullYear()).slice(2);
      const hh = String(now.getHours() % 12 || 12);
      const min = String(now.getMinutes()).padStart(2, "0");
      const ap = now.getHours() < 12 ? "am" : "pm";
      const slug = v.trim().replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 40) || "asr_run";
      set("output_dir", `./outputs/${slug}_${dd}${mm}${yy}_${hh}${min}${ap}`);
    }
  };

  const toggleModule = (mod: string) =>
    set("target_modules",
      form.target_modules.includes(mod)
        ? form.target_modules.filter((m) => m !== mod)
        : [...form.target_modules, mod]
    );

  const { mutate: start, isPending } = useMutation({
    mutationFn: () => createASRJob({
      name: form.name || `asr-whisper-${Date.now()}`,
      peft_method: form.training_method,
      dataset_id: Number(form.dataset_id) || undefined,
      val_dataset_id: Number(form.val_dataset_id) || undefined,
      config: {
        model_path: form.model_path,
        task: form.task,
        language: form.language,
        quantization: form.quantization === "none" ? null : form.quantization,
        training_method: form.training_method,
        lora_r: form.lora_r,
        lora_alpha: form.lora_alpha,
        lora_dropout: form.lora_dropout,
        target_modules: form.target_modules,
        audio_col: form.audio_col,
        text_col: form.text_col,
        sample_rate: form.sample_rate,
        val_split: form.val_split,
        use_max_steps: form.use_max_steps,
        max_steps: form.max_steps,
        warmup_steps: form.warmup_steps,
        num_epochs: form.num_epochs,
        warmup_ratio: form.warmup_ratio,
        batch_size: form.batch_size,
        eval_batch_size: form.eval_batch_size,
        gradient_accumulation_steps: form.gradient_accumulation_steps,
        learning_rate: form.learning_rate,
        eval_steps: form.eval_steps,
        save_steps: form.save_steps,
        save_total_limit: form.save_total_limit,
        logging_steps: form.logging_steps,
        predict_with_generate: form.predict_with_generate,
        generation_max_length: form.generation_max_length,
        load_best_model_at_end: form.load_best_model_at_end,
        fp16: form.fp16,
        bf16: form.bf16,
        gradient_checkpointing: form.gradient_checkpointing,
        gpu_id: form.gpu_id === "auto" ? null : form.gpu_id,
        output_dir: form.output_dir,
      },
    }),
    onSuccess: (job) => {
      setActiveJob(job);
      setLogs([
        `[system] ASR Job #${job.id} "${job.name}" created.`,
        `[system] Model: ${form.model_path}  |  Task: ${form.task}  |  Lang: ${form.language === "auto" ? "auto-detect (multilingual)" : form.language}`,
        `[system] Method: ${form.training_method}${form.training_method !== "sft" ? `  |  LoRA r=${form.lora_r} α=${form.lora_alpha}` : "  (full fine-tune)"}`,
        `[system] Waiting for worker…`,
      ]);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (e: unknown) => {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setLogs((p) => [...p, `[error] ${msg}`]);
    },
  });

  const { mutate: stop } = useMutation({
    mutationFn: () => cancelASRJob(activeJob!.id),
    onSuccess: () => {
      setActiveJob((j) => j ? { ...j, status: "cancelled" } : j);
      setLogs((p) => [...p, "[system] Job cancelled."]);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  // Append metrics to log
  useEffect(() => {
    if (metrics.length === 0) return;
    const last = metrics[metrics.length - 1];
    const parts = [`[step ${last.step}]`];
    if (last.loss != null) parts.push(`loss=${last.loss.toFixed(4)}`);
    if (last.eval_loss != null) parts.push(`eval_loss=${last.eval_loss.toFixed(4)}`);
    if (last.learning_rate != null) parts.push(`lr=${last.learning_rate.toExponential(2)}`);
    // WER stored in reward field from MetricLoggingCallback
    if (last.reward != null) parts.push(`wer=${last.reward.toFixed(2)}%`);
    setLogs((p) => [...p, parts.join("  ")]);
  }, [metrics.length]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const status = activeJob?.status ?? "idle";

  return (
    <div className="lf-train-layout">

      {/* ── LEFT: Config ── */}
      <div className="lf-train-config">

        {/* Header badge */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--accent)", background: "var(--accent-dim)", padding: "2px 7px", borderRadius: 2 }}>
            ASR
          </span>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>
            Whisper Fine-Tuning — separate from LLM text training
          </span>
        </div>

        <Section title="Model" tooltip="Select which Whisper model to fine-tune and how to load it." />
        <div style={{ marginBottom: 8 }}>
          <label className="lf-label" style={{ display: "flex", alignItems: "center" }}>
            whisper model<Tooltip text="The Whisper model size to fine-tune. Larger models are more accurate but need more VRAM. tiny/base are for quick experiments, small/medium for production use, large-v3 for highest accuracy. You can also paste a local path or any HuggingFace repo." />
          </label>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 6 }}>
            {whisperModels.map((m) => (
              <button key={m.id} className={`lf-chip ${form.model_path === m.id ? "lf-chip-active" : ""}`}
                onClick={() => set("model_path", m.id)}>
                {m.id.replace("openai/", "")}
                <span style={{ marginLeft: 4, opacity: 0.6, fontSize: 9 }}>{m.params}</span>
              </button>
            ))}
          </div>
          <input className="lf-input" value={form.model_path} onChange={(e) => set("model_path", e.target.value)} placeholder="openai/whisper-large-v3 or local path" />
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="task" tooltip="Transcribe: output speech as text in the same language as the audio (standard ASR). Translate: output English text regardless of the audio language — useful for cross-lingual speech translation.">
            <div style={{ display: "flex", gap: 4 }}>
              {TASKS.map((t) => (
                <button key={t} className={`lf-chip ${form.task === t ? "lf-chip-active" : ""}`} style={{ flex: 1, justifyContent: "center" }}
                  onClick={() => set("task", t)}>{t}</button>
              ))}
            </div>
          </Field>
          <Field label="quantization" tooltip="Load model weights in reduced precision. 4-bit saves ~4× VRAM and is recommended for Whisper-large-v3 on 24GB GPUs. Use with qlora training method. None = full bfloat16/float16 precision.">
            <select className="lf-input lf-select" value={form.quantization} onChange={(e) => set("quantization", e.target.value)}>
              {QUANT_OPTIONS.map((q) => <option key={q}>{q}</option>)}
            </select>
          </Field>
          <Field label="device" tooltip="Which GPU to use for training. Auto selects the first available CUDA device. On multi-GPU machines, pin to a specific GPU to avoid memory conflicts.">
            <select className="lf-input lf-select" value={form.gpu_id} onChange={(e) => set("gpu_id", e.target.value)}>
              <option value="auto">auto</option>
              {sysStats?.gpu.map((g) => (
                <option key={g.index} value={String(g.index)}>
                  GPU {g.index} ({(g.total_mb / 1024).toFixed(0)}GB)
                </option>
              ))}
              {!sysStats?.cuda_available && <option value="cpu" disabled>CPU only</option>}
            </select>
          </Field>
        </div>

        <div style={{ marginBottom: 8 }}>
          <label className="lf-label" style={{ display: "flex", alignItems: "center" }}>
            language<Tooltip text="Target language for transcription/translation. Auto-detect lets Whisper identify the language per segment — recommended for multilingual or code-mixed audio (e.g. Bahasa Rojak). Setting an explicit language speeds up training and inference by removing the language detection step." /></label>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 6 }}>
            {LANG_PRESETS.map((l) => (
              <button key={l.value} className={`lf-chip ${form.language === l.value ? "lf-chip-active" : ""}`}
                onClick={() => set("language", l.value)}>{l.label}</button>
            ))}
          </div>
          <input className="lf-input" value={form.language} onChange={(e) => set("language", e.target.value)} placeholder="malay / english / auto" />
          {form.language === "auto" && (
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--amber)", marginTop: 4 }}>
              ↳ Auto-detect: Whisper decodes each segment in its detected language — recommended for Bahasa Rojak / code-mixed audio
            </div>
          )}
        </div>

        <Section title="Training Method" tooltip="How model weights are updated during training — full fine-tuning vs. lightweight adapter methods." />
        <div style={{ marginBottom: 8 }}>
          <label className="lf-label" style={{ display: "flex", alignItems: "center" }}>
            method<Tooltip text="SFT (full): all Whisper encoder+decoder weights are updated — most expressive but needs most VRAM (40GB+ for large-v3). LoRA: attaches small trainable adapters, trains in ~8GB. QLoRA: LoRA + 4-bit base weights, enables large-v3 on 12–16GB." />
          </label>
          <div className="lf-checkbox-group">
            {TRAINING_METHODS.map((m) => (
              <span key={m} className="lf-tt-wrap" style={{ marginLeft: 0 }}>
                <button className={`lf-chip ${form.training_method === m ? "lf-chip-active" : ""}`}
                  onClick={() => {
                    set("training_method", m);
                    if (m === "qlora") set("quantization", "4bit");
                    if (m === "sft") set("quantization", "none");
                  }}>{m === "sft" ? "SFT (full)" : m}</button>
                <span className="lf-tt-box">{TRAINING_METHOD_TIPS[m]}</span>
              </span>
            ))}
          </div>
          {form.training_method === "sft" && (
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginTop: 4 }}>
              Full supervised fine-tuning — all model weights trained, no adapter
            </div>
          )}
        </div>

        {form.training_method !== "sft" && (
          <>
            <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
              <Field label="lora rank (r)" tooltip="Adapter capacity — number of trainable dimensions per LoRA matrix. r=32–64 works well for Whisper ASR. Higher r improves WER on larger datasets but uses more VRAM and trains slower.">
                <input className="lf-input" type="number" value={form.lora_r} onChange={(e) => set("lora_r", +e.target.value)} />
              </Field>
              <Field label="lora alpha" tooltip="Scaling factor for adapter outputs. Effective update strength = alpha/r. Setting alpha = 2×r is the standard convention. Increase alpha if the adapter seems to have little effect on outputs.">
                <input className="lf-input" type="number" value={form.lora_alpha} onChange={(e) => set("lora_alpha", +e.target.value)} />
              </Field>
              <Field label="dropout" tooltip="Dropout probability on LoRA layers. Regularizes the adapter to prevent overfitting on small audio datasets. 0.0–0.1 is typical. Has no effect at inference time.">
                <input className="lf-input" type="number" step="0.01" value={form.lora_dropout} onChange={(e) => set("lora_dropout", +e.target.value)} />
              </Field>
            </div>
            <div style={{ marginBottom: 8 }}>
              <label className="lf-label" style={{ display: "flex", alignItems: "center" }}>
                target modules<Tooltip text="Whisper attention projection layers that receive LoRA adapters. q_proj + v_proj targets cross-attention query and value — sufficient for most ASR domain adaptation. Adding k_proj and o_proj increases capacity for highly accented or low-resource language tasks." />
              </label>
              <div className="lf-checkbox-group">
                {TARGET_MODS.map((mod) => (
                  <span key={mod} className="lf-tt-wrap" style={{ marginLeft: 0 }}>
                    <button className={`lf-chip ${form.target_modules.includes(mod) ? "lf-chip-active" : ""}`}
                      onClick={() => toggleModule(mod)}>{mod}</button>
                    <span className="lf-tt-box">{MOD_TIPS[mod]}</span>
                  </span>
                ))}
              </div>
            </div>
          </>
        )}

        <Section title="Dataset" tooltip="Configure the audio-text pairs used for training and validation." />
        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="train CSV" tooltip="CSV file with audio_path and text columns, registered via ASR Datasets. Each row maps an audio file path to its ground-truth transcription.">
            <select className="lf-input lf-select" value={form.dataset_id} onChange={(e) => set("dataset_id", e.target.value)}>
              <option value="">— select —</option>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_samples?.toLocaleString() ?? "?"})</option>)}
            </select>
          </Field>
          <Field label="val CSV (optional)" tooltip="A separate held-out CSV for validation. If left empty, a fraction of the training set is split off automatically based on val_split. A dedicated val set gives more reliable WER estimates.">
            <select className="lf-input lf-select" value={form.val_dataset_id} onChange={(e) => set("val_dataset_id", e.target.value)}>
              <option value="">— auto split —</option>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </Field>
        </div>

        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          <Field label="audio column" tooltip="Name of the column in your CSV that contains the path to each audio file. Must match exactly — case-sensitive.">
            <input className="lf-input" value={form.audio_col} onChange={(e) => set("audio_col", e.target.value)} />
          </Field>
          <Field label="text column" tooltip="Name of the column in your CSV containing the ground-truth transcription for each audio file.">
            <input className="lf-input" value={form.text_col} onChange={(e) => set("text_col", e.target.value)} />
          </Field>
          <Field label="val split %" tooltip="Fraction of training data reserved for validation when no separate val CSV is provided. 0.1 = 10%. Larger splits give better validation signal but reduce effective training data.">
            <input className="lf-input" type="number" step="0.05" min="0.01" max="0.5" value={form.val_split} onChange={(e) => set("val_split", +e.target.value)} />
          </Field>
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="sample rate (hz)" tooltip="Expected audio sample rate. Whisper requires 16,000 Hz. Audio files at other sample rates (e.g. 44100 Hz from recordings) will be resampled automatically by librosa before feature extraction.">
            <input className="lf-input" type="number" value={form.sample_rate} onChange={(e) => set("sample_rate", +e.target.value)} />
          </Field>
        </div>

        <Section title="Training" tooltip="Optimization schedule and batch configuration. These parameters have the most direct effect on WER improvement and training stability." />

        <div style={{ marginBottom: 8 }}>
          <label className="lf-label" style={{ display: "flex", alignItems: "center" }}>
            step control<Tooltip text="max_steps: train for a fixed number of gradient updates regardless of dataset size — recommended for ASR since dataset sizes vary widely. epochs: complete N full passes through the data — more intuitive but can over/under-train on unusual dataset sizes." />
          </label>
          <div className="lf-checkbox-group">
            <button className={`lf-chip ${form.use_max_steps ? "lf-chip-active" : ""}`} onClick={() => set("use_max_steps", true)}>max_steps</button>
            <button className={`lf-chip ${!form.use_max_steps ? "lf-chip-active" : ""}`} onClick={() => set("use_max_steps", false)}>epochs</button>
          </div>
        </div>

        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          {form.use_max_steps ? (
            <>
              <Field label="max steps" tooltip="Total gradient update steps. For Whisper LoRA, 3000–10000 steps is typical depending on dataset size. More data → more steps needed to see each example multiple times.">
                <input className="lf-input" type="number" value={form.max_steps} onChange={(e) => set("max_steps", +e.target.value)} />
              </Field>
              <Field label="warmup steps" tooltip="Number of steps to linearly ramp up the learning rate from 0 to its peak value. Prevents large destabilizing gradient updates at the very start of training. Typically 5–10% of max_steps.">
                <input className="lf-input" type="number" value={form.warmup_steps} onChange={(e) => set("warmup_steps", +e.target.value)} />
              </Field>
            </>
          ) : (
            <>
              <Field label="epochs" tooltip="Number of complete passes through the training CSV. 3–10 epochs is typical for ASR fine-tuning. Monitor eval_loss — stop when it plateaus or rises.">
                <input className="lf-input" type="number" value={form.num_epochs} onChange={(e) => set("num_epochs", +e.target.value)} />
              </Field>
              <Field label="warmup ratio" tooltip="Fraction of total epoch-steps used for learning rate warmup. 0.05 = 5% warmup. Alternative to warmup_steps when using epoch-based training.">
                <input className="lf-input" type="number" step="0.01" value={form.warmup_ratio} onChange={(e) => set("warmup_ratio", +e.target.value)} />
              </Field>
            </>
          )}
          <Field label="learning rate" tooltip="Optimizer step size. 1e-4 is a common starting point for Whisper LoRA. If WER improves slowly, try 2e-4. If training is unstable (loss spikes), reduce to 5e-5.">
            <input className="lf-input" type="number" step="0.00001" value={form.learning_rate} onChange={(e) => set("learning_rate", +e.target.value)} />
          </Field>
        </div>

        <div className="lf-row lf-row-4" style={{ marginBottom: 8 }}>
          <Field label="batch size" tooltip="Audio samples per GPU per step. ASR batches are VRAM-intensive — keep at 1–4 for Whisper-large. Increase only if GPU memory allows. Use grad_accum to simulate larger effective batches.">
            <input className="lf-input" type="number" value={form.batch_size} onChange={(e) => set("batch_size", +e.target.value)} />
          </Field>
          <Field label="grad accum" tooltip="Accumulate gradients over N steps before updating weights. Effective batch = batch_size × grad_accum. Use to simulate batch_size=16 without needing 16 samples in VRAM at once.">
            <input className="lf-input" type="number" value={form.gradient_accumulation_steps} onChange={(e) => set("gradient_accumulation_steps", +e.target.value)} />
          </Field>
          <Field label="eval steps" tooltip="Run evaluation on the validation set every N training steps. Generates eval_loss and WER metrics. Frequent evaluation (e.g. every 200 steps) helps catch overfitting early but slows training slightly.">
            <input className="lf-input" type="number" value={form.eval_steps} onChange={(e) => set("eval_steps", +e.target.value)} />
          </Field>
          <Field label="save steps" tooltip="Save a checkpoint to disk every N steps. Allows resuming if training is interrupted. Each checkpoint for Whisper-large is ~3GB — use save_total_limit to cap disk usage.">
            <input className="lf-input" type="number" value={form.save_steps} onChange={(e) => set("save_steps", +e.target.value)} />
          </Field>
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="save total limit" tooltip="Maximum number of checkpoints to keep on disk. When exceeded, the oldest checkpoint is deleted. Set to 1–2 to save disk space, or higher if you need to compare multiple checkpoints.">
            <input className="lf-input" type="number" value={form.save_total_limit} onChange={(e) => set("save_total_limit", +e.target.value)} />
          </Field>
          <Field label="logging steps" tooltip="Emit training metrics (loss, learning rate) every N steps to the log console. Lower values give finer-grained loss curves. 25–100 is typical for ASR training.">
            <input className="lf-input" type="number" value={form.logging_steps} onChange={(e) => set("logging_steps", +e.target.value)} />
          </Field>
        </div>

        <Section title="Generation" tooltip="Settings for how Whisper decodes audio during evaluation runs." />
        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="generation max length" tooltip="Maximum number of tokens Whisper can generate per audio segment during evaluation. 225 tokens ≈ 30 seconds of speech at normal speaking pace. Increase for longer audio clips or slower speech.">
            <input className="lf-input" type="number" value={form.generation_max_length} onChange={(e) => set("generation_max_length", +e.target.value)} />
          </Field>
        </div>
        <div style={{ display: "flex", gap: 14, marginBottom: 8, flexWrap: "wrap" }}>
          <Toggle label="predict_with_generate" tooltip="Use model.generate() for evaluation instead of teacher-forced decoding. Required for computing real WER — teacher forcing gives artificially low loss and does not reflect actual transcription quality. Keep enabled." checked={form.predict_with_generate} onChange={(v) => set("predict_with_generate", v)} />
          <Toggle label="load_best_model_at_end" tooltip="After training completes, reload the checkpoint with the lowest eval_loss rather than the final step. Ensures you get the best-performing weights even if the model started overfitting near the end of training." checked={form.load_best_model_at_end} onChange={(v) => set("load_best_model_at_end", v)} />
        </div>

        <Section title="Precision" tooltip="Mixed-precision training settings that affect VRAM usage and numerical stability." />
        <div style={{ display: "flex", gap: 14, marginBottom: 8, flexWrap: "wrap" }}>
          <Toggle label="fp16" tooltip="Train in float16 precision. Compatible with most NVIDIA GPUs. May produce NaN losses on Whisper-large — switch to bf16 if you see NaN. Mutually exclusive with bf16." checked={form.fp16} onChange={(v) => { set("fp16", v); if (v) set("bf16", false); }} />
          <Toggle label="bf16" tooltip="Train in bfloat16 precision. More numerically stable than fp16 for large Whisper models — recommended for Whisper-medium and larger. Requires Ampere+ GPU (RTX 30xx, A100). Mutually exclusive with fp16." checked={form.bf16} onChange={(v) => { set("bf16", v); if (v) set("fp16", false); }} />
          <Toggle label="grad checkpointing" tooltip="Recomputes activations during the backward pass instead of storing them — reduces VRAM by ~30–40%. Especially important for Whisper-large on 24GB GPUs. Adds ~20% training time overhead." checked={form.gradient_checkpointing} onChange={(v) => set("gradient_checkpointing", v)} />
        </div>

        <Section title="Output" tooltip="Run labeling and checkpoint output location." />
        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="run name" tooltip="Label for this training run in the job list and log output. Does not affect training.">
            <input className="lf-input" value={form.name} onChange={(e) => handleNameChange(e.target.value)} placeholder="whisper-malay-ft" />
          </Field>
          <Field label="output dir" tooltip="Directory where checkpoints and the final adapter weights are saved. Use a unique path per run to avoid overwriting previous results. Large models can use significant disk space — ensure sufficient free space.">
            <input className="lf-input" value={form.output_dir} onChange={(e) => { dirTouchedRef.current = true; set("output_dir", e.target.value); }} />
          </Field>
        </div>

        {/* Start / Stop */}
        <div style={{ display: "flex", gap: 8, paddingTop: 12, borderTop: "1px solid var(--border)", marginTop: 4 }}>
          {status === "running" ? (
            <button className="lf-btn lf-btn-danger" style={{ flex: 1 }} onClick={() => stop()}>
              ■ Abort Training
            </button>
          ) : (
            <button
              className="lf-btn lf-btn-primary"
              style={{ flex: 1 }}
              disabled={isPending || !form.model_path || !form.dataset_id}
              onClick={() => start()}
            >
              {isPending ? <><span className="lf-spin" /> Starting…</> : "▶ Start ASR Training"}
            </button>
          )}
          {activeJob && (
            <Link href={`/jobs/${activeJob.id}`} className="lf-btn lf-btn-ghost">View Job →</Link>
          )}
          <Link href="/asr/datasets" className="lf-btn lf-btn-ghost" style={{ fontSize: 11 }}>Datasets ↗</Link>
        </div>
      </div>

      {/* ── RIGHT: Output ── */}
      <div className="lf-train-output">
        {/* Status bar */}
        <div style={{
          borderBottom: "1px solid var(--border)", padding: "0 14px", height: 32,
          display: "flex", alignItems: "center", gap: 16,
          background: "var(--bg-panel)", flexShrink: 0,
        }}>
          <StatusPill status={status} />
          {activeJob && (
            <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
              job #{activeJob.id} · asr_whisper · {form.training_method}
            </span>
          )}
          {metrics.length > 0 && (() => {
            const last = metrics[metrics.length - 1];
            return (
              <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
                step {last.step}
                {last.loss != null && ` · loss ${last.loss.toFixed(4)}`}
                {last.reward != null && ` · WER ${last.reward.toFixed(2)}%`}
              </span>
            );
          })()}
        </div>

        {/* WER note */}
        <div style={{ padding: "6px 14px", borderBottom: "1px solid var(--border)", background: "var(--bg-panel)", flexShrink: 0 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>
            Primary metric: <span style={{ color: "var(--amber)" }}>WER ↓</span> (Word Error Rate — lower is better) · displayed in reward channel
          </span>
        </div>

        {/* Charts */}
        <div style={{ flex: "0 0 auto", borderBottom: "1px solid var(--border)", padding: "10px 14px" }}>
          <MetricsPanel metrics={metrics} />
        </div>

        {/* Log */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", padding: "10px 14px" }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>
            Output Log
          </div>
          <div ref={logRef} className="lf-console" style={{ flex: 1 }}>
            {logs.map((line, i) => (
              <div key={i} style={{ color: line.startsWith("[error]") ? "var(--red)" : line.startsWith("[system]") ? "var(--text-dim)" : "var(--green)" }}>
                {line}
              </div>
            ))}
            {status === "running" && <span className="lf-cursor">█</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
