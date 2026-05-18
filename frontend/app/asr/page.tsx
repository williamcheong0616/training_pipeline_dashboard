"use client";
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { createASRJob, cancelASRJob, getASRModels, getASRDatasets } from "@/lib/api";
import { useMetricsStream } from "@/lib/sse";
import MetricsPanel from "@/components/MetricsPanel";
import type { Job } from "@/types";
import Link from "next/link";

const TRAINING_METHODS = ["sft", "lora", "qlora"] as const;
const QUANT_OPTIONS    = ["none", "4bit", "8bit"] as const;
const TASKS            = ["transcribe", "translate"] as const;
const TARGET_MODS      = ["q_proj", "k_proj", "v_proj", "o_proj"] as const;
const LANG_PRESETS     = [
  { label: "Auto-detect", value: "auto" },
  { label: "Malay",       value: "malay" },
  { label: "English",     value: "english" },
  { label: "Chinese",     value: "chinese" },
  { label: "Tamil",       value: "tamil" },
] as const;

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="lf-label">{label}</label>
      {children}
    </div>
  );
}

function Section({ title }: { title: string }) {
  return <div className="lf-section" style={{ marginTop: 12 }}>{title}</div>;
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="lf-toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="lf-toggle-track" />
      {label}
    </label>
  );
}

type FormState = {
  name: string;
  model_path: string;
  task: string;
  language: string;
  quantization: string;
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
  task: "transcribe", language: "auto", quantization: "none",
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
  output_dir: "./outputs/asr_run",
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
  const metrics = useMetricsStream(activeJob?.status === "running" ? activeJob.id : null);

  const { data: whisperModels = [] } = useQuery({ queryKey: ["asr-models"], queryFn: getASRModels });
  const { data: datasets = [] } = useQuery({ queryKey: ["asr-datasets"], queryFn: getASRDatasets });

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((p) => ({ ...p, [k]: v }));

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
    <div style={{ display: "grid", gridTemplateColumns: "420px 1fr", height: "calc(100vh - 40px)", overflow: "hidden" }}>

      {/* ── LEFT: Config ── */}
      <div style={{ borderRight: "1px solid var(--border)", overflowY: "auto", padding: "10px 12px" }}>

        {/* Header badge */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--accent)", background: "var(--accent-dim)", padding: "2px 7px", borderRadius: 2 }}>
            ASR
          </span>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>
            Whisper Fine-Tuning — separate from LLM text training
          </span>
        </div>

        <Section title="Model" />
        <div style={{ marginBottom: 8 }}>
          <label className="lf-label">whisper model</label>
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
          <Field label="task">
            <div style={{ display: "flex", gap: 4 }}>
              {TASKS.map((t) => (
                <button key={t} className={`lf-chip ${form.task === t ? "lf-chip-active" : ""}`} style={{ flex: 1, justifyContent: "center" }}
                  onClick={() => set("task", t)}>{t}</button>
              ))}
            </div>
          </Field>
          <Field label="quantization">
            <select className="lf-input lf-select" value={form.quantization} onChange={(e) => set("quantization", e.target.value)}>
              {QUANT_OPTIONS.map((q) => <option key={q}>{q}</option>)}
            </select>
          </Field>
        </div>

        <div style={{ marginBottom: 8 }}>
          <label className="lf-label">language</label>
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

        <Section title="Training Method" />
        <div style={{ marginBottom: 8 }}>
          <label className="lf-label">method</label>
          <div className="lf-checkbox-group">
            {TRAINING_METHODS.map((m) => (
              <button key={m} className={`lf-chip ${form.training_method === m ? "lf-chip-active" : ""}`}
                onClick={() => {
                  set("training_method", m);
                  // QLoRA auto-selects 4bit quantization
                  if (m === "qlora") set("quantization", "4bit");
                  if (m === "sft") set("quantization", "none");
                }}>{m === "sft" ? "SFT (full)" : m}</button>
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
              <Field label="lora rank (r)">
                <input className="lf-input" type="number" value={form.lora_r} onChange={(e) => set("lora_r", +e.target.value)} />
              </Field>
              <Field label="lora alpha">
                <input className="lf-input" type="number" value={form.lora_alpha} onChange={(e) => set("lora_alpha", +e.target.value)} />
              </Field>
              <Field label="dropout">
                <input className="lf-input" type="number" step="0.01" value={form.lora_dropout} onChange={(e) => set("lora_dropout", +e.target.value)} />
              </Field>
            </div>
            <div style={{ marginBottom: 8 }}>
              <label className="lf-label">target modules</label>
              <div className="lf-checkbox-group">
                {TARGET_MODS.map((mod) => (
                  <button key={mod} className={`lf-chip ${form.target_modules.includes(mod) ? "lf-chip-active" : ""}`}
                    onClick={() => toggleModule(mod)}>{mod}</button>
                ))}
              </div>
            </div>
          </>
        )}

        <Section title="Dataset" />
        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="train CSV">
            <select className="lf-input lf-select" value={form.dataset_id} onChange={(e) => set("dataset_id", e.target.value)}>
              <option value="">— select —</option>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_samples?.toLocaleString() ?? "?"})</option>)}
            </select>
          </Field>
          <Field label="val CSV (optional)">
            <select className="lf-input lf-select" value={form.val_dataset_id} onChange={(e) => set("val_dataset_id", e.target.value)}>
              <option value="">— auto split —</option>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </Field>
        </div>

        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          <Field label="audio column">
            <input className="lf-input" value={form.audio_col} onChange={(e) => set("audio_col", e.target.value)} />
          </Field>
          <Field label="text column">
            <input className="lf-input" value={form.text_col} onChange={(e) => set("text_col", e.target.value)} />
          </Field>
          <Field label="val split %">
            <input className="lf-input" type="number" step="0.05" min="0.01" max="0.5" value={form.val_split} onChange={(e) => set("val_split", +e.target.value)} />
          </Field>
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="sample rate (hz)">
            <input className="lf-input" type="number" value={form.sample_rate} onChange={(e) => set("sample_rate", +e.target.value)} />
          </Field>
        </div>

        <Section title="Training" />

        <div style={{ marginBottom: 8 }}>
          <label className="lf-label">step control</label>
          <div className="lf-checkbox-group">
            <button className={`lf-chip ${form.use_max_steps ? "lf-chip-active" : ""}`} onClick={() => set("use_max_steps", true)}>max_steps</button>
            <button className={`lf-chip ${!form.use_max_steps ? "lf-chip-active" : ""}`} onClick={() => set("use_max_steps", false)}>epochs</button>
          </div>
        </div>

        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          {form.use_max_steps ? (
            <>
              <Field label="max steps">
                <input className="lf-input" type="number" value={form.max_steps} onChange={(e) => set("max_steps", +e.target.value)} />
              </Field>
              <Field label="warmup steps">
                <input className="lf-input" type="number" value={form.warmup_steps} onChange={(e) => set("warmup_steps", +e.target.value)} />
              </Field>
            </>
          ) : (
            <>
              <Field label="epochs">
                <input className="lf-input" type="number" value={form.num_epochs} onChange={(e) => set("num_epochs", +e.target.value)} />
              </Field>
              <Field label="warmup ratio">
                <input className="lf-input" type="number" step="0.01" value={form.warmup_ratio} onChange={(e) => set("warmup_ratio", +e.target.value)} />
              </Field>
            </>
          )}
          <Field label="learning rate">
            <input className="lf-input" type="number" step="0.00001" value={form.learning_rate} onChange={(e) => set("learning_rate", +e.target.value)} />
          </Field>
        </div>

        <div className="lf-row lf-row-4" style={{ marginBottom: 8 }}>
          <Field label="batch size">
            <input className="lf-input" type="number" value={form.batch_size} onChange={(e) => set("batch_size", +e.target.value)} />
          </Field>
          <Field label="grad accum">
            <input className="lf-input" type="number" value={form.gradient_accumulation_steps} onChange={(e) => set("gradient_accumulation_steps", +e.target.value)} />
          </Field>
          <Field label="eval steps">
            <input className="lf-input" type="number" value={form.eval_steps} onChange={(e) => set("eval_steps", +e.target.value)} />
          </Field>
          <Field label="save steps">
            <input className="lf-input" type="number" value={form.save_steps} onChange={(e) => set("save_steps", +e.target.value)} />
          </Field>
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="save total limit">
            <input className="lf-input" type="number" value={form.save_total_limit} onChange={(e) => set("save_total_limit", +e.target.value)} />
          </Field>
          <Field label="logging steps">
            <input className="lf-input" type="number" value={form.logging_steps} onChange={(e) => set("logging_steps", +e.target.value)} />
          </Field>
        </div>

        <Section title="Generation" />
        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="generation max length">
            <input className="lf-input" type="number" value={form.generation_max_length} onChange={(e) => set("generation_max_length", +e.target.value)} />
          </Field>
        </div>
        <div style={{ display: "flex", gap: 14, marginBottom: 8, flexWrap: "wrap" }}>
          <Toggle label="predict_with_generate" checked={form.predict_with_generate} onChange={(v) => set("predict_with_generate", v)} />
          <Toggle label="load_best_model_at_end" checked={form.load_best_model_at_end} onChange={(v) => set("load_best_model_at_end", v)} />
        </div>

        <Section title="Precision" />
        <div style={{ display: "flex", gap: 14, marginBottom: 8, flexWrap: "wrap" }}>
          <Toggle label="fp16" checked={form.fp16} onChange={(v) => { set("fp16", v); if (v) set("bf16", false); }} />
          <Toggle label="bf16" checked={form.bf16} onChange={(v) => { set("bf16", v); if (v) set("fp16", false); }} />
          <Toggle label="grad checkpointing" checked={form.gradient_checkpointing} onChange={(v) => set("gradient_checkpointing", v)} />
        </div>

        <Section title="Output" />
        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="run name">
            <input className="lf-input" value={form.name} onChange={(e) => set("name", e.target.value)} placeholder="whisper-malay-ft" />
          </Field>
          <Field label="output dir">
            <input className="lf-input" value={form.output_dir} onChange={(e) => set("output_dir", e.target.value)} />
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
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
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
