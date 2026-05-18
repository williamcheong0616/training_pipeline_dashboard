"use client";
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { createJob, getModels, getDatasets, cancelJob } from "@/lib/api";
import { useMetricsStream } from "@/lib/sse";
import MetricsPanel from "@/components/MetricsPanel";
import type { Job } from "@/types";

const METHODS    = ["sft","unsupervised","dpo","rm","kto","orpo"] as const;
const PEFT       = ["lora","qlora","dora","full"] as const;
const QUANT      = ["none","4bit","8bit"] as const;
const SCHEDULERS = ["cosine","linear","constant","cosine_with_restarts","polynomial"] as const;
const TARGET_MODS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"] as const;

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
  model_id: string;
  template: string;
  quantization: string;
  flash_attention: boolean;
  training_method: string;
  peft_method: string;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  target_modules: string[];
  dataset_id: string;
  dataset_format: string;
  max_seq_length: number;
  packing: boolean;
  learning_rate: number;
  num_epochs: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  lr_scheduler: string;
  warmup_ratio: number;
  max_grad_norm: number;
  logging_steps: number;
  save_steps: number;
  seed: number;
  output_dir: string;
  bf16: boolean;
  fp16: boolean;
  gradient_checkpointing: boolean;
  dataloader_num_workers: number;
  resume_from_checkpoint: string;
};

const DEFAULT: FormState = {
  name: "", model_id: "", template: "alpaca", quantization: "none",
  flash_attention: false, training_method: "sft", peft_method: "lora",
  lora_r: 16, lora_alpha: 32, lora_dropout: 0.05,
  target_modules: ["q_proj","v_proj"],
  dataset_id: "", dataset_format: "alpaca", max_seq_length: 2048, packing: false,
  learning_rate: 2e-4, num_epochs: 3, batch_size: 4,
  gradient_accumulation_steps: 4, lr_scheduler: "cosine", warmup_ratio: 0.05,
  max_grad_norm: 1.0, logging_steps: 10, save_steps: 500, seed: 42,
  output_dir: "./outputs/run",
  bf16: true, fp16: false, gradient_checkpointing: true,
  dataloader_num_workers: 4, resume_from_checkpoint: "",
};

export default function TrainPage() {
  const router = useRouter();
  const qc = useQueryClient();
  const [form, setForm] = useState<FormState>(DEFAULT);
  const [activeJob, setActiveJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState<string[]>(["[system] Ready. Configure parameters and press Start."]);
  const logRef = useRef<HTMLDivElement>(null);
  const metrics = useMetricsStream(activeJob?.status === "running" ? activeJob.id : null);

  const { data: models = [] } = useQuery({ queryKey: ["models"], queryFn: getModels });
  const { data: datasets = [] } = useQuery({ queryKey: ["datasets"], queryFn: getDatasets });

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setForm((p) => ({ ...p, [k]: v }));

  const toggleModule = (mod: string) =>
    set("target_modules",
      form.target_modules.includes(mod)
        ? form.target_modules.filter((m) => m !== mod)
        : [...form.target_modules, mod]
    );

  const { mutate: start, isPending } = useMutation({
    mutationFn: () => {
      const m = models.find((x) => x.id === Number(form.model_id));
      const d = datasets.find((x) => x.id === Number(form.dataset_id));
      const jobName = form.name || `${form.training_method}-${Date.now()}`;
      return createJob({
        name: jobName,
        training_method: form.training_method,
        peft_method: form.peft_method,
        model_id: Number(form.model_id) || undefined,
        dataset_id: Number(form.dataset_id) || undefined,
        config: {
          model_path: m?.local_path ?? m?.hf_repo ?? "",
          dataset_path: d?.path ?? "",
          dataset_format: form.dataset_format,
          template: form.template,
          peft_method: form.peft_method,
          quantization: form.quantization === "none" ? null : form.quantization,
          use_flash_attention: form.flash_attention,
          lora_r: form.lora_r,
          lora_alpha: form.lora_alpha,
          lora_dropout: form.lora_dropout,
          target_modules: form.target_modules,
          max_seq_length: form.max_seq_length,
          packing: form.packing,
          learning_rate: form.learning_rate,
          num_epochs: form.num_epochs,
          batch_size: form.batch_size,
          gradient_accumulation_steps: form.gradient_accumulation_steps,
          lr_scheduler: form.lr_scheduler,
          warmup_ratio: form.warmup_ratio,
          max_grad_norm: form.max_grad_norm,
          logging_steps: form.logging_steps,
          save_steps: form.save_steps,
          seed: form.seed,
          output_dir: form.output_dir,
          bf16: form.bf16,
          fp16: form.fp16,
          gradient_checkpointing: form.gradient_checkpointing,
          dataloader_num_workers: form.dataloader_num_workers,
          resume_from_checkpoint: form.resume_from_checkpoint || null,
        },
      });
    },
    onSuccess: (job) => {
      setActiveJob(job);
      setLogs([
        `[system] Job #${job.id} "${job.name}" created.`,
        `[system] Method: ${job.training_method.toUpperCase()} | PEFT: ${job.peft_method}`,
        `[system] Waiting for worker to pick up job...`,
      ]);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: (e: unknown) => {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setLogs((p) => [...p, `[error] Failed to create job: ${msg}`]);
    },
  });

  const { mutate: stop } = useMutation({
    mutationFn: () => cancelJob(activeJob!.id),
    onSuccess: () => {
      setActiveJob((j) => j ? { ...j, status: "cancelled" } : j);
      setLogs((p) => [...p, "[system] Job cancelled."]);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  // Append metric updates to log
  useEffect(() => {
    if (metrics.length === 0) return;
    const last = metrics[metrics.length - 1];
    const parts = [`[step ${last.step}]`];
    if (last.loss != null) parts.push(`loss=${last.loss.toFixed(4)}`);
    if (last.learning_rate != null) parts.push(`lr=${last.learning_rate.toExponential(2)}`);
    if (last.epoch != null) parts.push(`epoch=${last.epoch.toFixed(2)}`);
    if (last.grad_norm != null) parts.push(`grad_norm=${last.grad_norm.toFixed(3)}`);
    setLogs((p) => [...p, parts.join("  ")]);
  }, [metrics.length]);

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const status = activeJob?.status ?? "idle";

  return (
    <div style={{ display: "grid", gridTemplateColumns: "420px 1fr", height: "calc(100vh - 40px)", overflow: "hidden" }}>

      {/* ── LEFT: Config ── */}
      <div style={{
        borderRight: "1px solid var(--border)",
        overflowY: "auto",
        padding: "10px 12px",
        display: "flex", flexDirection: "column", gap: 0,
      }}>
        <Section title="Model" />

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="run name">
            <input className="lf-input" value={form.name} onChange={(e) => set("name", e.target.value)} placeholder="my-sft-run" />
          </Field>
          <Field label="base model">
            <select className="lf-input lf-select" value={form.model_id} onChange={(e) => set("model_id", e.target.value)}>
              <option value="">— select —</option>
              {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </Field>
        </div>

        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          <Field label="template">
            <select className="lf-input lf-select" value={form.template} onChange={(e) => set("template", e.target.value)}>
              {["alpaca","chatml","llama3","mistral","qwen","phi3","gemma"].map((t) => <option key={t}>{t}</option>)}
            </select>
          </Field>
          <Field label="quantization">
            <select className="lf-input lf-select" value={form.quantization} onChange={(e) => set("quantization", e.target.value)}>
              {QUANT.map((q) => <option key={q}>{q}</option>)}
            </select>
          </Field>
          <div style={{ paddingTop: 16 }}>
            <Toggle label="flash attn" checked={form.flash_attention} onChange={(v) => set("flash_attention", v)} />
          </div>
        </div>

        <Section title="Method" />

        <div style={{ marginBottom: 8 }}>
          <label className="lf-label">training stage</label>
          <div className="lf-checkbox-group">
            {METHODS.map((m) => (
              <button key={m} className={`lf-chip ${form.training_method === m ? "lf-chip-active" : ""}`}
                onClick={() => set("training_method", m)}>{m}</button>
            ))}
          </div>
        </div>

        <div style={{ marginBottom: 8 }}>
          <label className="lf-label">finetuning type</label>
          <div className="lf-checkbox-group">
            {PEFT.map((p) => (
              <button key={p} className={`lf-chip ${form.peft_method === p ? "lf-chip-active" : ""}`}
                onClick={() => set("peft_method", p)}>{p}</button>
            ))}
          </div>
        </div>

        {form.peft_method !== "full" && (
          <>
            <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
              <Field label="lora rank (r)">
                <input className="lf-input" type="number" value={form.lora_r} onChange={(e) => set("lora_r", +e.target.value)} />
              </Field>
              <Field label="lora alpha">
                <input className="lf-input" type="number" value={form.lora_alpha} onChange={(e) => set("lora_alpha", +e.target.value)} />
              </Field>
              <Field label="lora dropout">
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
          <Field label="dataset">
            <select className="lf-input lf-select" value={form.dataset_id} onChange={(e) => set("dataset_id", e.target.value)}>
              <option value="">— select —</option>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_samples?.toLocaleString() ?? "?"})</option>)}
            </select>
          </Field>
          <Field label="format">
            <select className="lf-input lf-select" value={form.dataset_format} onChange={(e) => set("dataset_format", e.target.value)}>
              {["alpaca","sharegpt","plain_text"].map((f) => <option key={f}>{f}</option>)}
            </select>
          </Field>
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="cutoff / max seq length">
            <input className="lf-input" type="number" step="128" value={form.max_seq_length} onChange={(e) => set("max_seq_length", +e.target.value)} />
          </Field>
          <div style={{ paddingTop: 16 }}>
            <Toggle label="packing" checked={form.packing} onChange={(v) => set("packing", v)} />
          </div>
        </div>

        <Section title="Training Parameters" />

        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          <Field label="learning rate">
            <input className="lf-input" type="number" step="0.00001" value={form.learning_rate} onChange={(e) => set("learning_rate", +e.target.value)} />
          </Field>
          <Field label="epochs">
            <input className="lf-input" type="number" value={form.num_epochs} onChange={(e) => set("num_epochs", +e.target.value)} />
          </Field>
          <Field label="batch size / device">
            <input className="lf-input" type="number" value={form.batch_size} onChange={(e) => set("batch_size", +e.target.value)} />
          </Field>
        </div>

        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          <Field label="grad accum steps">
            <input className="lf-input" type="number" value={form.gradient_accumulation_steps} onChange={(e) => set("gradient_accumulation_steps", +e.target.value)} />
          </Field>
          <Field label="lr scheduler">
            <select className="lf-input lf-select" value={form.lr_scheduler} onChange={(e) => set("lr_scheduler", e.target.value)}>
              {SCHEDULERS.map((s) => <option key={s}>{s}</option>)}
            </select>
          </Field>
          <Field label="warmup ratio">
            <input className="lf-input" type="number" step="0.01" value={form.warmup_ratio} onChange={(e) => set("warmup_ratio", +e.target.value)} />
          </Field>
        </div>

        <div className="lf-row lf-row-4" style={{ marginBottom: 8 }}>
          <Field label="max grad norm">
            <input className="lf-input" type="number" step="0.1" value={form.max_grad_norm} onChange={(e) => set("max_grad_norm", +e.target.value)} />
          </Field>
          <Field label="logging steps">
            <input className="lf-input" type="number" value={form.logging_steps} onChange={(e) => set("logging_steps", +e.target.value)} />
          </Field>
          <Field label="save steps">
            <input className="lf-input" type="number" value={form.save_steps} onChange={(e) => set("save_steps", +e.target.value)} />
          </Field>
          <Field label="seed">
            <input className="lf-input" type="number" value={form.seed} onChange={(e) => set("seed", +e.target.value)} />
          </Field>
        </div>

        <Field label="output dir">
          <input className="lf-input" value={form.output_dir} onChange={(e) => set("output_dir", e.target.value)} />
        </Field>

        <Section title="Advanced" />

        <div style={{ display: "flex", flexWrap: "wrap", gap: 14, marginBottom: 8, paddingTop: 4 }}>
          <Toggle label="bf16" checked={form.bf16} onChange={(v) => { set("bf16", v); if (v) set("fp16", false); }} />
          <Toggle label="fp16" checked={form.fp16} onChange={(v) => { set("fp16", v); if (v) set("bf16", false); }} />
          <Toggle label="grad ckpt" checked={form.gradient_checkpointing} onChange={(v) => set("gradient_checkpointing", v)} />
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="dataloader workers">
            <input className="lf-input" type="number" value={form.dataloader_num_workers} onChange={(e) => set("dataloader_num_workers", +e.target.value)} />
          </Field>
          <Field label="resume from checkpoint">
            <input className="lf-input" value={form.resume_from_checkpoint} onChange={(e) => set("resume_from_checkpoint", e.target.value)} placeholder="path or empty" />
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
              disabled={isPending || !form.model_id || !form.dataset_id}
              onClick={() => start()}
            >
              {isPending ? <><span className="lf-spin" /> Starting…</> : "▶ Start Training"}
            </button>
          )}
          {activeJob && (
            <a href={`/jobs/${activeJob.id}`} className="lf-btn lf-btn-ghost">
              View Job →
            </a>
          )}
        </div>
      </div>

      {/* ── RIGHT: Output ── */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Status bar */}
        <div style={{
          borderBottom: "1px solid var(--border)",
          padding: "0 14px",
          height: 32,
          display: "flex", alignItems: "center", gap: 16,
          background: "var(--bg-panel)", flexShrink: 0,
        }}>
          <StatusPill status={status} />
          {activeJob && (
            <>
              <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
                job #{activeJob.id} · {activeJob.training_method.toUpperCase()} · {activeJob.peft_method}
              </span>
              {metrics.length > 0 && (
                <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
                  step {metrics[metrics.length - 1].step}
                  {metrics[metrics.length - 1].loss != null && ` · loss ${metrics[metrics.length - 1].loss!.toFixed(4)}`}
                </span>
              )}
            </>
          )}
        </div>

        {/* Charts */}
        <div style={{ flex: "0 0 auto", borderBottom: "1px solid var(--border)", padding: "10px 14px" }}>
          <MetricsPanel metrics={metrics} />
        </div>

        {/* Log console */}
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
      color: s.color, background: s.bg,
      padding: "2px 7px", borderRadius: 2,
    }}>
      {status}
    </span>
  );
}
