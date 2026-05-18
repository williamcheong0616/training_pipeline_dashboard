"use client";
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { createJob, getModels, getDatasets, cancelJob, getSystemStats } from "@/lib/api";
import { useMetricsStream } from "@/lib/sse";
import MetricsPanel from "@/components/MetricsPanel";
import Tooltip from "@/components/Tooltip";
import type { Job } from "@/types";

const METHODS    = ["sft","unsupervised","dpo","rm","kto","orpo"] as const;
const PEFT       = ["lora","qlora","dora","full"] as const;
const QUANT      = ["none","4bit","8bit"] as const;
const SCHEDULERS = ["cosine","linear","constant","cosine_with_restarts","polynomial"] as const;
const TARGET_MODS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"] as const;

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
  model_id: string;
  template: string;
  quantization: string;
  gpu_id: string;
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
  name: "", model_id: "", template: "alpaca", quantization: "none", gpu_id: "auto",
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

  const { data: models = [] }   = useQuery({ queryKey: ["models"],   queryFn: getModels });
  const { data: datasets = [] } = useQuery({ queryKey: ["datasets"], queryFn: getDatasets });
  const { data: sysStats }      = useQuery({ queryKey: ["system"],   queryFn: getSystemStats, refetchInterval: 2000 });

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
          gpu_id: form.gpu_id === "auto" ? null : form.gpu_id,
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
        <Section title="Model" tooltip="Configure which pre-trained model to load and how to load it." />

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="run name" tooltip="A label for this training run. Appears in the job list and log output. Does not affect training results.">
            <input className="lf-input" value={form.name} onChange={(e) => set("name", e.target.value)} placeholder="my-sft-run" />
          </Field>
          <Field label="base model" tooltip="The pre-trained model to fine-tune. Must be downloaded first via the Models page. The model's architecture determines which templates and modules are available.">
            <select className="lf-input lf-select" value={form.model_id} onChange={(e) => set("model_id", e.target.value)}>
              <option value="">— select —</option>
              {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </Field>
        </div>

        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          <Field label="template" tooltip="Chat prompt template that wraps instruction/response pairs before tokenization. Must match the model's original training format — a wrong template causes garbled outputs. E.g. llama3 for Meta-Llama-3, chatml for Qwen/Mistral.">
            <select className="lf-input lf-select" value={form.template} onChange={(e) => set("template", e.target.value)}>
              {["alpaca","chatml","llama3","mistral","qwen","phi3","gemma"].map((t) => <option key={t}>{t}</option>)}
            </select>
          </Field>
          <Field label="quantization" tooltip="Load model weights in reduced precision to save VRAM. 4-bit ≈ 4× VRAM reduction; 8-bit ≈ 2×. Required for fitting 13B+ models on consumer GPUs. Use with QLoRA for best results. None = full precision.">
            <select className="lf-input lf-select" value={form.quantization} onChange={(e) => set("quantization", e.target.value)}>
              {QUANT.map((q) => <option key={q}>{q}</option>)}
            </select>
          </Field>
          <Field label="device" tooltip="Which GPU to use for training. Auto selects the first available CUDA device. On multi-GPU machines, pin to a specific GPU to avoid contention with other workloads.">
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
          <Toggle label="flash attn" tooltip="FlashAttention-2 recomputes attention in fused CUDA kernels — speeds up training ~20–40% and reduces VRAM on long sequences. Requires Ampere+ GPU (RTX 30xx / A100) and the flash-attn package installed." checked={form.flash_attention} onChange={(v) => set("flash_attention", v)} />
        </div>

        <Section title="Method" tooltip="Choose the training objective and parameter-efficiency strategy." />

        <div style={{ marginBottom: 8 }}>
          <label className="lf-label" style={{ display: "flex", alignItems: "center" }}>
            training stage<Tooltip text="SFT: supervised fine-tuning on instruction-response pairs (most common). DPO: Direct Preference Optimization — trains on chosen vs rejected pairs without a reward model. RM: trains a reward model for RLHF. KTO: Kahneman-Tversky Optimization, a simpler preference objective. ORPO: Odds Ratio Preference Optimization. Unsupervised: causal LM on raw text." />
          </label>
          <div className="lf-checkbox-group">
            {METHODS.map((m) => (
              <button key={m} className={`lf-chip ${form.training_method === m ? "lf-chip-active" : ""}`}
                onClick={() => set("training_method", m)}>{m}</button>
            ))}
          </div>
        </div>

        <div style={{ marginBottom: 8 }}>
          <label className="lf-label" style={{ display: "flex", alignItems: "center" }}>
            finetuning type<Tooltip text="LoRA: attaches small trainable low-rank matrices — fast, low VRAM, recommended for most cases. QLoRA: LoRA + 4-bit base model — enables 70B+ on 24GB. DoRA: decomposed LoRA, marginally better quality. Full: all weights updated — highest expressiveness but requires full model VRAM." />
          </label>
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
              <Field label="lora rank (r)" tooltip="Number of trainable dimensions in each LoRA adapter matrix. Higher r = more expressive adapter but more VRAM and slower training. Range: 8–128. Doubling r roughly doubles adapter file size. Start at 16; increase if loss plateaus.">
                <input className="lf-input" type="number" value={form.lora_r} onChange={(e) => set("lora_r", +e.target.value)} />
              </Field>
              <Field label="lora alpha" tooltip="LoRA scaling factor applied to adapter outputs before adding to the base model. Effective update magnitude = alpha/r. Common convention: alpha = 2×r. Higher alpha makes the adapter more influential relative to frozen weights.">
                <input className="lf-input" type="number" value={form.lora_alpha} onChange={(e) => set("lora_alpha", +e.target.value)} />
              </Field>
              <Field label="lora dropout" tooltip="Dropout probability applied to LoRA layers during training. Acts as regularization to prevent overfitting on small datasets. Set 0.0 to disable; 0.05–0.1 is typical. Has no effect at inference time.">
                <input className="lf-input" type="number" step="0.01" value={form.lora_dropout} onChange={(e) => set("lora_dropout", +e.target.value)} />
              </Field>
            </div>
            <div style={{ marginBottom: 8 }}>
              <label className="lf-label" style={{ display: "flex", alignItems: "center" }}>
                target modules<Tooltip text="Which attention/FFN weight matrices receive LoRA adapters. q_proj + v_proj is the minimal effective set. Adding k_proj, o_proj, gate_proj, up_proj, down_proj increases adapter capacity at the cost of more VRAM. lm_head targets the output projection." />
              </label>
              <div className="lf-checkbox-group">
                {TARGET_MODS.map((mod) => (
                  <button key={mod} className={`lf-chip ${form.target_modules.includes(mod) ? "lf-chip-active" : ""}`}
                    onClick={() => toggleModule(mod)}>{mod}</button>
                ))}
              </div>
            </div>
          </>
        )}

        <Section title="Dataset" tooltip="Select training data and how it is tokenized." />

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="dataset" tooltip="The training dataset registered on the Datasets page. Upload JSON/JSONL files via Datasets ↗ before selecting here.">
            <select className="lf-input lf-select" value={form.dataset_id} onChange={(e) => set("dataset_id", e.target.value)}>
              <option value="">— select —</option>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_samples?.toLocaleString() ?? "?"})</option>)}
            </select>
          </Field>
          <Field label="format" tooltip="Parsing format for the dataset. Must match how the data is structured. alpaca: {instruction, input, output}. sharegpt: {conversations: [{from, value}]}. plain_text: {text}. Wrong format causes training on malformed sequences.">
            <select className="lf-input lf-select" value={form.dataset_format} onChange={(e) => set("dataset_format", e.target.value)}>
              {["alpaca","sharegpt","plain_text"].map((f) => <option key={f}>{f}</option>)}
            </select>
          </Field>
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="cutoff / max seq length" tooltip="Maximum token length per training example. Longer sequences are truncated at this limit. VRAM usage scales roughly quadratically with sequence length due to attention. 2048 is a safe default; increase for document-level tasks.">
            <input className="lf-input" type="number" step="128" value={form.max_seq_length} onChange={(e) => set("max_seq_length", +e.target.value)} />
          </Field>
          <div style={{ paddingTop: 16 }}>
            <Toggle label="packing" tooltip="Concatenate multiple short examples into single sequences up to max_seq_length. Eliminates padding waste and greatly improves GPU utilization for short-text datasets. Disable if your examples are already near the length limit." checked={form.packing} onChange={(v) => set("packing", v)} />
          </div>
        </div>

        <Section title="Training Parameters" tooltip="Hyperparameters that control the optimization process. These have the biggest impact on final model quality." />

        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          <Field label="learning rate" tooltip="Step size for the optimizer. Too high → loss spikes or divergence. Too low → slow convergence. Typical range: 1e-4 to 5e-4 for LoRA; 1e-5 to 5e-5 for full fine-tuning. Use cosine scheduler to decay it over training.">
            <input className="lf-input" type="number" step="0.00001" value={form.learning_rate} onChange={(e) => set("learning_rate", +e.target.value)} />
          </Field>
          <Field label="epochs" tooltip="Number of complete passes through the training dataset. More epochs = more training but higher overfitting risk. Watch eval loss — stop when it stops improving. 1–3 epochs is typical for instruction fine-tuning.">
            <input className="lf-input" type="number" value={form.num_epochs} onChange={(e) => set("num_epochs", +e.target.value)} />
          </Field>
          <Field label="batch size / device" tooltip="Examples processed per GPU per optimizer step. Higher batch = faster training but more VRAM. Effective batch = batch_size × grad_accum_steps × num_GPUs. Keep small (2–8) and use grad accumulation to simulate larger batches.">
            <input className="lf-input" type="number" value={form.batch_size} onChange={(e) => set("batch_size", +e.target.value)} />
          </Field>
        </div>

        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          <Field label="grad accum steps" tooltip="Accumulate gradients over N mini-batches before updating weights. Simulates a batch size of batch_size × N without extra VRAM. Use when you cannot fit a large batch in GPU memory. Effective batch = batch_size × grad_accum.">
            <input className="lf-input" type="number" value={form.gradient_accumulation_steps} onChange={(e) => set("gradient_accumulation_steps", +e.target.value)} />
          </Field>
          <Field label="lr scheduler" tooltip="How the learning rate changes over training. Cosine (recommended): smoothly decays to near 0 — best for most tasks. Linear: linear decay. Constant: no decay, useful for very short runs. Cosine with restarts: cyclic — can escape local minima.">
            <select className="lf-input lf-select" value={form.lr_scheduler} onChange={(e) => set("lr_scheduler", e.target.value)}>
              {SCHEDULERS.map((s) => <option key={s}>{s}</option>)}
            </select>
          </Field>
          <Field label="warmup ratio" tooltip="Fraction of total training steps used to linearly ramp the learning rate from 0 to its peak. Prevents large destabilizing updates at the very start. 0.03–0.1 is typical. Multiply by total steps to get warmup steps.">
            <input className="lf-input" type="number" step="0.01" value={form.warmup_ratio} onChange={(e) => set("warmup_ratio", +e.target.value)} />
          </Field>
        </div>

        <div className="lf-row lf-row-4" style={{ marginBottom: 8 }}>
          <Field label="max grad norm" tooltip="Clips the L2 norm of gradients to this value before each weight update. Prevents exploding gradients which cause loss spikes or NaN. 1.0 is the standard default. Reduce to 0.3 if you observe unstable early training.">
            <input className="lf-input" type="number" step="0.1" value={form.max_grad_norm} onChange={(e) => set("max_grad_norm", +e.target.value)} />
          </Field>
          <Field label="logging steps" tooltip="Emit training metrics (loss, learning rate, grad norm) to the log every N optimizer steps. Lower values produce finer-grained loss curves but add minor overhead. 10–50 is typical.">
            <input className="lf-input" type="number" value={form.logging_steps} onChange={(e) => set("logging_steps", +e.target.value)} />
          </Field>
          <Field label="save steps" tooltip="Save a full checkpoint to disk every N steps. Checkpoints allow resuming interrupted training. Combine with save_total_limit to cap disk usage. Large models can be 10–50GB per checkpoint.">
            <input className="lf-input" type="number" value={form.save_steps} onChange={(e) => set("save_steps", +e.target.value)} />
          </Field>
          <Field label="seed" tooltip="Random seed for reproducibility. Controls weight initialization, data shuffling order, and dropout. Using the same seed and config reproduces identical results. Change it to test variance between runs.">
            <input className="lf-input" type="number" value={form.seed} onChange={(e) => set("seed", +e.target.value)} />
          </Field>
        </div>

        <Field label="output dir" tooltip="Directory where checkpoints and the final merged/adapter weights are saved after training. Use a unique path per run to avoid overwriting previous results.">
          <input className="lf-input" value={form.output_dir} onChange={(e) => set("output_dir", e.target.value)} />
        </Field>

        <Section title="Advanced" tooltip="Precision and memory optimization settings. Defaults work for most cases." />

        <div style={{ display: "flex", flexWrap: "wrap", gap: 14, marginBottom: 8, paddingTop: 4 }}>
          <Toggle label="bf16" tooltip="Train in bfloat16 precision. Recommended for Ampere+ GPUs (RTX 30xx / A100+). Same throughput as fp16 but with a wider dynamic range — avoids NaN losses on very small gradients. Mutually exclusive with fp16." checked={form.bf16} onChange={(v) => { set("bf16", v); if (v) set("fp16", false); }} />
          <Toggle label="fp16" tooltip="Train in float16 precision. Compatible with Volta/Turing GPUs (V100, RTX 20xx). ~2× faster than fp32 with the same VRAM. Occasionally produces NaN on unstable runs — switch to bf16 if that happens. Mutually exclusive with bf16." checked={form.fp16} onChange={(v) => { set("fp16", v); if (v) set("bf16", false); }} />
          <Toggle label="grad ckpt" tooltip="Gradient checkpointing trades compute for memory. Instead of storing all intermediate activations, it recomputes them during the backward pass. Reduces VRAM usage by ~30–40% at the cost of ~20% slower training. Essential for fine-tuning large models on limited VRAM." checked={form.gradient_checkpointing} onChange={(v) => set("gradient_checkpointing", v)} />
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="dataloader workers" tooltip="Number of CPU subprocesses used to load and preprocess data in parallel. Higher values keep the GPU fed more consistently. Set to 0 if you see CUDA multiprocessing errors. Diminishing returns above 4.">
            <input className="lf-input" type="number" value={form.dataloader_num_workers} onChange={(e) => set("dataloader_num_workers", +e.target.value)} />
          </Field>
          <Field label="resume from checkpoint" tooltip="Path to a previously saved checkpoint directory to resume training from. The optimizer state and step count are restored so training continues seamlessly. Leave empty to start fresh from the base model.">
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
