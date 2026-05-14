"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { createJob, getModels, getDatasets } from "@/lib/api";

const METHODS = ["sft", "unsupervised", "dpo", "rm", "kto", "orpo"] as const;
const PEFT_METHODS = ["lora", "qlora", "dora", "full"] as const;
const QUANT_OPTIONS = ["none", "4bit", "8bit"] as const;

type Step = 1 | 2 | 3 | 4;

export default function NewJobPage() {
  const router = useRouter();
  const qc = useQueryClient();
  const [step, setStep] = useState<Step>(1);

  const { data: models = [] } = useQuery({ queryKey: ["models"], queryFn: getModels });
  const { data: datasets = [] } = useQuery({ queryKey: ["datasets"], queryFn: getDatasets });

  const [form, setForm] = useState({
    name: "",
    training_method: "sft" as typeof METHODS[number],
    peft_method: "lora" as typeof PEFT_METHODS[number],
    model_id: "" as string | number,
    dataset_id: "" as string | number,
    quantization: "none" as typeof QUANT_OPTIONS[number],
    lora_r: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    num_epochs: 3,
    batch_size: 4,
    learning_rate: 0.0002,
    max_seq_length: 2048,
    output_dir: "./outputs/run",
    template: "alpaca",
    dataset_format: "alpaca",
  });

  const set = (k: keyof typeof form, v: unknown) => setForm((p) => ({ ...p, [k]: v }));

  const { mutate: submit, isPending, error } = useMutation({
    mutationFn: () => {
      const selectedModel = models.find((m) => m.id === Number(form.model_id));
      return createJob({
        name: form.name,
        training_method: form.training_method,
        peft_method: form.peft_method,
        model_id: Number(form.model_id) || undefined,
        dataset_id: Number(form.dataset_id) || undefined,
        config: {
          model_path: selectedModel?.local_path ?? selectedModel?.hf_repo ?? "",
          dataset_path: datasets.find((d) => d.id === Number(form.dataset_id))?.path ?? "",
          dataset_format: form.dataset_format,
          template: form.template,
          peft_method: form.peft_method,
          quantization: form.quantization === "none" ? null : form.quantization,
          lora_r: form.lora_r,
          lora_alpha: form.lora_alpha,
          lora_dropout: form.lora_dropout,
          num_epochs: form.num_epochs,
          batch_size: form.batch_size,
          learning_rate: form.learning_rate,
          max_seq_length: form.max_seq_length,
          output_dir: form.output_dir,
        },
      });
    },
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      router.push(`/jobs/${job.id}`);
    },
  });

  const steps = ["Model", "Method", "Dataset", "Hyperparams"];

  return (
    <div className="max-w-2xl space-y-6">
      <h2 className="text-2xl font-bold">New Training Job</h2>

      {/* Stepper */}
      <ul className="steps steps-horizontal w-full">
        {steps.map((label, i) => (
          <li key={label} className={`step ${step > i ? "step-primary" : ""}`}>{label}</li>
        ))}
      </ul>

      <div className="card bg-base-200 border border-base-300">
        <div className="card-body gap-4">
          {/* Step 1 — Model */}
          {step === 1 && (
            <>
              <h3 className="text-lg font-semibold">Select Model</h3>
              <label className="form-control w-full">
                <div className="label"><span className="label-text">Job name</span></div>
                <input className="input input-bordered" value={form.name} onChange={(e) => set("name", e.target.value)} placeholder="my-finetune-run" />
              </label>
              <label className="form-control w-full">
                <div className="label"><span className="label-text">Base model</span></div>
                <select className="select select-bordered" value={form.model_id} onChange={(e) => set("model_id", e.target.value)}>
                  <option value="">— select a model —</option>
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>{m.name} ({m.hf_repo})</option>
                  ))}
                </select>
              </label>
              {models.length === 0 && (
                <div className="alert alert-warning text-sm">
                  No models registered yet. <a href="/models" className="link">Add one on the Models page.</a>
                </div>
              )}
            </>
          )}

          {/* Step 2 — Method */}
          {step === 2 && (
            <>
              <h3 className="text-lg font-semibold">Training Method & PEFT</h3>
              <label className="form-control w-full">
                <div className="label"><span className="label-text">Training method</span></div>
                <div className="grid grid-cols-3 gap-2">
                  {METHODS.map((m) => (
                    <button
                      key={m}
                      type="button"
                      className={`btn btn-sm uppercase ${form.training_method === m ? "btn-primary" : "btn-outline"}`}
                      onClick={() => set("training_method", m)}
                    >{m}</button>
                  ))}
                </div>
              </label>
              <label className="form-control w-full">
                <div className="label"><span className="label-text">PEFT method</span></div>
                <div className="grid grid-cols-4 gap-2">
                  {PEFT_METHODS.map((m) => (
                    <button
                      key={m}
                      type="button"
                      className={`btn btn-sm ${form.peft_method === m ? "btn-secondary" : "btn-outline"}`}
                      onClick={() => set("peft_method", m)}
                    >{m}</button>
                  ))}
                </div>
              </label>
              <label className="form-control w-full">
                <div className="label"><span className="label-text">Quantization</span></div>
                <select className="select select-bordered select-sm" value={form.quantization} onChange={(e) => set("quantization", e.target.value)}>
                  {QUANT_OPTIONS.map((q) => <option key={q} value={q}>{q}</option>)}
                </select>
              </label>
            </>
          )}

          {/* Step 3 — Dataset */}
          {step === 3 && (
            <>
              <h3 className="text-lg font-semibold">Dataset</h3>
              <label className="form-control w-full">
                <div className="label"><span className="label-text">Dataset</span></div>
                <select className="select select-bordered" value={form.dataset_id} onChange={(e) => set("dataset_id", e.target.value)}>
                  <option value="">— select a dataset —</option>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>{d.name} ({d.num_samples?.toLocaleString() ?? "?"} samples)</option>
                  ))}
                </select>
              </label>
              <label className="form-control w-full">
                <div className="label"><span className="label-text">Dataset format</span></div>
                <select className="select select-bordered select-sm" value={form.dataset_format} onChange={(e) => set("dataset_format", e.target.value)}>
                  {["alpaca", "sharegpt", "plain_text"].map((f) => <option key={f} value={f}>{f}</option>)}
                </select>
              </label>
              <label className="form-control w-full">
                <div className="label"><span className="label-text">Prompt template</span></div>
                <select className="select select-bordered select-sm" value={form.template} onChange={(e) => set("template", e.target.value)}>
                  {["alpaca", "chatml", "llama3", "mistral", "qwen", "phi3", "gemma"].map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </label>
            </>
          )}

          {/* Step 4 — Hyperparams */}
          {step === 4 && (
            <>
              <h3 className="text-lg font-semibold">Hyperparameters</h3>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Epochs", key: "num_epochs", type: "number", step: 1 },
                  { label: "Batch size", key: "batch_size", type: "number", step: 1 },
                  { label: "Learning rate", key: "learning_rate", type: "number", step: 0.00001 },
                  { label: "Max seq length", key: "max_seq_length", type: "number", step: 128 },
                  { label: "LoRA rank (r)", key: "lora_r", type: "number", step: 1 },
                  { label: "LoRA alpha", key: "lora_alpha", type: "number", step: 1 },
                ].map(({ label, key, type, step: s }) => (
                  <label key={key} className="form-control w-full">
                    <div className="label"><span className="label-text text-xs">{label}</span></div>
                    <input
                      className="input input-bordered input-sm"
                      type={type}
                      step={s}
                      value={form[key as keyof typeof form] as number}
                      onChange={(e) => set(key as keyof typeof form, parseFloat(e.target.value))}
                    />
                  </label>
                ))}
              </div>
              <label className="form-control w-full">
                <div className="label"><span className="label-text text-xs">Output directory</span></div>
                <input className="input input-bordered input-sm" value={form.output_dir} onChange={(e) => set("output_dir", e.target.value)} />
              </label>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="alert alert-error text-sm">Failed to create job. Check the API server.</div>
      )}

      {/* Navigation */}
      <div className="flex justify-between">
        <button className="btn btn-ghost" disabled={step === 1} onClick={() => setStep((s) => (s - 1) as Step)}>
          ← Back
        </button>
        {step < 4 ? (
          <button className="btn btn-primary" onClick={() => setStep((s) => (s + 1) as Step)}>
            Next →
          </button>
        ) : (
          <button className="btn btn-success" disabled={isPending || !form.name} onClick={() => submit()}>
            {isPending ? <span className="loading loading-spinner" /> : "🚀 Start Training"}
          </button>
        )}
      </div>
    </div>
  );
}
