"use client";
import { useRef, useState, useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { getModels, getDatasets, getJobs, getASRJobs, startEval } from "@/lib/api";
import type { Job, ModelEntry } from "@/types";

const QUANT_OPTIONS = ["none", "4bit", "8bit"] as const;
const TEMPLATES     = ["alpaca", "sharegpt", "llama3", "mistral", "qwen", "phi3", "chatml"] as const;

type EvalMode   = "evaluate" | "predict";
type SourceMode = "from_job" | "base_model" | "manual";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="lf-label">{label}</label>{children}</div>;
}
function Section({ title }: { title: string }) {
  return <div className="lf-section" style={{ marginTop: 12 }}>{title}</div>;
}

function resolveJobPaths(
  job: Job,
  models: ModelEntry[],
): { model_path: string; adapter_path: string } {
  const cfg = job.config_json ?? {};
  const isLoraLike = ["lora", "qlora", "dora"].includes(job.peft_method);

  if (job.training_method === "asr_whisper") {
    const base = String(cfg.model_path ?? "");
    return isLoraLike && job.output_dir
      ? { model_path: base, adapter_path: job.output_dir + "/final_adapter" }
      : { model_path: job.output_dir ? job.output_dir + "/final_model" : base, adapter_path: "" };
  }

  const reg = models.find((m) => m.id === job.model_id);
  const base = reg?.local_path || reg?.hf_repo || String(cfg.model_path ?? "");
  return isLoraLike && job.output_dir
    ? { model_path: base, adapter_path: job.output_dir + "/final_adapter" }
    : { model_path: job.output_dir ? job.output_dir + "/final_model" : base, adapter_path: "" };
}

export default function EvaluatePage() {
  const { data: models = [] }   = useQuery({ queryKey: ["models"],    queryFn: getModels });
  const { data: datasets = [] } = useQuery({ queryKey: ["datasets"],  queryFn: getDatasets });
  const { data: llmJobs = [] }  = useQuery({ queryKey: ["jobs"],      queryFn: getJobs });
  const { data: asrJobs = [] }  = useQuery({ queryKey: ["asr-jobs"],  queryFn: getASRJobs });

  const completedJobs = useMemo(
    () => [...llmJobs, ...asrJobs].filter((j) => j.status === "completed"),
    [llmJobs, asrJobs],
  );

  const [evalMode,   setEvalMode]   = useState<EvalMode>("evaluate");
  const [sourceMode, setSourceMode] = useState<SourceMode>("from_job");
  const [selectedJobId, setSelectedJobId] = useState("");

  const [form, setForm] = useState({
    model_path: "", adapter_path: "", quantization: "none", template: "alpaca",
    dataset_id: "", dataset_path: "",
    batch_size: 4, max_seq_len: 2048,
    predict_output: "./outputs/predictions.jsonl",
    // ASR
    audio_col: "audio_path", text_col: "text", language: "auto", task: "transcribe",
  });
  const set = <K extends keyof typeof form>(k: K, v: typeof form[K]) =>
    setForm((p) => ({ ...p, [k]: v }));

  // Track whether selected/manual source is ASR
  const [isASR, setIsASR] = useState(false);

  // Auto-fill paths when a job is selected
  useEffect(() => {
    if (sourceMode !== "from_job" || !selectedJobId) return;
    const job = completedJobs.find((j) => j.id === Number(selectedJobId));
    if (!job) return;
    const { model_path, adapter_path } = resolveJobPaths(job, models);
    const asr = job.training_method === "asr_whisper";
    setIsASR(asr);
    setForm((p) => ({ ...p, model_path, adapter_path }));
  }, [selectedJobId, sourceMode, completedJobs, models]);

  // Clear adapter path when switching to base_model mode
  useEffect(() => {
    if (sourceMode === "base_model") setForm((p) => ({ ...p, adapter_path: "" }));
  }, [sourceMode]);

  const [running, setRunning] = useState(false);
  const [runId,   setRunId]   = useState<string | null>(null);
  const [logs,    setLogs]    = useState<string[]>(["[system] Select a source and dataset, then start evaluation."]);
  const [result,  setResult]  = useState<Record<string, unknown> | null>(null);
  const [status,  setStatus]  = useState("idle");
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const canStart = !!form.model_path && !!(isASR ? form.dataset_path : (form.dataset_id || form.dataset_path));

  const handleStart = async () => {
    if (!canStart || running) return;
    setRunning(true);
    setResult(null);
    setLogs([`[system] Starting ${evalMode}…`]);
    setStatus("running");
    try {
      const body: Record<string, unknown> = {
        model_path: form.model_path,
        adapter_path: form.adapter_path || undefined,
        quantization: form.quantization === "none" ? null : form.quantization,
        mode: evalMode,
        batch_size: form.batch_size,
        max_seq_len: form.max_seq_len,
        predict_output: evalMode === "predict" ? form.predict_output : undefined,
        is_asr: isASR,
        ...(isASR && {
          audio_col: form.audio_col,
          text_col: form.text_col,
          language: form.language === "auto" ? null : form.language,
          task: form.task,
        }),
      };
      if (form.dataset_id)        body.dataset_id   = Number(form.dataset_id);
      else if (form.dataset_path) body.dataset_path = form.dataset_path;

      const { run_id } = await startEval(body);
      setRunId(run_id);

      const es = new EventSource(`/api/eval/${run_id}/stream`);
      es.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.line === "__done__") {
          setStatus(data.status);
          setResult(data.result ?? null);
          es.close();
          setRunning(false);
        } else {
          setLogs((p) => [...p, data.line]);
        }
      };
      es.onerror = () => {
        es.close();
        setRunning(false);
        setStatus("failed");
        setLogs((p) => [...p, "[error] Connection to server lost — check that the API is running"]);
      };
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setLogs((p) => [...p, `[error] ${msg}`]);
      setRunning(false);
      setStatus("failed");
    }
  };

  const statusColor: Record<string, string> = {
    idle: "var(--text-dim)", running: "var(--accent)", completed: "var(--green)", failed: "var(--red)",
  };

  const selectedJob = completedJobs.find((j) => j.id === Number(selectedJobId));

  return (
    <div style={{ display: "grid", gridTemplateColumns: "430px 1fr", height: "calc(100vh - 40px)", overflow: "hidden" }}>

      {/* ── LEFT ── */}
      <div style={{ borderRight: "1px solid var(--border)", overflowY: "auto", padding: "10px 12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--accent)", background: "var(--accent-dim)", padding: "2px 7px", borderRadius: 2 }}>Evaluate & Predict</span>
        </div>

        {/* Eval mode */}
        <div style={{ marginBottom: 10 }}>
          <label className="lf-label">mode</label>
          <div style={{ display: "flex", gap: 4 }}>
            {(["evaluate", "predict"] as EvalMode[]).map((m) => (
              <button key={m} className={`lf-chip ${evalMode === m ? "lf-chip-active" : ""}`}
                style={{ flex: 1, justifyContent: "center" }} onClick={() => setEvalMode(m)}>{m}</button>
            ))}
          </div>
        </div>

        <Section title="Source" />

        {/* Source mode toggle */}
        <div style={{ marginBottom: 10 }}>
          <label className="lf-label">load from</label>
          <div style={{ display: "flex", gap: 4 }}>
            {([["from_job", "Run"], ["base_model", "Base Model"], ["manual", "Manual"]] as [SourceMode, string][]).map(
              ([m, label]) => (
                <button key={m} className={`lf-chip ${sourceMode === m ? "lf-chip-active" : ""}`}
                  style={{ flex: 1, justifyContent: "center", fontSize: 10 }}
                  onClick={() => setSourceMode(m)}>{label}</button>
              )
            )}
          </div>
        </div>

        {/* FROM JOB */}
        {sourceMode === "from_job" && (
          <div style={{ marginBottom: 8 }}>
            <Field label="completed run">
              <select className="lf-input lf-select" value={selectedJobId}
                onChange={(e) => setSelectedJobId(e.target.value)}>
                <option value="">— select a completed run —</option>
                {completedJobs.map((j) => (
                  <option key={j.id} value={j.id}>
                    #{j.id} {j.name} [{j.training_method}/{j.peft_method}]
                  </option>
                ))}
              </select>
            </Field>
            {selectedJob && (
              <div style={{ marginTop: 6, padding: "6px 8px", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 3, display: "flex", flexDirection: "column", gap: 3 }}>
                <div style={{ fontFamily: "var(--mono)", fontSize: 10 }}>
                  <span style={{ color: "var(--text-dim)", minWidth: 52, display: "inline-block" }}>model</span>
                  <span style={{ color: "var(--text-hi)", wordBreak: "break-all" }}>{form.model_path || "—"}</span>
                </div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 10 }}>
                  <span style={{ color: "var(--text-dim)", minWidth: 52, display: "inline-block" }}>adapter</span>
                  {form.adapter_path
                    ? <span style={{ color: "var(--green)", wordBreak: "break-all" }}>{form.adapter_path}</span>
                    : <span style={{ color: "var(--amber, #f59e0b)" }}>none — full model eval</span>}
                </div>
              </div>
            )}
            {completedJobs.length === 0 && (
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginTop: 4 }}>
                No completed runs yet. Train a model first.
              </div>
            )}
          </div>
        )}

        {/* BASE MODEL ONLY */}
        {sourceMode === "base_model" && (
          <div style={{ marginBottom: 8 }}>
            <label className="lf-label">model</label>
            <select className="lf-input lf-select" value={form.model_path}
              onChange={(e) => set("model_path", e.target.value)} style={{ marginBottom: 4 }}>
              <option value="">— select registered model —</option>
              {models.map((m) => <option key={m.id} value={m.local_path || m.hf_repo}>{m.name}</option>)}
            </select>
            <input className="lf-input" value={form.model_path}
              onChange={(e) => set("model_path", e.target.value)} placeholder="or enter path / HF repo" />
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginTop: 4 }}>
              Base model only — no adapter applied
            </div>
          </div>
        )}

        {/* MANUAL */}
        {sourceMode === "manual" && (
          <>
            <div style={{ marginBottom: 8 }}>
              <label className="lf-label">model type</label>
              <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
                {([false, true] as const).map((asr) => (
                  <button key={String(asr)} className={`lf-chip ${isASR === asr ? "lf-chip-active" : ""}`}
                    style={{ flex: 1, justifyContent: "center" }} onClick={() => setIsASR(asr)}>
                    {asr ? "ASR (Whisper)" : "LLM"}
                  </button>
                ))}
              </div>
            </div>
            <div style={{ marginBottom: 8 }}>
              <label className="lf-label">model path</label>
              <select className="lf-input lf-select" value={form.model_path}
                onChange={(e) => set("model_path", e.target.value)} style={{ marginBottom: 4 }}>
                <option value="">— select registered model —</option>
                {models.map((m) => <option key={m.id} value={m.local_path || m.hf_repo}>{m.name}</option>)}
              </select>
              <input className="lf-input" value={form.model_path}
                onChange={(e) => set("model_path", e.target.value)} placeholder="or enter path / HF repo" />
            </div>
            <div style={{ marginBottom: 8 }}>
              <Field label="adapter path (optional)">
                <input className="lf-input" value={form.adapter_path}
                  onChange={(e) => set("adapter_path", e.target.value)} placeholder="./outputs/run1/final_adapter" />
              </Field>
            </div>
          </>
        )}

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="quantization">
            <select className="lf-input lf-select" value={form.quantization} onChange={(e) => set("quantization", e.target.value)}>
              {QUANT_OPTIONS.map((q) => <option key={q}>{q}</option>)}
            </select>
          </Field>
          {!isASR && (
            <Field label="template">
              <select className="lf-input lf-select" value={form.template} onChange={(e) => set("template", e.target.value)}>
                {TEMPLATES.map((t) => <option key={t}>{t}</option>)}
              </select>
            </Field>
          )}
          {isASR && (
            <Field label="task">
              <select className="lf-input lf-select" value={form.task} onChange={(e) => set("task", e.target.value)}>
                <option value="transcribe">transcribe</option>
                <option value="translate">translate</option>
              </select>
            </Field>
          )}
        </div>

        {isASR && (
          <>
            <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
              <Field label="language">
                <input className="lf-input" value={form.language} onChange={(e) => set("language", e.target.value)} placeholder="auto / en / ms / zh…" />
              </Field>
              <Field label="audio column">
                <input className="lf-input" value={form.audio_col} onChange={(e) => set("audio_col", e.target.value)} />
              </Field>
            </div>
            <div style={{ marginBottom: 8 }}>
              <Field label="text column (ground truth)">
                <input className="lf-input" value={form.text_col} onChange={(e) => set("text_col", e.target.value)} />
              </Field>
            </div>
          </>
        )}

        <Section title="Dataset" />
        <div style={{ marginBottom: 8 }}>
          <label className="lf-label">{isASR ? "dataset CSV path" : "dataset"}</label>
          {!isASR && (
            <select className="lf-input lf-select" value={form.dataset_id}
              onChange={(e) => set("dataset_id", e.target.value)} style={{ marginBottom: 4 }}>
              <option value="">— select from registry —</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>{d.name} ({d.num_samples?.toLocaleString() ?? "?"})</option>
              ))}
            </select>
          )}
          <input className="lf-input" value={form.dataset_path}
            onChange={(e) => set("dataset_path", e.target.value)}
            placeholder={isASR ? "path/to/dataset.csv" : "or enter file path"} />
          {isASR && (
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginTop: 3 }}>
              CSV must have columns matching audio column + text column above
            </div>
          )}
        </div>

        {!isASR && (
          <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
            <Field label="max seq length">
              <input className="lf-input" type="number" value={form.max_seq_len} onChange={(e) => set("max_seq_len", +e.target.value)} />
            </Field>
            <Field label="batch size">
              <input className="lf-input" type="number" value={form.batch_size} onChange={(e) => set("batch_size", +e.target.value)} />
            </Field>
          </div>
        )}

        {evalMode === "predict" && (
          <>
            <Section title="Predict Output" />
            <div style={{ marginBottom: 8 }}>
              <Field label="save path (.jsonl)">
                <input className="lf-input" value={form.predict_output} onChange={(e) => set("predict_output", e.target.value)} />
              </Field>
            </div>
          </>
        )}

        <div style={{ display: "flex", gap: 8, paddingTop: 12, borderTop: "1px solid var(--border)", marginTop: 4 }}>
          <button className="lf-btn lf-btn-primary" style={{ flex: 1 }}
            disabled={running || !canStart} onClick={handleStart}>
            {running
              ? <><span className="lf-spin" /> {evalMode === "evaluate" ? "Evaluating…" : "Predicting…"}</>
              : `▶ Start ${evalMode === "evaluate" ? "Evaluate" : "Predict"}`}
          </button>
        </div>
      </div>

      {/* ── RIGHT ── */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Status bar */}
        <div style={{ borderBottom: "1px solid var(--border)", padding: "0 14px", height: 32, display: "flex", alignItems: "center", gap: 14, background: "var(--bg-panel)", flexShrink: 0 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: statusColor[status] ?? "var(--text-dim)" }}>{status}</span>
          {isASR && <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--accent)", background: "var(--accent-dim)", padding: "1px 5px", borderRadius: 2 }}>ASR · WER</span>}
          {runId && <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>run {runId.slice(0, 8)}</span>}
        </div>

        {/* Result card */}
        {result && (
          <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--border)", background: "var(--bg-panel)", flexShrink: 0 }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>Results</div>
            <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
              {result.wer != null && (
                <div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>WER</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 700, color: +result.wer < 20 ? "var(--green)" : +result.wer < 40 ? "var(--amber)" : "var(--red)" }}>
                    {(+result.wer).toFixed(2)}%
                  </div>
                </div>
              )}
              {result.n_samples != null && (
                <div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>Samples</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 700, color: "var(--text-hi)" }}>{String(result.n_samples)}</div>
                </div>
              )}
              {result.loss != null && (
                <div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>Loss</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 700, color: "var(--accent)" }}>{(+result.loss).toFixed(4)}</div>
                </div>
              )}
              {result.perplexity != null && (
                <div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>Perplexity</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 20, fontWeight: 700, color: "var(--green)" }}>{(+result.perplexity).toFixed(2)}</div>
                </div>
              )}
              {!!result.output_file && (
                <div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>Output file</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-hi)" }}>{String(result.output_file)}</div>
                </div>
              )}
              {!!result.error && (
                <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)" }}>{String(result.error)}</div>
              )}
            </div>
          </div>
        )}

        {/* Log */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", padding: "10px 14px" }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>Output Log</div>
          <div ref={logRef} className="lf-console" style={{ flex: 1 }}>
            {logs.map((line, i) => (
              <div key={i} style={{ color: line.startsWith("[error]") ? "var(--red)" : line.startsWith("[system]") ? "var(--text-dim)" : "var(--green)" }}>{line}</div>
            ))}
            {running && <span className="lf-cursor">█</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
