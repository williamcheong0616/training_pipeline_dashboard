"use client";
import { useRef, useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getModels, getDatasets, startEval } from "@/lib/api";

const QUANT_OPTIONS = ["none", "4bit", "8bit"] as const;
const TEMPLATES     = ["alpaca", "sharegpt", "llama3", "mistral", "qwen", "phi3", "chatml"] as const;

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="lf-label">{label}</label>{children}</div>;
}
function Section({ title }: { title: string }) {
  return <div className="lf-section" style={{ marginTop: 12 }}>{title}</div>;
}

type EvalMode = "evaluate" | "predict";

export default function EvaluatePage() {
  const { data: models = [] } = useQuery({ queryKey: ["models"], queryFn: getModels });
  const { data: datasets = [] } = useQuery({ queryKey: ["datasets"], queryFn: getDatasets });

  const [mode, setMode] = useState<EvalMode>("evaluate");
  const [form, setForm] = useState({
    model_path: "", adapter_path: "", quantization: "none", template: "alpaca",
    dataset_id: "", dataset_path: "",
    batch_size: 4, max_seq_len: 2048,
    predict_output: "./outputs/predictions.jsonl",
  });
  const set = <K extends keyof typeof form>(k: K, v: typeof form[K]) =>
    setForm((p) => ({ ...p, [k]: v }));

  const [running, setRunning] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>(["[system] Select a model and dataset, then start evaluation."]);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [status, setStatus] = useState("idle");
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const handleStart = async () => {
    if (!form.model_path) return;
    setRunning(true);
    setResult(null);
    setLogs([`[system] Starting ${mode}…`]);
    setStatus("running");
    try {
      const body: Record<string, unknown> = {
        model_path: form.model_path,
        adapter_path: form.adapter_path || undefined,
        quantization: form.quantization === "none" ? null : form.quantization,
        mode,
        batch_size: form.batch_size,
        max_seq_len: form.max_seq_len,
        predict_output: mode === "predict" ? form.predict_output : undefined,
      };
      if (form.dataset_id) body.dataset_id = Number(form.dataset_id);
      else if (form.dataset_path) body.dataset_path = form.dataset_path;

      const { run_id } = await startEval(body);
      setRunId(run_id);

      // SSE stream
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
      es.onerror = () => { es.close(); setRunning(false); setStatus("failed"); };
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

  return (
    <div style={{ display: "grid", gridTemplateColumns: "420px 1fr", height: "calc(100vh - 40px)", overflow: "hidden" }}>

      {/* LEFT */}
      <div style={{ borderRight: "1px solid var(--border)", overflowY: "auto", padding: "10px 12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--accent)", background: "var(--accent-dim)", padding: "2px 7px", borderRadius: 2 }}>Evaluate & Predict</span>
        </div>

        {/* Mode */}
        <div style={{ marginBottom: 10 }}>
          <label className="lf-label">mode</label>
          <div style={{ display: "flex", gap: 4 }}>
            {(["evaluate", "predict"] as EvalMode[]).map((m) => (
              <button key={m} className={`lf-chip ${mode === m ? "lf-chip-active" : ""}`} style={{ flex: 1, justifyContent: "center" }}
                onClick={() => setMode(m)}>{m}</button>
            ))}
          </div>
        </div>

        <Section title="Model" />
        <div style={{ marginBottom: 8 }}>
          <label className="lf-label">model</label>
          <select className="lf-input lf-select" value={form.model_path} onChange={(e) => set("model_path", e.target.value)} style={{ marginBottom: 4 }}>
            <option value="">— select registered model —</option>
            {models.map((m) => <option key={m.id} value={m.local_path || m.hf_repo}>{m.name}</option>)}
          </select>
          <input className="lf-input" value={form.model_path} onChange={(e) => set("model_path", e.target.value)} placeholder="or enter path / HF repo" />
        </div>

        <div style={{ marginBottom: 8 }}>
          <Field label="adapter path (optional)">
            <input className="lf-input" value={form.adapter_path} onChange={(e) => set("adapter_path", e.target.value)} placeholder="./outputs/run1/final_adapter" />
          </Field>
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="quantization">
            <select className="lf-input lf-select" value={form.quantization} onChange={(e) => set("quantization", e.target.value)}>
              {QUANT_OPTIONS.map((q) => <option key={q}>{q}</option>)}
            </select>
          </Field>
          <Field label="template">
            <select className="lf-input lf-select" value={form.template} onChange={(e) => set("template", e.target.value)}>
              {TEMPLATES.map((t) => <option key={t}>{t}</option>)}
            </select>
          </Field>
        </div>

        <Section title="Dataset" />
        <div style={{ marginBottom: 8 }}>
          <label className="lf-label">dataset</label>
          <select className="lf-input lf-select" value={form.dataset_id} onChange={(e) => set("dataset_id", e.target.value)} style={{ marginBottom: 4 }}>
            <option value="">— select from registry —</option>
            {datasets.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.num_samples?.toLocaleString() ?? "?"})</option>)}
          </select>
          <input className="lf-input" value={form.dataset_path} onChange={(e) => set("dataset_path", e.target.value)} placeholder="or enter file path" />
        </div>

        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="max seq length">
            <input className="lf-input" type="number" value={form.max_seq_len} onChange={(e) => set("max_seq_len", +e.target.value)} />
          </Field>
          <Field label="batch size">
            <input className="lf-input" type="number" value={form.batch_size} onChange={(e) => set("batch_size", +e.target.value)} />
          </Field>
        </div>

        {mode === "predict" && (
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
            disabled={running || !form.model_path || (!form.dataset_id && !form.dataset_path)}
            onClick={handleStart}>
            {running ? <><span className="lf-spin" /> {mode === "evaluate" ? "Evaluating…" : "Predicting…"}</> : `▶ Start ${mode === "evaluate" ? "Evaluate" : "Predict"}`}
          </button>
        </div>
      </div>

      {/* RIGHT */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Status bar */}
        <div style={{ borderBottom: "1px solid var(--border)", padding: "0 14px", height: 32, display: "flex", alignItems: "center", gap: 14, background: "var(--bg-panel)", flexShrink: 0 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: statusColor[status] ?? "var(--text-dim)" }}>{status}</span>
          {runId && <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>run {runId.slice(0, 8)}</span>}
        </div>

        {/* Result card */}
        {result && (
          <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--border)", background: "var(--bg-panel)", flexShrink: 0 }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6 }}>Results</div>
            <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
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
