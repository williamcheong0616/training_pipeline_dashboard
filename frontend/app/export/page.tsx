"use client";
import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getJobs, getASRJobs, getExports, exportJob, exportFromPath } from "@/lib/api";

type ExportMode = "job" | "path";

function Section({ title }: { title: string }) {
  return <div className="lf-section" style={{ marginTop: 12 }}>{title}</div>;
}
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="lf-label">{label}</label>{children}</div>;
}

export default function ExportPage() {
  const qc = useQueryClient();
  const { data: llmJobs = [] } = useQuery({ queryKey: ["jobs"],     queryFn: getJobs });
  const { data: asrJobs = [] } = useQuery({ queryKey: ["asr-jobs"], queryFn: getASRJobs });
  const { data: exports = [], refetch: refetchExports } = useQuery({ queryKey: ["exports"], queryFn: getExports, refetchInterval: 5000 });

  const completedJobs = useMemo(
    () => [...llmJobs, ...asrJobs].filter((j) => j.status === "completed" && j.peft_method !== "sft" && j.peft_method !== "full"),
    [llmJobs, asrJobs],
  );

  const [mode, setMode] = useState<ExportMode>("job");
  const [form, setForm] = useState({ job_id: "", adapter_path: "", output_name: "" });
  const [mergeStatus, setMergeStatus] = useState<"idle" | "merging" | "done" | "error">("idle");
  const [mergeMsg, setMergeMsg] = useState("");

  const set = <K extends keyof typeof form>(k: K, v: typeof form[K]) =>
    setForm((p) => ({ ...p, [k]: v }));

  const { mutate: doExport, isPending } = useMutation({
    mutationFn: async () => {
      if (mode === "job") {
        if (!form.job_id) throw new Error("Select a job");
        return exportJob(Number(form.job_id), form.output_name || undefined);
      } else {
        if (!form.adapter_path) throw new Error("Enter adapter path");
        return exportFromPath({ adapter_path: form.adapter_path, output_name: form.output_name || undefined });
      }
    },
    onMutate: () => { setMergeStatus("merging"); setMergeMsg(""); },
    onSuccess: (data: { save_path?: string; message?: string } | void) => {
      setMergeStatus("done");
      const savePath = data && typeof data === "object" && "save_path" in data ? data.save_path : undefined;
      setMergeMsg(savePath ? `Merging to ${savePath}` : "Merge started");
      setTimeout(() => refetchExports(), 3000);
    },
    onError: (e: unknown) => {
      setMergeStatus("error");
      setMergeMsg(e instanceof Error ? e.message : "Export failed");
    },
  });

  const statusColor: Record<string, string> = {
    idle: "var(--text-dim)", merging: "var(--amber)", done: "var(--green)", error: "var(--red)",
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", height: "calc(100vh - 40px)", overflow: "hidden" }}>

      {/* LEFT */}
      <div style={{ borderRight: "1px solid var(--border)", overflowY: "auto", padding: "10px 12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--accent)", background: "var(--accent-dim)", padding: "2px 7px", borderRadius: 2 }}>Export</span>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>Merge LoRA adapter → full model</span>
        </div>

        <Section title="Source" />
        <div style={{ marginBottom: 10 }}>
          <label className="lf-label">mode</label>
          <div style={{ display: "flex", gap: 4 }}>
            {(["job", "path"] as ExportMode[]).map((m) => (
              <button key={m} className={`lf-chip ${mode === m ? "lf-chip-active" : ""}`} style={{ flex: 1, justifyContent: "center" }}
                onClick={() => setMode(m)}>
                {m === "job" ? "From Completed Job" : "From Adapter Path"}
              </button>
            ))}
          </div>
        </div>

        {mode === "job" ? (
          <div style={{ marginBottom: 8 }}>
            <Field label="completed job">
              <select className="lf-input lf-select" value={form.job_id} onChange={(e) => set("job_id", e.target.value)}>
                <option value="">— select job —</option>
                {completedJobs.map((j) => (
                  <option key={`${j.training_method}-${j.id}`} value={j.id}>
                    #{j.id} {j.name} [{j.training_method}/{j.peft_method}]
                  </option>
                ))}
              </select>
            </Field>
            {completedJobs.length === 0 && (
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginTop: 4 }}>
                No completed jobs yet. Train a model first.
              </div>
            )}
          </div>
        ) : (
          <div style={{ marginBottom: 8 }}>
            <Field label="adapter directory path">
              <input className="lf-input" value={form.adapter_path} onChange={(e) => set("adapter_path", e.target.value)} placeholder="./outputs/run1/final_adapter" />
            </Field>
          </div>
        )}

        <Section title="Output" />
        <div style={{ marginBottom: 8 }}>
          <Field label="output name (optional)">
            <input className="lf-input" value={form.output_name} onChange={(e) => set("output_name", e.target.value)} placeholder={`merged_${mode === "job" ? "job_" + (form.job_id || "?") : "model"}`} />
          </Field>
        </div>

        <div style={{ marginBottom: 8, fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", padding: "6px 8px", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 3 }}>
          Exports saved to: <span style={{ color: "var(--text-hi)" }}>./exports/</span>
        </div>

        <div style={{ paddingTop: 12, borderTop: "1px solid var(--border)" }}>
          <button className="lf-btn lf-btn-primary" style={{ width: "100%" }}
            disabled={isPending || (mode === "job" ? !form.job_id : !form.adapter_path)}
            onClick={() => doExport()}>
            {isPending ? <><span className="lf-spin" /> Merging…</> : "⇓ Export / Merge Adapter"}
          </button>

          {mergeMsg && (
            <div style={{ marginTop: 8, fontFamily: "var(--mono)", fontSize: 10, padding: "6px 8px", borderRadius: 3, border: "1px solid var(--border)", color: statusColor[mergeStatus] }}>
              {mergeMsg}
            </div>
          )}
        </div>

        <div style={{ marginTop: 14, fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", lineHeight: 1.8 }}>
          <div style={{ color: "var(--accent)", marginBottom: 4 }}>What this does</div>
          <div>Loads the base model + LoRA adapter, merges weights via merge_and_unload(), and saves a standalone full-precision model usable without PEFT. Auto-detects Whisper vs LLM from the adapter config.</div>
        </div>
      </div>

      {/* RIGHT — exported models */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ borderBottom: "1px solid var(--border)", padding: "0 14px", height: 32, display: "flex", alignItems: "center", gap: 14, background: "var(--bg-panel)", flexShrink: 0 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
            Exported Models ({exports.length})
          </span>
          {mergeStatus === "merging" && (
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--amber)" }}>
              <span className="lf-spin" style={{ marginRight: 6 }} />merging in background…
            </span>
          )}
        </div>

        <div style={{ flex: 1, overflow: "auto", padding: "10px 14px" }}>
          <div className="lf-panel" style={{ overflow: "auto" }}>
            {exports.length === 0 ? (
              <div style={{ padding: 32, textAlign: "center", fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
                No exported models yet. Merge an adapter above.
              </div>
            ) : (
              <table className="lf-table">
                <thead>
                  <tr><th>Name</th><th>Size</th><th>Created</th><th>Path</th></tr>
                </thead>
                <tbody>
                  {exports.map((exp) => (
                    <tr key={exp.name}>
                      <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-hi)", fontWeight: 500 }}>{exp.name}</td>
                      <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>{exp.size_mb.toFixed(0)} MB</td>
                      <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{new Date(exp.created_at).toLocaleDateString()}</td>
                      <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{exp.path}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
