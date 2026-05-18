"use client";
import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getJob, cancelJob, exportJob } from "@/lib/api";
import { useMetricsStream } from "@/lib/sse";
import MetricsPanel from "@/components/MetricsPanel";
import type { JobStatus } from "@/types";

function badge(status: JobStatus) {
  const map: Record<JobStatus, string> = {
    pending: "lf-badge-pending", running: "lf-badge-running",
    completed: "lf-badge-done", failed: "lf-badge-failed", cancelled: "lf-badge-cancelled",
  };
  return <span className={`lf-badge ${map[status]}`}>{status}</span>;
}

export default function JobDetailPage() {
  const { id } = useParams<{ id: string }>();
  const jobId = Number(id);
  const qc = useQueryClient();
  const logRef = useRef<HTMLDivElement>(null);
  const [logs, setLogs] = useState<string[]>([]);

  const { data: job } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    refetchInterval: (q) => (q.state.data?.status === "running" ? 3000 : false),
  });

  const metrics = useMetricsStream(job?.status === "running" ? jobId : null);

  const { mutate: cancel } = useMutation({
    mutationFn: () => cancelJob(jobId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["job", jobId] }),
  });

  const { mutate: doExport, isPending: exporting } = useMutation({
    mutationFn: () => exportJob(jobId),
  });

  useEffect(() => {
    if (!job) return;
    const lines = [
      `[system] Job #${job.id}: ${job.name}`,
      `[system] Method=${job.training_method.toUpperCase()}  PEFT=${job.peft_method}  Status=${job.status}`,
    ];
    if (job.started_at) lines.push(`[system] Started: ${new Date(job.started_at).toLocaleString()}`);
    if (job.error_msg) lines.push(`[error] ${job.error_msg}`);
    setLogs(lines);
  }, [job?.id]);

  useEffect(() => {
    if (metrics.length === 0) return;
    const last = metrics[metrics.length - 1];
    const parts = [`[step ${last.step}]`];
    if (last.loss != null) parts.push(`loss=${last.loss.toFixed(4)}`);
    if (last.eval_loss != null) parts.push(`eval_loss=${last.eval_loss.toFixed(4)}`);
    if (last.learning_rate != null) parts.push(`lr=${last.learning_rate.toExponential(2)}`);
    if (last.epoch != null) parts.push(`epoch=${last.epoch.toFixed(2)}`);
    if (last.grad_norm != null) parts.push(`grad_norm=${last.grad_norm.toFixed(3)}`);
    if (last.reward != null) parts.push(`reward=${last.reward.toFixed(4)}`);
    setLogs((p) => [...p, parts.join("  ")]);
  }, [metrics.length]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  if (!job) return (
    <div style={{ padding: 24, fontFamily: "var(--mono)", fontSize: 12, color: "var(--text-dim)" }}>loading…</div>
  );

  const cfg = (job as unknown as { config_json?: Record<string, unknown> }).config_json ?? {};

  return (
    <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", height: "calc(100vh - 40px)", overflow: "hidden" }}>
      {/* ── LEFT: Config details ── */}
      <div style={{ borderRight: "1px solid var(--border)", overflowY: "auto", padding: "10px 12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
          {badge(job.status)}
          <span style={{ fontFamily: "var(--mono)", fontSize: 12, fontWeight: 600, color: "var(--text-hi)" }}>#{job.id} {job.name}</span>
        </div>

        {job.error_msg && (
          <div style={{ background: "var(--red-dim)", border: "1px solid var(--red)", borderRadius: 3, padding: "8px 10px", marginBottom: 10, fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)" }}>
            {job.error_msg}
          </div>
        )}

        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          {(job.status === "running" || job.status === "pending") && (
            <button className="lf-btn lf-btn-danger" style={{ flex: 1 }} onClick={() => cancel()}>■ Abort</button>
          )}
          {job.status === "completed" && (
            <button className="lf-btn lf-btn-success" style={{ flex: 1 }} disabled={exporting} onClick={() => doExport()}>
              {exporting ? <><span className="lf-spin" /> Merging…</> : "⇓ Export & Merge"}
            </button>
          )}
        </div>

        <div className="lf-section">Job Info</div>
        <ConfigTable rows={[
          ["method",    job.training_method.toUpperCase()],
          ["peft",      job.peft_method],
          ["created",   new Date(job.created_at).toLocaleString()],
          ["started",   job.started_at ? new Date(job.started_at).toLocaleString() : "—"],
          ["finished",  job.finished_at ? new Date(job.finished_at).toLocaleString() : "—"],
          ["output",    job.output_dir ?? "—"],
        ]} />

        <div className="lf-section" style={{ marginTop: 12 }}>Config</div>
        {Object.entries(cfg).map(([k, v]) => (
          <div key={k} style={{ display: "flex", gap: 8, padding: "3px 0", borderBottom: "1px solid var(--border)" }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", minWidth: 120, flexShrink: 0 }}>{k}</span>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-hi)", wordBreak: "break-all" }}>
              {Array.isArray(v) ? v.join(", ") : v == null ? "null" : String(v)}
            </span>
          </div>
        ))}
      </div>

      {/* ── RIGHT: Metrics + Log ── */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Charts */}
        <div style={{ flex: "0 0 auto", padding: "10px 14px", borderBottom: "1px solid var(--border)" }}>
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
            {job.status === "running" && <span className="lf-cursor">█</span>}
          </div>
        </div>
      </div>
    </div>
  );
}

function ConfigTable({ rows }: { rows: [string, string][] }) {
  return (
    <div>
      {rows.map(([k, v]) => (
        <div key={k} style={{ display: "flex", gap: 8, padding: "4px 0", borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", minWidth: 80, flexShrink: 0 }}>{k}</span>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-hi)", wordBreak: "break-all" }}>{v}</span>
        </div>
      ))}
    </div>
  );
}
