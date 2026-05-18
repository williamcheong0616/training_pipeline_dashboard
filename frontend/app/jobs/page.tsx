"use client";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { getJobs, cancelJob } from "@/lib/api";
import type { JobStatus } from "@/types";

function badge(status: JobStatus) {
  const map: Record<JobStatus, string> = {
    pending: "lf-badge-pending", running: "lf-badge-running",
    completed: "lf-badge-done", failed: "lf-badge-failed", cancelled: "lf-badge-cancelled",
  };
  return <span className={`lf-badge ${map[status]}`}>{status}</span>;
}

function elapsed(started: string | null, finished: string | null) {
  if (!started) return "—";
  const ms = (finished ? new Date(finished) : new Date()).getTime() - new Date(started).getTime();
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s/60)}m${s%60}s`;
  return `${Math.floor(s/3600)}h${Math.floor((s%3600)/60)}m`;
}

export default function JobsPage() {
  const qc = useQueryClient();
  const { data: jobs = [], isLoading } = useQuery({ queryKey: ["jobs"], queryFn: getJobs, refetchInterval: 4000 });
  const { mutate: cancel } = useMutation({ mutationFn: cancelJob, onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }) });

  return (
    <div style={{ padding: "12px 16px" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <span style={{ fontFamily: "var(--mono)", fontSize: 12, fontWeight: 600, color: "var(--text-hi)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          Job History
        </span>
        <Link href="/" className="lf-btn lf-btn-primary" style={{ height: 26, fontSize: 11 }}>▶ New Training</Link>
      </div>

      <div className="lf-panel" style={{ overflow: "hidden" }}>
        {isLoading ? (
          <div style={{ padding: 24, textAlign: "center", fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>loading…</div>
        ) : jobs.length === 0 ? (
          <div style={{ padding: 40, textAlign: "center", fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
            No jobs yet. <Link href="/" style={{ color: "var(--accent)" }}>Start training →</Link>
          </div>
        ) : (
          <table className="lf-table">
            <thead>
              <tr>
                <th>#</th><th>Name</th><th>Method</th><th>PEFT</th>
                <th>Status</th><th>Duration</th><th>Created</th><th style={{ width: 80 }} />
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id}>
                  <td style={{ fontFamily: "var(--mono)", color: "var(--text-dim)", fontSize: 11 }}>{j.id}</td>
                  <td>
                    <Link href={`/jobs/${j.id}`} style={{ color: "var(--accent)", fontFamily: "var(--mono)", fontSize: 12, textDecoration: "none" }}>
                      {j.name}
                    </Link>
                  </td>
                  <td><span style={{ fontFamily: "var(--mono)", fontSize: 11, textTransform: "uppercase", color: "var(--text-dim)" }}>{j.training_method}</span></td>
                  <td><span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>{j.peft_method}</span></td>
                  <td>{badge(j.status)}</td>
                  <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>{elapsed(j.started_at, j.finished_at)}</td>
                  <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{new Date(j.created_at).toLocaleString()}</td>
                  <td style={{ textAlign: "right" }}>
                    {(j.status === "running" || j.status === "pending") ? (
                      <button className="lf-btn lf-btn-danger" style={{ height: 22, fontSize: 10, padding: "0 8px" }} onClick={() => cancel(j.id)}>stop</button>
                    ) : (
                      <Link href={`/jobs/${j.id}`} className="lf-btn lf-btn-ghost" style={{ height: 22, fontSize: 10, padding: "0 8px" }}>view</Link>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
