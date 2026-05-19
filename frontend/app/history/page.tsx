"use client";
import { useState, useMemo, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getJobs, getASRJobs, getJobMetrics, updateJobRemarks, purgeJob } from "@/lib/api";
import { fmtDateTime } from "@/lib/datetime";
import type { Job, JobStatus } from "@/types";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

// ── helpers ──────────────────────────────────────────────────────────────────

function elapsed(started: string | null, finished: string | null) {
  if (!started) return "—";
  const ms = (finished ? new Date(finished) : new Date()).getTime() - new Date(started).getTime();
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

const fmtDate = fmtDateTime;

const STATUS_COLORS: Record<JobStatus, string> = {
  pending: "var(--amber)", running: "var(--accent)",
  completed: "var(--green)", failed: "var(--red)", cancelled: "var(--text-dim)",
};
const STATUS_BG: Record<JobStatus, string> = {
  pending: "lf-badge-pending", running: "lf-badge-running",
  completed: "lf-badge-done", failed: "lf-badge-failed", cancelled: "lf-badge-cancelled",
};

function Badge({ status }: { status: JobStatus }) {
  return <span className={`lf-badge ${STATUS_BG[status]}`}>{status}</span>;
}

// ── chart ────────────────────────────────────────────────────────────────────

function MetricsChart({ jobId }: { jobId: number }) {
  const { data: raw = [], isLoading } = useQuery({
    queryKey: ["job-metrics", jobId],
    queryFn: () => getJobMetrics(jobId),
    staleTime: 30_000,
  });

  if (isLoading) return (
    <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
      loading metrics…
    </div>
  );
  if (raw.length === 0) return (
    <div style={{ height: 60, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>
      no metrics recorded
    </div>
  );

  const hasLoss    = raw.some((r) => r.loss != null);
  const hasEval    = raw.some((r) => r.eval_loss != null);
  const hasReward  = raw.some((r) => r.reward != null);
  const hasLR      = raw.some((r) => r.learning_rate != null);
  const hasGNorm   = raw.some((r) => r.grad_norm != null);

  const data = raw.map((r) => ({ ...r }));

  const tickStyle = { fontFamily: "var(--mono)", fontSize: 10, fill: "var(--text-dim)" };
  const tooltipStyle = {
    background: "var(--bg-panel)", border: "1px solid var(--border)",
    fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-hi)",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {(hasLoss || hasEval) && (
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>Loss</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={data} margin={{ top: 2, right: 8, bottom: 2, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="step" tick={tickStyle} />
              <YAxis tick={tickStyle} width={48} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontFamily: "var(--mono)", fontSize: 10 }} />
              {hasLoss && <Line type="monotone" dataKey="loss" stroke="var(--accent)" dot={false} strokeWidth={1.5} name="train loss" />}
              {hasEval && <Line type="monotone" dataKey="eval_loss" stroke="var(--amber)" dot={false} strokeWidth={1.5} name="eval loss" />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {hasReward && (
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>Reward</div>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={data} margin={{ top: 2, right: 8, bottom: 2, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="step" tick={tickStyle} />
              <YAxis tick={tickStyle} width={48} />
              <Tooltip contentStyle={tooltipStyle} />
              {hasReward && <Line type="monotone" dataKey="reward" stroke="var(--green)" dot={false} strokeWidth={1.5} name="reward" />}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {(hasLR || hasGNorm) && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
          {hasLR && (
            <div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>Learning Rate</div>
              <ResponsiveContainer width="100%" height={100}>
                <LineChart data={data} margin={{ top: 2, right: 4, bottom: 2, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="step" tick={tickStyle} />
                  <YAxis tick={tickStyle} width={52} tickFormatter={(v) => v.toExponential(1)} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toExponential(3)} />
                  <Line type="monotone" dataKey="learning_rate" stroke="var(--accent)" dot={false} strokeWidth={1.5} name="lr" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
          {hasGNorm && (
            <div>
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>Grad Norm</div>
              <ResponsiveContainer width="100%" height={100}>
                <LineChart data={data} margin={{ top: 2, right: 4, bottom: 2, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="step" tick={tickStyle} />
                  <YAxis tick={tickStyle} width={40} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Line type="monotone" dataKey="grad_norm" stroke="var(--red)" dot={false} strokeWidth={1.5} name="grad norm" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── config summary ────────────────────────────────────────────────────────────

const CONFIG_KEYS = [
  "model_path", "learning_rate", "batch_size", "gradient_accumulation_steps",
  "num_epochs", "max_steps", "warmup_steps", "warmup_ratio", "lr_scheduler_type",
  "lora_r", "lora_alpha", "lora_dropout", "quantization", "fp16", "bf16",
  "max_seq_length", "output_dir", "language", "task",
];

function ConfigSummary({ cfg }: { cfg: Record<string, unknown> | null }) {
  if (!cfg) return <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>no config recorded</div>;
  const entries = CONFIG_KEYS.filter((k) => cfg[k] != null).map((k) => [k, cfg[k]]);
  if (entries.length === 0) return <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>—</div>;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2px 16px" }}>
      {entries.map(([k, v]) => (
        <div key={String(k)} style={{ display: "flex", gap: 6, fontFamily: "var(--mono)", fontSize: 11 }}>
          <span style={{ color: "var(--text-dim)", minWidth: 140, flexShrink: 0 }}>{String(k)}</span>
          <span style={{ color: "var(--text-hi)", wordBreak: "break-all" }}>{String(v)}</span>
        </div>
      ))}
    </div>
  );
}

// ── remarks ───────────────────────────────────────────────────────────────────

function RemarksEditor({ job }: { job: Job }) {
  const qc = useQueryClient();
  const [text, setText] = useState(job.remarks ?? "");
  const [saved, setSaved] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => { setText(job.remarks ?? ""); setSaved(true); }, [job.id, job.remarks]);
  useEffect(() => { return () => { if (timerRef.current) clearTimeout(timerRef.current); }; }, []);

  const { mutate: save } = useMutation({
    mutationFn: (r: string) => updateJobRemarks(job.id, r),
    onSuccess: () => {
      setSaved(true);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["asr-jobs"] });
    },
  });

  const handleChange = (v: string) => {
    setText(v);
    setSaved(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => save(v), 1200);
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Remarks</span>
        {saved
          ? <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--green)" }}>saved</span>
          : <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--amber)" }}>unsaved…</span>}
      </div>
      <textarea
        className="lf-textarea"
        value={text}
        onChange={(e) => handleChange(e.target.value)}
        placeholder="Add notes, observations, or next steps for this run…"
        style={{ minHeight: 80, resize: "vertical" }}
      />
    </div>
  );
}

// ── detail panel ──────────────────────────────────────────────────────────────

function DetailPanel({ job }: { job: Job }) {
  const isASR = job.training_method === "asr_whisper";
  const [tab, setTab] = useState<"metrics" | "config">("metrics");

  const bestLoss = useMemo(() => {
    // shown from config or we rely on metrics endpoint
    return null;
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Header */}
      <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--border)", background: "var(--bg-panel)", flexShrink: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 13, fontWeight: 700, color: "var(--text-hi)" }}>#{job.id} {job.name}</span>
          <Badge status={job.status} />
          {isASR && <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--accent)", background: "var(--accent-dim)", padding: "1px 6px", borderRadius: 2 }}>ASR</span>}
        </div>
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          {[
            ["method",   job.training_method],
            ["peft",     job.peft_method],
            ["duration", elapsed(job.started_at, job.finished_at)],
            ["created",  fmtDate(job.created_at)],
          ].map(([k, v]) => (
            <span key={k} style={{ fontFamily: "var(--mono)", fontSize: 11 }}>
              <span style={{ color: "var(--text-dim)" }}>{k} </span>
              <span style={{ color: "var(--text-hi)" }}>{v}</span>
            </span>
          ))}
        </div>
        {job.error_msg && (
          <div style={{ marginTop: 6, fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)", padding: "4px 8px", background: "var(--red-dim)", borderRadius: 3 }}>
            {job.error_msg}
          </div>
        )}
      </div>

      {/* Tab bar */}
      <div style={{ display: "flex", borderBottom: "1px solid var(--border)", background: "var(--bg-panel)", flexShrink: 0 }}>
        {(["metrics", "config"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`lf-tab ${tab === t ? "lf-tab-active" : ""}`}
            style={{ height: 32, fontSize: 11 }}>
            {t}
          </button>
        ))}
      </div>

      {/* Scrollable content */}
      <div style={{ flex: 1, overflowY: "auto", padding: "12px 14px" }}>
        {tab === "metrics" && <MetricsChart jobId={job.id} />}
        {tab === "config" && (
          <div style={{ marginBottom: 16 }}>
            <ConfigSummary cfg={job.config_json} />
          </div>
        )}

        {/* Remarks always visible below the tab content */}
        <div style={{ marginTop: 16, paddingTop: 14, borderTop: "1px solid var(--border)" }}>
          <RemarksEditor job={job} />
        </div>
      </div>
    </div>
  );
}

// ── main page ─────────────────────────────────────────────────────────────────

type FilterStatus = "all" | JobStatus;
type FilterType   = "all" | "llm" | "asr";

export default function HistoryPage() {
  const qc = useQueryClient();
  const { data: llmJobs = [] } = useQuery({ queryKey: ["jobs"],     queryFn: getJobs,    refetchInterval: 8000, staleTime: 6000 });
  const { data: asrJobs = [] } = useQuery({ queryKey: ["asr-jobs"], queryFn: getASRJobs, refetchInterval: 8000, staleTime: 6000 });

  const { mutate: doDelete } = useMutation({
    mutationFn: purgeJob,
    onSuccess: (_data, deletedId) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["asr-jobs"] });
      setSelectedId((prev) => (prev === deletedId ? null : prev));
    },
  });

  const allJobs = useMemo(
    () => [...llmJobs, ...asrJobs].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    [llmJobs, asrJobs],
  );

  const [filterStatus, setFilterStatus] = useState<FilterStatus>("all");
  const [filterType,   setFilterType]   = useState<FilterType>("all");
  const [search,       setSearch]       = useState("");
  const [selectedId,   setSelectedId]   = useState<number | null>(null);

  const filtered = useMemo(() => allJobs.filter((j) => {
    if (filterStatus !== "all" && j.status !== filterStatus) return false;
    if (filterType === "llm" && j.training_method === "asr_whisper") return false;
    if (filterType === "asr" && j.training_method !== "asr_whisper") return false;
    if (search && !j.name.toLowerCase().includes(search.toLowerCase()) && !String(j.id).includes(search)) return false;
    return true;
  }), [allJobs, filterStatus, filterType, search]);

  const selectedJob = allJobs.find((j) => j.id === selectedId) ?? null;

  // Auto-select first job if nothing selected
  useEffect(() => {
    if (!selectedId && filtered.length > 0) setSelectedId(filtered[0].id);
  }, [filtered, selectedId]);

  const STATUS_FILTERS: FilterStatus[] = ["all", "completed", "running", "failed", "pending", "cancelled"];
  const TYPE_FILTERS: [FilterType, string][] = [["all", "All"], ["llm", "LLM"], ["asr", "ASR"]];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "340px 1fr", height: "calc(100vh - 40px)", overflow: "hidden" }}>

      {/* ── LEFT: job list ── */}
      <div style={{ borderRight: "1px solid var(--border)", display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Filters */}
        <div style={{ padding: "8px 10px", borderBottom: "1px solid var(--border)", background: "var(--bg-panel)", flexShrink: 0 }}>
          <input
            className="lf-input"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="search by name or id…"
            style={{ marginBottom: 7 }}
          />
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 5 }}>
            {TYPE_FILTERS.map(([t, label]) => (
              <button key={t} className={`lf-chip ${filterType === t ? "lf-chip-active" : ""}`}
                style={{ height: 20, fontSize: 10, padding: "0 7px" }} onClick={() => setFilterType(t)}>
                {label}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {STATUS_FILTERS.map((s) => (
              <button key={s} className={`lf-chip ${filterStatus === s ? "lf-chip-active" : ""}`}
                style={{ height: 20, fontSize: 10, padding: "0 7px",
                  ...(filterStatus !== s && s !== "all" ? { color: STATUS_COLORS[s as JobStatus] } : {}) }}
                onClick={() => setFilterStatus(s)}>
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Count */}
        <div style={{ padding: "4px 10px", borderBottom: "1px solid var(--border)", fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", flexShrink: 0 }}>
          {filtered.length} / {allJobs.length} runs
        </div>

        {/* List */}
        <div style={{ flex: 1, overflowY: "auto" }}>
          {filtered.length === 0 ? (
            <div style={{ padding: 24, textAlign: "center", fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>no runs match filter</div>
          ) : filtered.map((j) => {
            const isASR = j.training_method === "asr_whisper";
            const active = j.id === selectedId;
            return (
              <div key={j.id}
                style={{
                  position: "relative",
                  padding: "8px 10px",
                  borderBottom: "1px solid var(--border)",
                  background: active ? "var(--bg-hover)" : "transparent",
                  cursor: "pointer",
                  borderLeft: active ? "2px solid var(--accent)" : "2px solid transparent",
                  transition: "background 0.1s",
                }}
                onClick={() => setSelectedId(j.id)}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>#{j.id}</span>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 12, fontWeight: 500, color: "var(--text-hi)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{j.name}</span>
                  <Badge status={j.status} />
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      if (confirm(`Delete job #${j.id} "${j.name}" permanently?`)) doDelete(j.id);
                    }}
                    title="Delete permanently"
                    style={{
                      background: "none", border: "none", cursor: "pointer",
                      color: "var(--text-dim)", padding: "2px 4px", borderRadius: 2,
                      fontFamily: "var(--mono)", fontSize: 12, lineHeight: 1, flexShrink: 0,
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = "var(--red)")}
                    onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-dim)")}
                  >✕</button>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: isASR ? "var(--accent)" : "var(--text-dim)", background: isASR ? "var(--accent-dim)" : undefined, padding: isASR ? "0 4px" : 0, borderRadius: 2 }}>
                    {isASR ? "ASR" : j.training_method}
                  </span>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{j.peft_method}</span>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginLeft: "auto" }}>{elapsed(j.started_at, j.finished_at)}</span>
                </div>
                {j.remarks && (
                  <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--amber)", marginTop: 3, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    📝 {j.remarks}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ── RIGHT: detail ── */}
      <div style={{ overflow: "hidden" }}>
        {selectedJob
          ? <DetailPanel key={selectedJob.id} job={selectedJob} />
          : (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", fontFamily: "var(--mono)", fontSize: 12, color: "var(--text-dim)" }}>
              Select a run to view its metrics and details
            </div>
          )}
      </div>
    </div>
  );
}
