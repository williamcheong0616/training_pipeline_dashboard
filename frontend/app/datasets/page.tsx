"use client";
import { useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getDatasets, uploadDataset, deleteDataset, previewDataset } from "@/lib/api";
import type { Dataset } from "@/types";

// ── format auto-detection ────────────────────────────────────────────────────

function detectFormat(records: Record<string, unknown>[]): string {
  if (!records.length) return "plain_text";
  const first = records[0];
  if ("conversations" in first || "messages" in first) return "sharegpt";
  if ("prompt" in first && "chosen" in first && "rejected" in first) return "dpo";
  if ("prompt" in first && "completion" in first && "label" in first) return "kto";
  if ("instruction" in first || "output" in first) return "alpaca";
  return "plain_text";
}

function formatLabel(fmt: string) {
  const colors: Record<string, string> = {
    alpaca: "var(--green)", sharegpt: "var(--accent)", plain_text: "var(--text-dim)",
    dpo: "var(--amber)", kto: "var(--amber)",
  };
  return (
    <span className="lf-badge" style={{ background: "var(--bg-input)", color: colors[fmt] ?? "var(--text-dim)", border: "1px solid var(--border)" }}>
      {fmt}
    </span>
  );
}

// ── sample card renderer ──────────────────────────────────────────────────────

function SampleCard({ record, index }: { record: Record<string, unknown>; index: number }) {
  const KEY_ORDER = ["instruction", "input", "output", "prompt", "chosen", "rejected",
    "completion", "label", "text", "conversations", "messages"];
  const keys = [...new Set([...KEY_ORDER.filter((k) => k in record), ...Object.keys(record)])];

  return (
    <div style={{
      marginBottom: 8, padding: "8px 10px",
      background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 3,
    }}>
      <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 6 }}>
        sample #{index + 1}
      </div>
      {keys.map((k) => {
        const val = record[k];
        const display = typeof val === "string" ? val : JSON.stringify(val, null, 2);
        return (
          <div key={k} style={{ marginBottom: 5 }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--accent)", display: "block", marginBottom: 2 }}>{k}</span>
            <div style={{
              fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-hi)",
              background: "var(--bg-input)", padding: "4px 6px", borderRadius: 2,
              whiteSpace: "pre-wrap", wordBreak: "break-word",
              maxHeight: 120, overflowY: "auto", lineHeight: 1.5,
            }}>
              {display.length > 400 ? display.slice(0, 400) + "…" : display}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── preview panel ─────────────────────────────────────────────────────────────

function PreviewPanel({ dataset, onClose }: { dataset: Dataset; onClose: () => void }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["dataset-preview", dataset.id],
    queryFn: () => previewDataset(dataset.id),
    staleTime: 60_000,
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-hi)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {dataset.name}
        </span>
        <button className="lf-btn lf-btn-ghost" style={{ height: 22, fontSize: 10, padding: "0 8px" }} onClick={onClose}>✕ close</button>
      </div>

      {data && (
        <div style={{ display: "flex", gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
          {formatLabel(data.format)}
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>
            {data.total?.toLocaleString() ?? "?"} samples total · showing {data.samples.length}
          </span>
        </div>
      )}

      <div style={{ flex: 1, overflowY: "auto" }}>
        {isLoading && <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)", padding: 12 }}>loading preview…</div>}
        {isError && <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)", padding: 12 }}>Failed to load preview — file may have moved.</div>}
        {data?.samples.map((rec, i) => <SampleCard key={i} record={rec} index={i} />)}
      </div>
    </div>
  );
}

// ── upload panel ──────────────────────────────────────────────────────────────

const FORMATS = ["alpaca", "sharegpt", "plain_text", "dpo", "kto"];

function UploadPanel({ onUploaded }: { onUploaded: () => void }) {
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [form, setForm] = useState({ name: "", format: "alpaca", description: "" });
  const [detected, setDetected] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const handleFileChange = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    // Auto-fill name if empty
    if (!form.name) {
      setForm((p) => ({ ...p, name: file.name.replace(/\.(json|jsonl)$/i, "") }));
    }
    // Detect format
    try {
      const text = await file.text();
      let records: Record<string, unknown>[] = [];
      if (file.name.endsWith(".jsonl")) {
        const lines = text.split("\n").filter((l) => l.trim());
        records = lines.slice(0, 3).map((l) => JSON.parse(l));
      } else {
        const parsed = JSON.parse(text);
        records = Array.isArray(parsed) ? parsed.slice(0, 3) : [parsed];
      }
      const fmt = detectFormat(records);
      setDetected(fmt);
      setForm((p) => ({ ...p, format: fmt }));
    } catch {
      setDetected(null);
    }
  };

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file || !form.name) return;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("name", form.name);
    fd.append("format", form.format);
    fd.append("description", form.description);
    setUploading(true);
    try {
      await uploadDataset(fd);
      qc.invalidateQueries({ queryKey: ["datasets"] });
      setForm({ name: "", format: "alpaca", description: "" });
      setDetected(null);
      if (fileRef.current) fileRef.current.value = "";
      onUploaded();
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div>
        <label className="lf-label">name</label>
        <input className="lf-input" value={form.name} onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))} placeholder="my-dataset" />
      </div>
      <div>
        <label className="lf-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
          format
          {detected && (
            <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--green)", background: "var(--green-dim)", padding: "1px 5px", borderRadius: 2 }}>
              auto-detected
            </span>
          )}
        </label>
        <select className="lf-input lf-select" value={form.format} onChange={(e) => { setForm((p) => ({ ...p, format: e.target.value })); setDetected(null); }}>
          {FORMATS.map((f) => <option key={f}>{f}</option>)}
        </select>
      </div>
      <div>
        <label className="lf-label">description</label>
        <input className="lf-input" value={form.description} onChange={(e) => setForm((p) => ({ ...p, description: e.target.value }))} />
      </div>
      <div>
        <label className="lf-label">file (.json / .jsonl)</label>
        <input ref={fileRef} type="file" accept=".json,.jsonl" onChange={handleFileChange}
          style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text)", width: "100%", cursor: "pointer" }} />
      </div>
      <button className="lf-btn lf-btn-primary" style={{ width: "100%", marginTop: 4 }} disabled={uploading || !form.name} onClick={handleUpload}>
        {uploading ? <><span className="lf-spin" /> Uploading…</> : "⇑ Upload"}
      </button>

      <div style={{ marginTop: 10, borderTop: "1px solid var(--border)", paddingTop: 10 }}>
        <div className="lf-section" style={{ marginTop: 0 }}>Format Guide</div>
        <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", lineHeight: 1.8 }}>
          {[
            ["alpaca",     `{"instruction":"…","input":"…","output":"…"}`],
            ["sharegpt",   `{"conversations":[{"from":"human","value":"…"},…]}`],
            ["dpo",        `{"prompt":"…","chosen":"…","rejected":"…"}`],
            ["kto",        `{"prompt":"…","completion":"…","label":true}`],
            ["plain_text", `{"text":"raw content…"}`],
          ].map(([fmt, ex]) => (
            <div key={fmt} style={{ marginBottom: 6 }}>
              <span style={{ color: "var(--accent)" }}>{fmt}</span>
              <div style={{ color: "var(--text-dim)", fontSize: 9, marginTop: 1 }}>{ex}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── main page ─────────────────────────────────────────────────────────────────

export default function DatasetsPage() {
  const qc = useQueryClient();
  const { data: datasets = [], isLoading } = useQuery({ queryKey: ["datasets"], queryFn: getDatasets });
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [rightPanel, setRightPanel] = useState<"upload" | "preview">("upload");

  const { mutate: remove } = useMutation({
    mutationFn: deleteDataset,
    onSuccess: (_d, id) => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      if (selectedId === id) { setSelectedId(null); setRightPanel("upload"); }
    },
  });

  const selectDataset = (d: Dataset) => {
    setSelectedId(d.id);
    setRightPanel("preview");
  };

  const selected = datasets.find((d) => d.id === selectedId) ?? null;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 12, padding: "12px 14px", height: "calc(100vh - 40px)", overflow: "hidden" }}>

      {/* ── left: table ── */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Datasets ({datasets.length})
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", fontWeight: 400, marginLeft: 10, textTransform: "none" }}>
            click a row to preview
          </span>
        </div>
        <div className="lf-panel" style={{ flex: 1, overflow: "auto" }}>
          {isLoading ? (
            <div style={{ padding: 20, fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)", textAlign: "center" }}>loading…</div>
          ) : (
            <table className="lf-table">
              <thead>
                <tr><th>Name</th><th>Format</th><th>Samples</th><th>Description</th><th>Uploaded</th><th /></tr>
              </thead>
              <tbody>
                {datasets.map((d) => (
                  <tr key={d.id}
                    onClick={() => selectDataset(d)}
                    style={{
                      cursor: "pointer",
                      background: d.id === selectedId ? "var(--bg-hover)" : undefined,
                      borderLeft: d.id === selectedId ? "2px solid var(--accent)" : "2px solid transparent",
                    }}>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-hi)", fontWeight: 500 }}>{d.name}</td>
                    <td>{formatLabel(d.format)}</td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>{d.num_samples?.toLocaleString() ?? "—"}</td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {d.description ?? "—"}
                    </td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{new Date(d.created_at).toLocaleDateString()}</td>
                    <td onClick={(e) => e.stopPropagation()}>
                      <button className="lf-btn lf-btn-danger" style={{ height: 20, fontSize: 10, padding: "0 6px" }} onClick={() => remove(d.id)}>✕</button>
                    </td>
                  </tr>
                ))}
                {datasets.length === 0 && (
                  <tr><td colSpan={6} style={{ textAlign: "center", color: "var(--text-dim)", fontFamily: "var(--mono)", fontSize: 11, padding: 32 }}>
                    No datasets. Upload a JSON or JSONL file.
                  </td></tr>
                )}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* ── right: upload / preview ── */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
          {(["upload", "preview"] as const).map((p) => (
            <button key={p} className={`lf-chip ${rightPanel === p ? "lf-chip-active" : ""}`}
              style={{ flex: 1, justifyContent: "center" }}
              onClick={() => setRightPanel(p)}
              disabled={p === "preview" && !selected}>
              {p === "upload" ? "⇑ Upload" : `Preview${selected ? ` · ${selected.name}` : ""}`}
            </button>
          ))}
        </div>

        <div className="lf-panel" style={{ flex: 1, padding: "12px", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          {rightPanel === "upload"
            ? <UploadPanel onUploaded={() => {}} />
            : selected
              ? <PreviewPanel dataset={selected} onClose={() => setRightPanel("upload")} />
              : <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>Select a dataset to preview.</div>
          }
        </div>
      </div>
    </div>
  );
}
