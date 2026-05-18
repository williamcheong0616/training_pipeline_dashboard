"use client";
import { useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getDatasets, uploadDataset, deleteDataset } from "@/lib/api";

export default function DatasetsPage() {
  const qc = useQueryClient();
  const { data: datasets = [], isLoading } = useQuery({ queryKey: ["datasets"], queryFn: getDatasets });
  const fileRef = useRef<HTMLInputElement>(null);
  const [form, setForm] = useState({ name: "", format: "alpaca", description: "" });
  const [uploading, setUploading] = useState(false);

  const { mutate: remove } = useMutation({
    mutationFn: deleteDataset,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }),
  });

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
      if (fileRef.current) fileRef.current.value = "";
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 12, padding: "12px 14px", height: "calc(100vh - 40px)", overflow: "hidden" }}>
      {/* Datasets table */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Datasets ({datasets.length})
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
                  <tr key={d.id}>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-hi)", fontWeight: 500 }}>{d.name}</td>
                    <td><span className="lf-badge" style={{ background: "var(--bg-input)", color: "var(--text-dim)", border: "1px solid var(--border)" }}>{d.format}</span></td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>{d.num_samples?.toLocaleString() ?? "—"}</td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {d.description ?? "—"}
                    </td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{new Date(d.created_at).toLocaleDateString()}</td>
                    <td>
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

      {/* Upload panel */}
      <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Upload Dataset
        </div>
        <div className="lf-panel" style={{ padding: "12px 12px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div>
              <label className="lf-label">name</label>
              <input className="lf-input" value={form.name} onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))} placeholder="my-dataset" />
            </div>
            <div>
              <label className="lf-label">format</label>
              <select className="lf-input lf-select" value={form.format} onChange={(e) => setForm((p) => ({ ...p, format: e.target.value }))}>
                {["alpaca","sharegpt","plain_text"].map((f) => <option key={f}>{f}</option>)}
              </select>
            </div>
            <div>
              <label className="lf-label">description</label>
              <input className="lf-input" value={form.description} onChange={(e) => setForm((p) => ({ ...p, description: e.target.value }))} />
            </div>
            <div>
              <label className="lf-label">file (.json / .jsonl)</label>
              <input
                ref={fileRef}
                type="file"
                accept=".json,.jsonl"
                style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text)", width: "100%", cursor: "pointer" }}
              />
            </div>
            <button
              className="lf-btn lf-btn-primary"
              style={{ width: "100%", marginTop: 4 }}
              disabled={uploading || !form.name}
              onClick={handleUpload}
            >
              {uploading ? <><span className="lf-spin" /> Uploading…</> : "⇑ Upload"}
            </button>
          </div>

          <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
            <div className="lf-section">Format Guide</div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", lineHeight: 1.8 }}>
              <div style={{ color: "var(--accent)", marginBottom: 4 }}>alpaca</div>
              <div>{`{"instruction":"...","input":"...","output":"..."}`}</div>
              <div style={{ color: "var(--accent)", marginBottom: 4, marginTop: 8 }}>sharegpt</div>
              <div>{`{"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}`}</div>
              <div style={{ color: "var(--accent)", marginBottom: 4, marginTop: 8 }}>plain_text</div>
              <div>{`{"text":"raw content..."}`}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
