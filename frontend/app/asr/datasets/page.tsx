"use client";
import { useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getASRDatasets, uploadASRDataset, uploadASRZip, deleteASRDataset, previewASRDataset } from "@/lib/api";
import Link from "next/link";
import type { Dataset } from "@/types";

type UploadMode = "zip" | "csv";

// ── preview panel ─────────────────────────────────────────────────────────────

function PreviewPanel({ dataset, onClose }: { dataset: Dataset; onClose: () => void }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["asr-dataset-preview", dataset.id],
    queryFn: () => previewASRDataset(dataset.id),
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
        <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 8 }}>
          {data.total?.toLocaleString() ?? "?"} samples · columns: {data.columns.join(", ")}
        </div>
      )}

      <div style={{ flex: 1, overflowY: "auto" }}>
        {isLoading && <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)", padding: 12 }}>loading preview…</div>}
        {isError && <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)", padding: 12 }}>Failed to load — CSV may have moved.</div>}
        {data?.samples.map((row, i) => (
          <div key={i} style={{ marginBottom: 8, padding: "8px 10px", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 3 }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 5 }}>row #{i + 1}</div>
            {Object.entries(row).map(([col, val]) => {
              const isAudio = col.toLowerCase().includes("audio") || col.toLowerCase().includes("path");
              return (
                <div key={col} style={{ marginBottom: 4 }}>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--accent)", display: "block", marginBottom: 1 }}>{col}</span>
                  <div style={{
                    fontFamily: "var(--mono)", fontSize: 11,
                    color: isAudio ? "var(--text-dim)" : "var(--text-hi)",
                    background: "var(--bg-input)", padding: "3px 6px", borderRadius: 2,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }} title={val}>
                    {isAudio ? "📂 " : ""}{val || "—"}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── upload panel ──────────────────────────────────────────────────────────────

function UploadPanel({ onUploaded }: { onUploaded: () => void }) {
  const qc = useQueryClient();
  const fileRef = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<UploadMode>("zip");
  const [form, setForm] = useState({ name: "", description: "", audio_col: "audio_path", text_col: "text" });
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<string | null>(null);

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file || !form.name) return;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("name", form.name);
    fd.append("description", form.description);
    if (mode === "zip") {
      fd.append("audio_col", form.audio_col);
      fd.append("text_col", form.text_col);
    }
    setUploading(true);
    setUploadResult(null);
    try {
      const result = mode === "zip" ? await uploadASRZip(fd) : await uploadASRDataset(fd);
      qc.invalidateQueries({ queryKey: ["asr-datasets"] });
      setUploadResult(`Registered "${result.name}" — ${result.num_samples?.toLocaleString() ?? "?"} samples`);
      setForm({ name: "", description: "", audio_col: "audio_path", text_col: "text" });
      if (fileRef.current) fileRef.current.value = "";
      onUploaded();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      setUploadResult(`Error: ${msg}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, overflowY: "auto" }}>
      <div style={{ display: "flex", gap: 4 }}>
        {(["zip", "csv"] as UploadMode[]).map((m) => (
          <button key={m} className={`lf-chip ${mode === m ? "lf-chip-active" : ""}`} style={{ flex: 1, justifyContent: "center" }}
            onClick={() => { setMode(m); setUploadResult(null); if (fileRef.current) fileRef.current.value = ""; }}>
            {m === "zip" ? "ZIP (with audio)" : "CSV (server path)"}
          </button>
        ))}
      </div>

      <div>
        <label className="lf-label">name</label>
        <input className="lf-input" value={form.name} onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))} placeholder="my-malay-asr" />
      </div>
      <div>
        <label className="lf-label">description</label>
        <input className="lf-input" value={form.description} onChange={(e) => setForm((p) => ({ ...p, description: e.target.value }))} />
      </div>

      {mode === "zip" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
          <div>
            <label className="lf-label">audio column</label>
            <input className="lf-input" value={form.audio_col} onChange={(e) => setForm((p) => ({ ...p, audio_col: e.target.value }))} />
          </div>
          <div>
            <label className="lf-label">text column</label>
            <input className="lf-input" value={form.text_col} onChange={(e) => setForm((p) => ({ ...p, text_col: e.target.value }))} />
          </div>
        </div>
      )}

      <div>
        <label className="lf-label">{mode === "zip" ? "ZIP file (.zip)" : "CSV file (.csv)"}</label>
        <input ref={fileRef} type="file" accept={mode === "zip" ? ".zip" : ".csv"}
          style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text)", width: "100%", cursor: "pointer" }} />
      </div>

      <button className="lf-btn lf-btn-primary" style={{ width: "100%" }} disabled={uploading || !form.name} onClick={handleUpload}>
        {uploading ? <><span className="lf-spin" /> {mode === "zip" ? "Extracting…" : "Uploading…"}</> : `⇑ Upload ${mode === "zip" ? "ZIP" : "CSV"}`}
      </button>

      {uploadResult && (
        <div style={{ fontFamily: "var(--mono)", fontSize: 10, padding: "6px 8px", borderRadius: 3, border: "1px solid var(--border)", color: uploadResult.startsWith("Error") ? "var(--red)" : "var(--green)", background: "var(--bg)" }}>
          {uploadResult}
        </div>
      )}

      <div style={{ marginTop: 6, borderTop: "1px solid var(--border)", paddingTop: 10 }}>
        {mode === "zip" ? (
          <>
            <div className="lf-section" style={{ marginTop: 0 }}>ZIP Format</div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", lineHeight: 1.9 }}>
              <div style={{ background: "var(--console-bg)", padding: "6px 8px", borderRadius: 3, border: "1px solid var(--border)", marginBottom: 8, color: "var(--console-text)" }}>
                <div>my_dataset.zip</div>
                <div style={{ color: "var(--text-dim)" }}>├── manifest.csv</div>
                <div style={{ color: "var(--text-dim)" }}>├── 001.wav</div>
                <div style={{ color: "var(--text-dim)" }}>└── ...</div>
              </div>
              <div>Audio paths matched by filename. Supported: .wav .mp3 .flac .ogg .m4a</div>
            </div>
          </>
        ) : (
          <>
            <div className="lf-section" style={{ marginTop: 0 }}>CSV Format</div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", lineHeight: 1.9 }}>
              <div style={{ background: "var(--console-bg)", padding: "6px 8px", borderRadius: 3, border: "1px solid var(--border)", color: "var(--console-text)" }}>
                <div>audio_path,text</div>
                <div style={{ color: "var(--text-dim)" }}>/data/001.wav,hello world</div>
                <div style={{ color: "var(--text-dim)" }}>/data/002.wav,terima kasih</div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── main page ─────────────────────────────────────────────────────────────────

export default function ASRDatasetsPage() {
  const qc = useQueryClient();
  const { data: datasets = [], isLoading } = useQuery({ queryKey: ["asr-datasets"], queryFn: getASRDatasets });
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [rightPanel, setRightPanel] = useState<"upload" | "preview">("upload");

  const { mutate: remove } = useMutation({
    mutationFn: deleteASRDataset,
    onSuccess: (_d, id) => {
      qc.invalidateQueries({ queryKey: ["asr-datasets"] });
      if (selectedId === id) { setSelectedId(null); setRightPanel("upload"); }
    },
  });

  const selectDataset = (d: Dataset) => {
    setSelectedId(d.id);
    setRightPanel("preview");
  };

  const selected = datasets.find((d) => d.id === selectedId) ?? null;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 12, padding: "12px 14px", height: "calc(100vh - 40px)", overflow: "hidden" }}>

      {/* ── left: table ── */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              ASR Datasets ({datasets.length})
            </span>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--accent)", background: "var(--accent-dim)", padding: "1px 6px", borderRadius: 2 }}>asr_csv</span>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>click a row to preview</span>
          </div>
          <Link href="/asr" className="lf-btn lf-btn-ghost" style={{ height: 24, fontSize: 10, padding: "0 10px" }}>← Back to ASR</Link>
        </div>

        <div className="lf-panel" style={{ flex: 1, overflow: "auto" }}>
          {isLoading ? (
            <div style={{ padding: 20, fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)", textAlign: "center" }}>loading…</div>
          ) : (
            <table className="lf-table">
              <thead>
                <tr><th>Name</th><th>Samples</th><th>Description</th><th>Uploaded</th><th /></tr>
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
                    <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>{d.num_samples?.toLocaleString() ?? "—"}</td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {d.description ?? "—"}
                    </td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{new Date(d.created_at).toLocaleDateString()}</td>
                    <td onClick={(e) => e.stopPropagation()}>
                      <button className="lf-btn lf-btn-danger" style={{ height: 20, fontSize: 10, padding: "0 6px" }} onClick={() => remove(d.id)}>✕</button>
                    </td>
                  </tr>
                ))}
                {datasets.length === 0 && (
                  <tr><td colSpan={5} style={{ textAlign: "center", color: "var(--text-dim)", fontFamily: "var(--mono)", fontSize: 11, padding: 32 }}>
                    No ASR datasets yet.
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
