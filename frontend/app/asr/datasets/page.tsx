"use client";
import { useRef, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getASRDatasets, uploadASRDataset, uploadASRZip, deleteASRDataset } from "@/lib/api";
import Link from "next/link";

type UploadMode = "zip" | "csv";

export default function ASRDatasetsPage() {
  const qc = useQueryClient();
  const { data: datasets = [], isLoading } = useQuery({ queryKey: ["asr-datasets"], queryFn: getASRDatasets });
  const fileRef = useRef<HTMLInputElement>(null);
  const [mode, setMode] = useState<UploadMode>("zip");
  const [form, setForm] = useState({ name: "", description: "", audio_col: "audio_path", text_col: "text" });
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<string | null>(null);

  const { mutate: remove } = useMutation({
    mutationFn: deleteASRDataset,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["asr-datasets"] }),
  });

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
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      setUploadResult(`Error: ${msg}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 12, padding: "12px 14px", height: "calc(100vh - 40px)", overflow: "hidden" }}>
      {/* Dataset table */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
              ASR Datasets
            </span>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--accent)", background: "var(--accent-dim)", padding: "1px 6px", borderRadius: 2 }}>
              asr_csv
            </span>
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
                  <tr key={d.id}>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-hi)", fontWeight: 500 }}>{d.name}</td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)" }}>{d.num_samples?.toLocaleString() ?? "—"}</td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {d.description ?? "—"}
                    </td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{new Date(d.created_at).toLocaleDateString()}</td>
                    <td>
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

      {/* Upload panel */}
      <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Upload Dataset
        </div>
        <div className="lf-panel" style={{ padding: "12px" }}>

          {/* Mode toggle */}
          <div style={{ display: "flex", gap: 4, marginBottom: 12 }}>
            {(["zip", "csv"] as UploadMode[]).map((m) => (
              <button key={m} className={`lf-chip ${mode === m ? "lf-chip-active" : ""}`} style={{ flex: 1, justifyContent: "center" }}
                onClick={() => { setMode(m); setUploadResult(null); if (fileRef.current) fileRef.current.value = ""; }}>
                {m === "zip" ? "ZIP (with audio)" : "CSV (server path)"}
              </button>
            ))}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
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

            <button className="lf-btn lf-btn-primary" style={{ width: "100%", marginTop: 4 }} disabled={uploading || !form.name} onClick={handleUpload}>
              {uploading ? <><span className="lf-spin" /> {mode === "zip" ? "Extracting…" : "Uploading…"}</> : `⇑ Upload ${mode === "zip" ? "ZIP" : "CSV"}`}
            </button>

            {uploadResult && (
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, padding: "6px 8px", borderRadius: 3, border: "1px solid var(--border)", color: uploadResult.startsWith("Error") ? "var(--red)" : "var(--green)", background: "var(--bg)" }}>
                {uploadResult}
              </div>
            )}
          </div>

          <div style={{ marginTop: 14, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
            {mode === "zip" ? (
              <>
                <div className="lf-section" style={{ marginTop: 0 }}>ZIP Format</div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", lineHeight: 1.9 }}>
                  <div style={{ color: "var(--accent)", marginBottom: 2 }}>structure</div>
                  <div style={{ background: "#090b10", padding: "6px 8px", borderRadius: 3, border: "1px solid var(--border)", marginBottom: 8 }}>
                    <div style={{ color: "var(--green)" }}>my_dataset.zip</div>
                    <div style={{ color: "var(--text-dim)" }}>├── manifest.csv</div>
                    <div style={{ color: "var(--text-dim)" }}>├── 001.wav</div>
                    <div style={{ color: "var(--text-dim)" }}>├── 002.mp3</div>
                    <div style={{ color: "var(--text-dim)" }}>└── ...</div>
                  </div>
                  <div style={{ color: "var(--text-dim)" }}>Audio paths in the CSV are matched by <span style={{ color: "var(--accent)" }}>filename</span> — original machine paths do not need to match.</div>
                  <div style={{ marginTop: 6, color: "var(--text-dim)" }}>Supported audio: .wav .mp3 .flac .ogg .m4a</div>
                </div>
              </>
            ) : (
              <>
                <div className="lf-section" style={{ marginTop: 0 }}>CSV Format</div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", lineHeight: 1.9 }}>
                  <div style={{ color: "var(--accent)", marginBottom: 2 }}>required columns</div>
                  <div>audio_path — absolute path on the server to .wav / .mp3</div>
                  <div>text — ground truth transcript</div>
                  <div style={{ background: "#090b10", padding: "6px 8px", borderRadius: 3, border: "1px solid var(--border)", marginTop: 6 }}>
                    <div style={{ color: "var(--green)" }}>audio_path,text</div>
                    <div style={{ color: "var(--text-dim)" }}>/data/audio/001.wav,hello world</div>
                    <div style={{ color: "var(--text-dim)" }}>/data/audio/002.wav,terima kasih</div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
