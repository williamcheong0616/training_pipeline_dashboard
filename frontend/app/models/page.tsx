"use client";
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getModels, registerModel, downloadModel, searchHub } from "@/lib/api";
import type { HFSearchResult } from "@/types";

export default function ModelsPage() {
  const qc = useQueryClient();
  const { data: models = [], isLoading } = useQuery({ queryKey: ["models"], queryFn: getModels });
  const [q, setQ] = useState("");
  const [results, setResults] = useState<HFSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [manual, setManual] = useState({ name: "", hf_repo: "", template: "alpaca", architecture: "" });

  const { mutate: download } = useMutation({ mutationFn: downloadModel, onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }) });
  const { mutate: register } = useMutation({ mutationFn: registerModel, onSuccess: () => { qc.invalidateQueries({ queryKey: ["models"] }); setManual({ name: "", hf_repo: "", template: "alpaca", architecture: "" }); } });

  const doSearch = async () => {
    if (!q.trim()) return;
    setSearching(true);
    try { setResults(await searchHub(q)); } finally { setSearching(false); }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, padding: "12px 14px", height: "calc(100vh - 40px)", overflow: "hidden" }}>
      {/* LEFT: registered models */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Registered Models
        </div>
        <div className="lf-panel" style={{ flex: 1, overflow: "auto" }}>
          {isLoading ? (
            <div style={{ padding: 20, fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)", textAlign: "center" }}>loading…</div>
          ) : (
            <table className="lf-table">
              <thead><tr><th>Name</th><th>Repo</th><th>Template</th><th>Status</th><th /></tr></thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.id}>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-hi)" }}>{m.name}</td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{m.hf_repo}</td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{m.template}</td>
                    <td>
                      <span className={`lf-badge ${m.is_downloaded === "true" ? "lf-badge-done" : "lf-badge-pending"}`}>
                        {m.is_downloaded === "true" ? "ready" : "not downloaded"}
                      </span>
                    </td>
                    <td>
                      {m.is_downloaded !== "true" && (
                        <button className="lf-btn lf-btn-ghost" style={{ height: 22, fontSize: 10, padding: "0 8px" }} onClick={() => download(m.id)}>
                          download
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
                {models.length === 0 && (
                  <tr><td colSpan={5} style={{ textAlign: "center", color: "var(--text-dim)", fontFamily: "var(--mono)", fontSize: 11, padding: 24 }}>
                    No models. Add from HF Hub or manually.
                  </td></tr>
                )}
              </tbody>
            </table>
          )}
        </div>

        {/* Manual add */}
        <div style={{ marginTop: 10 }}>
          <div className="lf-section">Add Manually</div>
          <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
            {(["name","hf_repo","template","architecture"] as const).map((k) => (
              <div key={k}>
                <label className="lf-label">{k.replace("_"," ")}</label>
                <input className="lf-input" value={manual[k]} onChange={(e) => setManual((p) => ({ ...p, [k]: e.target.value }))} placeholder={k === "hf_repo" ? "org/model-name" : ""} />
              </div>
            ))}
          </div>
          <button className="lf-btn lf-btn-primary" style={{ width: "100%" }} disabled={!manual.name || !manual.hf_repo} onClick={() => register(manual)}>
            + Register Model
          </button>
        </div>
      </div>

      {/* RIGHT: HF Hub search */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          HuggingFace Hub Search
        </div>
        <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
          <input className="lf-input" style={{ flex: 1 }} value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && doSearch()} placeholder="Search models… (e.g. llama, mistral, qwen)" />
          <button className="lf-btn lf-btn-primary" disabled={searching} onClick={doSearch}>
            {searching ? <span className="lf-spin" /> : "Search"}
          </button>
        </div>
        <div className="lf-panel" style={{ flex: 1, overflow: "auto" }}>
          <table className="lf-table">
            <thead><tr><th>Model ID</th><th>Tag</th><th>Downloads</th><th /></tr></thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.model_id}>
                  <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-hi)" }}>{r.model_id}</td>
                  <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{r.pipeline_tag ?? "—"}</td>
                  <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{r.downloads?.toLocaleString() ?? "—"}</td>
                  <td>
                    <button className="lf-btn lf-btn-ghost" style={{ height: 22, fontSize: 10, padding: "0 8px" }}
                      onClick={() => register({ name: r.model_id.split("/").pop()!, hf_repo: r.model_id, template: "alpaca" })}>
                      + Add
                    </button>
                  </td>
                </tr>
              ))}
              {results.length === 0 && (
                <tr><td colSpan={4} style={{ textAlign: "center", color: "var(--text-dim)", fontFamily: "var(--mono)", fontSize: 11, padding: 24 }}>
                  Search HuggingFace Hub for models to import.
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
