"use client";
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getModels, registerModel, downloadModel, deleteModel, searchHub } from "@/lib/api";
import type { HFSearchResult } from "@/types";

export default function ModelsPage() {
  const qc = useQueryClient();
  const { data: models = [], isLoading } = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
    refetchInterval: (q) => {
      // Poll every 5s if any model is downloading (not yet "true")
      const list = q.state.data ?? [];
      return list.some((m) => m.is_downloaded !== "true" && m.local_path) ? 5000 : false;
    },
  });

  const [q, setQ] = useState("");
  const [results, setResults] = useState<HFSearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [manual, setManual] = useState({ name: "", hf_repo: "", template: "alpaca", architecture: "", version: "" });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [downloadingIds, setDownloadingIds] = useState<Set<number>>(new Set());

  const { mutate: doDownload } = useMutation({
    mutationFn: (id: number) => downloadModel(id),
    onMutate: (id) => setDownloadingIds((s) => new Set(s).add(id)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      setErrors((e) => { const n = { ...e }; delete n.download; return n; });
    },
    onError: (err, id) => {
      setDownloadingIds((s) => { const n = new Set(s); n.delete(id); return n; });
      setErrors((e) => ({ ...e, download: err instanceof Error ? err.message : "Download failed" }));
    },
  });

  const { mutate: register, isPending: registering } = useMutation({
    mutationFn: registerModel,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      setManual({ name: "", hf_repo: "", template: "alpaca", architecture: "" });
      setErrors((e) => { const n = { ...e }; delete n.register; return n; });
    },
    onError: (err) => {
      setErrors((e) => ({ ...e, register: err instanceof Error ? err.message : "Registration failed" }));
    },
  });

  const { mutate: doDelete } = useMutation({
    mutationFn: (id: number) => deleteModel(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }),
    onError: (err) => setErrors((e) => ({ ...e, delete: err instanceof Error ? err.message : "Delete failed" })),
  });

  const doSearch = async () => {
    if (!q.trim()) return;
    setSearching(true);
    setErrors((e) => { const n = { ...e }; delete n.search; return n; });
    try {
      setResults(await searchHub(q));
    } catch (err) {
      setErrors((e) => ({ ...e, search: err instanceof Error ? err.message : "Search failed" }));
    } finally {
      setSearching(false);
    }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, padding: "12px 14px", height: "calc(100vh - 40px)", overflow: "hidden" }}>
      {/* LEFT: registered models */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Registered Models
        </div>

        {(errors.download || errors.delete) && (
          <div style={{ marginBottom: 6, padding: "6px 8px", background: "var(--red-dim)", border: "1px solid var(--red)", borderRadius: 3, fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)" }}>
            {errors.download ?? errors.delete}
          </div>
        )}

        <div className="lf-panel" style={{ flex: 1, overflow: "auto" }}>
          {isLoading ? (
            <div style={{ padding: 20, fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)", textAlign: "center" }}>loading…</div>
          ) : (
            <table className="lf-table">
              <thead><tr><th>Name</th><th>Repo</th><th>Template</th><th>Version</th><th>Status</th><th /></tr></thead>
              <tbody>
                {models.map((m) => {
                  const ready = m.is_downloaded === "true";
                  const downloading = downloadingIds.has(m.id) && !ready;
                  return (
                    <tr key={m.id}>
                      <td style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-hi)" }}>{m.name}</td>
                      <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{m.hf_repo}</td>
                      <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>{m.template}</td>
                      <td style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--accent)" }}>{m.version ?? "—"}</td>
                      <td>
                        <span className={`lf-badge ${ready ? "lf-badge-done" : downloading ? "lf-badge-running" : "lf-badge-pending"}`}>
                          {ready ? "ready" : downloading ? "downloading…" : "not downloaded"}
                        </span>
                      </td>
                      <td style={{ display: "flex", gap: 4, justifyContent: "flex-end" }}>
                        {!ready && (
                          <button className="lf-btn lf-btn-ghost" style={{ height: 22, fontSize: 10, padding: "0 8px" }}
                            disabled={downloading} onClick={() => doDownload(m.id)}>
                            {downloading ? <span className="lf-spin" /> : "download"}
                          </button>
                        )}
                        <button className="lf-btn lf-btn-ghost" style={{ height: 22, fontSize: 10, padding: "0 8px", color: "var(--red)" }}
                          onClick={() => { if (confirm(`Delete "${m.name}" from registry?`)) doDelete(m.id); }}>
                          ✕
                        </button>
                      </td>
                    </tr>
                  );
                })}
                {models.length === 0 && (
                  <tr><td colSpan={6} style={{ textAlign: "center", color: "var(--text-dim)", fontFamily: "var(--mono)", fontSize: 11, padding: 24 }}>
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
          {errors.register && (
            <div style={{ marginBottom: 6, padding: "4px 8px", background: "var(--red-dim)", border: "1px solid var(--red)", borderRadius: 3, fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)" }}>
              {errors.register}
            </div>
          )}
          <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
            {(["name","hf_repo","template","architecture","version"] as const).map((k) => (
              <div key={k}>
                <label className="lf-label">{k.replace("_"," ")}</label>
                <input className="lf-input" value={manual[k]} onChange={(e) => setManual((p) => ({ ...p, [k]: e.target.value }))}
                  placeholder={k === "hf_repo" ? "org/model-name" : k === "version" ? "e.g. v1.0, 2024-06 (optional)" : ""} />
              </div>
            ))}
          </div>
          <button className="lf-btn lf-btn-primary" style={{ width: "100%" }} disabled={!manual.name || !manual.hf_repo || registering} onClick={() => register(manual)}>
            {registering ? <><span className="lf-spin" /> Registering…</> : "+ Register Model"}
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
        {errors.search && (
          <div style={{ marginBottom: 6, padding: "4px 8px", background: "var(--red-dim)", border: "1px solid var(--red)", borderRadius: 3, fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)" }}>
            {errors.search}
          </div>
        )}
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
