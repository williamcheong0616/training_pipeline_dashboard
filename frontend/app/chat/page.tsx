"use client";
import { useRef, useState, useEffect, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getModels, loadChatModel, getChatStatus, unloadChatModel,
  getConversations, getConversation, createConversation, updateConversation,
  deleteConversation, addConversationMessage,
  getPromptProfiles, createPromptProfile, deletePromptProfile,
} from "@/lib/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";

const QUANT_OPTIONS = ["none", "4bit", "8bit"] as const;

type Message = { role: "user" | "assistant" | "system"; content: string };
type GenParams = { max_new_tokens: number; temperature: number; top_p: number; top_k: number; repetition_penalty: number };
const DEFAULT_GEN_PARAMS: GenParams = { max_new_tokens: 512, temperature: 0.7, top_p: 0.9, top_k: 50, repetition_penalty: 1.1 };

function Section({ title, right }: { title: string; right?: React.ReactNode }) {
  return (
    <div className="lf-section" style={{ marginTop: 12, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
      <span>{title}</span>
      {right}
    </div>
  );
}
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="lf-label">{label}</label>{children}</div>;
}

function MarkdownMessage({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{
        code({ className, children, ...props }) {
          const isBlock = className?.includes("language-");
          return isBlock ? (
            <code className={className} {...props}>{children}</code>
          ) : (
            <code style={{ fontFamily: "var(--mono)", fontSize: 11, background: "var(--bg-input)", border: "1px solid var(--border)", borderRadius: 2, padding: "1px 5px", color: "var(--green)" }} {...props}>{children}</code>
          );
        },
        pre({ children }) { return <pre style={{ margin: "8px 0", borderRadius: 4, overflow: "auto", border: "1px solid var(--border)", fontSize: 11 }}>{children}</pre>; },
        table({ children }) { return <div style={{ overflowX: "auto", margin: "8px 0" }}><table style={{ borderCollapse: "collapse", fontFamily: "var(--mono)", fontSize: 11, width: "100%" }}>{children}</table></div>; },
        th({ children }) { return <th style={{ padding: "4px 10px", borderBottom: "1px solid var(--border-hi)", color: "var(--text-dim)", textAlign: "left", fontWeight: 600 }}>{children}</th>; },
        td({ children }) { return <td style={{ padding: "4px 10px", borderBottom: "1px solid var(--border)", color: "var(--text)" }}>{children}</td>; },
        h1({ children }) { return <div style={{ fontFamily: "var(--mono)", fontSize: 15, fontWeight: 700, color: "var(--text-hi)", margin: "10px 0 4px" }}>{children}</div>; },
        h2({ children }) { return <div style={{ fontFamily: "var(--mono)", fontSize: 13, fontWeight: 600, color: "var(--text-hi)", margin: "8px 0 4px" }}>{children}</div>; },
        h3({ children }) { return <div style={{ fontFamily: "var(--mono)", fontSize: 12, fontWeight: 600, color: "var(--text)", margin: "6px 0 3px" }}>{children}</div>; },
        ul({ children }) { return <ul style={{ paddingLeft: 18, margin: "4px 0", fontSize: 12 }}>{children}</ul>; },
        ol({ children }) { return <ol style={{ paddingLeft: 18, margin: "4px 0", fontSize: 12 }}>{children}</ol>; },
        li({ children }) { return <li style={{ marginBottom: 2, color: "var(--text)" }}>{children}</li>; },
        blockquote({ children }) { return <blockquote style={{ borderLeft: "3px solid var(--accent)", paddingLeft: 10, margin: "6px 0", color: "var(--text-dim)", fontStyle: "italic" }}>{children}</blockquote>; },
        p({ children }) { return <p style={{ margin: "4px 0", lineHeight: 1.7 }}>{children}</p>; },
        hr() { return <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: "8px 0" }} />; },
        strong({ children }) { return <strong style={{ color: "var(--text-hi)", fontWeight: 600 }}>{children}</strong>; },
        em({ children }) { return <em style={{ color: "var(--text-dim)" }}>{children}</em>; },
        a({ href, children }) { return <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent)", textDecoration: "underline" }}>{children}</a>; },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function StatusDot({ status }: { status: string }) {
  const color = status === "ready" ? "var(--green)" : status === "loading" ? "var(--amber)" : status === "error" ? "var(--red)" : "var(--text-dim)";
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 5, fontFamily: "var(--mono)", fontSize: 10 }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: color, boxShadow: status === "ready" ? `0 0 6px ${color}` : "none" }} />
      <span style={{ color }}>{status}</span>
    </span>
  );
}

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export default function ChatPage() {
  const qc = useQueryClient();
  const { data: models = [] } = useQuery({ queryKey: ["models"], queryFn: getModels });
  const { data: chatStatus, refetch: refetchStatus } = useQuery({
    queryKey: ["chat-status"], queryFn: getChatStatus, refetchInterval: 2000,
  });
  const { data: conversations = [], refetch: refetchConvs } = useQuery({
    queryKey: ["conversations"], queryFn: getConversations,
  });
  const { data: profiles = [], refetch: refetchProfiles } = useQuery({
    queryKey: ["prompt-profiles"], queryFn: getPromptProfiles,
  });

  const [activeConvId, setActiveConvId] = useState<number | null>(null);
  const [loadForm, setLoadForm] = useState({ model_path: "", adapter_path: "", quantization: "none" });
  const [genParams, setGenParams] = useState<GenParams>(DEFAULT_GEN_PARAMS);
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant.");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const [settingsSaved, setSettingsSaved] = useState(true);
  const [savingProfile, setSavingProfile] = useState(false);
  const [profileName, setProfileName] = useState("");
  const chatRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => { return () => { abortRef.current?.abort(); }; }, []);
  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages]);

  // Debounce-save system prompt + gen params to DB when conversation is active
  useEffect(() => {
    if (!activeConvId) return;
    setSettingsSaved(false);
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(async () => {
      await updateConversation(activeConvId, {
        system_prompt: systemPrompt,
        gen_params: genParams as unknown as Record<string, unknown>,
      });
      setSettingsSaved(true);
    }, 1000);
    return () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [systemPrompt, genParams, activeConvId]);

  const loadConv = useCallback(async (convId: number) => {
    const conv = await getConversation(convId);
    setActiveConvId(convId);
    setMessages(conv.messages.filter((m) => m.role !== "system") as Message[]);
    if (conv.system_prompt != null) setSystemPrompt(conv.system_prompt);
    if (conv.gen_params) setGenParams(conv.gen_params as unknown as GenParams);
    if (conv.model_path) setLoadForm((p) => ({ ...p, model_path: conv.model_path ?? "", adapter_path: conv.adapter_path ?? "" }));
    setSettingsSaved(true);
  }, []);

  // Auto-load most recent conversation on mount
  useEffect(() => {
    if (conversations.length > 0 && activeConvId === null) {
      loadConv(conversations[0].id);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversations.length]);

  const startNewChat = () => {
    setActiveConvId(null);
    setMessages([]);
    setInput("");
    setSettingsSaved(true);
  };

  const { mutate: doLoad, isPending: isLoading } = useMutation({
    mutationFn: () => loadChatModel({
      model_path: loadForm.model_path,
      adapter_path: loadForm.adapter_path || undefined,
      quantization: loadForm.quantization === "none" ? undefined : loadForm.quantization,
    }),
    onSuccess: () => { setTimeout(() => refetchStatus(), 500); },
  });

  const { mutate: doUnload } = useMutation({
    mutationFn: unloadChatModel,
    onSuccess: () => refetchStatus(),
  });

  const { mutate: doDeleteConv } = useMutation({
    mutationFn: (id: number) => deleteConversation(id),
    onSuccess: (_data, deletedId) => {
      refetchConvs();
      if (activeConvId === deletedId) startNewChat();
    },
  });

  const { mutate: doDeleteProfile } = useMutation({
    mutationFn: (id: number) => deletePromptProfile(id),
    onSuccess: () => refetchProfiles(),
  });

  const handleSaveProfile = async () => {
    if (!profileName.trim()) return;
    await createPromptProfile({
      name: profileName.trim(),
      system_prompt: systemPrompt,
      gen_params: genParams as unknown as Record<string, unknown>,
    });
    setProfileName("");
    setSavingProfile(false);
    refetchProfiles();
  };

  const applyProfile = (profileId: string) => {
    const p = profiles.find((x) => x.id === Number(profileId));
    if (!p) return;
    if (p.system_prompt != null) setSystemPrompt(p.system_prompt);
    if (p.gen_params) setGenParams(p.gen_params as unknown as GenParams);
  };

  const isReady = chatStatus?.status === "ready";
  const modelStatus = chatStatus?.status ?? "unloaded";

  const sendMessage = async () => {
    if (!input.trim() || !isReady || generating) return;

    const userText = input.trim();
    const userMsg: Message = { role: "user", content: userText };
    setMessages((p) => [...p, userMsg, { role: "assistant", content: "" }]);
    setInput("");
    setGenerating(true);

    let convId = activeConvId;
    if (!convId) {
      const title = userText.slice(0, 60) + (userText.length > 60 ? "…" : "");
      const conv = await createConversation({
        title,
        model_path: loadForm.model_path || undefined,
        adapter_path: loadForm.adapter_path || undefined,
        system_prompt: systemPrompt || undefined,
        gen_params: genParams as unknown as Record<string, unknown>,
      });
      convId = conv.id;
      setActiveConvId(convId);
      refetchConvs();
    }

    await addConversationMessage(convId, "user", userText);

    const allMsgs: Message[] = [
      ...(systemPrompt ? [{ role: "system" as const, content: systemPrompt }] : []),
      ...messages,
      userMsg,
    ];

    const body = JSON.stringify({ messages: allMsgs, ...genParams });
    const controller = new AbortController();
    abortRef.current = controller;

    const resp = await fetch("/api/chat/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      signal: controller.signal,
    });

    if (!resp.ok || !resp.body) {
      const errText = resp.ok ? "No response body" : (await resp.text().catch(() => `HTTP ${resp.status}`));
      const errContent = `[Error: ${errText}]`;
      setMessages((p) => [...p.slice(0, -1), { role: "assistant", content: errContent }]);
      await addConversationMessage(convId, "assistant", errContent);
      setGenerating(false);
      abortRef.current = null;
      refetchConvs();
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let assistantContent = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const { token } = JSON.parse(line.slice(6));
              if (token === "__done__") {
                await addConversationMessage(convId!, "assistant", assistantContent);
                refetchConvs();
                setGenerating(false);
                abortRef.current = null;
                return;
              }
              assistantContent += token;
              setMessages((p) => {
                const updated = [...p];
                updated[updated.length - 1] = { ...updated[updated.length - 1], content: assistantContent };
                return updated;
              });
            } catch { /* ignore malformed SSE frame */ }
          }
        }
      }
    } catch { /* AbortError — save partial */ }

    if (assistantContent) {
      await addConversationMessage(convId!, "assistant", assistantContent);
      refetchConvs();
    }
    setGenerating(false);
    abortRef.current = null;
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const bannerInfo = (() => {
    if (modelStatus === "ready") return null;
    if (modelStatus === "loading") return { text: "Model loading…", color: "var(--amber)", bg: "var(--amber-dim)" };
    if (modelStatus === "error") return { text: chatStatus?.error ?? "Model failed to load — check logs", color: "var(--red)", bg: "var(--red-dim)" };
    return { text: "No model loaded — select a model from the left panel and click Load Model", color: "var(--amber)", bg: "var(--amber-dim)" };
  })();

  return (
    <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", height: "calc(100vh - 40px)", overflow: "hidden" }}>

      {/* LEFT — config */}
      <div style={{ borderRight: "1px solid var(--border)", overflowY: "auto", padding: "10px 12px" }}>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--accent)", background: "var(--accent-dim)", padding: "2px 7px", borderRadius: 2 }}>Chat</span>
          <StatusDot status={modelStatus} />
        </div>

        {/* Sessions */}
        <div style={{ marginBottom: 10 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Sessions</span>
            <button className="lf-btn lf-btn-ghost" style={{ height: 20, fontSize: 10, padding: "0 8px" }} onClick={startNewChat}>+ New</button>
          </div>
          <div style={{ maxHeight: 160, overflowY: "auto", display: "flex", flexDirection: "column", gap: 2 }}>
            {conversations.length === 0 && (
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", padding: "4px 0" }}>No sessions yet</div>
            )}
            {conversations.map((c) => (
              <div key={c.id}
                style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 7px", borderRadius: 3, cursor: "pointer", background: activeConvId === c.id ? "var(--bg-hover)" : "transparent", border: `1px solid ${activeConvId === c.id ? "var(--border-hi)" : "transparent"}` }}
                onClick={() => loadConv(c.id)}
              >
                <div style={{ flex: 1, overflow: "hidden" }}>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: activeConvId === c.id ? "var(--text-hi)" : "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.title}</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text-dim)", marginTop: 1 }}>{c.message_count} msgs · {timeAgo(c.updated_at)}</div>
                </div>
                <button onClick={(e) => { e.stopPropagation(); if (confirm(`Delete "${c.title}"?`)) doDeleteConv(c.id); }}
                  style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-dim)", fontSize: 11, padding: "1px 3px", borderRadius: 2, flexShrink: 0, lineHeight: 1 }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = "var(--red)")}
                  onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-dim)")}
                >✕</button>
              </div>
            ))}
          </div>
        </div>

        {/* Profiles */}
        <div style={{ marginBottom: 10, paddingTop: 8, borderTop: "1px solid var(--border)" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Profiles</span>
            <button className="lf-btn lf-btn-ghost" style={{ height: 20, fontSize: 10, padding: "0 8px" }} onClick={() => setSavingProfile((p) => !p)}>
              {savingProfile ? "Cancel" : "+ Save"}
            </button>
          </div>
          {savingProfile && (
            <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
              <input className="lf-input" value={profileName} onChange={(e) => setProfileName(e.target.value)}
                placeholder="Profile name…" onKeyDown={(e) => e.key === "Enter" && handleSaveProfile()}
                style={{ flex: 1, height: 26, fontSize: 11 }} />
              <button className="lf-btn lf-btn-primary" style={{ height: 26, fontSize: 10, padding: "0 10px" }} onClick={handleSaveProfile} disabled={!profileName.trim()}>Save</button>
            </div>
          )}
          {profiles.length === 0 ? (
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>No profiles saved</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {profiles.map((p) => (
                <div key={p.id} style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 6px", borderRadius: 3, background: "var(--bg-input)", border: "1px solid var(--border)" }}>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.name}</span>
                  <button className="lf-btn lf-btn-ghost" style={{ height: 18, fontSize: 9, padding: "0 6px" }} onClick={() => applyProfile(String(p.id))}>apply</button>
                  <button onClick={() => { if (confirm(`Delete profile "${p.name}"?`)) doDeleteProfile(p.id); }}
                    style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-dim)", fontSize: 11, padding: "1px 2px", lineHeight: 1 }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = "var(--red)")}
                    onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-dim)")}
                  >✕</button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ borderTop: "1px solid var(--border)", marginBottom: 8 }} />

        {/* Model */}
        <div style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-dim)", marginBottom: 6 }}>Model</div>
        <div style={{ marginBottom: 8 }}>
          <label className="lf-label">model</label>
          <select className="lf-input lf-select" value={loadForm.model_path} onChange={(e) => setLoadForm((p) => ({ ...p, model_path: e.target.value }))} style={{ marginBottom: 4 }}>
            <option value="">— select registered model —</option>
            {models.map((m) => <option key={m.id} value={m.local_path || m.hf_repo}>{m.name}</option>)}
          </select>
          <input className="lf-input" value={loadForm.model_path} onChange={(e) => setLoadForm((p) => ({ ...p, model_path: e.target.value }))} placeholder="or path / HF repo ID" />
        </div>
        <div style={{ marginBottom: 8 }}>
          <Field label="adapter path (optional)">
            <input className="lf-input" value={loadForm.adapter_path} onChange={(e) => setLoadForm((p) => ({ ...p, adapter_path: e.target.value }))} placeholder="./outputs/run1/final_adapter" />
          </Field>
        </div>
        <div style={{ marginBottom: 8 }}>
          <Field label="quantization">
            <select className="lf-input lf-select" value={loadForm.quantization} onChange={(e) => setLoadForm((p) => ({ ...p, quantization: e.target.value }))}>
              {QUANT_OPTIONS.map((q) => <option key={q}>{q}</option>)}
            </select>
          </Field>
        </div>
        <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
          <button className="lf-btn lf-btn-primary" style={{ flex: 1 }}
            disabled={isLoading || !loadForm.model_path || chatStatus?.status === "loading"}
            onClick={() => doLoad()}>
            {chatStatus?.status === "loading" ? <><span className="lf-spin" /> Loading…</> : "Load Model"}
          </button>
          <button className="lf-btn lf-btn-danger" disabled={!isReady} style={{ padding: "0 12px" }}
            onClick={() => { if (window.confirm("Unload the model? This clears GPU memory.")) doUnload(); }}>
            Unload
          </button>
        </div>
        {chatStatus?.model_path && (
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 8, padding: "4px 8px", background: "var(--bg)", borderRadius: 3, border: "1px solid var(--border)", wordBreak: "break-all" }}>
            loaded: {chatStatus.model_path}
          </div>
        )}

        {/* Generation Params */}
        <div style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-dim)", marginTop: 10, marginBottom: 6 }}>Generation Params</div>
        <div className="lf-row lf-row-2" style={{ marginBottom: 8 }}>
          <Field label="max new tokens">
            <input className="lf-input" type="number" value={genParams.max_new_tokens} onChange={(e) => setGenParams((p) => ({ ...p, max_new_tokens: +e.target.value }))} />
          </Field>
          <Field label="temperature">
            <input className="lf-input" type="number" step="0.05" min="0" max="2" value={genParams.temperature} onChange={(e) => setGenParams((p) => ({ ...p, temperature: +e.target.value }))} />
          </Field>
        </div>
        <div className="lf-row lf-row-3" style={{ marginBottom: 8 }}>
          <Field label="top_p">
            <input className="lf-input" type="number" step="0.05" min="0" max="1" value={genParams.top_p} onChange={(e) => setGenParams((p) => ({ ...p, top_p: +e.target.value }))} />
          </Field>
          <Field label="top_k">
            <input className="lf-input" type="number" value={genParams.top_k} onChange={(e) => setGenParams((p) => ({ ...p, top_k: +e.target.value }))} />
          </Field>
          <Field label="rep. penalty">
            <input className="lf-input" type="number" step="0.05" min="1" max="2" value={genParams.repetition_penalty} onChange={(e) => setGenParams((p) => ({ ...p, repetition_penalty: +e.target.value }))} />
          </Field>
        </div>

        {/* System Prompt */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 10, marginBottom: 4 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-dim)" }}>System Prompt</span>
          {activeConvId && (
            <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: settingsSaved ? "var(--green)" : "var(--amber)" }}>
              {settingsSaved ? "saved" : "saving…"}
            </span>
          )}
        </div>
        <textarea
          className="lf-input lf-console"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          rows={4}
          style={{ width: "100%", resize: "vertical", fontFamily: "var(--mono)", fontSize: 11 }}
          placeholder="You are a helpful assistant."
        />
      </div>

      {/* RIGHT — chat */}
      <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Toolbar */}
        <div style={{ borderBottom: "1px solid var(--border)", padding: "0 14px", height: 32, display: "flex", alignItems: "center", justifyContent: "space-between", background: "var(--bg-panel)", flexShrink: 0 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>
            {activeConvId ? `#${activeConvId} · ${messages.filter((m) => m.role !== "system").length} messages` : "new session"}
          </span>
          <button className="lf-btn lf-btn-ghost" style={{ height: 22, fontSize: 10, padding: "0 8px" }} onClick={startNewChat}>New Chat</button>
        </div>

        {/* No-model warning banner */}
        {bannerInfo && (
          <div style={{ padding: "7px 14px", background: bannerInfo.bg, borderBottom: `1px solid ${bannerInfo.color}`, flexShrink: 0, display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: bannerInfo.color }}>{bannerInfo.text}</span>
          </div>
        )}

        {/* Messages */}
        <div ref={chatRef} style={{ flex: 1, overflowY: "auto", padding: "12px 14px", display: "flex", flexDirection: "column", gap: 10 }}>
          {messages.length === 0 && (
            <div style={{ textAlign: "center", fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-dim)", marginTop: 40 }}>
              {isReady ? "Model loaded. Type a message below." : "Load a model from the left panel to start chatting."}
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
              <div style={{
                maxWidth: "80%", padding: "8px 12px", borderRadius: 4, fontSize: 12, lineHeight: 1.7,
                background: msg.role === "user" ? "var(--accent-dim)" : "var(--bg-panel)",
                border: `1px solid ${msg.role === "user" ? "var(--accent)" : "var(--border)"}`,
                color: msg.role === "user" ? "var(--accent)" : "var(--text)",
                fontFamily: msg.role === "user" ? "var(--mono)" : "var(--sans)",
                whiteSpace: msg.role === "user" ? "pre-wrap" : undefined,
              }}>
                <div style={{ fontSize: 9, color: "var(--text-dim)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em", fontFamily: "var(--mono)" }}>{msg.role}</div>
                {msg.role === "assistant" ? (
                  <><MarkdownMessage content={msg.content} />{generating && i === messages.length - 1 && <span className="lf-cursor" style={{ marginLeft: 2 }}>█</span>}</>
                ) : (
                  <>{msg.content}{generating && i === messages.length - 1 && <span className="lf-cursor" style={{ marginLeft: 2 }}>█</span>}</>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Input bar */}
        <div style={{ borderTop: "1px solid var(--border)", padding: "10px 14px", display: "flex", gap: 8, background: "var(--bg-panel)", flexShrink: 0 }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isReady ? "Type a message… (Enter to send, Shift+Enter for newline)" : "Load a model first"}
            disabled={!isReady || generating}
            rows={2}
            style={{ flex: 1, resize: "none", fontFamily: "var(--mono)", fontSize: 12, background: "var(--bg-input)", border: "1px solid var(--border)", borderRadius: 3, color: "var(--text)", padding: "6px 8px", outline: "none" }}
          />
          {generating ? (
            <button className="lf-btn lf-btn-danger" style={{ alignSelf: "flex-end", height: 36, padding: "0 16px" }}
              onClick={() => { abortRef.current?.abort(); setGenerating(false); }}>■ Stop</button>
          ) : (
            <button className="lf-btn lf-btn-primary" style={{ alignSelf: "flex-end", height: 36, padding: "0 16px" }}
              disabled={!isReady || !input.trim()} onClick={sendMessage}>Send</button>
          )}
        </div>
      </div>
    </div>
  );
}
