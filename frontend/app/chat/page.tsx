"use client";
import { useRef, useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getModels, loadChatModel, getChatStatus, unloadChatModel } from "@/lib/api";

const QUANT_OPTIONS = ["none", "4bit", "8bit"] as const;

type Message = { role: "user" | "assistant" | "system"; content: string };

function Section({ title }: { title: string }) {
  return <div className="lf-section" style={{ marginTop: 12 }}>{title}</div>;
}
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div><label className="lf-label">{label}</label>{children}</div>;
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

export default function ChatPage() {
  const qc = useQueryClient();
  const { data: models = [] } = useQuery({ queryKey: ["models"], queryFn: getModels });
  const { data: chatStatus, refetch: refetchStatus } = useQuery({
    queryKey: ["chat-status"],
    queryFn: getChatStatus,
    refetchInterval: 2000,
  });

  const [loadForm, setLoadForm] = useState({ model_path: "", adapter_path: "", quantization: "none" });
  const [genParams, setGenParams] = useState({ max_new_tokens: 512, temperature: 0.7, top_p: 0.9, top_k: 50, repetition_penalty: 1.1 });
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant.");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages]);

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

  const isReady = chatStatus?.status === "ready";

  const sendMessage = async () => {
    if (!input.trim() || !isReady || generating) return;

    const userMsg: Message = { role: "user", content: input.trim() };
    const allMsgs: Message[] = [
      ...(systemPrompt ? [{ role: "system" as const, content: systemPrompt }] : []),
      ...messages,
      userMsg,
    ];
    setMessages((p) => [...p, userMsg, { role: "assistant", content: "" }]);
    setInput("");
    setGenerating(true);

    const body = JSON.stringify({
      messages: allMsgs,
      ...genParams,
    });

    const es = new EventSource(`/api/chat/generate`);
    // Use fetch+SSE since we need POST with body
    es.close();

    // Use fetch with SSE manually
    const resp = await fetch("/api/chat/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });

    if (!resp.ok || !resp.body) {
      setGenerating(false);
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const read = async () => {
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
              if (token === "__done__") { setGenerating(false); return; }
              setMessages((p) => {
                const updated = [...p];
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  content: updated[updated.length - 1].content + token,
                };
                return updated;
              });
            } catch { /* ignore parse errors */ }
          }
        }
      }
      setGenerating(false);
    };
    read();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", height: "calc(100vh - 40px)", overflow: "hidden" }}>

      {/* LEFT — config */}
      <div style={{ borderRight: "1px solid var(--border)", overflowY: "auto", padding: "10px 12px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid var(--border)" }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--accent)", background: "var(--accent-dim)", padding: "2px 7px", borderRadius: 2 }}>Chat</span>
          <StatusDot status={chatStatus?.status ?? "unloaded"} />
        </div>

        <Section title="Model" />
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
          <button className="lf-btn lf-btn-danger" disabled={!isReady} onClick={() => doUnload()} style={{ padding: "0 12px" }}>
            Unload
          </button>
        </div>

        {chatStatus?.model_path && (
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 8, padding: "4px 8px", background: "var(--bg)", borderRadius: 3, border: "1px solid var(--border)", wordBreak: "break-all" }}>
            loaded: {chatStatus.model_path}
          </div>
        )}

        <Section title="Generation Params" />
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

        <Section title="System Prompt" />
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
            {messages.filter((m) => m.role !== "system").length} messages
          </span>
          <button className="lf-btn lf-btn-ghost" style={{ height: 22, fontSize: 10, padding: "0 8px" }}
            onClick={() => setMessages([])}>Clear</button>
        </div>

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
                maxWidth: "72%",
                padding: "8px 12px",
                borderRadius: 4,
                fontFamily: "var(--mono)",
                fontSize: 12,
                lineHeight: 1.7,
                whiteSpace: "pre-wrap",
                background: msg.role === "user" ? "var(--accent-dim)" : "var(--bg-panel)",
                border: `1px solid ${msg.role === "user" ? "var(--accent)" : "var(--border)"}`,
                color: msg.role === "user" ? "var(--accent)" : "var(--text)",
              }}>
                <div style={{ fontSize: 9, color: "var(--text-dim)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  {msg.role}
                </div>
                {msg.content}
                {msg.role === "assistant" && generating && i === messages.length - 1 && (
                  <span className="lf-cursor" style={{ marginLeft: 2 }}>█</span>
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
            style={{
              flex: 1, resize: "none", fontFamily: "var(--mono)", fontSize: 12,
              background: "var(--bg-input)", border: "1px solid var(--border)", borderRadius: 3,
              color: "var(--text)", padding: "6px 8px", outline: "none",
            }}
          />
          <button className="lf-btn lf-btn-primary" style={{ alignSelf: "flex-end", height: 36, padding: "0 16px" }}
            disabled={!isReady || !input.trim() || generating}
            onClick={sendMessage}>
            {generating ? <span className="lf-spin" /> : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
