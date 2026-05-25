"use client";
import { useRef, useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getModels, getASRModels, transcribeAudio } from "@/lib/api";

const WHISPER_PRESETS = [
  "openai/whisper-tiny",
  "openai/whisper-base",
  "openai/whisper-small",
  "openai/whisper-medium",
  "openai/whisper-large-v2",
  "openai/whisper-large-v3",
];
const LANG_PRESETS = [
  { label: "Auto-detect", value: "" },
  { label: "Malay",       value: "malay" },
  { label: "English",     value: "english" },
  { label: "Chinese",     value: "chinese" },
  { label: "Tamil",       value: "tamil" },
];

// ── helpers ──────────────────────────────────────────────────────────────────

function Section({ title }: { title: string }) {
  return (
    <div style={{ fontFamily: "var(--mono)", fontSize: 10, fontWeight: 700, textTransform: "uppercase",
      letterSpacing: "0.08em", color: "var(--accent)", background: "var(--accent-dim)",
      padding: "2px 7px", borderRadius: 2, display: "inline-block", marginBottom: 10, marginTop: 4 }}>
      {title}
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 3 }}>{children}</div>;
}

// ── ASR Panel ─────────────────────────────────────────────────────────────────

function ASRPanel() {
  const { data: registeredModels = [] } = useQuery({ queryKey: ["models"], queryFn: getModels });
  const { data: asrModelList = [] }     = useQuery({ queryKey: ["asr-models"], queryFn: getASRModels });

  const [modelPath, setModelPath] = useState("openai/whisper-base");
  const [language, setLanguage]   = useState("");
  const [task, setTask]           = useState<"transcribe" | "translate">("transcribe");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl]   = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [running, setRunning]     = useState(false);
  const [error, setError]         = useState("");
  const [elapsed, setElapsed]     = useState<number | null>(null);

  const fileRef    = useRef<HTMLInputElement>(null);
  const mediaRef   = useRef<MediaRecorder | null>(null);
  const chunksRef  = useRef<Blob[]>([]);

  // Clean up audio URL on unmount / change
  useEffect(() => { return () => { if (audioUrl) URL.revokeObjectURL(audioUrl); }; }, [audioUrl]);

  const handleFile = (f: File) => {
    setAudioFile(f);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(URL.createObjectURL(f));
    setTranscript("");
    setError("");
  };

  const startRecord = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone access not available in this browser.");
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunksRef.current = [];
    const mr = new MediaRecorder(stream);
    mediaRef.current = mr;
    mr.ondataavailable = (e) => chunksRef.current.push(e.data);
    mr.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      const file = new File([blob], "recording.webm", { type: "audio/webm" });
      handleFile(file);
    };
    mr.start();
    setRecording(true);
    setTranscript("");
  };

  const stopRecord = () => {
    mediaRef.current?.stop();
    setRecording(false);
  };

  const handleTranscribe = async () => {
    if (!audioFile) return;
    setRunning(true);
    setError("");
    setTranscript("");
    const t0 = Date.now();
    try {
      const fd = new FormData();
      fd.append("audio", audioFile);
      fd.append("model_path", modelPath);
      if (language) fd.append("language", language);
      fd.append("task", task);
      const res = await transcribeAudio(fd);
      setTranscript(res.transcript);
      setElapsed(Math.round((Date.now() - t0) / 100) / 10);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Transcription failed");
    } finally {
      setRunning(false);
    }
  };

  const asrModelOptions = [
    ...WHISPER_PRESETS,
    ...asrModelList.map((m) => m.id).filter((id) => !WHISPER_PRESETS.includes(id)),
    ...registeredModels
      .filter((m) => m.hf_repo.toLowerCase().includes("whisper") || m.architecture?.toLowerCase().includes("whisper"))
      .map((m) => m.local_path || m.hf_repo),
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div style={{ padding: "10px 14px", overflowY: "auto", flex: 1 }}>
        <Section title="ASR · Speech to Text" />

        <Label>model</Label>
        <select className="lf-input lf-select" value={modelPath} onChange={(e) => setModelPath(e.target.value)}
          style={{ marginBottom: 8 }}>
          <optgroup label="Whisper Presets">
            {WHISPER_PRESETS.map((m) => <option key={m} value={m}>{m}</option>)}
          </optgroup>
          {registeredModels.filter((m) => m.is_downloaded === "true" && m.local_path).length > 0 && (
            <optgroup label="Downloaded Models">
              {registeredModels
                .filter((m) => m.is_downloaded === "true" && m.local_path)
                .map((m) => <option key={m.id} value={m.local_path!}>{m.name}{m.version ? ` (${m.version})` : ""}</option>)}
            </optgroup>
          )}
        </select>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 8 }}>
          <div>
            <Label>language</Label>
            <select className="lf-input lf-select" value={language} onChange={(e) => setLanguage(e.target.value)}>
              {LANG_PRESETS.map((l) => <option key={l.value} value={l.value}>{l.label}</option>)}
            </select>
          </div>
          <div>
            <Label>task</Label>
            <select className="lf-input lf-select" value={task} onChange={(e) => setTask(e.target.value as "transcribe" | "translate")}>
              <option value="transcribe">Transcribe</option>
              <option value="translate">Translate → EN</option>
            </select>
          </div>
        </div>

        <Label>audio input</Label>
        <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
          <button className="lf-btn lf-btn-ghost" style={{ flex: 1 }}
            onClick={() => fileRef.current?.click()}>
            ↑ Upload File
          </button>
          <input ref={fileRef} type="file" accept="audio/*" style={{ display: "none" }}
            onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); e.target.value = ""; }} />
          <button className={`lf-btn ${recording ? "lf-btn-danger" : "lf-btn-ghost"}`} style={{ flex: 1 }}
            onClick={recording ? stopRecord : startRecord}>
            {recording ? "⏹ Stop Recording" : "⏺ Record"}
          </button>
        </div>

        {audioUrl && (
          <div style={{ marginBottom: 10 }}>
            <audio controls src={audioUrl} style={{ width: "100%", height: 36 }} />
            {audioFile && (
              <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginTop: 3 }}>
                {audioFile.name} · {(audioFile.size / 1024).toFixed(0)} KB
              </div>
            )}
          </div>
        )}

        <button className="lf-btn lf-btn-primary" style={{ width: "100%", marginBottom: 10 }}
          disabled={!audioFile || running} onClick={handleTranscribe}>
          {running ? <><span className="lf-spin" /> Transcribing…</> : "▶ Transcribe"}
        </button>
        {!audioFile && (
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textAlign: "center", marginBottom: 8 }}>
            Upload or record audio first
          </div>
        )}

        {error && (
          <div style={{ padding: "6px 8px", background: "var(--red-dim)", border: "1px solid var(--red)",
            borderRadius: 3, fontFamily: "var(--mono)", fontSize: 11, color: "var(--red)", marginBottom: 8 }}>
            {error}
          </div>
        )}

        {transcript && (
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Transcript</span>
              {elapsed !== null && (
                <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--green)" }}>{elapsed}s</span>
              )}
              <button className="lf-btn lf-btn-ghost" style={{ height: 20, fontSize: 10, padding: "0 7px", marginLeft: "auto" }}
                onClick={() => navigator.clipboard?.writeText(transcript)}>
                copy
              </button>
            </div>
            <textarea readOnly className="lf-textarea" value={transcript}
              style={{ minHeight: 120, resize: "vertical", color: "var(--text-hi)" }} />
          </div>
        )}
      </div>
    </div>
  );
}

// ── TTS Panel ─────────────────────────────────────────────────────────────────

function TTSPanel() {
  const [text, setText]           = useState("");
  const [voices, setVoices]       = useState<SpeechSynthesisVoice[]>([]);
  const [voiceIdx, setVoiceIdx]   = useState(0);
  const [rate, setRate]           = useState(1.0);
  const [pitch, setPitch]         = useState(1.0);
  const [speaking, setSpeaking]   = useState(false);
  const [supported, setSupported] = useState(true);

  useEffect(() => {
    if (!window.speechSynthesis) { setSupported(false); return; }
    const load = () => setVoices(window.speechSynthesis.getVoices());
    load();
    window.speechSynthesis.onvoiceschanged = load;
    return () => { window.speechSynthesis.cancel(); };
  }, []);

  const speak = () => {
    if (!text.trim() || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    if (voices[voiceIdx]) utt.voice = voices[voiceIdx];
    utt.rate  = rate;
    utt.pitch = pitch;
    utt.onstart = () => setSpeaking(true);
    utt.onend   = () => setSpeaking(false);
    utt.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utt);
  };

  const stop = () => {
    window.speechSynthesis?.cancel();
    setSpeaking(false);
  };

  if (!supported) {
    return (
      <div style={{ padding: 24, fontFamily: "var(--mono)", fontSize: 12, color: "var(--text-dim)", textAlign: "center" }}>
        Speech Synthesis is not supported in this browser.
      </div>
    );
  }

  // Group voices by language
  const voicesByLang = voices.reduce<Record<string, { idx: number; v: SpeechSynthesisVoice }[]>>((acc, v, i) => {
    const lang = v.lang || "unknown";
    if (!acc[lang]) acc[lang] = [];
    acc[lang].push({ idx: i, v });
    return acc;
  }, {});

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div style={{ padding: "10px 14px", overflowY: "auto", flex: 1 }}>
        <Section title="TTS · Text to Speech" />

        <Label>text to speak</Label>
        <textarea className="lf-textarea" value={text} onChange={(e) => setText(e.target.value)}
          placeholder="Type or paste text here to synthesize speech…"
          style={{ minHeight: 140, resize: "vertical", marginBottom: 10 }} />

        <Label>voice</Label>
        <select className="lf-input lf-select" value={voiceIdx} onChange={(e) => setVoiceIdx(+e.target.value)}
          style={{ marginBottom: 8 }}>
          {Object.entries(voicesByLang).sort(([a], [b]) => a.localeCompare(b)).map(([lang, items]) => (
            <optgroup key={lang} label={lang}>
              {items.map(({ idx, v }) => (
                <option key={idx} value={idx}>{v.name}{v.localService ? "" : " ☁"}</option>
              ))}
            </optgroup>
          ))}
          {voices.length === 0 && <option value={0}>Loading voices…</option>}
        </select>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
          <div>
            <Label>speed — {rate.toFixed(1)}×</Label>
            <input type="range" min={0.5} max={2} step={0.1} value={rate}
              onChange={(e) => setRate(+e.target.value)}
              style={{ width: "100%", accentColor: "var(--accent)" }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: 9, color: "var(--text-dim)" }}>
              <span>0.5×</span><span>1×</span><span>2×</span>
            </div>
          </div>
          <div>
            <Label>pitch — {pitch.toFixed(1)}</Label>
            <input type="range" min={0} max={2} step={0.1} value={pitch}
              onChange={(e) => setPitch(+e.target.value)}
              style={{ width: "100%", accentColor: "var(--accent)" }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: 9, color: "var(--text-dim)" }}>
              <span>low</span><span>normal</span><span>high</span>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <button className="lf-btn lf-btn-primary" style={{ flex: 1 }}
            disabled={!text.trim() || speaking} onClick={speak}>
            {speaking ? <><span className="lf-spin" /> Speaking…</> : "▶ Speak"}
          </button>
          {speaking && (
            <button className="lf-btn lf-btn-danger" onClick={stop}>■ Stop</button>
          )}
        </div>

        <div style={{ marginTop: 16, padding: "8px 10px", background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 3 }}>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 4 }}>
            Using browser built-in speech synthesis. Voices marked ☁ require an internet connection.
          </div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>
            {voices.length} voice{voices.length !== 1 ? "s" : ""} available in your browser.
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function SandboxPage() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", height: "calc(100vh - 40px)", overflow: "hidden" }}>
      {/* ASR */}
      <div style={{ borderRight: "1px solid var(--border)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <div style={{ borderBottom: "1px solid var(--border)", padding: "0 14px", height: 32, display: "flex", alignItems: "center", background: "var(--bg-panel)", flexShrink: 0 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-hi)" }}>ASR Testing</span>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginLeft: 8 }}>Whisper speech-to-text</span>
        </div>
        <div style={{ flex: 1, overflow: "hidden" }}>
          <ASRPanel />
        </div>
      </div>

      {/* TTS */}
      <div style={{ overflow: "hidden", display: "flex", flexDirection: "column" }}>
        <div style={{ borderBottom: "1px solid var(--border)", padding: "0 14px", height: 32, display: "flex", alignItems: "center", background: "var(--bg-panel)", flexShrink: 0 }}>
          <span style={{ fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600, color: "var(--text-hi)" }}>TTS Testing</span>
          <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginLeft: 8 }}>Browser text-to-speech</span>
        </div>
        <div style={{ flex: 1, overflow: "hidden" }}>
          <TTSPanel />
        </div>
      </div>
    </div>
  );
}
