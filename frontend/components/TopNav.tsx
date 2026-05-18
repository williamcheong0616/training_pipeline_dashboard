"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { getJobs, getSystemStats } from "@/lib/api";
import type { SystemStats } from "@/types";

const TABS = [
  { href: "/",          label: "LLM" },
  { href: "/asr",       label: "ASR" },
  { href: "/evaluate",  label: "Evaluate" },
  { href: "/chat",      label: "Chat" },
  { href: "/export",    label: "Export" },
  { href: "/jobs",      label: "Jobs" },
  { href: "/models",    label: "Models" },
  { href: "/datasets",  label: "Datasets" },
];

function utilColor(pct: number): string {
  if (pct >= 85) return "var(--red)";
  if (pct >= 60) return "#f59e0b";
  return "var(--green)";
}

function MiniBar({ pct }: { pct: number }) {
  return (
    <span style={{
      display: "inline-block", width: 32, height: 4, borderRadius: 2,
      background: "var(--bg-input)", verticalAlign: "middle", position: "relative", overflow: "hidden",
    }}>
      <span style={{
        position: "absolute", left: 0, top: 0, height: "100%",
        width: `${Math.min(pct, 100)}%`,
        background: utilColor(pct),
        transition: "width 0.6s ease",
        borderRadius: 2,
      }} />
    </span>
  );
}

function MetricsSkeleton() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, fontFamily: "var(--mono)", fontSize: 10 }}>
      {["CPU", "RAM", "GPU"].map((label) => (
        <span key={label} style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--text-dim)", opacity: 0.4 }}>
          <span>{label}</span>
          <span style={{ display: "inline-block", width: 32, height: 4, borderRadius: 2, background: "var(--bg-input)" }} />
          <span style={{ minWidth: 28 }}>--</span>
        </span>
      ))}
      <span style={{ width: 1, height: 14, background: "var(--border)", display: "inline-block" }} />
    </div>
  );
}

function SystemMetrics({ stats }: { stats: SystemStats }) {
  const cpuPct   = Math.round(stats.cpu_percent);
  const ramPct   = Math.round((stats.ram_used_mb / stats.ram_total_mb) * 100);
  const ramUsed  = (stats.ram_used_mb / 1024).toFixed(1);
  const ramTotal = (stats.ram_total_mb / 1024).toFixed(0);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, fontFamily: "var(--mono)", fontSize: 10 }}>
      {/* CPU */}
      <span style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--text-dim)" }}>
        <span>CPU</span>
        <MiniBar pct={cpuPct} />
        <span style={{ color: utilColor(cpuPct), minWidth: 28 }}>{cpuPct}%</span>
      </span>

      {/* RAM */}
      <span style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--text-dim)" }}>
        <span>RAM</span>
        <MiniBar pct={ramPct} />
        <span style={{ color: utilColor(ramPct) }}>{ramUsed}/{ramTotal}G</span>
      </span>

      {/* GPU(s) */}
      {stats.gpu.map((g) => {
        const gpuPct = Math.round((g.used_mb / g.total_mb) * 100);
        const usedG  = (g.used_mb  / 1024).toFixed(1);
        const totalG = (g.total_mb / 1024).toFixed(0);
        return (
          <span key={g.index} style={{ display: "flex", alignItems: "center", gap: 4, color: "var(--text-dim)" }}
            title={g.name}>
            <span>GPU{stats.gpu.length > 1 ? g.index : ""}</span>
            <MiniBar pct={gpuPct} />
            <span style={{ color: utilColor(gpuPct) }}>{usedG}/{totalG}G</span>
          </span>
        );
      })}

      {!stats.cuda_available && (
        <span style={{ color: "var(--text-dim)" }}>CPU only</span>
      )}

      <span style={{ width: 1, height: 14, background: "var(--border)", display: "inline-block" }} />
    </div>
  );
}

export default function TopNav() {
  const path = usePathname();
  const { data: jobs = [] } = useQuery({ queryKey: ["jobs"], queryFn: getJobs, refetchInterval: 5000 });
  const { data: stats }     = useQuery({ queryKey: ["system"], queryFn: getSystemStats, refetchInterval: 2000 });

  const running = jobs.filter((j) => j.status === "running").length;

  const active = (href: string) =>
    href === "/" ? path === "/" : path.startsWith(href);

  return (
    <header style={{
      background: "var(--bg-panel)",
      borderBottom: "1px solid var(--border)",
      display: "flex",
      alignItems: "stretch",
      height: 40,
      paddingLeft: 12,
      gap: 0,
      flexShrink: 0,
    }}>
      {/* Brand */}
      <div style={{
        display: "flex", alignItems: "center", gap: 8,
        paddingRight: 20, borderRight: "1px solid var(--border)", marginRight: 4,
      }}>
        <span style={{ fontSize: 14, color: "var(--accent)" }}>⚡</span>
        <span style={{
          fontFamily: "var(--mono)", fontWeight: 600, fontSize: 12,
          color: "var(--text-hi)", letterSpacing: "0.04em",
        }}>Forge</span>
      </div>

      {/* Tabs */}
      {TABS.map(({ href, label }) => (
        <Link
          key={href}
          href={href}
          className={`lf-tab ${active(href) ? "lf-tab-active" : ""}`}
        >
          {label}
          {label === "Jobs" && running > 0 && (
            <span style={{
              marginLeft: 6, background: "var(--accent)", color: "#000",
              borderRadius: 8, padding: "0 5px", fontSize: 10, fontWeight: 700,
            }}>{running}</span>
          )}
        </Link>
      ))}

      {/* Right side: system metrics + training status */}
      <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", paddingRight: 14, gap: 10 }}>
        {stats ? <SystemMetrics stats={stats} /> : <MetricsSkeleton />}
        <TrainingDot running={running > 0} />
      </div>
    </header>
  );
}

function TrainingDot({ running }: { running: boolean }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 5, fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)" }}>
      <span style={{
        width: 6, height: 6, borderRadius: "50%",
        background: running ? "var(--green)" : "var(--text-dim)",
        boxShadow: running ? "0 0 6px var(--green)" : "none",
      }} />
      {running ? "TRAINING" : "IDLE"}
    </span>
  );
}
