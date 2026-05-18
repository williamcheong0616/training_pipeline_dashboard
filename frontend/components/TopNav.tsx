"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { getJobs } from "@/lib/api";

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

export default function TopNav() {
  const path = usePathname();
  const { data: jobs = [] } = useQuery({ queryKey: ["jobs"], queryFn: getJobs, refetchInterval: 5000 });
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

      {/* Right side: system status */}
      <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", paddingRight: 14, gap: 14 }}>
        <SystemDot />
      </div>
    </header>
  );
}

function SystemDot() {
  const { data: jobs = [] } = useQuery({ queryKey: ["jobs"], queryFn: getJobs, refetchInterval: 3000 });
  const running = jobs.some((j) => j.status === "running");
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
