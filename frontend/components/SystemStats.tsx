"use client";
import { useQuery } from "@tanstack/react-query";
import { getSystemStats } from "@/lib/api";

function Bar({ label, value, max }: { label: string; value: number; max: number }) {
  const pct = Math.min(100, Math.round((value / max) * 100));
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", marginBottom: 3 }}>
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="lf-progress-track">
        <div className="lf-progress-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function SystemStats() {
  const { data } = useQuery({ queryKey: ["system"], queryFn: getSystemStats, refetchInterval: 5000 });
  if (!data) return null;
  return (
    <div className="lf-panel" style={{ padding: "10px 12px" }}>
      <div className="lf-section" style={{ marginTop: 0 }}>System</div>
      <Bar label="CPU" value={data.cpu_percent} max={100} />
      <Bar label={`RAM ${data.ram_used_mb}/${data.ram_total_mb} MB`} value={data.ram_used_mb} max={data.ram_total_mb} />
      <Bar label={`Disk ${data.disk_used_gb}/${data.disk_total_gb} GB`} value={data.disk_used_gb} max={data.disk_total_gb} />
      {data.gpu.map((g) => (
        <Bar key={g.index} label={`GPU${g.index} ${g.name} ${g.used_mb}/${g.total_mb} MB`} value={g.used_mb} max={g.total_mb} />
      ))}
      {!data.cuda_available && (
        <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--amber)", marginTop: 4 }}>no CUDA GPU</div>
      )}
    </div>
  );
}
