"use client";
import { useQuery } from "@tanstack/react-query";
import { getSystemStats } from "@/lib/api";

function ProgressBar({ value, max, label }: { value: number; max: number; label: string }) {
  const pct = Math.round((value / max) * 100);
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-base-content/70">{label}</span>
        <span className="font-mono">{pct}%</span>
      </div>
      <progress className="progress progress-primary w-full h-2" value={pct} max="100" />
    </div>
  );
}

export default function SystemStats() {
  const { data, isLoading } = useQuery({ queryKey: ["system"], queryFn: getSystemStats, refetchInterval: 5000 });

  if (isLoading) return <div className="skeleton h-28 w-full rounded-box" />;
  if (!data) return null;

  return (
    <div className="card bg-base-200 border border-base-300">
      <div className="card-body p-4 gap-3">
        <h3 className="card-title text-sm">System Resources</h3>
        <ProgressBar value={data.cpu_percent} max={100} label="CPU" />
        <ProgressBar value={data.ram_used_mb} max={data.ram_total_mb} label={`RAM (${data.ram_used_mb} / ${data.ram_total_mb} MB)`} />
        <ProgressBar value={data.disk_used_gb} max={data.disk_total_gb} label={`Disk (${data.disk_used_gb} / ${data.disk_total_gb} GB)`} />
        {data.gpu.map((g) => (
          <ProgressBar key={g.index} value={g.used_mb} max={g.total_mb} label={`GPU ${g.index} ${g.name} (${g.used_mb} / ${g.total_mb} MB)`} />
        ))}
        {!data.cuda_available && (
          <p className="text-xs text-warning">No CUDA GPU detected</p>
        )}
      </div>
    </div>
  );
}
