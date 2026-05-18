"use client";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import type { TrainingMetric } from "@/types";

const TICK_STYLE = { fontSize: 10, fontFamily: "JetBrains Mono, monospace", fill: "#5e6478" };
const TOOLTIP_STYLE = { background: "#090b10", border: "1px solid #252a3a", fontSize: 11, fontFamily: "JetBrains Mono, monospace", color: "#c8ccd8" };

function MiniChart({ data, keys, height = 110 }: {
  data: TrainingMetric[];
  keys: { key: keyof TrainingMetric; color: string; label: string }[];
  height?: number;
}) {
  if (data.length < 2) return (
    <div style={{ height, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-dim)", fontFamily: "var(--mono)", fontSize: 11 }}>
      — no data —
    </div>
  );
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="2 4" stroke="#252a3a" />
        <XAxis dataKey="step" tick={TICK_STYLE} stroke="transparent" />
        <YAxis tick={TICK_STYLE} stroke="transparent" width={52} />
        <Tooltip contentStyle={TOOLTIP_STYLE} />
        {keys.map((k) => (
          <Line
            key={k.key as string}
            type="monotone"
            dataKey={k.key as string}
            stroke={k.color}
            dot={false}
            name={k.label}
            strokeWidth={1.5}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

function StatBox({ label, value }: { label: string; value: string | number | null }) {
  return (
    <div style={{
      background: "var(--bg-input)",
      border: "1px solid var(--border)",
      borderRadius: 3,
      padding: "6px 10px",
      minWidth: 90,
    }}>
      <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 2 }}>{label}</div>
      <div style={{ fontFamily: "var(--mono)", fontSize: 14, fontWeight: 600, color: "var(--text-hi)" }}>
        {value == null ? "—" : typeof value === "number" && value < 0.01 ? value.toExponential(2) : typeof value === "number" ? value.toFixed(4) : value}
      </div>
    </div>
  );
}

export default function MetricsPanel({ metrics }: { metrics: TrainingMetric[] }) {
  const last = metrics[metrics.length - 1];
  const hasDPO = metrics.some((m) => m.reward != null);

  return (
    <div>
      {/* Live stat boxes */}
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
        <StatBox label="step"      value={last?.step ?? 0} />
        <StatBox label="loss"      value={last?.loss ?? null} />
        <StatBox label="eval loss" value={last?.eval_loss ?? null} />
        <StatBox label="lr"        value={last?.learning_rate ?? null} />
        <StatBox label="grad norm" value={last?.grad_norm ?? null} />
        {hasDPO && <StatBox label="reward" value={last?.reward ?? null} />}
        <StatBox label="epoch"     value={last?.epoch != null ? last.epoch.toFixed(2) : null} />
      </div>

      {/* Charts grid */}
      <div style={{ display: "grid", gridTemplateColumns: hasDPO ? "1fr 1fr 1fr" : "1fr 1fr", gap: 10 }}>
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>Loss</div>
          <MiniChart
            data={metrics}
            keys={[
              { key: "loss",      color: "#4a9eff", label: "train" },
              { key: "eval_loss", color: "#e8a820", label: "eval" },
            ]}
          />
        </div>
        <div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>Learning Rate</div>
          <MiniChart
            data={metrics}
            keys={[{ key: "learning_rate", color: "#3dd68c", label: "lr" }]}
          />
        </div>
        {hasDPO && (
          <div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>Reward</div>
            <MiniChart
              data={metrics}
              keys={[{ key: "reward", color: "#b06fff", label: "reward" }]}
            />
          </div>
        )}
      </div>
    </div>
  );
}
