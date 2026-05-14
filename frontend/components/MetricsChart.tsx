"use client";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { TrainingMetric } from "@/types";

interface Props {
  metrics: TrainingMetric[];
}

export default function MetricsChart({ metrics }: Props) {
  if (metrics.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-base-content/40 text-sm">
        No metrics yet — waiting for training to start…
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <ChartPanel
        title="Loss"
        data={metrics}
        lines={[
          { key: "loss", color: "#6366f1", name: "Train Loss" },
          { key: "eval_loss", color: "#f59e0b", name: "Eval Loss" },
        ]}
      />
      <ChartPanel
        title="Learning Rate"
        data={metrics}
        lines={[{ key: "learning_rate", color: "#10b981", name: "LR" }]}
        yDomain={["auto", "auto"]}
      />
      {metrics.some((m) => m.reward != null) && (
        <ChartPanel
          title="Reward"
          data={metrics}
          lines={[{ key: "reward", color: "#ec4899", name: "Reward" }]}
        />
      )}
    </div>
  );
}

function ChartPanel({
  title,
  data,
  lines,
  yDomain,
}: {
  title: string;
  data: TrainingMetric[];
  lines: { key: keyof TrainingMetric; color: string; name: string }[];
  yDomain?: [number | string, number | string];
}) {
  return (
    <div>
      <p className="text-sm font-semibold mb-2 text-base-content/80">{title}</p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" />
          <XAxis dataKey="step" tick={{ fontSize: 11 }} stroke="#ffffff30" />
          <YAxis domain={yDomain} tick={{ fontSize: 11 }} stroke="#ffffff30" />
          <Tooltip
            contentStyle={{ background: "#1d232a", border: "1px solid #ffffff20", fontSize: 12 }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {lines.map((l) => (
            <Line
              key={l.key as string}
              type="monotone"
              dataKey={l.key as string}
              stroke={l.color}
              dot={false}
              name={l.name}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
