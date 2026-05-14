import { useEffect, useRef, useState } from "react";
import type { TrainingMetric } from "@/types";

export function useMetricsStream(jobId: number | null) {
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!jobId) return;
    const es = new EventSource(`/api/jobs/${jobId}/metrics`);
    esRef.current = es;

    es.onmessage = (e) => {
      try {
        const m: TrainingMetric = JSON.parse(e.data);
        setMetrics((prev) => [...prev, m]);
      } catch {}
    };

    es.onerror = () => es.close();

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [jobId]);

  return metrics;
}
