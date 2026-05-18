import { useEffect, useRef, useState } from "react";
import type { TrainingMetric } from "@/types";

export function useMetricsStream(jobId: number | null, apiPrefix: string = "/api/jobs") {
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const esRef = useRef<EventSource | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seenIds = useRef<Set<number>>(new Set());

  useEffect(() => {
    if (!jobId) return;
    seenIds.current = new Set();

    const connect = () => {
      const es = new EventSource(`${apiPrefix}/${jobId}/metrics`);
      esRef.current = es;

      es.onmessage = (e) => {
        try {
          const m: TrainingMetric = JSON.parse(e.data);
          if (!seenIds.current.has(m.id)) {
            seenIds.current.add(m.id);
            setMetrics((prev) => [...prev, m]);
          }
        } catch {
          // malformed frame — ignore
        }
      };

      // Server emits a named "done" event when the job reaches a terminal state
      es.addEventListener("done", () => {
        es.close();
        esRef.current = null;
      });

      es.onerror = () => {
        es.close();
        esRef.current = null;
        // Reconnect after 3 s; deduplication via seenIds prevents duplicate metrics
        timerRef.current = setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      esRef.current?.close();
      esRef.current = null;
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [jobId, apiPrefix]);

  return metrics;
}
