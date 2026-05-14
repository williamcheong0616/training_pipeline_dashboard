"use client";
import { useParams } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getJob, cancelJob, exportJob } from "@/lib/api";
import { useMetricsStream } from "@/lib/sse";
import MetricsChart from "@/components/MetricsChart";
import JobStatusBadge from "@/components/JobStatusBadge";

export default function JobDetailPage() {
  const { id } = useParams<{ id: string }>();
  const jobId = Number(id);
  const qc = useQueryClient();

  const { data: job, isLoading } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    refetchInterval: (q) => (q.state.data?.status === "running" ? 5000 : false),
  });

  const metrics = useMetricsStream(job?.status === "running" ? jobId : null);

  const { mutate: cancel } = useMutation({
    mutationFn: () => cancelJob(jobId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["job", jobId] }),
  });

  const { mutate: doExport, isPending: exporting } = useMutation({
    mutationFn: () => exportJob(jobId),
  });

  if (isLoading) return <div className="skeleton h-64 w-full rounded-box" />;
  if (!job) return <div className="alert alert-error">Job not found</div>;

  const config = job as unknown as { config_json?: Record<string, unknown> };

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold">{job.name}</h2>
          <p className="text-sm text-base-content/50 mt-1">Job #{job.id}</p>
        </div>
        <div className="flex gap-2">
          {(job.status === "running" || job.status === "pending") && (
            <button className="btn btn-error btn-sm btn-outline" onClick={() => cancel()}>Stop</button>
          )}
          {job.status === "completed" && (
            <button className="btn btn-success btn-sm" disabled={exporting} onClick={() => doExport()}>
              {exporting ? <span className="loading loading-spinner" /> : "Export / Merge"}
            </button>
          )}
        </div>
      </div>

      {/* Status strip */}
      <div className="flex flex-wrap gap-4 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-base-content/50">Status</span>
          <JobStatusBadge status={job.status} />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-base-content/50">Method</span>
          <span className="badge badge-ghost badge-sm uppercase font-mono">{job.training_method}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-base-content/50">PEFT</span>
          <span className="font-mono text-xs">{job.peft_method}</span>
        </div>
        {job.started_at && (
          <div className="flex items-center gap-2">
            <span className="text-base-content/50">Started</span>
            <span className="text-xs">{new Date(job.started_at).toLocaleString()}</span>
          </div>
        )}
      </div>

      {job.error_msg && (
        <div className="alert alert-error text-sm">
          <span className="font-semibold">Error:</span> {job.error_msg}
        </div>
      )}

      {/* Metrics */}
      <div className="card bg-base-200 border border-base-300">
        <div className="card-body p-4">
          <h3 className="card-title text-sm mb-2">Training Metrics</h3>
          <MetricsChart metrics={metrics} />
        </div>
      </div>

      {/* Config */}
      <div className="collapse collapse-arrow bg-base-200 border border-base-300">
        <input type="checkbox" />
        <div className="collapse-title font-medium text-sm">Training Config</div>
        <div className="collapse-content">
          <pre className="text-xs overflow-auto bg-base-300 rounded p-3 max-h-64">
            {JSON.stringify(config.config_json ?? {}, null, 2)}
          </pre>
        </div>
      </div>

      {job.output_dir && (
        <div className="text-xs text-base-content/40">
          Output: <span className="font-mono">{job.output_dir}</span>
        </div>
      )}
    </div>
  );
}
