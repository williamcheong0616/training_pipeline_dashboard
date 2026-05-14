"use client";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { getJobs, cancelJob } from "@/lib/api";
import JobStatusBadge from "@/components/JobStatusBadge";

function elapsed(job: { started_at: string | null; finished_at: string | null }) {
  if (!job.started_at) return "—";
  const end = job.finished_at ? new Date(job.finished_at) : new Date();
  const secs = Math.round((end.getTime() - new Date(job.started_at).getTime()) / 1000);
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

export default function JobsPage() {
  const qc = useQueryClient();
  const { data: jobs = [], isLoading } = useQuery({ queryKey: ["jobs"], queryFn: getJobs, refetchInterval: 5000 });
  const { mutate: cancel } = useMutation({
    mutationFn: cancelJob,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }),
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Jobs</h2>
        <Link href="/jobs/new" className="btn btn-primary btn-sm">+ New Training Job</Link>
      </div>

      {isLoading ? (
        <div className="skeleton h-48 w-full rounded-box" />
      ) : jobs.length === 0 ? (
        <div className="card bg-base-200 border border-base-300">
          <div className="card-body items-center py-16 text-base-content/40">
            <p>No training jobs yet.</p>
            <Link href="/jobs/new" className="btn btn-primary btn-sm mt-2">Start your first job</Link>
          </div>
        </div>
      ) : (
        <div className="card bg-base-200 border border-base-300">
          <div className="card-body p-0">
            <div className="overflow-x-auto">
              <table className="table table-sm">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Method</th>
                    <th>PEFT</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Created</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job) => (
                    <tr key={job.id} className="hover">
                      <td className="font-mono text-xs text-base-content/50">#{job.id}</td>
                      <td>
                        <Link href={`/jobs/${job.id}`} className="link link-hover font-medium">
                          {job.name}
                        </Link>
                      </td>
                      <td className="badge badge-ghost badge-sm uppercase font-mono">{job.training_method}</td>
                      <td className="text-xs text-base-content/60">{job.peft_method}</td>
                      <td><JobStatusBadge status={job.status} /></td>
                      <td className="font-mono text-xs">{elapsed(job)}</td>
                      <td className="text-xs text-base-content/50">
                        {new Date(job.created_at).toLocaleString()}
                      </td>
                      <td>
                        {job.status === "running" || job.status === "pending" ? (
                          <button
                            className="btn btn-xs btn-error btn-outline"
                            onClick={() => cancel(job.id)}
                          >
                            Stop
                          </button>
                        ) : (
                          <Link href={`/jobs/${job.id}`} className="btn btn-xs btn-ghost">
                            View
                          </Link>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
