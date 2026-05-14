"use client";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { getJobs, getModels, getDatasets } from "@/lib/api";
import JobStatusBadge from "@/components/JobStatusBadge";
import SystemStats from "@/components/SystemStats";

export default function DashboardPage() {
  const { data: jobs = [] } = useQuery({ queryKey: ["jobs"], queryFn: getJobs, refetchInterval: 5000 });
  const { data: models = [] } = useQuery({ queryKey: ["models"], queryFn: getModels });
  const { data: datasets = [] } = useQuery({ queryKey: ["datasets"], queryFn: getDatasets });

  const activeJobs = jobs.filter((j) => j.status === "running").length;
  const recentJobs = jobs.slice(0, 5);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Dashboard</h2>
        <Link href="/jobs/new" className="btn btn-primary btn-sm">
          + New Training Job
        </Link>
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="stat bg-base-200 rounded-box border border-base-300">
          <div className="stat-title">Active Jobs</div>
          <div className="stat-value text-info">{activeJobs}</div>
          <div className="stat-desc">{jobs.length} total</div>
        </div>
        <div className="stat bg-base-200 rounded-box border border-base-300">
          <div className="stat-title">Models</div>
          <div className="stat-value text-primary">{models.length}</div>
          <div className="stat-desc">{models.filter((m) => m.is_downloaded === "true").length} downloaded</div>
        </div>
        <div className="stat bg-base-200 rounded-box border border-base-300">
          <div className="stat-title">Datasets</div>
          <div className="stat-value text-secondary">{datasets.length}</div>
          <div className="stat-desc">
            {datasets.reduce((s, d) => s + (d.num_samples ?? 0), 0).toLocaleString()} samples
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent jobs */}
        <div className="lg:col-span-2 card bg-base-200 border border-base-300">
          <div className="card-body p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="card-title text-sm">Recent Jobs</h3>
              <Link href="/jobs" className="btn btn-ghost btn-xs">View all →</Link>
            </div>
            {recentJobs.length === 0 ? (
              <p className="text-sm text-base-content/40 py-6 text-center">No jobs yet</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="table table-xs">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Method</th>
                      <th>Status</th>
                      <th>Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentJobs.map((job) => (
                      <tr key={job.id} className="hover cursor-pointer">
                        <td>
                          <Link href={`/jobs/${job.id}`} className="link link-hover font-medium">
                            {job.name}
                          </Link>
                        </td>
                        <td className="uppercase text-xs font-mono">{job.training_method}</td>
                        <td><JobStatusBadge status={job.status} /></td>
                        <td className="text-xs text-base-content/50">
                          {new Date(job.created_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* System stats */}
        <SystemStats />
      </div>
    </div>
  );
}
