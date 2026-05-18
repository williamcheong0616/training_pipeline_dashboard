import type { JobStatus } from "@/types";

const STYLE: Record<JobStatus, string> = {
  pending:   "lf-badge-pending",
  running:   "lf-badge-running",
  completed: "lf-badge-done",
  failed:    "lf-badge-failed",
  cancelled: "lf-badge-cancelled",
};

export default function JobStatusBadge({ status }: { status: JobStatus }) {
  return <span className={`lf-badge ${STYLE[status]}`}>{status}</span>;
}
