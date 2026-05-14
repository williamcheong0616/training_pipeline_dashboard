import type { JobStatus } from "@/types";
import clsx from "clsx";

const STATUS_STYLE: Record<JobStatus, string> = {
  pending: "badge-warning",
  running: "badge-info",
  completed: "badge-success",
  failed: "badge-error",
  cancelled: "badge-ghost",
};

export default function JobStatusBadge({ status }: { status: JobStatus }) {
  return (
    <span className={clsx("badge badge-sm font-semibold", STATUS_STYLE[status])}>
      {status}
    </span>
  );
}
