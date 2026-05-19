const MYT = "Asia/Kuala_Lumpur";

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-MY", {
    timeZone: MYT,
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-MY", {
    timeZone: MYT,
    dateStyle: "medium",
  });
}
