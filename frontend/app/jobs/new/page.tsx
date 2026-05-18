"use client";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function NewJobRedirect() {
  const router = useRouter();
  if (typeof window !== "undefined") router.replace("/");
  return (
    <div style={{ padding: 24, fontFamily: "var(--mono)", fontSize: 12, color: "var(--text-dim)" }}>
      Redirecting to Train… <Link href="/" style={{ color: "var(--accent)" }}>click here</Link>
    </div>
  );
}
