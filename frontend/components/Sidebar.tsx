"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

const links = [
  { href: "/", label: "Dashboard", icon: "🏠" },
  { href: "/jobs", label: "Jobs", icon: "🚀" },
  { href: "/models", label: "Models", icon: "🤖" },
  { href: "/datasets", label: "Datasets", icon: "🗄️" },
];

export default function Sidebar() {
  const path = usePathname();
  return (
    <aside className="w-56 min-h-screen bg-base-200 flex flex-col py-6 px-3 gap-1 border-r border-base-300">
      <div className="px-3 mb-6">
        <h1 className="text-lg font-bold text-primary">🔥 TrainPipeline</h1>
        <p className="text-xs text-base-content/50 mt-0.5">LLM Fine-tuning</p>
      </div>
      <ul className="menu menu-sm gap-1 w-full">
        {links.map(({ href, label, icon }) => (
          <li key={href}>
            <Link
              href={href}
              className={clsx(
                "flex items-center gap-2",
                path === href && "active"
              )}
            >
              <span>{icon}</span>
              {label}
            </Link>
          </li>
        ))}
      </ul>
    </aside>
  );
}
