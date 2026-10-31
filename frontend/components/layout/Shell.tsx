"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Activity, BarChart3, Moon, RefreshCw, Search, Sun } from "lucide-react";
import { RefreshProvider, useRefresh } from "./RefreshContext";

const NAV = [
  { href: "/", label: "Overview", icon: BarChart3 },
  { href: "/reviews", label: "Reviews", icon: Search },
  { href: "/pipeline", label: "Pipeline", icon: Activity },
];

function Header() {
  const { refresh, lastUpdated } = useRefresh();
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    try {
      const saved = localStorage.getItem("theme");
      if (saved === "dark" || saved === "light") setTheme(saved);
    } catch {}
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {}
  }, [theme]);

  return (
    <div className="flex items-center gap-3 text-xs text-muted">
      <span>{lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : "Loading…"}</span>
      <button
        onClick={refresh}
        className="flex items-center gap-1.5 rounded border border-border px-2.5 py-1.5 text-ink hover:bg-border/40"
      >
        <RefreshCw size={13} /> Refresh
      </button>
      <button
        aria-label="Toggle dark mode"
        onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        className="rounded border border-border p-1.5 text-ink hover:bg-border/40"
      >
        {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
      </button>
    </div>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  return (
    <RefreshProvider>
      <div className="min-h-screen overflow-x-hidden md:flex">
        <aside className="border-b border-border md:sticky md:top-0 md:h-screen md:w-48 md:shrink-0 md:border-b-0 md:border-r">
          <div className="px-4 py-4 text-sm font-semibold tracking-tight">SelfHealing<span className="text-muted">Pipeline</span></div>
          <nav className="flex gap-1 overflow-x-auto px-2 pb-2 md:flex-col md:pb-0">
            {NAV.map(({ href, label, icon: Icon }) => {
              const active = href === "/" ? path === "/" : path.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  className={`flex items-center gap-2 whitespace-nowrap rounded-md px-3 py-2 text-sm ${
                    active ? "bg-border/50 font-medium text-ink" : "text-muted hover:text-ink"
                  }`}
                >
                  <Icon size={15} /> {label}
                </Link>
              );
            })}
          </nav>
        </aside>
        <div className="min-w-0 flex-1">
          <header className="flex justify-end px-4 py-3 md:px-8">
            <Header />
          </header>
          <main className="mx-auto max-w-5xl px-4 py-6 md:px-8">{children}</main>
        </div>
      </div>
    </RefreshProvider>
  );
}
