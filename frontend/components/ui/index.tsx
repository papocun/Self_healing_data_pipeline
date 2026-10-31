"use client";
import { AlertTriangle, Loader2 } from "lucide-react";
import type { ApiState } from "@/lib/useApi";

export function PageHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: React.ReactNode }) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-0.5 text-sm text-muted">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

/** Unboxed section separated from its neighbours by a hairline. */
export function Section({
  title,
  action,
  children,
  className = "",
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`border-t border-border py-6 ${className}`}>
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="text-xs font-medium uppercase tracking-wider text-muted">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

/** Row of plain-typography metrics, no boxes. */
export function Metrics({ items }: { items: { label: string; value: string; tone?: "pos" | "neg" | "warn"; sub?: string }[] }) {
  return (
    <dl className="mb-8 grid grid-cols-2 gap-x-8 gap-y-5 sm:grid-cols-4">
      {items.map((m) => (
        <div key={m.label}>
          <dt className="text-xs text-muted">{m.label}</dt>
          <dd
            className={`mt-1 text-2xl font-medium tabular-nums tracking-tight ${
              m.tone === "pos" ? "text-pos" : m.tone === "neg" ? "text-neg" : m.tone === "warn" ? "text-warn" : ""
            }`}
          >
            {m.value}
          </dd>
          {m.sub && <div className="text-xs text-muted">{m.sub}</div>}
        </div>
      ))}
    </dl>
  );
}

export function Loading() {
  return (
    <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted">
      <Loader2 size={16} className="animate-spin" /> Loadingâ€¦
    </div>
  );
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="py-10 text-center">
      <p className="text-sm">{title}</p>
      {hint && <p className="mx-auto mt-1 max-w-sm text-xs text-muted">{hint}</p>}
    </div>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 border-l-2 border-neg py-1 pl-3 text-sm">
      <AlertTriangle size={15} className="mt-0.5 shrink-0 text-neg" />
      <div>
        <p>Could not load data</p>
        <p className="text-xs text-muted">
          {message}. Start the API with <code className="rounded bg-border/50 px-1">uvicorn api.main:app --port 8000</code>, then press Refresh.
        </p>
      </div>
    </div>
  );
}

export function Async<T>({ state, children }: { state: ApiState<T>; children: (d: T) => React.ReactNode }) {
  if (state.error) return <ErrorBox message={state.error} />;
  if (!state.data) return <Loading />;
  return <>{children(state.data)}</>;
}

const DOT: Record<string, string> = {
  positive: "bg-pos",
  negative: "bg-neg",
  neutral: "bg-neu",
  ok: "bg-pos",
  warn: "bg-warn",
  fail: "bg-neg",
  idle: "bg-neu",
  missing: "bg-neu",
};
/** Status as a dot plus label. */
export function Status({ kind, children }: { kind: string; children?: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-sm first-letter:uppercase">
      <span className={`h-1.5 w-1.5 rounded-full ${DOT[kind] ?? "bg-neu"}`} />
      {children ?? kind}
    </span>
  );
}

export function Pagination({ page, size, total, onPage }: { page: number; size: number; total: number; onPage: (p: number) => void }) {
  const pages = Math.max(1, Math.ceil(total / size));
  const btn = "rounded px-2 py-1 text-xs hover:bg-border/50 disabled:opacity-30";
  return (
    <div className="mt-3 flex items-center justify-between text-xs text-muted">
      <span>
        {total ? (page - 1) * size + 1 : 0}â€“{Math.min(page * size, total)} of {total.toLocaleString()}
      </span>
      <div className="flex items-center gap-1">
        <button className={btn} disabled={page <= 1} onClick={() => onPage(page - 1)}>
          Previous
        </button>
        <span className="px-1">
          {page} / {pages.toLocaleString()}
        </span>
        <button className={btn} disabled={page >= pages} onClick={() => onPage(page + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}

