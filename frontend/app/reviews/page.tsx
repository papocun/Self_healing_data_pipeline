"use client";
import { useEffect, useState } from "react";
import { ArrowDown, ArrowUp, X } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { fmtDate, fmtDateTime, fmtInt } from "@/lib/formatters";
import { Async, Empty, ErrorBox, Loading, PageHeader, Pagination, Status } from "@/components/ui";
import type { ReviewRow } from "@/lib/types";

interface Filters {
  q: string;
  sentiment: string;
  stars: string;
  conf_min: string;
  conf_max: string;
  date_from: string;
  date_to: string;
  business_id: string;
}
const EMPTY: Filters = { q: "", sentiment: "", stars: "", conf_min: "", conf_max: "", date_from: "", date_to: "", business_id: "" };
const SIZE = 25;

function useDebounced<T>(v: T, ms = 350) {
  const [d, setD] = useState(v);
  useEffect(() => {
    const t = setTimeout(() => setD(v), ms);
    return () => clearTimeout(t);
  }, [v, ms]);
  return d;
}

function Detail({ id, onClose }: { id: string; onClose: () => void }) {
  const st = useApi(() => api.review(id), [id]);
  return (
    <div className="fixed inset-0 z-30 flex justify-end bg-black/30" onClick={onClose}>
      <aside
        className="h-full w-full max-w-lg overflow-y-auto border-l border-border bg-surface p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold">Review detail</h2>
          <button aria-label="Close" onClick={onClose} className="rounded p-1 hover:bg-border/40">
            <X size={16} />
          </button>
        </div>
        <Async state={st}>
          {(r) => (
            <div className="space-y-5 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <Status kind={r.sentiment} />
                <span className="text-muted">
                  {r.stars}★ · confidence {(r.confidence * 100).toFixed(0)}% · {fmtDate(r.date)}
                </span>
              </div>
              <Section label="Original text">{r.original_text}</Section>
              {r.cleaned_text && r.cleaned_text !== r.original_text && <Section label="Cleaned text">{r.cleaned_text}</Section>}
              <Section label="Model reasoning">{r.reasoning || "Not stored."}</Section>
              <div className="grid grid-cols-2 gap-4">
                <Chips label="Positive aspects" items={r.positive_aspects} tone="positive" />
                <Chips label="Negative aspects" items={r.negative_aspects} tone="negative" />
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 border-t border-border pt-4 text-xs">
                <Meta k="Review ID" v={r.review_id} />
                <Meta k="Business ID" v={r.business_id} />
                <Meta k="Model" v={r.model_name} />
                <Meta k="Batch" v={r.batch_id != null ? String(r.batch_id) : "—"} />
                <Meta k="Inferred at" v={fmtDateTime(r.inference_timestamp)} />
              </dl>
            </div>
          )}
        </Async>
      </aside>
    </div>
  );
}
const Section = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div>
    <div className="mb-1 text-xs font-medium uppercase tracking-wide text-muted">{label}</div>
    <p className="whitespace-pre-wrap leading-relaxed">{children}</p>
  </div>
);
const Meta = ({ k, v }: { k: string; v: string }) => (
  <div className="min-w-0">
    <dt className="text-muted">{k}</dt>
    <dd className="break-all">{v}</dd>
  </div>
);
function Chips({ label, items, tone }: { label: string; items: string[]; tone: "positive" | "negative" }) {
  return (
    <div>
      <div className="mb-1 text-xs font-medium uppercase tracking-wide text-muted">{label}</div>
      {items.length ? (
        <ul className="space-y-1">
          {items.map((a) => (
            <li key={a} className={`rounded px-2 py-1 text-xs ${tone === "positive" ? "bg-pos/10" : "bg-neg/10"}`}>
              {a}
            </li>
          ))}
        </ul>
      ) : (
        <span className="text-xs text-muted">None</span>
      )}
    </div>
  );
}

const COLS: { key: string; label: string; sortable: boolean; cls?: string }[] = [
  { key: "stars", label: "Stars", sortable: true },
  { key: "sentiment", label: "Sentiment", sortable: true },
  { key: "confidence", label: "Conf.", sortable: true },
  { key: "text", label: "Review", sortable: false, cls: "min-w-[280px]" },
  { key: "date", label: "Date", sortable: true },
];

export default function ReviewsPage() {
  const [f, setF] = useState<Filters>(EMPTY);
  const [sort, setSort] = useState("date");
  const [order, setOrder] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(1);
  const [open, setOpen] = useState<string | null>(null);
  const df = useDebounced(f);

  useEffect(() => setPage(1), [df, sort, order]);

  const st = useApi(
    () =>
      api.reviews({
        ...df,
        conf_min: df.conf_min ? Number(df.conf_min) / 100 : undefined,
        conf_max: df.conf_max ? Number(df.conf_max) / 100 : undefined,
        sort,
        order,
        page,
        size: SIZE,
      }),
    [df, sort, order, page]
  );

  const set = (k: keyof Filters) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setF((p) => ({ ...p, [k]: e.target.value }));
  const input = "rounded border border-border bg-surface px-2.5 py-1.5 text-sm";
  const dirty = JSON.stringify(f) !== JSON.stringify(EMPTY);

  const toggleSort = (k: string) => {
    if (sort === k) setOrder(order === "asc" ? "desc" : "asc");
    else {
      setSort(k);
      setOrder("desc");
    }
  };

  return (
    <>
      <PageHeader title="Reviews" subtitle="Click a review to inspect the model’s prediction" />
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <input className={`${input} min-w-[220px] flex-1`} value={f.q} onChange={set("q")} placeholder="Search review text or ID" aria-label="Search" />
        <select className={`${input} w-auto`} value={f.sentiment} onChange={set("sentiment")} aria-label="Sentiment">
          <option value="">All sentiment</option>
          <option value="positive">Positive</option>
          <option value="negative">Negative</option>
          <option value="neutral">Neutral</option>
        </select>
        <select className={`${input} w-auto`} value={f.stars} onChange={set("stars")} aria-label="Star rating">
          <option value="">All ratings</option>
          {[5, 4, 3, 2, 1].map((n) => (
            <option key={n} value={n}>{n} ★</option>
          ))}
        </select>
        {dirty && (
          <button onClick={() => setF(EMPTY)} className="px-2 text-xs text-muted hover:text-ink">Clear</button>
        )}
      </div>

      <div className="border-t border-border pt-4">
        {st.error ? (
          <ErrorBox message={st.error} />
        ) : !st.data ? (
          <Loading />
        ) : st.data.total === 0 ? (
          <Empty title={dirty ? "No reviews match these filters" : "No predictions yet"} hint={dirty ? "Loosen or clear a filter." : "Run the pipeline, then press Refresh."} />
        ) : (
          <>
            <div className="mb-2 text-xs text-muted">{fmtInt(st.data.total)} matching reviews</div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-border text-xs text-muted">
                    {COLS.map((c) => (
                      <th key={c.key} className={`whitespace-nowrap px-2 py-2 font-medium ${c.cls ?? ""}`}>
                        {c.sortable ? (
                          <button onClick={() => toggleSort(c.key)} className="inline-flex items-center gap-1 hover:text-ink">
                            {c.label}
                            {sort === c.key && (order === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                          </button>
                        ) : (
                          c.label
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className={st.loading ? "opacity-50" : ""}>
                  {st.data.items.map((r: ReviewRow) => (
                    <tr key={r.review_id} onClick={() => setOpen(r.review_id)} className="cursor-pointer border-b border-border/60 last:border-0 hover:bg-border/40">
                      <td className="px-2 py-2 tabular-nums">{r.stars}★</td>
                      <td className="px-2 py-2"><Status kind={r.sentiment} /></td>
                      <td className={`px-2 py-2 tabular-nums ${r.confidence < 0.6 ? "text-warn" : ""}`}>{(r.confidence * 100).toFixed(0)}%</td>
                      <td className="px-2 py-2"><span className="line-clamp-2 text-muted">{r.text}</span></td>
                      <td className="whitespace-nowrap px-2 py-2 text-xs text-muted">{fmtDate(r.date)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Pagination page={page} size={SIZE} total={st.data.total} onPage={setPage} />
          </>
        )}
      </div>
      {open && <Detail id={open} onClose={() => setOpen(null)} />}
    </>
  );
}
