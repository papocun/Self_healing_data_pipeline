"use client";
import { useRef, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { fmtDateTime, fmtInt, fmtNum, fmtPct } from "@/lib/formatters";
import { Async, Empty, Loading, Metrics, PageHeader, Section, Status } from "@/components/ui";
import { SentimentDonut, TrendChart } from "@/components/charts";
import { pipelineState } from "@/lib/status";

const PRESETS = [
  { key: "all", label: "All time", months: 0 },
  { key: "12", label: "Last 12 months", months: 12 },
  { key: "3", label: "Last 3 months", months: 3 },
] as const;

const iso = (d: Date) => d.toISOString().slice(0, 10);

function Drivers({ title, items, tone }: { title: string; items: { aspect: string; count: number }[]; tone: "pos" | "neg" }) {
  const max = Math.max(1, ...items.map((i) => i.count));
  return (
    <div>
      <h3 className="mb-3 text-sm font-medium">{title}</h3>
      {items.length ? (
        <ul className="space-y-2.5">
          {items.map((i) => (
            <li key={i.aspect} className="text-sm">
              <div className="flex justify-between gap-3">
                <span className="truncate capitalize" title={i.aspect}>{i.aspect}</span>
                <span className="tabular-nums text-muted">{i.count}</span>
              </div>
              <div className="mt-1 h-1 rounded-full bg-border/60">
                <div className={`h-1 rounded-full ${tone === "pos" ? "bg-pos" : "bg-neg"}`} style={{ width: `${(100 * i.count) / max}%` }} />
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-muted">No aspects extracted.</p>
      )}
    </div>
  );
}

function Activity() {
  const p = useApi(api.pipeline);
  return (
    <Section title="Recent pipeline activity" action={<Link href="/pipeline" className="text-xs text-muted hover:text-ink">Details →</Link>}>
      <Async state={p}>
        {(d) => {
          const s = pipelineState(d);
          return (
            <dl className="grid grid-cols-2 gap-x-8 gap-y-4 text-sm sm:grid-cols-4">
              <div><dt className="text-xs text-muted">Status</dt><dd className="mt-1"><Status kind={s.kind}>{s.label}</Status></dd></div>
              <div><dt className="text-xs text-muted">Last successful run</dt><dd className="mt-1">{fmtDateTime(d.health.last_successful_execution)}</dd></div>
              <div><dt className="text-xs text-muted">Batches processed</dt><dd className="mt-1 tabular-nums">{fmtInt(d.kpis.completed_batches)}</dd></div>
              <div><dt className="text-xs text-muted">Failed batches</dt><dd className={`mt-1 tabular-nums ${d.kpis.failed_batches ? "text-neg" : ""}`}>{fmtInt(d.kpis.failed_batches)}</dd></div>
            </dl>
          );
        }}
      </Async>
    </Section>
  );
}

export default function OverviewPage() {
  const [preset, setPreset] = useState<(typeof PRESETS)[number]["key"]>("all");
  const maxDate = useRef<string | null>(null);
  const [gran, setGran] = useState<"daily" | "weekly" | "monthly">("monthly");

  let range: Record<string, string> = {};
  const months = PRESETS.find((x) => x.key === preset)!.months;
  if (months && maxDate.current) {
    const to = new Date(maxDate.current);
    const from = new Date(to);
    from.setMonth(from.getMonth() - months);
    range = { date_from: iso(from), date_to: iso(to) };
  }

  const st = useApi(async () => {
    const o = await api.overview(range);
    if (o.all_range?.max) maxDate.current = o.all_range.max;
    return o;
  }, [preset]);
  const tr = useApi(() => api.trends(gran, range), [gran, preset]);

  const select = (
    <select
      value={preset}
      onChange={(e) => setPreset(e.target.value as typeof preset)}
      aria-label="Date range"
      className="rounded border border-border bg-surface px-2.5 py-1.5 text-xs"
    >
      {PRESETS.map((x) => (
        <option key={x.key} value={x.key}>{x.label}</option>
      ))}
    </select>
  );

  return (
    <>
      <PageHeader title="Overview" subtitle="AI-powered review sentiment analytics" right={select} />
      <Async state={st}>
        {(o) =>
          o.empty ? (
            <Empty
              title={o.total_all_time ? "No reviews in this date range" : "No predictions yet"}
              hint={o.total_all_time ? "Choose a wider range." : "Run the pipeline so predictions land in output/predictions, then press Refresh."}
            />
          ) : (
            <>
              <Metrics
                items={[
                  { label: "Reviews processed", value: fmtInt(o.kpis.total) },
                  { label: "Positive", value: fmtPct(o.kpis.positive.pct), tone: "pos" },
                  { label: "Negative", value: fmtPct(o.kpis.negative.pct), tone: "neg" },
                  { label: "Average rating", value: fmtNum(o.kpis.avg_stars), sub: "out of 5" },
                ]}
              />

              <Section
                title="Sentiment trend"
                action={
                  <div className="flex gap-3 text-xs">
                    {(["daily", "weekly", "monthly"] as const).map((k) => (
                      <button key={k} onClick={() => setGran(k)} className={`capitalize ${gran === k ? "font-medium text-ink" : "text-muted hover:text-ink"}`}>
                        {k}
                      </button>
                    ))}
                  </div>
                }
              >
                {!tr.data ? <Loading /> : tr.data.points.length > 1 ? <TrendChart data={tr.data.points} /> : <Empty title="Not enough dated reviews to show a trend" />}
              </Section>

              <div className="grid gap-x-12 md:grid-cols-2">
                <Section title="Sentiment distribution">
                  <SentimentDonut data={o.distribution} total={o.kpis.total} />
                </Section>
                <Section title="Top review drivers">
                  <div className="grid gap-6 sm:grid-cols-2">
                    <Drivers title="Positive" items={o.aspects.positive.slice(0, 5)} tone="pos" />
                    <Drivers title="Negative" items={o.aspects.negative.slice(0, 5)} tone="neg" />
                  </div>
                </Section>
              </div>
            </>
          )
        }
      </Async>
      <Activity />
    </>
  );
}
