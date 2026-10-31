"use client";
import { CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Overview, Trends } from "@/lib/types";
import { fmtInt, fmtTick } from "@/lib/formatters";

export const COLOR = {
  positive: "var(--pos)",
  negative: "var(--neg)",
  neutral: "var(--neu)",
};

const tooltipStyle = {
  contentStyle: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 4,
    fontSize: 12,
    color: "var(--ink)",
    boxShadow: "none",
  },
  labelStyle: { color: "var(--ink)" },
  itemStyle: { color: "var(--ink)" },
};

/** Main focal chart: sentiment per period. */
export function TrendChart({ data }: { data: Trends["points"] }) {
  return (
    <div className="h-[300px]">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ left: -16, right: 4, top: 8 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="period" tickFormatter={fmtTick} tickLine={false} axisLine={false} minTickGap={40} />
          <YAxis tickLine={false} axisLine={false} allowDecimals={false} />
          <Tooltip {...tooltipStyle} labelFormatter={(p: string) => new Date(p).toLocaleDateString("en-US", { dateStyle: "medium" })} />
          <Legend iconType="plainline" wrapperStyle={{ fontSize: 12, color: "var(--muted)" }} />
          {(["positive", "negative", "neutral"] as const).map((s) => (
            <Line
              key={s}
              isAnimationActive={false}
              dataKey={s}
              stroke={COLOR[s]}
              strokeWidth={s === "neutral" ? 1.5 : 2}
              dot={false}
              type="monotone"
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function SentimentDonut({ data, total }: { data: Overview["distribution"]; total: number }) {
  return (
    <div className="flex items-center gap-8">
      <div className="relative h-[168px] w-[168px] shrink-0">
        <ResponsiveContainer>
          <PieChart>
            <Pie
              isAnimationActive={false}
              data={data}
              dataKey="count"
              nameKey="sentiment"
              innerRadius={58}
              outerRadius={80}
              stroke="var(--surface)"
              strokeWidth={2}
            >
              {data.map((d) => (
                <Cell key={d.sentiment} fill={COLOR[d.sentiment]} />
              ))}
            </Pie>
            <Tooltip {...tooltipStyle} formatter={(v: number) => fmtInt(v)} />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-lg font-medium tabular-nums">{fmtInt(total)}</span>
          <span className="text-xs text-muted">reviews</span>
        </div>
      </div>
      <ul className="flex-1 space-y-2.5 text-sm">
        {data.map((d) => (
          <li key={d.sentiment} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-2 capitalize">
              <span className="h-2 w-2 rounded-full" style={{ background: COLOR[d.sentiment] }} />
              {d.sentiment}
            </span>
            <span className="tabular-nums text-muted">
              {fmtInt(d.count)} <span className="ml-1 text-ink">{d.pct.toFixed(0)}%</span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
