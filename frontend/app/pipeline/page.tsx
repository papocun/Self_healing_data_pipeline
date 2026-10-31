"use client";
import { ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { fmtDateTime, fmtInt } from "@/lib/formatters";
import { pipelineState } from "@/lib/status";
import { Async, Empty, Metrics, PageHeader, Section, Status } from "@/components/ui";
import type { PipelineStatus } from "@/lib/types";

/** Spec stages -> API stage names. */
const FLOW: [string, string][] = [
  ["Raw Data", "Raw Data"],
  ["Validation", "Validation"],
  ["Cleaning", "Cleaning"],
  ["Deduplication", "Deduplication"],
  ["Ollama", "Ollama Inference"],
  ["Parquet", "Persistence"],
];

function Flow({ stages }: { stages: PipelineStatus["stages"] }) {
  return (
    <ol className="flex flex-col gap-3 lg:flex-row lg:items-start lg:gap-0">
      {FLOW.map(([label, key], i) => {
        const s = stages.find((x) => x.stage === key);
        return (
          <li key={label} className="flex flex-1 items-start lg:pr-2">
            <div className="flex-1">
              <div className="text-xs text-muted">{label}</div>
              <div className="mt-1 text-lg font-medium tabular-nums">{s?.count != null ? fmtInt(s.count) : "â€”"}</div>
              <div className="mt-1">
                <Status kind={s?.status ?? "idle"}>{s?.note ?? "no data"}</Status>
              </div>
            </div>
            {i < FLOW.length - 1 && <ChevronRight size={14} className="mt-6 hidden shrink-0 text-border lg:block" />}
          </li>
        );
      })}
    </ol>
  );
}

function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border/60 py-2 text-sm last:border-0">
      <span className="text-muted">{k}</span>
      <span className="text-right">{children}</span>
    </div>
  );
}

export default function PipelinePage() {
  const st = useApi(api.pipeline);
  const bt = useApi(api.batches);
  return (
    <>
      <PageHeader title="Pipeline" subtitle="Is the ETL and inference system working, and does it recover?" />
      <Async state={st}>
        {(p) => {
          const s = pipelineState(p);
          const k = p.kpis;
          const h = p.health;
          if (!k.total_batches)
            return <Empty title="No pipeline runs recorded" hint="Run main.py or trigger the Airflow DAG; checkpoints will appear here." />;
          const processed = k.successful_inferences + k.failed_inferences;
          return (
            <>
              <dl className="mb-8 grid grid-cols-2 gap-x-8 gap-y-4 text-sm sm:grid-cols-3">
                <div>
                  <dt className="text-xs text-muted">Pipeline status</dt>
                  <dd className="mt-1 text-lg"><Status kind={s.kind}>{s.label}</Status></dd>
                </div>
                <div>
                  <dt className="text-xs text-muted">Last execution</dt>
                  <dd className="mt-1 text-lg">{fmtDateTime(h.latest_checkpoint?.timestamp)}</dd>
                </div>
              </dl>
              <Metrics
                items={[
                  { label: "Records processed", value: fmtInt(processed), sub: "reached inference" },
                  { label: "Completed batches", value: fmtInt(k.completed_batches), sub: `of ${fmtInt(k.total_batches)}` },
                  { label: "Failed inferences", value: fmtInt(k.failed_inferences), tone: k.failed_inferences ? "neg" : undefined },
                  { label: "Invalid records", value: fmtInt(k.invalid_records), tone: k.invalid_records ? "warn" : undefined },
                ]}
              />

              <Section title="Pipeline flow">
                <Flow stages={p.stages} />
                <p className="mt-4 text-xs text-muted">Cumulative records reaching each stage across all checkpointed batches.</p>
              </Section>

              <div className="grid gap-x-12 md:grid-cols-2">
                <Section title="Retry and recovery">
                  <Row k="Failed records logged">{fmtInt(p.recovery.failed_records_logged)}</Row>
                  <Row k="Recovered by a later run"><span className="text-pos">{fmtInt(p.recovery.recovered)}</span></Row>
                  <Row k="Unresolved"><span className={p.recovery.unresolved ? "text-neg" : ""}>{fmtInt(p.recovery.unresolved)}</span></Row>
                  <Row k="Retry attempts"><span className="text-muted">not recorded</span></Row>
                  {Object.entries(p.inference.error_classes).length > 0 && (
                    <div className="mt-3 text-xs text-muted">
                      Causes: {Object.entries(p.inference.error_classes).map(([n, c]) => `${n} (${fmtInt(c)})`).join(" Â· ")}
                    </div>
                  )}
                </Section>
                <Section title="Health and checkpoint">
                  <Row k="Ollama">
                    <Status kind={h.ollama.reachable ? "ok" : "fail"}>{h.ollama.reachable ? "Reachable" : "Unreachable"}</Status>
                  </Row>
                  <Row k={`Model ${h.ollama.model}`}>
                    <Status kind={!h.ollama.reachable ? "idle" : h.ollama.model_available ? "ok" : "warn"}>
                      {!h.ollama.reachable ? "unknown" : h.ollama.model_available ? "Installed" : "Not installed"}
                    </Status>
                  </Row>
                  <Row k="Latest checkpoint">{h.latest_checkpoint ? `Batch ${h.latest_checkpoint.batch_id}` : "â€”"}</Row>
                  <Row k="Last failure">{fmtDateTime(h.last_failed_execution)}</Row>
                  <Row k="Quarantined records">{fmtInt(h.quarantine_count)}</Row>
                </Section>
              </div>

              <Section title="Recent executions">
                {!bt.data ? null : bt.data.items.length === 0 ? (
                  <Empty title="No batches" />
                ) : (
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-border text-xs text-muted">
                        <th className="px-2 py-2 font-medium">Batch</th>
                        <th className="px-2 py-2 font-medium">Source</th>
                        <th className="px-2 py-2 font-medium">Status</th>
                        <th className="px-2 py-2 text-right font-medium">Predicted</th>
                        <th className="px-2 py-2 text-right font-medium">Failed</th>
                        <th className="hidden px-2 py-2 font-medium sm:table-cell">Completed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {bt.data.items.map((b) => (
                        <tr key={b.key} className="border-b border-border/60 last:border-0">
                          <td className="px-2 py-2 tabular-nums">{b.batch_id}</td>
                          <td className="max-w-[160px] truncate px-2 py-2 text-muted" title={b.source_file}>{b.source_file}</td>
                          <td className="px-2 py-2"><Status kind={b.status === "COMPLETED" ? (b.failed ? "warn" : "ok") : "fail"}>{b.status.toLowerCase()}</Status></td>
                          <td className="px-2 py-2 text-right tabular-nums">{fmtInt(b.ok)}</td>
                          <td className={`px-2 py-2 text-right tabular-nums ${b.failed ? "text-neg" : ""}`}>{fmtInt(b.failed)}</td>
                          <td className="hidden whitespace-nowrap px-2 py-2 text-xs text-muted sm:table-cell">{fmtDateTime(b.timestamp)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                <p className="mt-3 text-xs text-muted">Inference latency is not recorded by the pipeline.</p>
              </Section>
            </>
          );
        }}
      </Async>
    </>
  );
}

