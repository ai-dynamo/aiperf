/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ports `docs/canvases/aiperf-metrics-accumulator.canvas.tsx` (a real, hand-authored Cursor
//! Canvas) onto aiperf-flow's component vocabulary. Single-view deck (no `PageTabs`): the
//! seven-stage record-to-export pipeline, the `ColumnStore` layout, `RaggedSeries` flat-buffer
//! storage, the sweep-line curve engine, effective-vs-active windowing, time slicing, the metric
//! taxonomy, and egress — one stacked scroll, matching the source canvas's structure.

import { useState } from "react";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Callout } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { Stat } from "../../prose/Stat.js";
import { Table, type TableColumn, type TableRow } from "../../prose/Table.js";
import { CollapsibleSection } from "../../prose/CollapsibleSection.js";
import { Swatch } from "../../prose/Swatch.js";
import { TopBar } from "../../shell/TopBar.js";
import {
  categoryBgClassName,
  categoryClassName,
  categoryFillClassName,
  inkClassName,
  strokeClassName,
  surfaceClassName,
  type CategoryRole,
} from "../../theme/tokens.js";
import {
  buildEvents,
  niceMax,
  stepPathD,
  stepPoints,
  type SweepCurveId,
  type SweepRequest,
} from "./sweepMath.js";

// ---------------------------------------------------------------------------
// Illustrative datasets (synthetic — chosen to make the concepts legible),
// ported verbatim from the source canvas.
// ---------------------------------------------------------------------------

const T_MAX = 52;

const REQUESTS: SweepRequest[] = [
  { id: "A", start: 0, gen: 6, end: 20, tokens: 120 },
  { id: "B", start: 3, gen: 10, end: 30, tokens: 200 },
  { id: "C", start: 8, gen: 12, end: 24, tokens: 90 },
  { id: "D", start: 14, gen: 22, end: 40, tokens: 260 },
  { id: "E", start: 18, gen: 25, end: 35, tokens: 150 },
  { id: "F", start: 28, gen: 34, end: 50, tokens: 180 },
];

// Ragged list-metric example: inter_chunk_latency (one variable-length list per record).
const ICL_LISTS: number[][] = [
  [12, 9, 11],
  [], // non-streaming / single chunk -> no inter-chunk gaps
  [8, 10, 9, 13, 7],
  [10, 12],
  [9],
];

type PipelineStage = {
  short: string;
  title: string;
  label: string;
  body: string;
};

const PIPELINE: PipelineStage[] = [
  {
    short: "Ingress",
    title: "1 · Ingress",
    label: "RequestRecord",
    body: "A Worker finishes an inference request and pushes an InferenceResultsMessage carrying the raw RequestRecord (HTTP/SSE payload + timestamps) onto the ZMQ bus.",
  },
  {
    short: "Parse",
    title: "2 · Parse & Extract",
    label: "MetricRecordDict",
    body: "The distributed RecordProcessor turns raw bytes into a ParsedResponseRecord (tokenization, error classification), then every registered metric runs its parse_record hook to emit a per-request MetricRecordDict of tag → scalar or list.",
  },
  {
    short: "Store",
    title: "3 · Columnar Store",
    label: "ColumnStore",
    body: "RecordsManager routes each MetricRecordsData into the MetricsAccumulator's ColumnStore: NaN-sparse float64 columns for scalars, RaggedSeries for list metrics, and separate metadata columns (phase, worker, correlation id).",
  },
  {
    short: "Sweep",
    title: "4 · Sweep Engine",
    label: "SweepLineCurves",
    body: "Interval boundaries become +/- delta events, sorted by time and cumulatively summed into step functions — concurrency, prefill/decode throughput, tokens-in-flight — the Effective and Active curves.",
  },
  {
    short: "Aggregate",
    title: "5 · Aggregation",
    label: "MetricResult",
    body: "RECORD metrics collapse arrays into min/max/avg/std/percentiles; AGGREGATE metrics fold with SUM/MAX/MIN; DERIVED metrics compute from other results in dependency order. Each tag becomes a MetricResult.",
  },
  {
    short: "Window",
    title: "6 · Windowing",
    label: "TimesliceResult",
    body: "An ExportContext masks records by phase (warmup vs profiling) and time. When slice_duration is set, a uniform grid buckets records into TimesliceResults, re-running the same stats per slice.",
  },
  {
    short: "Export",
    title: "7 · Egress",
    label: "ProfileResults",
    body: "Assembled into ProfileResults and handed to the ExporterManager: metrics JSON/CSV, timeslice JSON/CSV, console tables, and streamed per-record JSONL.",
  },
];

const CURVES: { id: SweepCurveId; label: string; unit: string; axis: string }[] = [
  { id: "concurrency", label: "Concurrency", unit: "in-flight requests", axis: "requests" },
  { id: "tokens", label: "Tokens in flight", unit: "tokens", axis: "tokens" },
  { id: "throughput", label: "Decode throughput", unit: "tokens / unit time", axis: "tok/t" },
];

const SLICE_DURATIONS = [10, 15, 20, 25];

// ---------------------------------------------------------------------------
// Section: interactive pipeline flow
// ---------------------------------------------------------------------------

function PipelineFlow(): React.JSX.Element {
  const [sel, setSel] = useState(3);
  const active = PIPELINE[sel];

  return (
    <Stack gap={12}>
      <Row gap={0} wrap={false} className="overflow-x-auto">
        {PIPELINE.map((stage, i) => (
          <button
            key={stage.short}
            type="button"
            onClick={() => setSel(i)}
            className={clsxLocal(
              "flex min-w-[104px] flex-1 flex-col items-center gap-1 border px-2 py-3 text-center transition-colors",
              strokeClassName("secondary"),
              i === sel
                ? "bg-accent-primary text-white"
                : `${surfaceClassName("elevated")} ${inkClassName("secondary")}`,
            )}
          >
            <span className="text-xs font-bold">{i + 1}</span>
            <span className="text-[11px]">{stage.short}</span>
          </button>
        ))}
      </Row>

      <div className={clsxLocal("border px-4 py-3", strokeClassName("secondary"), surfaceClassName("elevated"))}>
        <Row align="center" justify="space-between">
          <span className={`text-sm font-semibold ${inkClassName("primary")}`}>{active.title}</span>
          <span
            className={clsxLocal(
              "border px-2 py-0.5 font-mono text-[11px]",
              strokeClassName("secondary"),
              inkClassName("secondary"),
            )}
          >
            {active.label}
          </span>
        </Row>
        <p className={`mt-2 text-sm ${inkClassName("secondary")}`}>{active.body}</p>
      </div>
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Section: ColumnStore layout
// ---------------------------------------------------------------------------

const COLUMN_STORE_COLUMNS: TableColumn[] = [
  { key: "idx", label: "idx", align: "end" },
  { key: "start_ns", label: "start_ns", align: "end" },
  { key: "gen_start_ns", label: "gen_start_ns", align: "end" },
  { key: "end_ns", label: "end_ns", align: "end" },
  { key: "ttft", label: "ttft", align: "end" },
  { key: "output_tokens", label: "output_tokens", align: "end" },
  { key: "inter_chunk_latency", label: "inter_chunk_latency" },
  { key: "benchmark_phase", label: "benchmark_phase" },
];

const NAN_CELL = (
  <span className={`italic ${inkClassName("quaternary")}`}>NaN</span>
);
const RAGGED_CELL = <span className={categoryClassName("purple")}>{"→ragged"}</span>;
function phaseCell(p: string): React.JSX.Element {
  return <span className={categoryClassName("cyan")}>{p}</span>;
}

const COLUMN_STORE_ROWS: TableRow[] = [
  { idx: 0, start_ns: 0, gen_start_ns: 6, end_ns: 20, ttft: 6, output_tokens: 120, inter_chunk_latency: RAGGED_CELL, benchmark_phase: phaseCell("WARMUP") },
  { idx: 1, start_ns: 3, gen_start_ns: 10, end_ns: 30, ttft: 7, output_tokens: 200, inter_chunk_latency: RAGGED_CELL, benchmark_phase: phaseCell("PROFILING") },
  { idx: 2, start_ns: 8, gen_start_ns: 12, end_ns: 24, ttft: NAN_CELL, output_tokens: 90, inter_chunk_latency: RAGGED_CELL, benchmark_phase: phaseCell("PROFILING") },
  { idx: 3, start_ns: 14, gen_start_ns: 22, end_ns: 40, ttft: 8, output_tokens: 260, inter_chunk_latency: RAGGED_CELL, benchmark_phase: phaseCell("PROFILING") },
  { idx: 4, start_ns: 18, gen_start_ns: 25, end_ns: 35, ttft: 7, output_tokens: NAN_CELL, inter_chunk_latency: RAGGED_CELL, benchmark_phase: phaseCell("PROFILING") },
  { idx: 5, start_ns: 28, gen_start_ns: 34, end_ns: 50, ttft: 6, output_tokens: 180, inter_chunk_latency: RAGGED_CELL, benchmark_phase: phaseCell("PROFILING") },
];

function ColumnStoreView(): React.JSX.Element {
  return (
    <Stack gap={10}>
      <Row gap={16} wrap align="center">
        <LegendChip color="gray" label="fixed columns" />
        <LegendChip color="green" label="numeric (NaN = missing)" />
        <LegendChip color="purple" label="ragged / list" />
        <LegendChip color="cyan" label="metadata" />
      </Row>
      <Table columns={COLUMN_STORE_COLUMNS} rows={COLUMN_STORE_ROWS} />
      <p className={`text-sm ${inkClassName("tertiary")}`}>
        Append-only row index (never reused across phases). Metric columns are created lazily on
        first sighting; running per-column sums and counts give O(1) full-dataset totals without
        rescanning. Missing scalars are stored as NaN rather than reshaping the array.
      </p>
    </Stack>
  );
}

function LegendChip({ color, label }: { color: CategoryRole; label: string }): React.JSX.Element {
  return (
    <span className="inline-flex items-center gap-2">
      <Swatch color={color} />
      <span className={`text-sm ${inkClassName("secondary")}`}>{label}</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Section: RaggedSeries (flat buffer + offsets + record_indices)
// ---------------------------------------------------------------------------

const RAGGED_PALETTE: CategoryRole[] = ["blue", "orange", "green", "purple", "yellow"];

function RaggedSeriesView(): React.JSX.Element {
  const [sel, setSel] = useState(2);

  const values: number[] = [];
  const recordIndices: number[] = [];
  const offsets: number[] = [];
  ICL_LISTS.forEach((list, rec) => {
    if (list.length === 0) {
      offsets.push(-1);
      return;
    }
    offsets.push(values.length);
    for (const v of list) {
      values.push(v);
      recordIndices.push(rec);
    }
  });

  const selColor = RAGGED_PALETTE[sel % RAGGED_PALETTE.length];
  const selList = ICL_LISTS[sel];
  const cumsum: number[] = [];
  let acc = 0;
  for (const v of selList) {
    acc += v;
    cumsum.push(acc);
  }

  return (
    <Stack gap={14}>
      <Row gap={8} wrap align="center">
        <span className={`text-sm ${inkClassName("secondary")}`}>Select a record to trace its slice:</span>
        {ICL_LISTS.map((_, rec) => (
          <RecordPill key={rec} rec={rec} active={rec === sel} onClick={() => setSel(rec)} />
        ))}
      </Row>

      <Stack gap={6}>
        <span className={`text-sm font-semibold ${inkClassName("primary")}`}>
          Per-record lists (ragged — each request has a different length)
        </span>
        <Row gap={16} wrap>
          {ICL_LISTS.map((list, rec) => (
            <Row key={rec} gap={4} align="center">
              <span
                className={`w-6 text-sm font-semibold ${categoryClassName(RAGGED_PALETTE[rec % RAGGED_PALETTE.length])}`}
              >
                {`r${rec}`}
              </span>
              {list.length === 0 ? (
                <span className={`italic ${inkClassName("quaternary")}`}>empty</span>
              ) : (
                <Row gap={3}>
                  {list.map((v, k) => (
                    <RaggedCell key={k} value={v} color={RAGGED_PALETTE[rec % RAGGED_PALETTE.length]} highlight={rec === sel} />
                  ))}
                </Row>
              )}
            </Row>
          ))}
        </Row>
      </Stack>

      <Divider />

      <Stack gap={6}>
        <span className={`text-sm font-semibold ${inkClassName("primary")}`}>
          Flat storage — one contiguous buffer, no padding
        </span>
        <Row gap={10} align="start">
          <span className={`w-24 text-sm ${inkClassName("tertiary")}`}>values</span>
          <Row gap={3} wrap>
            {values.map((v, k) => (
              <RaggedCell key={k} value={v} color={selColor} highlight={recordIndices[k] === sel} />
            ))}
          </Row>
        </Row>
        <Row gap={10} align="start">
          <span className={`w-24 text-sm ${inkClassName("tertiary")}`}>record_indices</span>
          <Row gap={3} wrap>
            {recordIndices.map((r, k) => (
              <span
                key={k}
                className={clsxLocal(
                  "min-w-[30px] text-center text-[11px] tabular-nums",
                  r === sel ? categoryClassName(selColor) : inkClassName("quaternary"),
                )}
              >
                {r}
              </span>
            ))}
          </Row>
        </Row>
        <Row gap={10} align="start">
          <span className={`w-24 text-sm ${inkClassName("tertiary")}`}>offsets</span>
          <Row gap={3} wrap>
            {offsets.map((o, rec) => (
              <RaggedCell key={rec} value={o < 0 ? "−1" : o} color={RAGGED_PALETTE[rec % RAGGED_PALETTE.length]} highlight={rec === sel} />
            ))}
          </Row>
        </Row>
        <p className={`text-sm ${inkClassName("tertiary")}`}>
          <span className={`font-semibold ${categoryClassName(selColor)}`}>{`r${sel}`}</span> starts at offset{" "}
          <Code inline>{offsets[sel] < 0 ? "−1 (absent)" : String(offsets[sel])}</Code> and owns every flat value
          whose <Code inline>record_indices</Code> equals {sel}. A boolean mask over records selects matching values
          in bulk — no per-request Python loop.
        </p>
      </Stack>

      {selList.length > 0 && (
        <>
          <Divider />
          <Stack gap={6}>
            <span className={`text-sm font-semibold ${inkClassName("primary")}`}>
              grouped_cumsum on {`r${sel}`} — gaps become chunk boundaries
            </span>
            <Row gap={10} align="center" wrap>
              <span className={`w-24 text-sm ${inkClassName("tertiary")}`}>gaps</span>
              <Row gap={3}>
                {selList.map((v, k) => (
                  <RaggedCell key={k} value={v} color={selColor} highlight={false} />
                ))}
              </Row>
            </Row>
            <Row gap={10} align="center" wrap>
              <span className={`w-24 text-sm ${inkClassName("tertiary")}`}>cumulative</span>
              <Row gap={3}>
                {cumsum.map((v, k) => (
                  <RaggedCell key={k} value={v} color={selColor} highlight />
                ))}
              </Row>
            </Row>
            <p className={`text-sm ${inkClassName("tertiary")}`}>
              The prefix sum resets at each record boundary (via <Code inline>offsets</Code>), turning per-chunk
              latencies into absolute chunk end-times — the foundation of the ICL-aware throughput sweep.
            </p>
          </Stack>
        </>
      )}
    </Stack>
  );
}

function RecordPill({ rec, active, onClick }: { rec: number; active: boolean; onClick: () => void }): React.JSX.Element {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsxLocal(
        "inline-flex items-center gap-1.5 border px-2 py-1 text-xs font-semibold",
        strokeClassName("secondary"),
        active ? "bg-accent-primary text-white" : `${surfaceClassName("elevated")} ${inkClassName("secondary")}`,
      )}
    >
      <Swatch color={RAGGED_PALETTE[rec % RAGGED_PALETTE.length]} />
      {`r${rec}`}
    </button>
  );
}

function RaggedCell({
  value,
  color,
  highlight,
}: {
  value: number | string;
  color: CategoryRole;
  highlight: boolean;
}): React.JSX.Element {
  return (
    <span
      className={clsxLocal(
        "min-w-[30px] border px-1.5 py-1 text-center text-xs tabular-nums",
        strokeClassName("secondary"),
        highlight ? `${categoryBgClassName(color)} text-white` : `${surfaceClassName("elevated")} ${inkClassName("secondary")}`,
      )}
    >
      {value}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Section: Sweep-line — the interactive centerpiece
// ---------------------------------------------------------------------------

function SweepLineView(): React.JSX.Element {
  const [hidden, setHidden] = useState<string[]>([]);
  const [curve, setCurve] = useState<SweepCurveId>("concurrency");

  const isHidden = (id: string) => hidden.includes(id);
  const toggle = (id: string) =>
    setHidden((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  const activeReqs = REQUESTS.filter((r) => !isHidden(r.id));
  const evts = buildEvents(activeReqs, curve);
  const pts = stepPoints(evts);
  const rawMax = pts.reduce((m, p) => Math.max(m, p.v), 0);
  const vMax = niceMax(rawMax);
  const curveMeta = CURVES.find((c) => c.id === curve)!;

  const W = 760;
  const marginL = 54;
  const marginR = 20;
  const top = 12;
  const rowH = 20;
  const ganttH = REQUESTS.length * rowH;
  const stepTop = top + ganttH + 30;
  const stepH = 150;
  const axisY = stepTop + stepH;
  const H = axisY + 34;

  const xLeft = marginL;
  const xRight = W - marginR;
  const x = (t: number) => xLeft + (t / T_MAX) * (xRight - xLeft);
  const y = (v: number) => stepTop + stepH - (v / vMax) * stepH;

  const ticks = [0, 10, 20, 30, 40, 50];
  const yTicks = [0, vMax / 2, vMax];

  return (
    <Stack gap={14}>
      <Row gap={8} wrap align="center">
        <span className={`text-sm ${inkClassName("secondary")}`}>Curve:</span>
        {CURVES.map((c) => (
          <CurvePill key={c.id} active={c.id === curve} label={c.label} onClick={() => setCurve(c.id)} />
        ))}
        <span className="flex-1" />
        <Stat value={activeReqs.length} label="active requests" />
      </Row>

      <Row gap={6} wrap align="center">
        <span className={`text-sm ${inkClassName("secondary")}`}>Toggle requests:</span>
        {REQUESTS.map((r) => (
          <CurvePill key={r.id} active={!isHidden(r.id)} label={r.id} onClick={() => toggle(r.id)} />
        ))}
      </Row>

      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label="Sweep-line concurrency chart">
        {ticks.map((t) => (
          <line key={`g${t}`} x1={x(t)} y1={top} x2={x(t)} y2={axisY} className={strokeClassName("tertiary")} stroke="currentColor" />
        ))}

        {REQUESTS.map((r, i) => {
          const yTop = top + i * rowH + 3;
          const barH = rowH - 8;
          const dim = isHidden(r.id);
          return (
            <g key={r.id} opacity={dim ? 0.25 : 1}>
              <text x={xLeft - 8} y={yTop + barH - 2} textAnchor="end" fontSize={11} fontWeight={600} className={inkClassName("secondary")} fill="currentColor">
                {r.id}
              </text>
              <rect x={x(r.start)} y={yTop} width={Math.max(1, x(r.gen) - x(r.start))} height={barH} className={categoryFillClassName("gray")} />
              <rect x={x(r.gen)} y={yTop} width={Math.max(1, x(r.end) - x(r.gen))} height={barH} className={categoryFillClassName("blue")} opacity={dim ? 0.25 : 0.85} />
            </g>
          );
        })}

        <line x1={xLeft} y1={top + ganttH + 12} x2={xRight} y2={top + ganttH + 12} className={strokeClassName("secondary")} stroke="currentColor" strokeDasharray="3 3" />

        {yTicks.map((v, k) => (
          <g key={`y${k}`}>
            <line x1={xLeft} y1={y(v)} x2={xRight} y2={y(v)} className={strokeClassName("tertiary")} stroke="currentColor" />
            <text x={xLeft - 8} y={y(v) + 3} textAnchor="end" fontSize={10} className={inkClassName("tertiary")} fill="currentColor">
              {Number.isInteger(v) ? v : v.toFixed(1)}
            </text>
          </g>
        ))}

        <path d={`${stepPathD(pts, x, y, 0, T_MAX)} L ${x(T_MAX)} ${y(0)} L ${x(0)} ${y(0)} Z`} className={categoryFillClassName("blue")} opacity={0.12} />
        <path d={stepPathD(pts, x, y, 0, T_MAX)} fill="none" className={categoryClassName("blue")} stroke="currentColor" strokeWidth={2} />

        {evts.map((e, k) => (
          <line key={`e${k}`} x1={x(e.t)} y1={axisY - 4} x2={x(e.t)} y2={axisY + 4} className={e.d > 0 ? categoryClassName("green") : categoryClassName("orange")} stroke="currentColor" strokeWidth={1.5} />
        ))}

        <line x1={xLeft} y1={axisY} x2={xRight} y2={axisY} className={strokeClassName("primary")} stroke="currentColor" />
        {ticks.map((t) => (
          <text key={`t${t}`} x={x(t)} y={axisY + 18} textAnchor="middle" fontSize={10} className={inkClassName("tertiary")} fill="currentColor">
            {t}
          </text>
        ))}
        <text x={(xLeft + xRight) / 2} y={H - 2} textAnchor="middle" fontSize={10} className={inkClassName("tertiary")} fill="currentColor">
          time (relative ns)
        </text>
        <text x={14} y={stepTop + stepH / 2} textAnchor="middle" fontSize={10} className={inkClassName("tertiary")} fill="currentColor" transform={`rotate(-90 14 ${stepTop + stepH / 2})`}>
          {curveMeta.axis}
        </text>
      </svg>

      <Row gap={16} wrap>
        <LegendRow color="blue" label="decode window (generation_start → end)" />
        <LegendRow color="gray" label="prefill window (start → generation_start)" />
        <LegendRow color="green" label="+delta (interval start)" />
        <LegendRow color="orange" label="−delta (interval end)" />
      </Row>

      <Callout tone="info" title="Why a sweep line?">
        Each interval contributes a <Code inline>+weight</Code> at its start and a{" "}
        <Code inline>−weight</Code> at its end. Sorting these events by time and taking a running{" "}
        <Code inline>cumsum</Code> yields the exact step function in <Code inline>O(E log E)</Code> — never
        scanning the timeline point by point. The same machinery produces {curveMeta.label.toLowerCase()} (
        {curveMeta.unit}), concurrency, and throughput just by changing the weight.
      </Callout>
    </Stack>
  );
}

function CurvePill({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }): React.JSX.Element {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsxLocal(
        "border px-2.5 py-1 text-xs font-semibold",
        strokeClassName("secondary"),
        active ? "bg-accent-primary text-white" : `${surfaceClassName("elevated")} ${inkClassName("secondary")}`,
      )}
    >
      {label}
    </button>
  );
}

function LegendRow({ color, label }: { color: CategoryRole; label: string }): React.JSX.Element {
  return (
    <Row gap={5} align="center">
      <Swatch color={color} />
      <span className={`text-sm ${inkClassName("tertiary")}`}>{label}</span>
    </Row>
  );
}

// ---------------------------------------------------------------------------
// Section: sweep events -> cumsum table
// ---------------------------------------------------------------------------

const CUMSUM_COLUMNS: TableColumn[] = [
  { key: "time", label: "time", align: "end" },
  { key: "request", label: "request", align: "center" },
  { key: "delta", label: "delta", align: "end" },
  { key: "sum", label: "running sum (concurrency)", align: "end" },
];

function CumsumTable(): React.JSX.Element {
  const evts = buildEvents(REQUESTS, "concurrency");
  let acc = 0;
  const rows: TableRow[] = evts.map((e) => {
    acc += e.d;
    return {
      time: e.t,
      request: e.id,
      delta: (
        <span className={e.d > 0 ? categoryClassName("green") : categoryClassName("orange")}>
          {e.d > 0 ? `+${e.d}` : e.d}
        </span>
      ),
      sum: <span className="font-semibold">{acc}</span>,
    };
  });
  return <Table columns={CUMSUM_COLUMNS} rows={rows} />;
}

// ---------------------------------------------------------------------------
// Section: Time slicing
// ---------------------------------------------------------------------------

const SLICE_PALETTE: CategoryRole[] = ["blue", "green", "orange", "purple", "gray", "cyan", "yellow"];

function TimeSliceView(): React.JSX.Element {
  const [dur, setDur] = useState(15);

  const spanEnd = Math.max(...REQUESTS.map((r) => r.end));
  const spanStart = Math.min(...REQUESTS.map((r) => r.start));
  const nSlices = Math.ceil((spanEnd - spanStart) / dur);
  const edges: number[] = [];
  for (let k = 0; k <= nSlices; k++) edges.push(spanStart + k * dur);
  const gridEnd = edges[edges.length - 1];
  const lastIncomplete = gridEnd > spanEnd;

  const binOf = (start: number) => Math.min(nSlices - 1, Math.floor((start - spanStart) / dur));

  const W = 760;
  const marginL = 40;
  const marginR = 20;
  const top = 10;
  const rowH = 22;
  const ganttH = REQUESTS.length * rowH;
  const axisY = top + ganttH + 6;
  const H = axisY + 40;
  const xLeft = marginL;
  const xRight = W - marginR;
  const x = (t: number) => xLeft + (t / T_MAX) * (xRight - xLeft);

  return (
    <Stack gap={14}>
      <Row gap={8} wrap align="center">
        <span className={`text-sm ${inkClassName("secondary")}`}>slice_duration:</span>
        {SLICE_DURATIONS.map((d) => (
          <CurvePill key={d} active={d === dur} label={`${d} ns`} onClick={() => setDur(d)} />
        ))}
        <span className="flex-1" />
        <Stat value={nSlices} label="slices" />
      </Row>

      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label="Time slice chart">
        {edges.slice(0, -1).map((e, k) => {
          const start = e;
          const rawEnd = edges[k + 1];
          const clipped = Math.min(rawEnd, spanEnd);
          const incomplete = rawEnd > spanEnd;
          return (
            <g key={`slice${k}`}>
              <rect x={x(start)} y={top} width={x(clipped) - x(start)} height={ganttH} className={k % 2 === 0 ? inkClassName("quaternary") : ""} fill={k % 2 === 0 ? "currentColor" : "transparent"} opacity={k % 2 === 0 ? 0.4 : 1} />
              {incomplete && (
                <rect x={x(clipped)} y={top} width={x(rawEnd) - x(clipped)} height={ganttH} className={categoryFillClassName("orange")} opacity={0.15} />
              )}
              <line x1={x(start)} y1={top} x2={x(start)} y2={axisY} className={strokeClassName("secondary")} stroke="currentColor" strokeDasharray="2 2" />
              <text x={(x(start) + x(clipped)) / 2} y={axisY + 16} textAnchor="middle" fontSize={10} className={incomplete ? categoryClassName("orange") : inkClassName("tertiary")} fill="currentColor">
                {`slice ${k}${incomplete ? " *" : ""}`}
              </text>
            </g>
          );
        })}
        <line x1={x(Math.min(gridEnd, spanEnd))} y1={top} x2={x(Math.min(gridEnd, spanEnd))} y2={axisY} className={strokeClassName("secondary")} stroke="currentColor" strokeDasharray="2 2" />

        {REQUESTS.map((r, i) => {
          const yTop = top + i * rowH + 4;
          const barH = rowH - 9;
          const bin = binOf(r.start);
          const color = SLICE_PALETTE[bin % SLICE_PALETTE.length];
          return (
            <g key={r.id}>
              <rect x={x(r.start)} y={yTop} width={Math.max(2, x(r.end) - x(r.start))} height={barH} className={categoryFillClassName(color)} opacity={0.85} />
              <circle cx={x(r.start)} cy={yTop + barH / 2} r={3} className={inkClassName("primary")} fill="currentColor" />
              <text x={x(r.start) + 6} y={yTop + barH - 1} fontSize={10} fill="white" fontWeight={600}>
                {r.id}
              </text>
            </g>
          );
        })}

        <line x1={xLeft} y1={axisY} x2={xRight} y2={axisY} className={strokeClassName("primary")} stroke="currentColor" />
        <text x={(xLeft + xRight) / 2} y={H - 4} textAnchor="middle" fontSize={10} className={inkClassName("tertiary")} fill="currentColor">
          time (relative ns) · dot = record start (binning key)
        </text>
      </svg>

      {lastIncomplete ? (
        <Callout tone="warning" title="Incomplete trailing slice">
          The uniform grid extends to <Code inline>{String(gridEnd)}</Code> but activity ends at{" "}
          <Code inline>{String(spanEnd)}</Code>. The last slice is clipped to real activity and flagged{" "}
          <Code inline>is_complete = false</Code> so rate metrics aren&apos;t diluted by idle padding.
        </Callout>
      ) : (
        <Callout tone="success" title="Evenly divisible">
          The grid lands exactly on the activity span — every slice is complete.
        </Callout>
      )}
      <p className={`text-sm ${inkClassName("tertiary")}`}>
        Records are assigned to a bin by their <Code inline>start</Code> time via <Code inline>np.digitize</Code>.
        Each non-empty bin becomes a <Code inline>TimesliceResult</Code> that re-runs the full RECORD / AGGREGATE /
        DERIVED stats plus sweep metrics clipped to that window — sweeps are computed once, then queried per
        slice.
      </p>
    </Stack>
  );
}

// ---------------------------------------------------------------------------
// Section: metric taxonomy
// ---------------------------------------------------------------------------

type TaxonomyItem = {
  type: string;
  color: CategoryRole;
  desc: string;
  agg: string;
  ex: string;
};

const TAXONOMY: TaxonomyItem[] = [
  {
    type: "RECORD",
    color: "blue",
    desc: "One scalar or list per request.",
    agg: "Collapse the array → min / max / avg / std / p1…p99 / count / sum.",
    ex: "request_latency, inter_chunk_latency, output_token_count",
  },
  {
    type: "AGGREGATE",
    color: "green",
    desc: "One contribution per request, folded live.",
    agg: "Fold via AggregationKind — SUM, MAX, or MIN → a single scalar.",
    ex: "request_count, benchmark_duration",
  },
  {
    type: "DERIVED",
    color: "purple",
    desc: "No raw values — computed from other results.",
    agg: "derive_value() runs after RECORD + AGGREGATE, in dependency order.",
    ex: "output_token_throughput, request_throughput",
  },
];

function MetricTaxonomy(): React.JSX.Element {
  return (
    <Grid columns={3} gap={12}>
      {TAXONOMY.map((it) => (
        <div key={it.type} className={clsxLocal("border px-4 py-3", strokeClassName("secondary"), surfaceClassName("elevated"))}>
          <Row align="center" gap={8}>
            <Swatch color={it.color} />
            <span className={`text-sm font-bold ${inkClassName("primary")}`}>{it.type}</span>
          </Row>
          <Stack gap={8} className="mt-2">
            <p className={`text-sm ${inkClassName("secondary")}`}>{it.desc}</p>
            <p className={`text-sm ${inkClassName("primary")}`}>{it.agg}</p>
            <p className={`text-sm italic ${inkClassName("tertiary")}`}>e.g. {it.ex}</p>
          </Stack>
        </div>
      ))}
    </Grid>
  );
}

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

function clsxLocal(...parts: (string | false | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

/**
 * Single-view deck (no in-deck page tabs, matching the source canvas's structure): from raw
 * request records to exported percentiles — the columnar store, ragged list buffers,
 * sweep-line curves built on cumulative sums, and time-sliced windowing, as one visual system.
 */
export function AiperfMetricsAccumulatorDeck(): React.JSX.Element {
  return (
    <div className={clsxLocal("flex h-screen flex-col", surfaceClassName("chrome"))}>
      <TopBar section="Metrics Accumulator" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className={clsxLocal("mx-auto min-h-full max-w-6xl px-10 py-8", surfaceClassName("page"))}>
          <Stack gap={28}>
            <Stack gap={8}>
              <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>AIPerf Metrics Accumulator</h1>
              <p className={inkClassName("secondary")}>
                From raw request records to exported percentiles — the columnar store, the ragged list buffers,
                the sweep-line curves built on cumulative sums, and time-sliced windowing, as one visual system.
              </p>
              <Callout tone="info" title="Conceptual illustration">
                All numbers below are small synthetic examples chosen to make each mechanism legible. Controls are
                interactive — toggle requests, switch curves, pick a slice duration, and select a record to
                trace it through the ragged buffer.
              </Callout>
            </Stack>

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>The pipeline</h2>
              <p className={inkClassName("secondary")}>
                Seven stages carry a request from the wire to an exported artifact. Click a stage to inspect it.
              </p>
              <PipelineFlow />
            </Stack>

            <Divider />

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Stage 3 · The ColumnStore</h2>
              <p className={inkClassName("secondary")}>
                The accumulator holds the whole run as a record-indexed columnar table. Scalars live in NaN-sparse
                numeric columns, lists in RaggedSeries, and run metadata in separate typed columns that never enter
                the metric math.
              </p>
              <ColumnStoreView />
            </Stack>

            <Divider />

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>
                RaggedSeries · variable-length metrics, flat memory
              </h2>
              <p className={inkClassName("secondary")}>
                List metrics like inter-chunk latency have a different length per request. Instead of a padded
                matrix, everything is concatenated into one buffer with two index arrays —{" "}
                <Code inline>offsets</Code> and <Code inline>record_indices</Code> — enabling O(1) lookup, bulk
                masking, and per-request grouped prefix sums.
              </p>
              <RaggedSeriesView />
            </Stack>

            <Divider />

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Stage 4 · The sweep line</h2>
              <p className={inkClassName("secondary")}>
                Concurrency and throughput aren&apos;t per-request scalars — they&apos;re functions of time. The
                sweep line turns interval boundaries into signed events and rebuilds the exact step function with a
                single cumulative sum.
              </p>
              <SweepLineView />
              <CollapsibleSection title="Events → running sum (concurrency, all requests)">
                <Stack gap={8}>
                  <p className={inkClassName("secondary")}>
                    The sorted event stream and its prefix sum. The running total is exactly the height of the step
                    function above.
                  </p>
                  <CumsumTable />
                </Stack>
              </CollapsibleSection>
            </Stack>

            <Divider />

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Effective vs. Active</h2>
              <p className={inkClassName("secondary")}>
                Every sweep curve is reported two ways, both time-weighted over the window rather than counting
                requests equally.
              </p>
              <Grid columns={2} gap={12}>
                <div className={clsxLocal("border px-4 py-3", strokeClassName("secondary"), surfaceClassName("elevated"))}>
                  <span className={`text-sm font-bold ${inkClassName("primary")}`}>Effective</span>
                  <p className={`mt-2 text-sm ${inkClassName("secondary")}`}>
                    Averaged across the entire measurement window, including moments the system sat idle. Answers:
                    what did the endpoint sustain over wall clock?
                  </p>
                </div>
                <div className={clsxLocal("border px-4 py-3", strokeClassName("secondary"), surfaceClassName("elevated"))}>
                  <span className={`text-sm font-bold ${inkClassName("primary")}`}>Active</span>
                  <p className={`mt-2 text-sm ${inkClassName("secondary")}`}>
                    Averaged only over segments where the relevant phase mask is non-zero. Answers: how fast was it
                    while actually working?
                  </p>
                </div>
              </Grid>
            </Stack>

            <Divider />

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Stage 6 · Time slicing</h2>
              <p className={inkClassName("secondary")}>
                With a slice duration set, the window is cut into a uniform grid and each record is bucketed by its
                start time. Watch how the trailing slice becomes incomplete when the grid overshoots real activity.
              </p>
              <TimeSliceView />
            </Stack>

            <Divider />

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Stage 5 · Metric taxonomy</h2>
              <p className={inkClassName("secondary")}>
                Three metric kinds share one output envelope, the <Code inline>MetricResult</Code>, but aggregate
                differently.
              </p>
              <MetricTaxonomy />
            </Stack>

            <Divider />

            <Stack gap={12}>
              <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Stage 7 · Egress</h2>
              <Row gap={8} wrap align="center">
                <EgressChip label="AccumulatorMetricsSummary" active />
                <span className={inkClassName("tertiary")}>{"→"}</span>
                <EgressChip label="ProfileResults" active />
                <span className={inkClassName("tertiary")}>{"→"}</span>
                <EgressChip label="ExporterManager" active />
                <span className={inkClassName("tertiary")}>{"→"}</span>
                <EgressChip label="metrics JSON / CSV" />
                <EgressChip label="timeslice JSON / CSV" />
                <EgressChip label="console tables" />
                <EgressChip label="per-record JSONL" />
              </Row>
              <p className={`text-sm ${inkClassName("tertiary")}`}>
                Warmup and profiling phases summarize independently; GPU telemetry and server metrics flow through
                parallel accumulators that share the same protocol but hold different record types.
              </p>
            </Stack>
          </Stack>
        </div>
      </div>
    </div>
  );
}

function EgressChip({ label, active = false }: { label: string; active?: boolean }): React.JSX.Element {
  return (
    <span
      className={clsxLocal(
        "border px-2.5 py-1 text-xs font-semibold",
        strokeClassName("secondary"),
        active ? "bg-accent-primary text-white" : `${surfaceClassName("elevated")} ${inkClassName("secondary")}`,
      )}
    >
      {label}
    </span>
  );
}
