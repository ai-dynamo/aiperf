/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — watching both algorithms take the same values.
//!
//! Values arrive one at a time. On the left they are kept; on the right they are absorbed into
//! centroids. Then both are asked the same question, and the page shows each answering it —
//! reading two retained values, or walking a chain of centroids.

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  arrivalAt,
  createIngest,
  ingestOne,
  traceExact,
  traceFold,
  sortedAtSummarize,
  traceSketch,
  type ExactTrace,
  type IngestState,
} from "./ingest.js";
import { ControlBar, Panel, Readout, SourceNote, SpikeHeader, Toggle } from "../ui.js";

const CYAN = "var(--color-category-cyan)";
const GREEN = "var(--color-category-green)";
const ORANGE = "var(--color-category-orange)";
const PURPLE = "var(--color-category-purple)";
const DIM = "var(--color-ink-quaternary)";

const CELLS = 3;
/**
 * A far smaller δ than production's 100.
 *
 * At δ=100 a digest settles at ~50 centroids, which is too many to follow one at a time. At 12 it
 * settles at a handful, so each absorption is a visible event. The rule being demonstrated is
 * identical; only the budget is smaller.
 */
const COMPRESSION = 12;
/**
 * The compress threshold, turned right down.
 *
 * The Rust derives it as `max(64, δ × 10)` — 1000 at production's δ=100 — so a real digest sits
 * on a thousand raw unsorted centroids before doing anything. That is the true cost profile and
 * it is unwatchable, so the page uses 8. The rule is unchanged: append until the count is
 * exceeded, then sort and cluster the whole buffer at once.
 */
const THRESHOLD = 20;
/** How many recent values the arrival strip renders. The array behind it keeps growing. */
const STRIP_WINDOW = 90;
/** Most bars the sort animation draws. Beyond this it animates an evenly-spaced sample. */
const SORT_WINDOW = 260;
/** Interval, and how many values to take per tick. Batching is what makes "flood" unbounded in
 *  any useful sense — one React render per value caps the rate at about forty a second. */
const SPEEDS = [
  { ms: 700, batch: 1, label: "slow" },
  { ms: 350, batch: 1, label: "medium" },
  { ms: 120, batch: 1, label: "fast" },
  { ms: 40, batch: 25, label: "flood" },
] as const;
const CELL_COLOR = [CYAN, PURPLE, ORANGE];

export function SketchLiveSpike(): React.JSX.Element {
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(SPEEDS[1]);
  const [percentile, setPercentile] = useState(90);
  /** Set when the user asks to summarize, which can be at any point — not only at the end. */
  const [summarizing, setSummarizing] = useState(false);

  // Incremental, not recomputed. Replaying the whole stream on every tick is quadratic, which is
  // fine for a fixed sixty values and hopeless for a run with no end.
  const [state, setState] = useState<IngestState>(() => createIngest(CELLS, COMPRESSION, THRESHOLD));

  const advance = useCallback((count: number) => {
    setState((current) => {
      let next = current;
      for (let i = 0; i < count; i++) {
        next = ingestOne(next, arrivalAt(next.arrived.length, CELLS));
      }
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    setState(createIngest(CELLS, COMPRESSION, THRESHOLD));
  }, []);

  useEffect(() => {
    if (!running) return undefined;
    const handle = window.setInterval(() => advance(speed.batch), speed.ms);
    return () => window.clearInterval(handle);
  }, [running, speed, advance]);

  const next = summarizing ? undefined : arrivalAt(state.arrived.length, CELLS);
  // There is no end to reach, so a summary only ever happens because it was asked for.
  const showSummary = summarizing;
  const pendingAtSummarize = state.cells.reduce(
    (n, cell, i) => n + cell.centroids.length - (state.settled[i] ?? 0),
    0,
  );

  return (
    <div className="min-h-screen bg-surface-page px-8 py-7 text-ink-primary">
      <SpikeHeader title="One value at a time, into both">
        <p>
          The same latency values arrive at two structures. The exact path <strong>keeps</strong>{" "}
          every one, in sorted order. The sketch path <strong>absorbs</strong> each into a
          centroid, and when two centroids are close enough to share one cluster it merges them and
          the value is gone as a distinct thing. Watch the right-hand side stop growing.
        </p>
      </SpikeHeader>

      <ControlBar>
        <div className="flex items-center gap-1.5">
          {summarizing ? (
            <Toggle active onClick={() => setSummarizing(false)}>← back to ingest</Toggle>
          ) : (
            <>
              <Toggle active onClick={() => setRunning((r) => !r)}>
                {running ? "Pause" : "Play"}
              </Toggle>
              <Toggle onClick={() => { setRunning(false); advance(1); }}>Next value</Toggle>
            </>
          )}
          <Toggle onClick={() => { setRunning(false); setSummarizing(false); reset(); }}>
            Reset
          </Toggle>
          <Toggle disabled={state.arrived.length === 0 || showSummary}
            onClick={() => { setRunning(false); setSummarizing(true); }}
            title="Summarize whatever has arrived so far — a report can be asked for at any point">
            Summarize now
          </Toggle>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">speed</span>
          {SPEEDS.map((option) => (
            <Toggle key={option.label} active={speed.label === option.label}
              onClick={() => setSpeed(option)}>
              {option.label}
            </Toggle>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">summarize</span>
          {[50, 90, 99].map((p) => (
            <Toggle key={p} active={percentile === p} onClick={() => setPercentile(p)}>p{p}</Toggle>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-6">
          <Readout label="arrived" value={state.arrived.length.toLocaleString()} />
          <Readout label="kept" value={`${(state.arrived.length * 8 / 1024).toFixed(1)} KB`}
            color={GREEN} />
          <Readout label="centroids"
            value={state.cells.reduce((n, c) => n + c.centroids.length, 0)} color={CYAN} />
          <Readout label="δ" value={COMPRESSION} />
        </div>
      </ControlBar>

      <div className="mb-3 flex min-h-[52px] items-center gap-4 rounded-lg border border-white/10
        bg-surface-elevated px-4 py-3">
        {next === undefined ? (
          <span className="text-[15px] text-ink-quaternary">
            Every value has arrived. Both structures are asked for p{percentile} below.
          </span>
        ) : (
          <>
            <span className="text-[15px] text-ink-tertiary">next value</span>
            <span className="rounded px-3 py-1 font-mono text-[19px] font-bold text-black"
              style={{ background: CELL_COLOR[next.cell] }}>
              {next.value.toFixed(1)} ms
            </span>
            <span className="text-[15px] text-ink-quaternary">
              → kept on the left, and routed to cell {next.cell} on the right
            </span>
          </>
        )}
      </div>

      <div className="mb-4 grid grid-cols-[1fr_1fr] gap-4">
        <Panel label="EXACT" hint={`${state.arrived.length} values retained, in arrival order`}>
          <ArrivalStrip state={state} />
        </Panel>
        <Panel label="SKETCH" hint={`${CELLS} cells, each summarizing its own slice`}>
          <CellDigests state={state} />
        </Panel>
      </div>

      {showSummary && (
        <SummarizePhase state={state} percentile={percentile} pending={pendingAtSummarize}
          midRun />
      )}

      <SourceNote>
        Both structures are the pinned port of{" "}
        <code>rust/runtime/src/cellular/sketch.rs</code>, verified against the Rust through a
        golden fixture, and both ingest the way the runtime does. Adding a value is an append on
        either side: the exact path pushes onto an unsorted vector, and the digest pushes a
        weight-1 centroid onto the end of its own. Neither sorts as values land.{" "}
        <code>kernel.rs:117</code> sorts the retained values once, at summarize; the digest sorts
        and clusters only when its buffer exceeds <code>compress_threshold</code>, and its
        centroids are — in the words of the struct&apos;s own doc — &ldquo;sorted by mean after
        every compress&rdquo; while holding &ldquo;transient unsorted weight-1 centroids between
        compressions&rdquo;.
        <br />
        <span className="text-ink-quaternary">
          Two knobs are turned down so the structures fit on screen, and nothing else differs. δ is{" "}
          {COMPRESSION} rather than production&apos;s 100, so a digest settles at a handful of
          centroids instead of about fifty. The compress threshold is {THRESHOLD} rather than the{" "}
          <code>max(64, δ × 10)</code> the Rust derives — 1000 at δ=100, which would mean a
          thousand raw centroids accumulating before anything happened.
        </span>
      </SourceNote>
    </div>
  );
}

/**
 * The retained values in arrival order — unsorted, because nothing sorts them yet.
 *
 * `kernel.rs:117` sorts once at summarize time. During the run this is a plain append-only
 * vector, which is why keeping every value is cheap per record and expensive only in memory.
 */
function ArrivalStrip({ state }: { state: IngestState }): React.JSX.Element {
  return (
    <div>
      {state.arrived.length > STRIP_WINDOW && (
        <div className="mb-1 text-[13px]" style={{ color: DIM }}>
          + {(state.arrived.length - STRIP_WINDOW).toLocaleString()} earlier values, still held,
          not drawn
        </div>
      )}
      <div className="flex flex-wrap gap-1">
        {state.arrived.slice(-STRIP_WINDOW).map((value, i, window) => {
          const fresh = i === window.length - 1;
          return (
            <span key={i} className="rounded px-1.5 py-0.5 font-mono text-[12px] tabular-nums"
              style={{
                background: fresh ? GREEN : "rgba(255,255,255,0.06)",
                color: fresh ? "black" : "var(--color-ink-secondary)",
                fontWeight: fresh ? 700 : 400,
              }}>
              {value.toFixed(0)}
            </span>
          );
        })}
        {state.arrived.length === 0 && (
          <span className="text-[15px] text-ink-quaternary">Empty. Play, or step one value in.</span>
        )}
      </div>
      <p className="mt-3 text-[14px] leading-relaxed text-ink-quaternary">
        Appended, not inserted — these are in the order they arrived and nothing has sorted them.
        Per record the exact path costs one push. Its expense is memory, and one sort at the end.
      </p>
    </div>
  );
}

/**
 * Per-cell centroids: the settled, clustered prefix and the raw appended tail behind it.
 *
 * The tail is the part an earlier version of this page hid. Values land there as weight-1
 * centroids in arrival order, unsorted against anything, and stay that way until the count crosses
 * the threshold — at which point the whole buffer is sorted and clustered in one go.
 */
function CellDigests({ state }: { state: IngestState }): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2.5">
      {state.cells.map((cell, index) => {
        const justCompressed = state.compressedCells.includes(index);
        const settled = state.settled[index] ?? 0;
        const centroids = cell.centroids;
        const pending = centroids.length - settled;
        const maxWeight = Math.max(1, ...centroids.map((c) => c.weight));
        const headroom = cell.compressThreshold - centroids.length + 1;
        return (
          <div key={index} className="flex items-center gap-3">
            <span className="w-14 shrink-0 font-mono text-[13px]" style={{ color: CELL_COLOR[index] }}>
              cell {index}
            </span>
            <span className="flex flex-wrap items-end gap-[3px]">
              {centroids.map((c, i) => {
                const isPending = i >= settled;
                return (
                  <span key={i} className="flex flex-col items-center justify-end"
                    title={`${isPending ? "pending — " : ""}mean ${c.mean.toFixed(1)}, weight ${c.weight}`}>
                    <span className="rounded-t-[2px]"
                      style={{
                        width: 15,
                        height: 6 + (c.weight / maxWeight) * 24,
                        background: isPending ? "transparent" : CELL_COLOR[index],
                        outline: isPending ? `1px dashed ${CELL_COLOR[index]}` : "none",
                        outlineOffset: -1,
                        opacity: isPending ? 0.7 : 0.4 + 0.5 * (c.weight / maxWeight),
                        transition: "height 250ms ease-out",
                      }} />
                    <span className="font-mono text-[10px]" style={{ color: DIM }}>{c.weight}</span>
                  </span>
                );
              })}
              {centroids.length === 0 && (
                <span className="text-[14px]" style={{ color: DIM }}>empty</span>
              )}
            </span>
            <span className="ml-auto shrink-0 text-right font-mono text-[13px] tabular-nums">
              {justCompressed ? (
                <span style={{ color: GREEN, fontWeight: 700 }}>
                  compressed {state.collapsedFrom} → {centroids.length}
                </span>
              ) : (
                <span style={{ color: DIM }}>
                  {settled} settled · {pending} pending
                </span>
              )}
              <span className="block text-[12px]" style={{ color: pending > 0 ? ORANGE : DIM }}>
                {headroom > 0 ? `${headroom} more before compress` : "at capacity — compressing every value"}
              </span>
            </span>
          </div>
        );
      })}
      <p className="mt-1 text-[14px] leading-relaxed text-ink-quaternary">
        Solid bars are settled centroids — sorted and clustered by the last compress, height is the
        weight they carry. Dashed bars are the pending tail: raw weight-1 arrivals, in arrival
        order, unsorted. Nothing clusters until the buffer exceeds{" "}
        <strong>{THRESHOLD}</strong>, and then the whole thing collapses at once.
      </p>
    </div>
  );
}

/**
 * The summarize phase, staged so each operation is watchable.
 *
 * Stages: pool the cells' centroids, compress them, then walk the result one centroid at a time
 * until the running centre quantile passes the target. The walk is the part worth animating —
 * where it stops is not obvious until you see the centres go by.
 */
const STAGE_SORT = 0;
const STAGE_CONCAT = 1;
const STAGE_COMPRESS = 2;
const STAGE_WALK = 3;

function SummarizePhase({
  state,
  percentile,
  pending,
  midRun,
}: {
  state: IngestState;
  percentile: number;
  /** Centroids sitting in unsorted tails when the report was asked for. */
  pending: number;
  /** True when the user asked mid-run rather than at the end. */
  midRun: boolean;
}): React.JSX.Element {
  // The sort happens here, at summarize — not during the run.
  const sorted = useMemo(() => sortedAtSummarize(state), [state]);
  const exact = traceExact(sorted, percentile);
  const fold = useMemo(() => traceFold(state.cells, COMPRESSION), [state.cells]);
  const sketch = traceSketch(fold.folded, percentile / 100);

  const walkLength = sketch?.steps.length ?? 0;
  const lastStage = STAGE_WALK + walkLength;
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(true);

  // Restart whenever the question or the data changes, so the walk is always watched from the
  // beginning — asking mid-run twice at different points should replay, not resume.
  useEffect(() => {
    setStage(0);
    setPlaying(true);
  }, [percentile, state]);

  useEffect(() => {
    if (!playing) return undefined;
    const handle = window.setInterval(() => {
      setStage((s) => {
        if (s >= lastStage) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, stage === STAGE_SORT ? 1100 : 550);
    return () => window.clearInterval(handle);
  }, [playing, lastStage, stage]);

  if (exact === null || sketch === null) return <></>;

  const visited = Math.max(0, Math.min(walkLength, stage - STAGE_WALK + 1));
  const done = stage >= lastStage;
  const errorPct = ((sketch.result - exact.result) / exact.result) * 100;

  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-4 rounded-lg border px-5 py-3 text-base"
        style={{ borderColor: GREEN, background: "rgba(0,255,128,0.04)" }}>
        <strong style={{ color: GREEN }}>Now summarize.</strong>
        <span>
          Both are asked for p{percentile} over the {state.arrived.length} values that have arrived
          so far. The left <strong>sorts</strong> them — its first and only expensive step — then
          reads two. The right has no values left to sort, so it folds its cells and walks the
          result.
        </span>
        <span className="ml-auto flex items-center gap-1.5">
          <Toggle active onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Replay"}</Toggle>
          <Toggle onClick={() => { setPlaying(false); setStage((s) => Math.min(lastStage, s + 1)); }}>
            Step
          </Toggle>
          <Toggle onClick={() => { setStage(0); setPlaying(true); }}>Restart</Toggle>
        </span>
      </div>

      {midRun && (
        <div className="mb-3 rounded-lg border px-5 py-3 text-[15px] leading-relaxed"
          style={{ borderColor: ORANGE, background: "rgba(255,140,0,0.05)" }}>
          <strong style={{ color: ORANGE }}>Asked mid-run.</strong>{" "}
          A report can be requested at any point, and neither structure was ready for it.{" "}
          {pending > 0 ? (
            <>
              {pending} centroid{pending === 1 ? "" : "s"} were still sitting in unsorted tails; the
              fold sorts and clusters them before it can answer, because <code>clustered</code>{" "}
              sorts by mean first. That flush is work the run had been deferring, and asking early
              is what pays for it.
            </>
          ) : (
            <>Every cell happened to be freshly compressed, so nothing had to be flushed first.</>
          )}{" "}
          Keep going and ask again. Early on the answer swings about; it steadies once the digest
          has enough to work with, at whatever accuracy δ allows.
        </div>
      )}

      <div className="grid grid-cols-[1fr_1.3fr] gap-4">
        <Panel label="EXACT — sort, then interpolate" hint="sorts once, then reads two values">
          <SortAnimation arrived={state.arrived} sortedYet={stage > STAGE_SORT} />
          {stage > STAGE_SORT && (
            <ExactStrip trace={exact} sorted={sorted} reveal={done} />
          )}
        </Panel>

        <Panel label="SKETCH — fold, then walk" hint="no values left; only centroids">
          <FoldStages fold={fold} stage={stage} />
          {stage >= STAGE_WALK && (
            <WalkChain trace={sketch} visited={visited} percentile={percentile} />
          )}
          {done && (
            <div className="mt-3 flex flex-wrap items-baseline gap-x-3 border-t border-white/10 pt-2.5">
              <span className="text-[13px] font-bold tracking-widest text-ink-secondary">RESULT</span>
              <span className="font-mono text-[21px] font-bold" style={{ color: CYAN }}>
                {sketch.result.toFixed(2)} ms
              </span>
              <span className="ml-auto font-mono text-[16px]"
                style={{ color: Math.abs(errorPct) > 1 ? ORANGE : DIM }}>
                {errorPct >= 0 ? "+" : ""}{errorPct.toFixed(3)}% vs exact
              </span>
            </div>
          )}
          <p className="mt-2 text-[14px] leading-relaxed text-ink-quaternary">
            Neither endpoint of that blend is a value that occurred — both are cluster means. That
            is the whole difference: the exact path interpolates between two measurements, and this
            one interpolates between two summaries.
          </p>
          <p className="mt-2 text-[14px] leading-relaxed" style={{ color: ORANGE }}>
            That percentage is set by δ, not by how much data has arrived. δ is {COMPRESSION} here
            rather than production&apos;s 100, and more values do not rescue it — measured on this
            stream at δ&nbsp;={COMPRESSION} the p90 error runs 2.7%, −0.6%, 5.6%, 1.7% at 30, 300,
            3,000 and 30,000 values, wandering rather than shrinking. At δ&nbsp;=&nbsp;100 the same
            code gives 4.3%, 0.20%, 0.098%, 0.11% — it settles near a tenth of a percent once a few
            hundred values are in, and stays there. The budget sets the floor; the sample size only
            decides how quickly you reach it.
          </p>
        </Panel>
      </div>
    </>
  );
}

/**
 * The sort, animated.
 *
 * Every retained value is a bar. It starts where it arrived and slides to where it belongs, which
 * is the exact path's one expensive operation made visible — nothing maintained this order during
 * the run, and `kernel.rs:117` does it in a single pass when a percentile is finally asked for.
 *
 * Position is animated through a `transform` on each bar's group rather than the `x` attribute,
 * because transform transitions are reliable across engines where geometry-attribute transitions
 * are not.
 */
function SortAnimation({
  arrived,
  sortedYet,
}: {
  arrived: readonly number[];
  sortedYet: boolean;
}): React.JSX.Element {
  const width = 620;
  const height = 96;
  const pad = 6;
  // Beyond a few hundred the bars are thinner than a pixel and the animation says nothing, so an
  // evenly-spaced sample of arrival order stands in. Sampling by position preserves both the
  // distribution and the disorder, and the label says it is a sample.
  const sampled = useMemo(() => {
    if (arrived.length <= SORT_WINDOW) return [...arrived];
    const stride = arrived.length / SORT_WINDOW;
    return Array.from({ length: SORT_WINDOW }, (_, i) => arrived[Math.floor(i * stride)]!);
  }, [arrived]);
  const isSample = sampled.length < arrived.length;
  const step = (width - pad * 2) / Math.max(1, sampled.length);
  const max = Math.max(...sampled, 1);

  // Destination slot for each arrival. Duplicate values are resolved by consuming ranks in order,
  // so no two bars are ever assigned the same slot.
  const destination = useMemo(() => {
    const order = sampled.map((value, index) => ({ value, index }));
    order.sort((a, b) => a.value - b.value || a.index - b.index);
    const slots = new Array<number>(sampled.length);
    order.forEach((entry, rank) => {
      slots[entry.index] = rank;
    });
    return slots;
  }, [sampled]);

  return (
    <div className="mb-3">
      <div className="mb-1 text-[13px]" style={{ color: sortedYet ? DIM : GREEN }}>
        0 · sort the {arrived.length.toLocaleString()} retained values —{" "}
        {sortedYet ? "done, once" : "in arrival order, as they were kept"}
        {isSample && (
          <span style={{ color: DIM }}> · drawing an evenly-spaced {sampled.length} of them</span>
        )}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
        role="img" aria-label="retained values sorting into order">
        {sampled.map((value, i) => {
          const slot = sortedYet ? (destination[i] ?? i) : i;
          const barH = Math.max(2, (value / max) * (height - 20));
          return (
            <g key={i} style={{ transition: "transform 900ms cubic-bezier(0.4, 0, 0.2, 1)" }}
              transform={`translate(${pad + slot * step}, 0)`}>
              <rect x={0.5} y={height - 12 - barH} width={Math.max(1.5, step - 1)} height={barH}
                rx={1} fill={sortedYet ? "rgba(255,255,255,0.28)" : GREEN} opacity={0.85}
                style={{ transition: "fill 900ms ease-out" }} />
            </g>
          );
        })}
        <line x1={pad} x2={width - pad} y1={height - 10} y2={height - 10}
          stroke="rgba(255,255,255,0.12)" />
      </svg>
    </div>
  );
}

/**
 * A window of the sorted values around the virtual index.
 *
 * Drawing all sixty positions puts the two brackets a few pixels apart with their labels on top of
 * each other — the first version did exactly that and was unreadable. The computation only ever
 * touches two neighbours, so the strip zooms to a handful either side: enough context to see it is
 * a sorted array, wide enough to label the pair it actually reads.
 */
function ExactStrip({
  trace,
  sorted,
  reveal,
}: {
  trace: ExactTrace;
  sorted: readonly number[];
  reveal: boolean;
}): React.JSX.Element {
  const width = 620;
  const height = 168;
  const pad = 14;
  const context = 4;
  const from = Math.max(0, trace.lo - context);
  const to = Math.min(sorted.length - 1, trace.hi + context);
  const shown = to - from + 1;
  const step = (width - pad * 2) / shown;
  const max = Math.max(...sorted.slice(from, to + 1)) * 1.25;
  const x = (i: number) => pad + (i - from) * step;
  const barH = (v: number) => Math.max(3, (v / max) * (height - 74));
  const markerX = pad + (trace.virtualIndex - from + 0.5) * step;

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
        role="img" aria-label="sorted values around the virtual index">
        {Array.from({ length: shown }, (_, n) => {
          const i = from + n;
          const value = sorted[i]!;
          const bracket = i === trace.lo || i === trace.hi;
          return (
            <g key={i}>
              <rect x={x(i) + 1} y={height - 40 - barH(value)} width={Math.max(4, step - 3)}
                height={barH(value)} rx={2}
                fill={bracket ? GREEN : "rgba(255,255,255,0.14)"} opacity={bracket ? 0.9 : 1} />
              {bracket && (
                <text x={x(i) + step / 2} y={height - 46 - barH(value)} fontSize={13}
                  textAnchor="middle" fill={GREEN} fontFamily="var(--font-mono, monospace)">
                  {value.toFixed(2)}
                </text>
              )}
              <text x={x(i) + step / 2} y={height - 24} fontSize={11} textAnchor="middle"
                fill={bracket ? GREEN : DIM} fontFamily="var(--font-mono, monospace)">
                {i}
              </text>
            </g>
          );
        })}

        <line x1={pad} x2={width - pad} y1={height - 38} y2={height - 38}
          stroke="rgba(255,255,255,0.12)" />

        <line x1={markerX} x2={markerX} y1={20} y2={height - 34} stroke={ORANGE} strokeWidth={2}
          style={{ transition: "all 450ms ease-out" }} />
        <polygon points={`${markerX - 5},20 ${markerX + 5},20 ${markerX},27`} fill={ORANGE} />
        <text x={markerX} y={15} fontSize={12} textAnchor="middle" fill={ORANGE}
          fontFamily="var(--font-mono, monospace)">
          index {trace.virtualIndex.toFixed(2)}
        </text>

        <text x={width / 2} y={height - 6} fontSize={11} textAnchor="middle" fill={DIM}>
          {shown} of {trace.count} retained values · bar height is the value
        </text>
      </svg>

      <div className="mt-1 flex flex-col gap-1 font-mono text-[14px]">
        <Line label="virtual index"
          value={`${trace.percentile}/100 × (${trace.count} − 1) = ${trace.virtualIndex.toFixed(3)}`} />
        <Line label="brackets"
          value={`sorted[${trace.lo}] = ${trace.loValue.toFixed(2)} · sorted[${trace.hi}] = ${trace.hiValue.toFixed(2)}`} />
        <Line label="blend" value={`${trace.frac.toFixed(3)} of the way between them`} />
      </div>
      {reveal && (
        <div className="mt-3 flex flex-wrap items-baseline gap-2 border-t border-white/10 pt-2.5">
          <span className="text-[13px] font-bold tracking-widest text-ink-secondary">RESULT</span>
          <span className="font-mono text-[21px] font-bold">{trace.result.toFixed(2)} ms</span>
          <span className="text-[14px] text-ink-quaternary">
            — exact, because both endpoints genuinely occurred
          </span>
        </div>
      )}
    </div>
  );
}

/**
 * The fold, as its two operations: pool the centroids, then cluster them.
 *
 * Blocks are sized by weight, not drawn uniformly. A pooled buffer is mostly light centroids with
 * a few heavy ones, and rendering them all the same width says the opposite — that the fold is
 * combining equal things. It is combining summaries of wildly different populations, and the
 * compressed row is visibly made of fewer, fatter blocks covering the same total.
 */
function FoldStages({
  fold,
  stage,
}: {
  fold: ReturnType<typeof traceFold>;
  stage: number;
}): React.JSX.Element {
  const pooled = stage >= STAGE_CONCAT;
  const compressed = stage >= STAGE_COMPRESS;
  const pooledTotal = fold.concatenated.reduce((n, c) => n + c.weight, 0) || 1;
  const foldedTotal = fold.folded.centroids.reduce((n, c) => n + c.weight, 0) || 1;

  return (
    <div className="mb-3">
      <div className="mb-1 text-[13px]" style={{ color: stage === STAGE_CONCAT ? GREEN : DIM }}>
        1 · pool every cell&apos;s centroids — <strong>{fold.concatenated.length}</strong> of them,
        width is the weight each carries
      </div>
      <div className="flex h-5 w-full gap-[2px]">
        {fold.contributed.map((c) =>
          c.centroids.map((centroid, i) => (
            <span key={`${c.cell}-${i}`} className="h-5 rounded-[2px]"
              style={{
                flexGrow: centroid.weight,
                flexBasis: 0,
                minWidth: 2,
                background: CELL_COLOR[c.cell],
                opacity: !pooled ? 0.15 : compressed ? 0.22 : 0.85,
                transition: "opacity 400ms ease-out",
              }}
              title={`cell ${c.cell}: mean ${centroid.mean.toFixed(1)}, weight ${centroid.weight} of ${pooledTotal}`} />
          )),
        )}
      </div>

      <div className="mb-1 mt-2 text-[13px]" style={{ color: stage === STAGE_COMPRESS ? GREEN : DIM }}>
        2 · compress — <strong>{compressed ? fold.folded.centroids.length : "…"}</strong>{" "}
        {compressed ? "remain, same total weight" : "pending"}
      </div>
      <div className="flex h-5 w-full gap-[2px]">
        {compressed &&
          fold.folded.centroids.map((c, i) => (
            <span key={i} className="flex h-5 items-center justify-center rounded-[2px]"
              style={{
                flexGrow: c.weight,
                flexBasis: 0,
                minWidth: 2,
                background: GREEN,
                opacity: 0.8,
              }}
              title={`mean ${c.mean.toFixed(1)}, weight ${c.weight} of ${foldedTotal}`}>
              <span className="font-mono text-[10px] text-black">{c.weight}</span>
            </span>
          ))}
      </div>
    </div>
  );
}

/**
 * The walk, drawn on the quantile axis.
 *
 * Every centroid is a block whose width is its share of the total weight, so the strip spans
 * exactly q0 to q1 and each block's midpoint is the centre quantile the walk compares against.
 * The target line is where it is heading; the walk stops on the first centre past it, and the
 * answer is a blend of the two centres either side. Seeing the centres go by is what makes the
 * stopping rule obvious.
 */
function WalkChain({
  trace,
  visited,
  percentile,
}: {
  trace: NonNullable<ReturnType<typeof traceSketch>>;
  visited: number;
  percentile: number;
}): React.JSX.Element {
  const width = 760;
  const height = 128;
  const pad = 10;
  const inner = width - pad * 2;
  const total = trace.totalWeight;
  const x = (q: number) => pad + q * inner;
  const targetX = x(trace.quantile);
  const active = trace.steps[Math.max(0, visited - 1)];

  return (
    <div>
      <div className="mb-1 text-[13px]" style={{ color: GREEN }}>
        3 · walk until a centroid&apos;s centre passes q = {(percentile / 100).toFixed(2)}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
        role="img" aria-label="centroid chain along the quantile axis with the walk marker">
        {trace.steps.map((step, i) => {
          const q0 = step.cumulativeBefore / total;
          const q1 = (step.cumulativeBefore + step.centroid.weight) / total;
          const seen = i < visited;
          const isStop = step.stopped && seen;
          return (
            <g key={i}>
              <rect x={x(q0) + 0.5} y={44} width={Math.max(2, x(q1) - x(q0) - 1)} height={30} rx={2}
                fill={isStop ? GREEN : seen ? CYAN : "rgba(255,255,255,0.07)"}
                opacity={isStop ? 0.85 : seen ? 0.55 : 1}
                style={{ transition: "fill 300ms ease-out, opacity 300ms ease-out" }} />
              {seen && (
                <line x1={x(step.centerQ)} x2={x(step.centerQ)} y1={40} y2={78}
                  stroke={isStop ? GREEN : "var(--color-ink-secondary)"} strokeWidth={1.5} />
              )}
              {seen && x(q1) - x(q0) > 26 && (
                <text x={(x(q0) + x(q1)) / 2} y={64} fontSize={11} textAnchor="middle"
                  fill="black" fontFamily="var(--font-mono, monospace)">
                  {step.centroid.weight}
                </text>
              )}
            </g>
          );
        })}

        <line x1={targetX} x2={targetX} y1={16} y2={94} stroke={ORANGE} strokeWidth={2} />
        <polygon points={`${targetX - 5},16 ${targetX + 5},16 ${targetX},23`} fill={ORANGE} />
        <text x={targetX} y={12} fontSize={12} textAnchor="middle" fill={ORANGE}
          fontFamily="var(--font-mono, monospace)">
          p{percentile}
        </text>

        {active !== undefined && (
          <text x={x(active.centerQ)} y={92} fontSize={12} textAnchor="middle"
            fill={active.stopped ? GREEN : "var(--color-ink-secondary)"}
            fontFamily="var(--font-mono, monospace)">
            centre {active.centerQ.toFixed(4)}
          </text>
        )}

        <text x={pad} y={height - 6} fontSize={11} fill={DIM}>q0</text>
        <text x={width - pad} y={height - 6} fontSize={11} textAnchor="end" fill={DIM}>q1</text>
        <text x={width / 2} y={height - 6} fontSize={11} textAnchor="middle" fill={DIM}>
          block width = share of the run · tick = its centre quantile
        </text>

        {visited >= trace.steps.length && !trace.anchored && (
          <line x1={x(trace.fromQ)} x2={x(trace.toQ)} y1={36} y2={36}
            stroke={CYAN} strokeWidth={2.5} strokeLinecap="round" />
        )}
      </svg>
      {visited >= trace.steps.length && !trace.anchored && (
        <div className="font-mono text-[13px]" style={{ color: DIM }}>
          blend between ({trace.fromQ.toFixed(4)}, {trace.fromValue.toFixed(1)} ms) and (
          {trace.toQ.toFixed(4)}, {trace.toValue.toFixed(1)} ms)
        </div>
      )}
    </div>
  );
}

function Line({ label, value }: { label: string; value: string }): React.JSX.Element {
  return (
    <div className="flex gap-3">
      <span className="w-32 shrink-0" style={{ color: DIM }}>{label}</span>
      <span className="text-ink-secondary">{value}</span>
    </div>
  );
}
