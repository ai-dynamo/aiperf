/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — watching both algorithms take the same values.
//!
//! Values arrive one at a time. On the left they are kept; on the right they are absorbed into
//! centroids. Then both are asked the same question, and the page shows each answering it —
//! reading two retained values, or walking a chain of centroids.

import { useEffect, useMemo, useState } from "react";
import {
  arrivals,
  createIngest,
  ingestOne,
  traceExact,
  traceFold,
  traceSketch,
  type IngestState,
} from "./ingest.js";
import { latencySamples } from "./sketchSim.js";
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
const TOTAL = 60;
const SPEEDS = [700, 350, 120] as const;
const CELL_COLOR = [CYAN, PURPLE, ORANGE];

export function SketchLiveSpike(): React.JSX.Element {
  const [step, setStep] = useState(0);
  const [running, setRunning] = useState(false);
  const [interval, setIntervalMs] = useState<number>(350);
  const [percentile, setPercentile] = useState(90);

  const values = useMemo(() => latencySamples(TOTAL, 5), []);
  const feed = useMemo(() => arrivals(values, CELLS), [values]);

  const state = useMemo(() => {
    let s = createIngest(CELLS, COMPRESSION);
    for (let i = 0; i < step; i++) s = ingestOne(s, feed[i]!);
    return s;
  }, [feed, step]);

  useEffect(() => {
    if (!running) return undefined;
    const handle = window.setInterval(() => {
      setStep((s) => {
        if (s >= TOTAL) {
          setRunning(false);
          return s;
        }
        return s + 1;
      });
    }, interval);
    return () => window.clearInterval(handle);
  }, [running, interval]);

  const next = step < TOTAL ? feed[step] : undefined;
  const complete = step >= TOTAL;

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
          <Toggle active onClick={() => setRunning((r) => !r)}>{running ? "Pause" : "Play"}</Toggle>
          <Toggle onClick={() => { setRunning(false); setStep((s) => Math.min(TOTAL, s + 1)); }}>
            Next value
          </Toggle>
          <Toggle onClick={() => { setRunning(false); setStep(0); }}>Reset</Toggle>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">speed</span>
          {SPEEDS.map((ms, i) => (
            <Toggle key={ms} active={interval === ms} onClick={() => setIntervalMs(ms)}>
              {["slow", "medium", "fast"][i]}
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
          <Readout label="arrived" value={`${step} / ${TOTAL}`} />
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
        <Panel label="EXACT" hint={`${state.sorted.length} values retained, sorted`}>
          <SortedStrip state={state} />
        </Panel>
        <Panel label="SKETCH" hint={`${CELLS} cells, each summarizing its own slice`}>
          <CellDigests state={state} />
        </Panel>
      </div>

      {complete && (
        <SummarizePhase state={state} percentile={percentile} />
      )}

      <SourceNote>
        Both structures here are the pinned port of{" "}
        <code>rust/runtime/src/cellular/sketch.rs</code>, verified against the Rust through a
        golden fixture. Two presentational departures, stated because they matter: δ is{" "}
        {COMPRESSION} rather than production&apos;s 100, so a digest settles at a handful of
        centroids you can follow instead of about fifty; and the digest is compressed on every
        arrival where the runtime batches until a threshold. Clustering sorts first, so both reach
        the same digest — a test asserts exactly that — but compressing per value shows the
        absorption happening rather than hiding it inside one bulk step.
      </SourceNote>
    </div>
  );
}

/** The retained values, in sorted order, with the newest arrival highlighted where it landed. */
function SortedStrip({ state }: { state: IngestState }): React.JSX.Element {
  return (
    <div>
      <div className="flex flex-wrap gap-1">
        {state.sorted.map((value, i) => {
          const fresh = i === state.lastSortedIndex;
          return (
            <span key={`${i}-${value}`}
              className="rounded px-1.5 py-0.5 font-mono text-[12px] tabular-nums"
              style={{
                background: fresh ? GREEN : "rgba(255,255,255,0.06)",
                color: fresh ? "black" : "var(--color-ink-secondary)",
                fontWeight: fresh ? 700 : 400,
              }}>
              {value.toFixed(0)}
            </span>
          );
        })}
        {state.sorted.length === 0 && (
          <span className="text-[15px] text-ink-quaternary">Empty. Play, or step one value in.</span>
        )}
      </div>
      <p className="mt-3 text-[14px] leading-relaxed text-ink-quaternary">
        Every arrival is inserted at its sorted position and kept forever. The structure grows
        exactly as fast as the run does — which is what makes any percentile answerable later, and
        what makes it impossible to merge from another machine without shipping all of it.
      </p>
    </div>
  );
}

/** Per-cell centroids, sized by weight, with an absorption called out when it happens. */
function CellDigests({ state }: { state: IngestState }): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2">
      {state.cells.map((cell, index) => {
        const absorbed = state.compressedCells.includes(index);
        const centroids = cell.centroids;
        const maxWeight = Math.max(1, ...centroids.map((c) => c.weight));
        return (
          <div key={index} className="flex items-center gap-3">
            <span className="w-14 shrink-0 font-mono text-[13px]" style={{ color: CELL_COLOR[index] }}>
              cell {index}
            </span>
            <span className="flex flex-wrap items-end gap-[3px]">
              {centroids.map((c, i) => (
                <span key={i} className="flex flex-col items-center justify-end"
                  title={`mean ${c.mean.toFixed(1)} ms, weight ${c.weight}`}>
                  <span className="rounded-t-[2px]"
                    style={{
                      width: 16,
                      height: 6 + (c.weight / maxWeight) * 26,
                      background: CELL_COLOR[index],
                      opacity: 0.35 + 0.5 * (c.weight / maxWeight),
                    }} />
                  <span className="font-mono text-[10px]" style={{ color: DIM }}>{c.weight}</span>
                </span>
              ))}
              {centroids.length === 0 && (
                <span className="text-[14px]" style={{ color: DIM }}>empty</span>
              )}
            </span>
            <span className="ml-auto shrink-0 text-right font-mono text-[13px] tabular-nums"
              style={{ color: absorbed ? GREEN : DIM }}>
              {absorbed ? "absorbed — no new centroid" : `${centroids.length} centroids`}
              <span className="block text-[12px]" style={{ color: DIM }}>
                {cell.totalWeight} values
              </span>
            </span>
          </div>
        );
      })}
      <p className="mt-1 text-[14px] leading-relaxed text-ink-quaternary">
        Bar height is a centroid&apos;s weight — how many values it stands for. A new arrival enters
        as weight 1; when the K1 rule allows, it is folded into its neighbour and the two become one
        mean. From then on the individual value cannot be recovered.
      </p>
    </div>
  );
}

/** Both answers to the same question, each showing what it read to get there. */
function SummarizePhase({
  state,
  percentile,
}: {
  state: IngestState;
  percentile: number;
}): React.JSX.Element {
  const exact = traceExact(state.sorted, percentile);
  const fold = useMemo(() => traceFold(state.cells, COMPRESSION), [state.cells]);
  const sketch = traceSketch(fold.folded, percentile / 100);
  if (exact === null || sketch === null) return <></>;

  const errorPct = ((sketch.result - exact.result) / exact.result) * 100;

  return (
    <>
      <div className="mb-3 rounded-lg border px-5 py-3 text-base"
        style={{ borderColor: GREEN, background: "rgba(0,255,128,0.04)" }}>
        <strong style={{ color: GREEN }}>Now summarize.</strong>{" "}
        Both are asked for p{percentile}. The left reads two of the values it kept. The right has
        no values left to read, so it folds its cells and walks the result.
      </div>

      <div className="grid grid-cols-[1fr_1.25fr] gap-4">
        <Panel label="EXACT — type-7 interpolation" hint="reads two retained values">
          <div className="flex flex-col gap-1.5 font-mono text-[14px]">
            <Line label="virtual index" value={`${percentile}/100 × (${exact.count} − 1) = ${exact.virtualIndex.toFixed(3)}`} />
            <Line label="brackets" value={`sorted[${exact.lo}] and sorted[${exact.hi}]`} />
            <Line label="values" value={`${exact.loValue.toFixed(2)} and ${exact.hiValue.toFixed(2)}`} />
            <Line label="fraction" value={exact.frac.toFixed(3)} />
          </div>
          <div className="mt-3 flex items-baseline gap-2 border-t border-white/10 pt-2.5">
            <span className="text-[13px] font-bold tracking-widest text-ink-secondary">RESULT</span>
            <span className="font-mono text-[21px] font-bold">{exact.result.toFixed(2)} ms</span>
          </div>
          <p className="mt-2 text-[14px] leading-relaxed text-ink-quaternary">
            Two array reads and a blend. Exact because both endpoints are values that genuinely
            occurred.
          </p>
        </Panel>

        <Panel label="SKETCH — fold, then walk" hint="no values left; only centroids">
          <div className="mb-3">
            <div className="mb-1 text-[13px] text-ink-tertiary">
              1 · concatenate every cell&apos;s centroids —{" "}
              <strong>{fold.concatenated.length}</strong> of them
            </div>
            <div className="flex flex-wrap gap-[3px]">
              {fold.contributed.map((c) =>
                c.centroids.map((centroid, i) => (
                  <span key={`${c.cell}-${i}`} className="h-3.5 rounded-[2px]"
                    style={{ width: 10, background: CELL_COLOR[c.cell], opacity: 0.7 }}
                    title={`cell ${c.cell}: mean ${centroid.mean.toFixed(1)}, weight ${centroid.weight}`} />
                )),
              )}
            </div>
            <div className="mb-1 mt-2 text-[13px] text-ink-tertiary">
              2 · compress — <strong>{fold.folded.centroids.length}</strong> remain
            </div>
            <div className="flex flex-wrap gap-[3px]">
              {fold.folded.centroids.map((c, i) => (
                <span key={i} className="h-3.5 rounded-[2px]"
                  style={{ width: 10, background: GREEN, opacity: 0.75 }}
                  title={`mean ${c.mean.toFixed(1)}, weight ${c.weight}`} />
              ))}
            </div>
          </div>

          <div className="mb-1 text-[13px] text-ink-tertiary">
            3 · walk until the running centre passes q = {(percentile / 100).toFixed(2)}
          </div>
          <div className="font-mono text-[13px] leading-[1.75]">
            {sketch.steps.map((s, i) => (
              <div key={i} className="flex items-baseline gap-4 whitespace-nowrap"
                style={{ color: s.stopped ? GREEN : DIM, fontWeight: s.stopped ? 700 : 400 }}>
                <span className="w-7">#{i}</span>
                <span className="w-32">mean {s.centroid.mean.toFixed(1)}</span>
                <span className="w-14">w {s.centroid.weight}</span>
                <span className="w-36">centre q {s.centerQ.toFixed(4)}</span>
                <span>{s.stopped ? "◄ passed q — stop here" : ""}</span>
              </div>
            ))}
          </div>

          <div className="mt-3 flex flex-wrap items-baseline gap-x-3 gap-y-1 border-t border-white/10 pt-2.5">
            <span className="text-[13px] font-bold tracking-widest text-ink-secondary">RESULT</span>
            <span className="font-mono text-[21px] font-bold" style={{ color: CYAN }}>
              {sketch.result.toFixed(2)} ms
            </span>
            <span className="font-mono text-[14px]" style={{ color: DIM }}>
              interpolated between ({sketch.fromQ.toFixed(4)}, {sketch.fromValue.toFixed(1)}) and (
              {sketch.toQ.toFixed(4)}, {sketch.toValue.toFixed(1)})
            </span>
            <span className="ml-auto font-mono text-[16px]"
              style={{ color: Math.abs(errorPct) > 1 ? ORANGE : DIM }}>
              {errorPct >= 0 ? "+" : ""}{errorPct.toFixed(3)}% vs exact
            </span>
          </div>
          <p className="mt-2 text-[14px] leading-relaxed text-ink-quaternary">
            Neither endpoint is a value that occurred — both are cluster means. That is the whole
            difference: the exact path interpolates between two measurements, and this one
            interpolates between two summaries.
          </p>
          <p className="mt-2 text-[14px] leading-relaxed" style={{ color: ORANGE }}>
            Do not read that percentage as how wrong a sketch is. {TOTAL} values across {CELLS}{" "}
            cells at δ&nbsp;={" "}{COMPRESSION} is a deliberately tiny configuration chosen so the
            structures fit on screen, and it is the worst case for accuracy: a handful of centroids
            each standing for several values. At production scale — thousands of records, δ&nbsp;=
            100 — the same code lands well under a percent, which the companion page measures.
          </p>
        </Panel>
      </div>
    </>
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
