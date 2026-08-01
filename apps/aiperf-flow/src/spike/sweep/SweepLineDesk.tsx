/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the sweep-line desk: every stage of the metrics plane, narrated, on one dataset.
//!
//! Fully synchronized in the strict sense: there is exactly one position, derived from narration
//! word progress by `useBeatClock`, and every panel is a pure function of it. No panel owns a
//! timer, so nothing can drift out of step with the voice or with another panel.

import { useMemo } from "react";
import { PresentationShell } from "../../shell/PresentationShell.js";
import { useNarratedDeck } from "../../audio/index.js";
import type { SlideDefinition } from "../../deck/types.js";
import { useBeatClock, type BeatAnchor } from "../useBeatClock.js";
import { buildScenario, TARGET_CONCURRENCY, toSeconds } from "./scenario.js";
import {
  ColumnStorePanel,
  ConsensusPanel,
  CumsumPanel,
  CurvePanel,
  CusumPanel,
  EventSortPanel,
  Mser5Panel,
  Panel,
  StationarityPanel,
} from "./panels.js";

const GREEN = "var(--color-category-green)";
const ORANGE = "var(--color-category-orange)";
const BLUE = "var(--color-category-blue)";
const PURPLE = "var(--color-category-purple)";

/** Which panels a beat shows, and how far through the run its playhead has travelled. */
type Stage =
  | "columns"
  | "ragged"
  | "events"
  | "collision"
  | "cumsum"
  | "icl"
  | "threshold"
  | "cusum"
  | "mser"
  | "consensus";

type Beat = BeatAnchor & { stage: Stage; title: string; lede: string; caption: string };

const BEATS: Beat[] = [
  {
    endAt: 0.1,
    stage: "columns",
    title: "Records land in columns, addressed by index",
    lede: "Every metric is its own array. A missing value is a NaN sentinel, never a shorter column.",
    caption: "metrics_core/store.rs — NumericColumn, absolute-request-index aligned.",
    narration:
      "Everything starts with storage, and the shape of the storage decides what the rest of the plane can do cheaply. Each record's fields are split apart into separate arrays — one for start time, one for end, one for input tokens, and so on. A record is not a struct sitting in a list; it is the same index into every column. That is the invariant the whole plane leans on. When a value is missing, the column stores a not-a-number sentinel rather than skipping the row, because skipping would shift every later index and break the alignment between columns.",
  },
  {
    endAt: 0.2,
    stage: "ragged",
    title: "Lists do not fit a column, so they get a CSR",
    lede: "Inter-chunk latencies are one variable-length list per record: flat values plus offsets.",
    caption: "store.rs — RaggedSeries; sweepline/kv_cache.rs — IclSeries.",
    narration:
      "One metric refuses to fit. Inter-chunk latency is not a number per record, it is a list — one entry for every gap between streamed chunks — and each request produces a different number of them. So it gets a compressed sparse row layout instead: one flat array holding every value back to back, plus a per-record offset and length saying which slice belongs to whom. Notice that a record with no chunks still takes an offset. It has to, because the ragged series must stay index-aligned with the numeric columns beside it.",
  },
  {
    endAt: 0.32,
    stage: "events",
    title: "Every interval becomes two signed events",
    lede: "Plus one at each start, minus one at each end. Nothing scans the timeline.",
    caption: "sweepline/mod.rs — concurrency_sweep_line pushes +1 and -1 per record.",
    narration:
      "Now the sweep line itself. To know how many requests were in flight at any moment, you could walk the timeline and count — and that costs the length of the timeline. Instead each interval is turned into two signed events: plus one when it starts, minus one when it ends. The timeline is never scanned at all. The cost becomes sorting the events, which is a function of how many requests there were, not of how long the run lasted or how fine its resolution is.",
  },
  {
    endAt: 0.44,
    stage: "collision",
    title: "Collisions decide by sign, not by luck",
    lede: "Sorted by (timestamp, delta) — so at an equal instant, ends land before starts.",
    caption: "sweepline/mod.rs — 'end deltas before positive deltas at equal timestamps'.",
    narration:
      "And here is the subtle part, the one that is wrong in most hand-rolled versions. Requests routinely share a timestamp — one ends at the exact nanosecond another begins. If you sort only by time, the order of those two events is arbitrary, and half the time you will process the start first and briefly report one more request in flight than ever existed. The fix is to sort by timestamp and then by delta. Negative sorts before positive, so an end always lands before a start at the same instant, and a phantom unit of concurrency becomes unrepresentable rather than merely unlikely.",
  },
  {
    endAt: 0.56,
    stage: "cumsum",
    title: "One cumulative sum builds the whole curve",
    lede: "Running total after each event. Residuals below 1e-9 of the peak snap to zero.",
    caption: "sweepline/mod.rs — sweep_line_cumsum, then snap_small_residuals.",
    narration:
      "With the events ordered, the curve is just a running total. Add each delta in turn and the value after every event is the concurrency held from that moment until the next one — a right-continuous step function, exact, with no sampling and no interpolation anywhere. There is one piece of housekeeping. Adding and subtracting thousands of floating-point weights leaves a curve that should be exactly zero sitting at ten to the minus thirteen. So residuals below a billionth of the peak are snapped to zero, which is what makes the statement the curve returns to zero true rather than nearly true.",
  },
  {
    endAt: 0.68,
    stage: "icl",
    title: "ICL awareness: tokens arrive when they arrived",
    lede: "Each chunk enters at generation_start + cumsum(icl), not all at once.",
    caption: "sweepline/kv_cache.rs — tokens_in_flight_sweep_line_icl.",
    narration:
      "The same machinery answers a harder question once the ragged data is used. Tokens in flight, done coarsely, drops every output token into the curve the instant generation starts — which is a lie, because they arrive one chunk at a time over the whole response. The inter-chunk latencies say exactly when. Running a cumulative sum along a single record's own chunk latencies places each arrival at generation start plus the elapsed gaps, and a chunk that would land past the record's end is clamped to it. The dashed line is the coarse version, the solid line the aware one. Same totals, and a completely different shape.",
  },
  {
    endAt: 0.79,
    stage: "threshold",
    title: "Steady state, by threshold crossing",
    lede: "Open at the first rise to ceil(0.8 × target), close at the last descent below it.",
    caption: "metrics_core/steady_state.rs — detect_steady_window.",
    narration:
      "Which brings us to what the curve is for. A concurrency-target run has a ramp while the generator fills its slots and a drain while the last requests finish, and summarizing across both biases throughput low and tail latency high. So the window is detected from this very curve. The threshold is a fraction of the configured target — eighty percent by default, rounded up — compared with a half-unit margin so that residual snapping can never flip a boundary. The window opens the first time the curve reaches it and closes at the last time it falls back below. Not the first fall: the last, so a momentary dip in the middle does not end the measurement early.",
  },
  {
    endAt: 0.87,
    stage: "cusum",
    title: "CUSUM, and a detector that degenerates",
    lede: "Time-weighted deviations from a p95 target. On any run with a drain, it gives up.",
    caption: "new-config-kube — analysis/ramp_detection.py:24, cusum_steady_state_window.",
    narration:
      "The threshold detector needs a target to exist. A separate suite detects steady state from the data alone, and its first estimator is a retrospective cumulative sum. It measures each level's deviation from a time-weighted ninety-fifth percentile, weights it by how long that level was held, and looks for the turning points. But watch what it actually does here. The target is a p95, so by construction almost none of the run sits above it, every deviation is negative, the forward sum only ever falls, and its minimum lands at the very end. The ordering check fails and the detector returns the entire run. That is not a porting error — the shipped implementation does exactly this on any run with a drain.",
  },
  {
    endAt: 0.95,
    stage: "mser",
    title: "MSER-5 does the real work",
    lede: "Batch by five, then delete the prefix minimizing variance over remaining count.",
    caption: "new-config-kube — ramp_detection.py:106, mser5_truncation_point.",
    narration:
      "The estimator that earns its place is MSER-5, a warm-up truncation rule borrowed from discrete-event simulation. Batch the series into groups of five and take their means. Then for every possible number of leading batches you might delete, compute the variance of what remains divided by how many remain — the squared standard error of the mean you would end up reporting. Choose the truncation that minimizes it. The guard that makes it honest is the half rule: it may never delete more than half the batches, so it cannot manufacture convergence by throwing away everything and reporting the standard error of two points.",
  },
  {
    endAt: 1,
    stage: "consensus",
    title: "Consensus, then a test that can veto it",
    lede: "Latest start, earliest end — then reject the window if it is still trending.",
    caption: "ramp_detection.py:248 consensus; stationarity.py:178 batch-means trend test.",
    narration:
      "No single estimator is trustworthy, which is exactly why there are several. They are reconciled by taking the latest start any of them proposes and the earliest end — the most conservative reading available, so one late-settling signal shortens the window rather than being outvoted. And then the answer is checked rather than trusted. The window's records are split into ten batches, their means rank-correlated against batch index, and if that correlation is strong the window is still drifting and gets rejected. A detector that cannot fail is a detector that cannot be believed.",
  },
];

const SLIDES: readonly SlideDefinition[] = BEATS.map((b, i) => ({
  id: `sweep-${i}`,
  eyebrow: `${String(i + 1).padStart(2, "0")} · SWEEP LINE`,
  title: b.title,
  lede: b.lede,
  narration: b.narration,
  caption: b.caption,
  nodes: [],
  edges: [],
}));

export function SweepLineDesk(): React.JSX.Element {
  const scenario = useMemo(() => buildScenario(1), []);

  const narrated = useNarratedDeck({
    narrations: BEATS.map((b) => b.narration),
    storagePrefix: "spike:sweep-desk",
  });

  // One position for the whole desk. Every panel below reads it; none owns a timer.
  const { position } = useBeatClock(BEATS, narrated.index, narrated.activeWordIndex, 1);
  const beat = BEATS[narrated.index] ?? BEATS[0]!;

  // Progress *within* the current beat, so a panel can fill during its own explanation.
  const from = BEATS[narrated.index - 1]?.endAt ?? 0;
  const local = Math.max(0, Math.min(1, (position - from) / Math.max(1e-6, beat.endAt - from)));

  const span = scenario.runEndNs - scenario.runStartNs;
  const headNs = scenario.runStartNs + local * span;
  const window = scenario.thresholdWindow;

  return (
    <PresentationShell
      slides={SLIDES}
      slideIndex={narrated.index}
      onSlideIndexChange={narrated.goTo}
      narrated={narrated}
      title="The sweep-line desk"
    >
      <div className="flex h-full min-h-0 flex-col gap-3 px-6 pt-3 pb-2">
        <div className="flex items-center gap-5 text-[11px] tabular-nums">
          <span><span className="text-ink-tertiary">records</span> <strong>{scenario.records.length}</strong></span>
          <span><span className="text-ink-tertiary">events</span> <strong>{scenario.sortedEvents.length}</strong></span>
          <span><span className="text-ink-tertiary">collisions</span>{" "}
            <strong style={{ color: ORANGE }}>{scenario.collisions.length}</strong></span>
          <span><span className="text-ink-tertiary">target</span> <strong>{TARGET_CONCURRENCY}</strong></span>
          <span><span className="text-ink-tertiary">peak</span> <strong>{window?.peakConcurrency ?? "—"}</strong></span>
          <span className="text-ink-quaternary">t = {toSeconds(headNs, scenario.runStartNs).toFixed(1)}s</span>
        </div>

        {(beat.stage === "columns" || beat.stage === "ragged") && (
          <Panel title="COLUMN STORE — one array per metric, addressed by absolute request index"
            hint="dashed = NaN sentinel · purple = ragged ICL slices">
            <ColumnStorePanel scenario={scenario} rows={local * scenario.store.rows} />
          </Panel>
        )}

        {(beat.stage === "events" || beat.stage === "collision") && (
          <div className="grid min-h-0 flex-1 grid-cols-[1fr_1fr] gap-3">
            <Panel title="EVENTS — sorted by (timestamp asc, delta asc)"
              hint={`${scenario.collisions.length} share a timestamp`}>
              <EventSortPanel scenario={scenario} upTo={local * scenario.sortedEvents.length} />
            </Panel>
            <Panel title="CONCURRENCY — the curve these events build">
              <CurvePanel scenario={scenario} curve={scenario.concurrency} headNs={headNs} height={300}
                valueLabel="in flight" />
            </Panel>
          </div>
        )}

        {beat.stage === "cumsum" && (
          <div className="grid min-h-0 flex-1 grid-cols-[320px_1fr] gap-3">
            <Panel title="RUNNING TOTAL" hint="delta → held value">
              <CumsumPanel scenario={scenario} upTo={local * scenario.steps.length} />
            </Panel>
            <Panel title="CONCURRENCY — right-continuous step function">
              <CurvePanel scenario={scenario} curve={scenario.concurrency} headNs={headNs} height={300}
                valueLabel="in flight" />
            </Panel>
          </div>
        )}

        {beat.stage === "icl" && (
          <Panel title="TOKENS IN FLIGHT — ICL-aware (solid) against coarse (dashed)"
            hint="same totals, different shape">
            <CurvePanel scenario={scenario} curve={scenario.iclTokens} headNs={headNs} height={330}
              color={PURPLE} valueLabel="tokens"
              overlay={{ ghost: { curve: scenario.coarseTokens, color: BLUE, label: "coarse" } }} />
          </Panel>
        )}

        {beat.stage === "threshold" && (
          <Panel title="STEADY-STATE WINDOW — threshold crossing"
            hint={window !== null
              ? `threshold ${window.threshold} = ceil(0.8 × ${TARGET_CONCURRENCY})`
              : "no window"}>
            <CurvePanel scenario={scenario} curve={scenario.concurrency} headNs={headNs} height={340}
              valueLabel="in flight"
              overlay={{
                level: window !== null
                  ? { value: window.threshold, color: ORANGE, label: `threshold ${window.threshold}` }
                  : undefined,
                band: window !== null
                  ? { fromNs: window.startNs, toNs: window.endNs, color: GREEN, label: "steady window" }
                  : undefined,
              }} />
          </Panel>
        )}

        {beat.stage === "cusum" && (
          <div className="grid min-h-0 flex-1 grid-cols-[1fr_1fr] gap-3">
            <Panel title="CUSUM — forward (solid) and backward (dashed)"
              hint="argmin of each is a turning point">
              <CusumPanel scenario={scenario} />
            </Panel>
            <Panel title="CONCURRENCY — what it was reading">
              <CurvePanel scenario={scenario} curve={scenario.concurrency} headNs={null} height={230}
                valueLabel="in flight"
                overlay={{ level: { value: scenario.cusum.target, color: "var(--color-category-cyan)", label: "p95 target" } }} />
            </Panel>
          </div>
        )}

        {beat.stage === "mser" && (
          <div className="grid min-h-0 flex-1 grid-cols-[1fr_1fr] gap-3">
            <Panel title="MSER-5 — request latency" hint="means above, statistic below">
              <Mser5Panel trace={scenario.mser5Latency} label="latency" />
            </Panel>
            <Panel title="MSER-5 — time to first token">
              <Mser5Panel trace={scenario.mser5Ttft} label="ttft" />
            </Panel>
          </div>
        )}

        {beat.stage === "consensus" && (
          <div className="grid min-h-0 flex-1 grid-cols-[1fr_1fr] gap-3">
            <Panel title="CONSENSUS — latest start, earliest end" hint="every signal must agree">
              <ConsensusPanel scenario={scenario} />
            </Panel>
            <Panel title="STATIONARITY — batch means across the window"
              hint="a strong trend vetoes the window">
              <StationarityPanel scenario={scenario} />
            </Panel>
          </div>
        )}
      </div>
    </PresentationShell>
  );
}
