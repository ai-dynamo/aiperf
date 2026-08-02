/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the shortfall that only exists in the sum.
//!
//! Both panes run the identical workload on the identical worker count. Every lane in both obeys
//! the cap it was given; nothing is misbehaving anywhere. The difference appears only when the
//! lanes are added together, which is exactly why this is worth drawing rather than describing.

import { useEffect, useMemo, useState } from "react";
import {
  admissibleTotal,
  capsFor,
  createRun,
  DEFAULT_CONFIG,
  durationFor,
  fragmentation,
  runToEnd,
  step,
  strandedSlots,
  summarize,
  type Config,
  type HopRouting,
  type Mode,
  type RunState,
} from "./dispatchSim.js";
import { ControlBar, Legend, LegendItem, Panel, Readout, SourceNote, SpikeHeader, Toggle } from "../ui.js";

const CYAN = "var(--color-category-cyan)";
const ORANGE = "var(--color-category-orange)";
const RED = "var(--color-category-red)";
const GREEN = "var(--color-category-green)";
const DIM = "var(--color-ink-quaternary)";

const MODES: readonly Mode[] = ["sharded", "global", "global-hop"];
const MODE_ACCENT: Record<Mode, string> = {
  sharded: ORANGE,
  global: CYAN,
  "global-hop": "var(--color-category-purple)",
};
const MODE_SUBTITLE: Record<Mode, string> = {
  sharded: "runtime.dispatch: sharded",
  global: "runtime.dispatch: global",
  "global-hop": "runtime.dispatch: global-hop",
};
const ROUTINGS: readonly HopRouting[] = ["round-robin", "sticky", "least-loaded"];

/** The one-line answer for each mode, so the panes can be compared before they are read. */
const VERDICT: Record<Mode, { what: string; cost: string }> = {
  sharded: {
    what: "Split it up front.",
    cost: "No coordination at all — but a thread cannot lend a slot it is not using.",
  },
  global: {
    what: "Share one pool.",
    cost: "Always holds the target. Threads still race, so order varies run to run.",
  },
  "global-hop": {
    what: "One loop decides everything.",
    cost: "Exact, reproducible order — paid for with a cross-thread trip per request.",
  },
};

const WORKER_CHOICES = [2, 4, 8] as const;
const CONCURRENCY_CHOICES = [3, 8, 12] as const;

export function DispatchModesSpike(): React.JSX.Element {
  const [workers, setWorkers] = useState(DEFAULT_CONFIG.workers);
  const [concurrency, setConcurrency] = useState(DEFAULT_CONFIG.concurrency);
  const [running, setRunning] = useState(false);

  const config: Config = useMemo(
    () => ({ ...DEFAULT_CONFIG, workers, concurrency }),
    [workers, concurrency],
  );

  const [routing, setRouting] = useState<HopRouting>("round-robin");
  const [runs, setRuns] = useState<RunState[]>(() => MODES.map((m) => createRun(m, config, routing)));

  // Changing the shape or the routing is a different run, so all three restart together and stay
  // comparable — the whole page is only meaningful when the three are running the same workload.
  useEffect(() => {
    setRuns(MODES.map((m) => createRun(m, config, routing)));
    setRunning(false);
  }, [config, routing]);

  useEffect(() => {
    if (!running) return undefined;
    const handle = window.setInterval(() => {
      setRuns((current) => current.map((r) => step(r, config)));
    }, 80);
    return () => window.clearInterval(handle);
  }, [running, config]);

  useEffect(() => {
    if (runs.every((r) => r.done)) setRunning(false);
  }, [runs]);

  const finished = useMemo(
    () => MODES.map((m) => {
      const state = runToEnd(m, config, routing);
      return { mode: m, summary: summarize(state, config), state };
    }),
    [config, routing],
  );
  const overSubscribed = admissibleTotal("sharded", concurrency, workers) > concurrency;

  return (
    <div className="min-h-screen bg-surface-page px-8 py-7 text-ink-primary">
      <SpikeHeader title="Who gets to hold a slot?">
        <p>
          You ask for a concurrency of {concurrency}. There are {workers} worker threads. Something
          has to decide which thread may hold which slot, and AIPerf has three answers — running
          side by side below on identical work.
        </p>
      </SpikeHeader>

      <ControlBar>
        <div className="flex items-center gap-1.5">
          <Toggle active onClick={() => setRunning((r) => !r)}>{running ? "Pause" : "Run"}</Toggle>
          <Toggle onClick={() => { setRunning(false); setRuns((c) => c.map((r) => step(r, config))); }}>
            Step
          </Toggle>
          <Toggle onClick={() => { setRunning(false); setRuns(MODES.map((m) => createRun(m, config, routing))); }}>
            Reset
          </Toggle>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">workers</span>
          {WORKER_CHOICES.map((w) => (
            <Toggle key={w} active={workers === w} onClick={() => setWorkers(w)}>{w}</Toggle>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">concurrency target</span>
          {CONCURRENCY_CHOICES.map((c) => (
            <Toggle key={c} active={concurrency === c} onClick={() => setConcurrency(c)}>{c}</Toggle>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">hop routing</span>
          {ROUTINGS.map((r) => (
            <Toggle key={r} active={routing === r} onClick={() => setRouting(r)}
              title="Only meaningful for global-hop; inert under the other two modes">
              {r}
            </Toggle>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-6">
          <Readout label="tick" value={runs[0]?.tick ?? 0} />
          <Readout label="requests" value={config.requests} />
          <Readout label="sessions" value={config.sessions} />
        </div>
      </ControlBar>

      {overSubscribed && (
        <div className="mb-4 rounded-lg border px-5 py-4 text-base leading-relaxed"
          style={{ borderColor: ORANGE, background: "rgba(255,140,0,0.05)" }}>
          <strong style={{ color: ORANGE }}>Over-subscribed.</strong>{" "}
          A target of {concurrency} across {workers} workers gives each a share below one, and{" "}
          <code>owned_cap</code> floors every share at one so no thread is starved. The caps stop
          tiling and start exceeding: <strong>{admissibleTotal("sharded", concurrency, workers)}</strong>{" "}
          admissible against a target of <strong>{concurrency}</strong>. The shared pool cannot do
          this — there is only one counter to check.
        </div>
      )}

      <div className="mb-2 grid grid-cols-3 gap-4">
        {MODES.map((mode) => (
          <div key={mode} className="text-[15px] leading-snug text-ink-secondary">
            <span className="font-bold" style={{ color: MODE_ACCENT[mode] }}>{VERDICT[mode].what}</span>{" "}
            {VERDICT[mode].cost}
          </div>
        ))}
      </div>

      <Legend>
        <LegendItem mark="▰">a request in flight</LegendItem>
        <LegendItem mark="▰" color={RED}>a slow one</LegendItem>
        <LegendItem mark="▱">a free slot</LegendItem>
        <LegendItem mark="▱" color={RED}>free, and unreachable by the thread that needs it</LegendItem>
      </Legend>

      <div className="grid grid-cols-3 gap-4">
        {runs.map((state, i) => (
          <ModePane key={state.mode} state={state} config={config}
            accent={MODE_ACCENT[state.mode]}
            title={state.mode === "global" ? "global  (default)" : state.mode}
            subtitle={MODE_SUBTITLE[state.mode]}
            finished={finished[i]!.summary} />
        ))}
      </div>

      <div className="mt-4 rounded-lg border px-5 py-4"
        style={{ borderColor: GREEN, background: "rgba(0,255,128,0.03)" }}>
        <div className="mb-3 text-[12px] font-bold tracking-widest" style={{ color: GREEN }}>
WHAT EACH ONE BOUGHT, AND PAID
        </div>
        <table className="w-full text-base tabular-nums">
          <thead>
            <tr className="text-left text-[14px] text-ink-tertiary">
              <th className="w-[420px] font-normal" />
              {finished.map((f) => (
                <th key={f.mode} className="pb-1 font-bold" style={{ color: MODE_ACCENT[f.mode] }}>
                  {f.mode}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <Row label="Concurrency it actually held"
              values={finished.map((f) => `${f.summary.meanInFlight.toFixed(2)} of ${concurrency}`)}
              best={(i) => finished[i]!.summary.utilisation > 0.99} />
            <Row label="Time to finish identical work"
              values={finished.map((f) => `${f.state.curve.length} ticks`)}
              best={(i) => finished[i]!.state.curve.length ===
                Math.min(...finished.map((g) => g.state.curve.length))} />
            <Row label="Threads a session is spread over"
              hint="the connection pool is per-thread — 2 means two connections where one would do"
              values={finished.map((f) => fragmentation(f.state).mean.toFixed(2))}
              best={(i) => fragmentation(finished[i]!.state).worst === 1} />
            <Row label="Same order every run"
              values={finished.map((f) => (f.mode === "global-hop" ? "yes" : "no"))}
              best={(i) => finished[i]!.mode === "global-hop"} />
          </tbody>
        </table>
      </div>

      <SourceNote>
        Four things worth knowing, each easy to assume wrongly. The request <em>budget</em> is
        split per thread under <code>sharded</code> and <code>global</code> alike — the shared gate
        covers concurrency and rate only — while <code>global-hop</code> splits nothing.{" "}
        <code>global-hop</code> needs no shared gate at all: one loop holding the full cap{" "}
        <em>is</em> the global cap. A target smaller than the thread count over-subscribes rather
        than under-shooting, because each share is floored at one. And the mode only affects
        request-rate phases — <code>user_centric</code> and <code>fixed_schedule</code> ignore it.
        The default already accounts for this: <code>global</code> for one process,{" "}
        <code>sharded</code> once <code>cells &gt; 1</code>, where the shared gate buys parity that
        is already gone and costs ~7-8× for it.
        <br />
        <span className="text-ink-quaternary">
          Ported from <code>config/model/dispatch.rs</code>,{" "}
          <code>engine/sharded_scheduled.rs:128</code>, <code>engine/cell_launcher.rs:272</code>,{" "}
          <code>engine/turn_execution.rs:438</code>, <code>engine/global_hop.rs</code>, and{" "}
          <code>engine/protocol_v2.rs:255</code>. The hop cost charged here is illustrative; its
          shape — one bounded mpsc trip and a oneshot reply per request — is not.
        </span>
      </SourceNote>
    </div>
  );
}

/** One metric across the three modes, with the winners marked. */
function Row({
  label,
  values,
  best,
  hint,
}: {
  label: string;
  values: readonly string[];
  best: (index: number) => boolean;
  hint?: string;
}): React.JSX.Element {
  return (
    <tr className="border-t border-white/5">
      <td className="w-[420px] py-2 pr-8 align-baseline">
        <span className="text-[15px] text-ink-secondary">{label}</span>
        {hint !== undefined && (
          <span className="ml-2 text-[13px] text-ink-quaternary">{hint}</span>
        )}
      </td>
      {values.map((value, i) => (
        <td key={i} className="py-2 pr-4 align-baseline">
          <strong className="text-[18px]" style={{ color: best(i) ? GREEN : "inherit" }}>
            {value}
          </strong>
        </td>
      ))}
    </tr>
  );
}

function ModePane({
  state,
  config,
  accent,
  title,
  subtitle,
  finished,
}: {
  state: RunState;
  config: Config;
  accent: string;
  title: string;
  subtitle: string;
  finished: ReturnType<typeof summarize>;
}): React.JSX.Element {
  const live = state.workers.reduce((sum, w) => sum + w.inFlight.length, 0);
  const stranded = strandedSlots(state);
  const caps = capsFor(state.mode, config.concurrency, config.workers);

  return (
    <Panel label={title} hint={subtitle}>
      <div className="mb-3 flex items-baseline gap-6">
        <span className="text-lg tabular-nums">
          <span className="text-ink-tertiary">in flight</span>{" "}
          <strong style={{ color: accent }}>{live}</strong>
          <span className="text-ink-quaternary"> / {config.concurrency}</span>
        </span>
        {state.mode === "sharded" && (
          <span className="text-lg tabular-nums">
            <span className="text-ink-tertiary">slots stranded</span>{" "}
            <strong style={{ color: stranded > 0 ? RED : DIM }}>{stranded}</strong>
          </span>
        )}
        {state.mode === "global-hop" && (
          <span className="text-lg tabular-nums">
            <span className="text-ink-tertiary">hops</span>{" "}
            <strong style={{ color: accent }}>{state.hopTicksCharged}</strong>
            <span className="text-ink-quaternary"> ticks paid</span>
          </span>
        )}
        {state.done && (
          <span className="ml-auto rounded px-2.5 py-0.5 text-[12px] font-bold text-black"
            style={{ background: GREEN }}>DONE</span>
        )}
      </div>


      {state.mode === "sharded" && (
        <ShardedLanes state={state} config={config} accent={accent} caps={caps} />
      )}
      {state.mode === "global" && <SharedPool state={state} config={config} accent={accent} />}
      {state.mode === "global-hop" && <HopLanes state={state} config={config} accent={accent} />}

      <Curve state={state} config={config} accent={accent} finishedTicks={finished.ticks} />
    </Panel>
  );
}

const CURVE_H = 120;

/** Aggregate in-flight over time against the target. The only view where the modes differ. */
function Curve({
  state,
  config,
  accent,
  finishedTicks,
}: {
  state: RunState;
  config: Config;
  accent: string;
  finishedTicks: number;
}): React.JSX.Element {
  const width = 640;
  const pad = 26;
  // A fixed horizontal extent across both panes, so one curve running longer is visible as
  // running longer rather than being rescaled to look the same.
  const span = Math.max(finishedTicks, 40) * 1.1;
  const vMax = Math.max(config.concurrency + 2, ...state.curve, 1);
  const x = (i: number) => pad + (i / span) * (width - pad - 8);
  const y = (v: number) => CURVE_H - 12 - (v / vMax) * (CURVE_H - 26);

  const path = state.curve.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(v)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${CURVE_H}`} width="100%" height={CURVE_H}
      role="img" aria-label="aggregate in-flight concurrency over time">
      <line x1={pad} x2={width - 8} y1={y(config.concurrency)} y2={y(config.concurrency)}
        stroke={GREEN} strokeDasharray="5 4" strokeWidth={1} />
      <text x={width - 10} y={y(config.concurrency) - 5} fontSize={12} textAnchor="end" fill={GREEN}>
        target {config.concurrency}
      </text>

      {/* The shortfall, filled: everything between what was asked for and what was held. */}
      {state.curve.length > 1 && (
        <path
          d={`${path} L ${x(state.curve.length - 1)} ${y(config.concurrency)} L ${x(0)} ${y(config.concurrency)} Z`}
          fill={accent} opacity={0.12} />
      )}
      <path d={path} fill="none" stroke={accent} strokeWidth={1.75} />

      <line x1={pad} x2={pad} y1={6} y2={CURVE_H - 12} stroke="rgba(255,255,255,0.12)" />
      <line x1={pad} x2={width - 8} y1={CURVE_H - 12} y2={CURVE_H - 12} stroke="rgba(255,255,255,0.12)" />
      <text x={4} y={y(config.concurrency) + 4} fontSize={12} fill={DIM}>{config.concurrency}</text>
      <text x={4} y={CURVE_H - 14} fontSize={12} fill={DIM}>0</text>
    </svg>
  );
}

/** One slot box. Filled means occupied; a red outline means free but unusable. */
function Slot({
  filled,
  slow,
  accent,
  stranded,
}: {
  filled: boolean;
  slow: boolean;
  accent: string;
  stranded?: boolean;
}): React.JSX.Element {
  return (
    <span className="h-5 w-5 rounded-[3px]"
      style={{
        background: !filled ? "transparent" : slow ? RED : accent,
        outline: filled
          ? "none"
          : `1px dashed ${stranded === true ? RED : "rgba(255,255,255,0.16)"}`,
        outlineOffset: -1,
      }} />
  );
}

/**
 * Sharded: one row of slots per worker, because that is literally what the mode allocates.
 *
 * A worker whose queue is empty while another still has work has its free slots outlined in red.
 * Those slots exist, are idle, and cannot be given away — which is the entire shortfall, drawn.
 */
function ShardedLanes({
  state,
  config,
  accent,
  caps,
}: {
  state: RunState;
  config: Config;
  accent: string;
  caps: number[];
}): React.JSX.Element {
  const anyoneWaiting = state.workers.some((w) => w.queue.length > 0);
  return (
    <div className="mb-3 flex flex-col gap-1.5">
      {state.workers.map((worker) => {
        const idle = worker.queue.length === 0;
        return (
          <div key={worker.id} className="flex items-center gap-3">
            <span className="w-20 shrink-0 font-mono text-[13px] text-ink-tertiary">
              worker {worker.id}
            </span>
            <span className="flex items-center gap-[3px]">
              {Array.from({ length: caps[worker.id] ?? 0 }, (_, slot) => {
                const request = worker.inFlight[slot];
                return (
                  <Slot key={slot} accent={accent} filled={request !== undefined}
                    slow={request !== undefined && durationFor(request.id, config) > config.shortTicks}
                    stranded={request === undefined && idle && anyoneWaiting} />
                );
              })}
            </span>
            <span className="font-mono text-[13px] tabular-nums"
              style={{ color: idle && anyoneWaiting ? RED : DIM }}>
              {idle ? "queue empty" : `${worker.queue.length} queued`}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Global: one row, because there is one pool.
 *
 * Drawing a per-worker cap here would invent a structure the mode does not have. Slots are tinted
 * by which worker currently holds them, so the reallocation is visible as the mix shifting rather
 * than as lanes filling independently.
 */
function SharedPool({
  state,
  config,
  accent,
}: {
  state: RunState;
  config: Config;
  accent: string;
}): React.JSX.Element {
  const held = state.workers.flatMap((w) => w.inFlight.map((r) => r));
  return (
    <div className="mb-3 flex flex-col gap-1.5">
      <div className="flex items-center gap-3">
        <span className="w-20 shrink-0 font-mono text-[13px] text-ink-tertiary">shared pool</span>
        <span className="flex flex-wrap items-center gap-[3px]">
          {Array.from({ length: config.concurrency }, (_, slot) => {
            const request = held[slot];
            return (
              <Slot key={slot} accent={accent} filled={request !== undefined}
                slow={request !== undefined && durationFor(request.id, config) > config.shortTicks} />
            );
          })}
        </span>
        <span className="font-mono text-[13px] tabular-nums" style={{ color: DIM }}>
          any worker may take any slot
        </span>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-x-5 gap-y-1">
        {state.workers.map((worker) => (
          <span key={worker.id} className="font-mono text-[13px]" style={{ color: DIM }}>
            worker {worker.id}{" "}
            <span style={{ color: worker.inFlight.length > 0 ? accent : DIM }}>
              {worker.inFlight.length} held
            </span>
            {" · "}
            {worker.queue.length === 0 ? "queue empty" : `${worker.queue.length} queued`}
          </span>
        ))}
      </div>
    </div>
  );
}

/**
 * Global-hop: the coordinator's queue, then where each request was placed.
 *
 * The other two modes have no equivalent of the left-hand queue — their `W` loops each walk their
 * own partition and race. Here one loop issues in exact global order, so `turn i` reaching
 * `worker i % W` under round-robin is a guarantee rather than an accident, and that is what makes
 * jittered arrival statistics reproducible.
 */
function HopLanes({
  state,
  config,
  accent,
}: {
  state: RunState;
  config: Config;
  accent: string;
}): React.JSX.Element {
  const upcoming = state.coordinatorQueue.slice(0, 6);
  return (
    <div className="mb-3 flex flex-col gap-1.5">
      <div className="flex items-center gap-3">
        <span className="w-24 shrink-0 font-mono text-[13px] text-ink-tertiary">coordinator</span>
        <span className="flex items-center gap-1.5 font-mono text-[13px]">
          {upcoming.length === 0 && <span style={{ color: DIM }}>issued everything</span>}
          {upcoming.map((id) => (
            <span key={id} className="rounded px-1.5 py-0.5"
              style={{ background: "rgba(255,255,255,0.06)", color: DIM }}>
              #{id}
            </span>
          ))}
          {state.coordinatorQueue.length > upcoming.length && (
            <span style={{ color: DIM }}>+{state.coordinatorQueue.length - upcoming.length}</span>
          )}
        </span>
      </div>

      {state.workers.map((worker) => (
        <div key={worker.id} className="flex items-center gap-3">
          <span className="w-24 shrink-0 font-mono text-[13px] text-ink-tertiary">
            → worker {worker.id}
          </span>
          <span className="flex flex-wrap items-center gap-[3px]">
            {worker.inFlight.map((request) => (
              <Slot key={request.id} accent={accent} filled
                slow={durationFor(request.id, config) > config.shortTicks} />
            ))}
            {worker.inFlight.length === 0 && (
              <span className="font-mono text-[13px]" style={{ color: DIM }}>idle</span>
            )}
          </span>
          <span className="ml-auto font-mono text-[13px] tabular-nums" style={{ color: DIM }}>
            {new Set(worker.inFlight.map((r) => r.correlation)).size} sessions
          </span>
        </div>
      ))}
    </div>
  );
}
