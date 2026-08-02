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
  runToEnd,
  step,
  strandedSlots,
  summarize,
  type Config,
  type RunState,
} from "./dispatchSim.js";
import { ControlBar, Legend, LegendItem, Panel, Readout, SourceNote, SpikeHeader, Toggle } from "../ui.js";

const CYAN = "var(--color-category-cyan)";
const ORANGE = "var(--color-category-orange)";
const RED = "var(--color-category-red)";
const GREEN = "var(--color-category-green)";
const DIM = "var(--color-ink-quaternary)";

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

  const [sharded, setSharded] = useState<RunState>(() => createRun("sharded", config));
  const [global, setGlobal] = useState<RunState>(() => createRun("global", config));

  // Changing the shape is a different run, so both restart together and stay comparable.
  useEffect(() => {
    setSharded(createRun("sharded", config));
    setGlobal(createRun("global", config));
    setRunning(false);
  }, [config]);

  useEffect(() => {
    if (!running) return undefined;
    const handle = window.setInterval(() => {
      setSharded((s) => step(s, config));
      setGlobal((s) => step(s, config));
    }, 90);
    return () => window.clearInterval(handle);
  }, [running, config]);

  useEffect(() => {
    if (sharded.done && global.done) setRunning(false);
  }, [sharded.done, global.done]);

  const finishedSharded = useMemo(() => summarize(runToEnd("sharded", config), config), [config]);
  const finishedGlobal = useMemo(() => summarize(runToEnd("global", config), config), [config]);
  const overSubscribed = admissibleTotal("sharded", concurrency, workers) > concurrency;

  return (
    <div className="min-h-screen bg-surface-page px-8 py-7 text-ink-primary">
      <SpikeHeader title="The shortfall that only exists in the sum">
        <p>
          With more than one worker thread, something has to decide who may have a slot.{" "}
          <code>sharded</code> hands each thread a fixed <code>1/workers</code> share up front.{" "}
          <code>global</code> — the default — gives every thread one shared pool to admit from. Both
          panes below run the same requests on the same threads against the same target.
        </p>
        <p>
          Watch the lanes: <strong>every one of them is obeying its cap correctly, in both
          panes.</strong> Nothing is broken at the level anyone would look. Then watch the two
          curves. A sharded thread that finishes its own queue keeps its slots and cannot lend them
          to a thread still working, so the aggregate quietly sits below the number you asked for —
          and the run takes longer to do identical work.
        </p>
      </SpikeHeader>

      <ControlBar>
        <div className="flex items-center gap-1.5">
          <Toggle active onClick={() => setRunning((r) => !r)}>{running ? "Pause" : "Run"}</Toggle>
          <Toggle onClick={() => { setRunning(false); setSharded((s) => step(s, config)); setGlobal((s) => step(s, config)); }}>
            Step
          </Toggle>
          <Toggle onClick={() => { setRunning(false); setSharded(createRun("sharded", config)); setGlobal(createRun("global", config)); }}>
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

        <div className="ml-auto flex items-center gap-6">
          <Readout label="tick" value={sharded.tick} />
          <Readout label="requests" value={config.requests} />
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

      <div className="grid grid-cols-2 gap-4">
        <ModePane state={sharded} config={config} accent={ORANGE} title="sharded"
          subtitle="a fixed 1/workers share each, decided before the run starts"
          finished={finishedSharded} />
        <ModePane state={global} config={config} accent={CYAN} title="global  (default)"
          subtitle="one shared pool, reallocated on every completion"
          finished={finishedGlobal} />
      </div>

      <div className="mt-4 rounded-lg border px-5 py-4"
        style={{ borderColor: GREEN, background: "rgba(0,255,128,0.03)" }}>
        <div className="mb-2 text-[12px] font-bold tracking-widest" style={{ color: GREEN }}>
          THE SAME WORK, FINISHED
        </div>
        <div className="grid grid-cols-3 gap-6 text-base">
          <Comparison label="mean concurrency held"
            sharded={`${finishedSharded.meanInFlight.toFixed(2)} of ${concurrency}`}
            global={`${finishedGlobal.meanInFlight.toFixed(2)} of ${concurrency}`} />
          <Comparison label="utilisation of the target"
            sharded={`${(finishedSharded.utilisation * 100).toFixed(1)}%`}
            global={`${(finishedGlobal.utilisation * 100).toFixed(1)}%`} />
          <Comparison label="ticks spent dispatching"
            sharded={`${finishedSharded.ticks}`}
            global={`${finishedGlobal.ticks}`} />
        </div>
      </div>

      <SourceNote>
        Modelled on <code>rust/runtime/src/engine/</code>:{" "}
        <code>DispatchMode</code> at <code>protocol.rs:14</code>,{" "}
        <code>slice_phase_for_thread</code> at <code>sharded_scheduled.rs:128</code>,{" "}
        <code>owned_positions</code> at <code>cell_launcher.rs:272</code>, and the condition stated
        by <code>VariableLatencyMock</code> at <code>workers_characterization.rs:1325</code> — that
        uneven completion times are what make the static partition visibly wrong. Two details the
        page keeps faithful because they are easy to assume wrongly: the request <em>budget</em> is
        sliced under <strong>both</strong> modes, since the shared gate covers concurrency and rate
        only; and the shared gate applies to the request-rate phase shapes,{" "}
        <code>user_centric</code> and <code>fixed_schedule</code> being unaffected by the mode.
      </SourceNote>
    </div>
  );
}

function Comparison({
  label,
  sharded,
  global,
}: {
  label: string;
  sharded: string;
  global: string;
}): React.JSX.Element {
  return (
    <div>
      <div className="mb-1 text-[14px] text-ink-tertiary">{label}</div>
      <div className="flex items-baseline gap-4">
        <span className="tabular-nums" style={{ color: ORANGE }}>
          <span className="text-[13px] text-ink-quaternary">sharded </span>
          <strong className="text-[19px]">{sharded}</strong>
        </span>
        <span className="tabular-nums" style={{ color: CYAN }}>
          <span className="text-[13px] text-ink-quaternary">global </span>
          <strong className="text-[19px]">{global}</strong>
        </span>
      </div>
    </div>
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
        {state.done && (
          <span className="ml-auto rounded px-2.5 py-0.5 text-[12px] font-bold text-black"
            style={{ background: GREEN }}>DONE</span>
        )}
      </div>

      <Legend>
        <LegendItem mark="▰" color={accent}>in flight</LegendItem>
        <LegendItem mark="▰" color={RED}>a slow one</LegendItem>
        <LegendItem mark="▱">free</LegendItem>
        {state.mode === "sharded" && (
          <LegendItem mark="▱" color={RED}>free, and unreachable</LegendItem>
        )}
      </Legend>

      {state.mode === "global" ? (
        <SharedPool state={state} config={config} accent={accent} />
      ) : (
        <ShardedLanes state={state} config={config} accent={accent} caps={caps} />
      )}

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
  const anyoneWaiting = state.workers.some((w) => w.remaining > 0);
  return (
    <div className="mb-3 flex flex-col gap-1.5">
      {state.workers.map((worker) => {
        const idle = worker.remaining === 0;
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
              {idle ? "queue empty" : `${worker.remaining} queued`}
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
            {worker.remaining === 0 ? "queue empty" : `${worker.remaining} queued`}
          </span>
        ))}
      </div>
    </div>
  );
}
