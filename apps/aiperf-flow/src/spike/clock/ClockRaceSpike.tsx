/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the same workload on two clocks, running side by side.
//!
//! Both panes are the same tasks and the same `Clock` seam. The left one waits out the gaps
//! between events because real timers must; the right one jumps straight to each next event
//! because virtual time has no obligation to. Press Run and watch the right pane finish while the
//! left is still on its first sleep — then compare the event logs, which are identical.

import { useEffect, useRef, useState } from "react";
import {
  createClock,
  defaultTasks,
  nextEventTime,
  NS_PER_MS,
  runToEnd,
  spanNs,
  stepReal,
  stepSim,
  type ClockState,
} from "./clockSim.js";

const GREEN = "var(--color-category-green)";
const BLUE = "var(--color-category-blue)";
const ORANGE = "var(--color-category-orange)";
const CYAN = "var(--color-category-cyan)";
const DIM = "var(--color-ink-quaternary)";

const TASKS = defaultTasks();
const SPAN = spanNs(TASKS);
const SPEEDS = [1, 0.5, 0.25] as const;

export function ClockRaceSpike(): React.JSX.Element {
  const [real, setReal] = useState<ClockState>(() => createClock("real", TASKS));
  const [sim, setSim] = useState<ClockState>(() => createClock("sim", TASKS));
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(0.5);

  const runningRef = useRef(running);
  runningRef.current = running;
  const speedRef = useRef(speed);
  speedRef.current = speed;

  useEffect(() => {
    let handle = 0;
    let last = performance.now();
    const frame = (t: number) => {
      const dt = Math.min(64, t - last);
      last = t;
      if (runningRef.current) {
        // Real time advances only as fast as it actually passes.
        setReal((s) => (s.done ? s : stepReal(s, dt * speedRef.current * NS_PER_MS, TASKS)));
        // Virtual time advances one whole event per tick — the gap costs nothing.
        setSim((s) => (s.done ? s : stepSim(s, TASKS)));
      }
      handle = requestAnimationFrame(frame);
    };
    handle = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(handle);
  }, []);

  const reset = () => {
    setReal(createClock("real", TASKS));
    setSim(createClock("sim", TASKS));
    setRunning(false);
  };

  const finished = runToEnd("sim", TASKS);
  const identical =
    real.done && sim.done && JSON.stringify(real.events) === JSON.stringify(sim.events);

  return (
    <div className="min-h-screen bg-surface-page px-8 py-6 text-ink-primary">
      <div className="mb-1 flex items-baseline gap-3">
        <span className="text-xs font-bold uppercase tracking-[0.2em] text-ink-link">Spike</span>
        <h1 className="text-2xl font-extrabold">Two clocks, one workload</h1>
      </div>
      <p className="mb-4 max-w-4xl text-sm text-ink-secondary">
        Everything on the hot path takes time through the <code>Clock</code> seam, so the same
        workload runs unmodified on either. <strong>RealClock</strong> waits out the gaps between
        events because real timers must. <strong>SimClock</strong> parks sleepers in a heap ordered
        by <code>(at_ns, seq_no)</code> and jumps straight to the next deadline, so the empty space
        costs nothing. Press Run: the right pane finishes while the left is still on its first
        sleep — and the event logs come out identical.
      </p>

      <div className="mb-4 rounded-lg border border-white/10 bg-surface-elevated px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex items-center gap-1.5">
            <button type="button" onClick={() => setRunning((r) => !r)}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold">
              {running ? "Pause" : "Run"}
            </button>
            <button type="button" onClick={reset}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary">
              Reset
            </button>
            <button type="button"
              onClick={() => { setSim((s) => (s.done ? s : stepSim(s, TASKS))); setRunning(false); }}
              className="rounded border border-white/15 bg-surface-panel px-3 py-1.5 text-sm font-semibold text-ink-secondary">
              Step sim
            </button>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="mr-1 text-sm text-ink-tertiary">real-time speed</span>
            {SPEEDS.map((s) => (
              <button key={s} type="button" onClick={() => setSpeed(s)}
                className={`rounded border px-2.5 py-1 text-xs font-semibold tabular-nums ${
                  speed === s ? "border-transparent bg-accent-primary text-black"
                    : "border-white/15 bg-surface-panel text-ink-secondary"}`}>
                {s}×
              </button>
            ))}
          </div>
          <div className="ml-auto text-sm tabular-nums">
            <span className="text-ink-tertiary">workload spans</span>{" "}
            <strong>{(SPAN / NS_PER_MS).toFixed(0)} ms</strong>
            <span className="text-ink-quaternary"> of virtual time · {finished.events.length} events</span>
          </div>
        </div>
      </div>

      {identical && (
        <div className="mb-4 rounded-lg border px-4 py-3 text-sm"
          style={{ borderColor: GREEN, background: "rgba(0,255,128,0.05)" }}>
          <strong style={{ color: GREEN }}>Identical.</strong>{" "}
          Both clocks emitted the same {real.events.length} events in the same order and ended at
          the same virtual time. The simulated run got there in{" "}
          <strong>{sim.wallMs.toFixed(1)} ms</strong> of wall time against the real run&apos;s{" "}
          <strong>{real.wallMs.toFixed(0)} ms</strong> — a {(real.wallMs / Math.max(sim.wallMs, 0.01)).toFixed(0)}×
          difference in how long it took to learn exactly the same thing.
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <ClockPane state={real} label="RealClock" accent={BLUE}
          hint="default drive: current-thread tokio, real timers" />
        <ClockPane state={sim} label="SimClock" accent={CYAN}
          hint="drive_sim: advance_to(next_event_time), event by event" />
      </div>

      <p className="mt-3 text-[11px] text-ink-quaternary">
        Modelled on <code>rust/runtime/src/clock/</code>: the <code>Clock</code> trait at
        clock.rs:12, <code>SimClock</code>&apos;s <code>(at_ns, seq_no)</code> heap at
        sim_clock.rs:48, <code>next_event_time</code> at :92 and <code>advance_to</code> at :106.
        Sequence numbers are what make same-deadline wakes deterministic rather than arbitrary —
        the same idea as the sweep line&apos;s <code>(timestamp, delta)</code> tie-break.
      </p>
    </div>
  );
}

function ClockPane({
  state,
  label,
  accent,
  hint,
}: {
  state: ClockState;
  label: string;
  accent: string;
  hint: string;
}): React.JSX.Element {
  const progress = Math.min(1, state.nowNs / Math.max(1, SPAN));
  return (
    <section className="rounded-lg border border-white/10 bg-surface-elevated p-3">
      <div className="mb-2 flex items-baseline gap-3">
        <h2 className="text-sm font-bold" style={{ color: accent }}>{label}</h2>
        <span className="text-[10px] text-ink-quaternary">{hint}</span>
        {state.done && (
          <span className="ml-auto rounded px-2 py-0.5 text-[10px] font-bold text-black"
            style={{ background: state.deadlocked ? ORANGE : GREEN }}>
            {state.deadlocked ? "DEADLOCKED" : "DONE"}
          </span>
        )}
      </div>

      <div className="mb-2 flex items-baseline gap-5 text-sm tabular-nums">
        <span><span className="text-ink-tertiary">clock time</span>{" "}
          <strong>{(state.nowNs / NS_PER_MS).toFixed(0)} ms</strong></span>
        <span><span className="text-ink-tertiary">wall time</span>{" "}
          <strong style={{ color: accent }}>{state.wallMs.toFixed(1)} ms</strong></span>
        <span><span className="text-ink-tertiary">events</span> <strong>{state.events.length}</strong></span>
      </div>

      <div className="mb-3 h-2 w-full rounded" style={{ background: "rgba(255,255,255,0.07)" }}>
        <div className="h-2 rounded" style={{ width: `${progress * 100}%`, background: accent }} />
      </div>

      <div className="mb-1 text-[10px] font-bold tracking-widest text-ink-secondary">
        PARKED SLEEPERS — heap order (at_ns, seq_no)
      </div>
      <div className="mb-3 flex min-h-[68px] flex-col gap-1">
        {state.heap.length === 0 && (
          <span className="text-[11px]" style={{ color: DIM }}>
            {state.done ? "empty — nothing left to wake" : "—"}
          </span>
        )}
        {state.heap.slice(0, 5).map((s, i) => (
          <div key={`${s.taskId}-${s.seqNo}`} className="flex items-center gap-2 font-mono text-[10px]">
            <span className="w-4 text-right" style={{ color: DIM }}>{i}</span>
            <span className="w-16" style={{ color: i === 0 ? accent : undefined }}>{s.taskId}</span>
            <span className="w-20 tabular-nums" style={{ color: DIM }}>
              at {(s.atNs / NS_PER_MS).toFixed(0)}ms
            </span>
            <span className="tabular-nums" style={{ color: DIM }}>seq {s.seqNo}</span>
            {i === 0 && <span style={{ color: accent }}>◄ next</span>}
          </div>
        ))}
      </div>

      <div className="mb-1 flex items-baseline gap-2">
        <span className="text-[10px] font-bold tracking-widest text-ink-secondary">EVENT LOG</span>
        <span className="text-[10px]" style={{ color: DIM }}>
          {state.kind === "sim" && !state.done
            ? `next event at ${((nextEventTime(state) ?? 0) / NS_PER_MS).toFixed(0)}ms`
            : ""}
        </span>
      </div>
      <div className="h-40 overflow-y-auto font-mono text-[10px] leading-[1.6]">
        {state.events.length === 0 && <span style={{ color: DIM }}>Nothing yet.</span>}
        {state.events.map((e, i) => (
          <div key={i} style={{ color: i === state.events.length - 1 ? accent : undefined }}>
            {String(i).padStart(2, "0")} {e}
          </div>
        ))}
      </div>
    </section>
  );
}
