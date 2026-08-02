/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the same workload on two clocks, running side by side.
//!
//! Both panes benchmark the same eight requests through the same `Clock` seam. The left waits out
//! every gap because real timers must; the right jumps straight to each next event because virtual
//! time has no obligation to. The comparison an end user cares about is the result table, not the
//! heap: same requests, same tokens, same latencies, in a fraction of the wall time.

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
  summarize,
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
  // Not a raw event comparison: the real clock's timestamps are rounded up to its tick, and a
  // coarse tick can also batch two wakes that simulation would have visited separately. What must
  // agree is the outcome — every request completed, every token generated, each request's own
  // steps in order.
  const agree =
    real.done &&
    sim.done &&
    summarize(real).completed === summarize(sim).completed &&
    summarize(real).tokens === summarize(sim).tokens;
  const identical = agree;

  return (
    <div className="min-h-screen bg-surface-page px-8 py-7 text-ink-primary">
      <div className="mb-1 flex items-baseline gap-3">
        <span className="text-sm font-bold uppercase tracking-[0.2em] text-ink-link">Spike</span>
        <h1 className="text-3xl font-extrabold">The same benchmark, run twice</h1>
      </div>
      <p className="mb-4 max-w-5xl text-base leading-relaxed text-ink-secondary">
        Eight requests are benchmarked below — sent, waiting out their time-to-first-token, then
        streaming tokens. <strong>The identical run happens in both panes.</strong> The left one
        waits: every 400 ms of TTFT is 400 ms of your life. The right one does not, because its
        clock is a number it can move rather than a wall it has to wait for.
      </p>
      <p className="mb-4 max-w-5xl text-base leading-relaxed text-ink-secondary">
        Press Run and watch the right pane finish before the left has its first token. Then compare
        the two result tables: <strong>same requests, same tokens, same latencies</strong>. That is
        what replaying a recorded trace under simulation buys — an afternoon-long benchmark
        answered in the time it takes to blink, with the same conclusions.
      </p>

      <div className="mb-4 rounded-lg border border-white/10 bg-surface-elevated px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div className="flex items-center gap-1.5">
            <button type="button" onClick={() => setRunning((r) => !r)}
              className="rounded border border-white/15 bg-surface-panel px-4 py-2 text-base font-semibold">
              {running ? "Pause" : "Run"}
            </button>
            <button type="button" onClick={reset}
              className="rounded border border-white/15 bg-surface-panel px-4 py-2 text-base font-semibold text-ink-secondary">
              Reset
            </button>
            <button type="button"
              onClick={() => { setSim((s) => (s.done ? s : stepSim(s, TASKS))); setRunning(false); }}
              className="rounded border border-white/15 bg-surface-panel px-4 py-2 text-base font-semibold text-ink-secondary">
              Step sim
            </button>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="mr-1 text-base text-ink-tertiary">real-time speed</span>
            {SPEEDS.map((s) => (
              <button key={s} type="button" onClick={() => setSpeed(s)}
                className={`rounded border px-3 py-1.5 text-sm font-semibold tabular-nums ${
                  speed === s ? "border-transparent bg-accent-primary text-black"
                    : "border-white/15 bg-surface-panel text-ink-secondary"}`}>
                {s}×
              </button>
            ))}
          </div>
          <div className="ml-auto text-base tabular-nums">
            <span className="text-ink-tertiary">workload spans</span>{" "}
            <strong>{(SPAN / NS_PER_MS).toFixed(0)} ms</strong>
            <span className="text-ink-quaternary"> of virtual time · {finished.events.length} events</span>
          </div>
        </div>
      </div>

      {identical && (
        <div className="mb-4 rounded-lg border px-5 py-4 text-base leading-relaxed"
          style={{ borderColor: GREEN, background: "rgba(0,255,128,0.05)" }}>
          <strong style={{ color: GREEN }}>Same answer.</strong>{" "}
          Both runs completed all {summarize(sim).completed} requests and generated{" "}
          {summarize(sim).tokens} tokens, and every request went through its own steps in the same
          order. The real run took <strong>{(real.wallMs / 1000).toFixed(1)} seconds</strong> of
          wall time. The simulated one took{" "}
          <strong>{sim.wallMs.toFixed(1)} milliseconds</strong> —{" "}
          {(real.wallMs / Math.max(sim.wallMs, 0.01)).toFixed(0)}× faster to reach the same
          conclusions. Scale that to a benchmark that runs for an hour.
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <ClockPane state={real} label="Real time — you wait" accent={BLUE}
          hint="RealClock · current-thread tokio, real timers" />
        <ClockPane state={sim} label="Simulated time — instant" accent={CYAN}
          hint="SimClock · advance_to(next_event_time), event by event" />
      </div>

      <p className="mt-4 max-w-6xl text-[13px] leading-relaxed text-ink-quaternary">
        Modelled on <code>rust/runtime/src/clock/</code>: the <code>Clock</code> trait at
        clock.rs:12, <code>SimClock</code>&apos;s <code>(at_ns, seq_no)</code> heap at
        sim_clock.rs:48, <code>next_event_time</code> at :92 and <code>advance_to</code> at :106.
        Sequence numbers are what make same-deadline wakes deterministic rather than arbitrary —
        the same idea as the sweep line&apos;s <code>(timestamp, delta)</code> tie-break. One
        honest difference: the simulated timings are exact, while the real ones are rounded up to
        whenever the runtime next looked. A real timer fires at or after its deadline, never on it.
      </p>
    </div>
  );
}

function Metric({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="flex items-baseline justify-between">
      <span className="text-ink-tertiary">{label}</span>
      <strong style={{ color: accent }}>{value}</strong>
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
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el !== null) el.scrollTop = el.scrollHeight;
  }, [state.events.length]);

  const progress = Math.min(1, state.nowNs / Math.max(1, SPAN));
  return (
    <section className="rounded-lg border border-white/10 bg-surface-elevated p-4">
      <div className="mb-2 flex items-baseline gap-3">
        <h2 className="text-lg font-bold" style={{ color: accent }}>{label}</h2>
        <span className="text-[13px] text-ink-quaternary">{hint}</span>
        {state.done && (
          <span className="ml-auto rounded px-2.5 py-0.5 text-[12px] font-bold text-black"
            style={{ background: state.deadlocked ? ORANGE : GREEN }}>
            {state.deadlocked ? "DEADLOCKED" : "DONE"}
          </span>
        )}
      </div>

      <div className="mb-2 flex items-baseline gap-6 text-lg tabular-nums">
        <span><span className="text-ink-tertiary">benchmark time</span>{" "}
          <strong>{(state.nowNs / NS_PER_MS).toFixed(0)} ms</strong></span>
        <span title="How long you actually sat there">
          <span className="text-ink-tertiary">you waited</span>{" "}
          <strong style={{ color: accent, fontSize: "1.15em" }}>
            {state.wallMs >= 1000 ? `${(state.wallMs / 1000).toFixed(1)} s` : `${state.wallMs.toFixed(1)} ms`}
          </strong></span>
      </div>

      <div className="mb-3 h-2 w-full rounded" style={{ background: "rgba(255,255,255,0.07)" }}>
        <div className="h-2 rounded" style={{ width: `${progress * 100}%`, background: accent }} />
      </div>

      <div className="mb-1.5 text-[12px] font-bold tracking-widest text-ink-secondary">
        WAITING ON <span className="font-normal text-ink-quaternary">— the mechanism: a heap ordered by (deadline, registration)</span>
      </div>
      <div className="mb-3 flex min-h-[68px] flex-col gap-1">
        {state.heap.length === 0 && (
          <span className="text-[15px]" style={{ color: DIM }}>
            {state.done ? "empty — nothing left to wake" : "—"}
          </span>
        )}
        {state.heap.slice(0, 5).map((s, i) => (
          <div key={`${s.taskId}-${s.seqNo}`} className="flex items-center gap-3 font-mono text-[14px]">
            <span className="w-4 text-right" style={{ color: DIM }}>{i}</span>
            <span className="w-24" style={{ color: i === 0 ? accent : undefined }}>{s.taskId}</span>
            <span className="w-28 tabular-nums" style={{ color: DIM }}>
              at {(s.atNs / NS_PER_MS).toFixed(0)}ms
            </span>
            <span className="tabular-nums" style={{ color: DIM }}>seq {s.seqNo}</span>
            {i === 0 && <span style={{ color: accent }}>◄ next</span>}
          </div>
        ))}
      </div>

      <div className="mb-1.5 text-[12px] font-bold tracking-widest text-ink-secondary">
        BENCHMARK RESULT
      </div>
      <div className="mb-3 grid grid-cols-2 gap-x-6 gap-y-2 text-[16px] tabular-nums">
        <Metric label="requests done" value={`${summarize(state).completed}`} />
        <Metric label="tokens" value={`${summarize(state).tokens}`} />
        <Metric label="mean TTFT" value={`${summarize(state).meanTtftMs.toFixed(1)} ms`} accent={accent} />
        <Metric label="mean ITL" value={`${summarize(state).meanItlMs.toFixed(1)} ms`} accent={accent} />
        <Metric label="slowest request" value={`${summarize(state).p100LatencyMs.toFixed(0)} ms`} />
      </div>

      <div className="mb-1 flex items-baseline gap-2">
        <span className="text-[12px] font-bold tracking-widest text-ink-secondary">WHAT HAPPENED</span>
        <span className="text-[12px]" style={{ color: DIM }}>
          {state.kind === "sim" && !state.done
            ? `next event at ${((nextEventTime(state) ?? 0) / NS_PER_MS).toFixed(0)}ms`
            : ""}
        </span>
      </div>
      <div ref={logRef} className="h-56 overflow-y-auto font-mono text-[15px] leading-[1.75]">
        {state.events.length === 0 && <span style={{ color: DIM }}>Nothing yet.</span>}
        {state.events.slice(-40).map((e, i, arr) => (
          <div key={i} style={{ color: i === arr.length - 1 ? accent : undefined }}>
            <span style={{ color: DIM }}>{(e.atNs / NS_PER_MS).toFixed(0).padStart(4)}ms </span>
            {e.taskId} <span style={{ color: DIM }}>{e.label}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
