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

import { useEffect, useMemo, useRef, useState } from "react";
import {
  createClock,
  defaultTasks,
  instantsOf,
  stretchTasks,
  nextEventTime,
  NS_PER_MS,
  runToEnd,
  spanNs,
  stepReal,
  stepSim,
  summarize,
  type ClockState,
  type Task,
} from "./clockSim.js";

const GREEN = "var(--color-category-green)";
const BLUE = "var(--color-category-blue)";
const ORANGE = "var(--color-category-orange)";
const CYAN = "var(--color-category-cyan)";
const DIM = "var(--color-ink-quaternary)";

const BASE_TASKS = defaultTasks();
const SPEEDS = [1, 0.5, 0.25] as const;

/**
 * How far apart to push the events.
 *
 * Nothing else about the run changes — same requests, same order, same event count. Only the
 * emptiness between them grows, which is precisely the thing one clock is billed for and the
 * other is not.
 */
const STRETCHES = [
  { factor: 1, label: "as recorded" },
  { factor: 100, label: "3 minutes" },
  { factor: 2000, label: "1 hour" },
] as const;

function humanNs(ns: number): string {
  const ms = ns / NS_PER_MS;
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  if (ms < 3_600_000) return `${(ms / 60_000).toFixed(1)} min`;
  return `${(ms / 3_600_000).toFixed(1)} hours`;
}

export function ClockRaceSpike(): React.JSX.Element {
  const [stretch, setStretch] = useState(1);
  const tasks = useMemo(() => stretchTasks(BASE_TASKS, stretch), [stretch]);
  const span = useMemo(() => spanNs(tasks), [tasks]);
  const finished = useMemo(() => runToEnd("sim", tasks), [tasks]);
  const allInstants = useMemo(() => instantsOf(finished), [finished]);
  const instants = allInstants.length;

  const [real, setReal] = useState<ClockState>(() => createClock("real", tasks));
  const [sim, setSim] = useState<ClockState>(() => createClock("sim", tasks));
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(0.5);

  const tasksRef = useRef(tasks);
  tasksRef.current = tasks;

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
        setReal((s) => (s.done ? s : stepReal(s, dt * speedRef.current * NS_PER_MS, tasksRef.current)));
        // Virtual time advances one whole event per tick — the gap costs nothing.
        setSim((s) => (s.done ? s : stepSim(s, tasksRef.current)));
      }
      handle = requestAnimationFrame(frame);
    };
    handle = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(handle);
  }, []);

  const reset = (next: readonly typeof tasks[number][] = tasks) => {
    setReal(createClock("real", next));
    setSim(createClock("sim", next));
    setRunning(false);
  };

  // Changing the stretch is a different workload, so both clocks start over on it.
  useEffect(() => {
    setReal(createClock("real", tasks));
    setSim(createClock("sim", tasks));
    setRunning(false);
  }, [tasks]);

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
        The faint vertical lines are the <em>only</em> moments anything happens — all{" "}
        <strong>{instants}</strong> of them. Everything between is empty, however long the
        benchmark is. That is what the two clocks disagree about the price of. Stretch it to
        an hour and watch which pane&apos;s cost moves.
      </p>
      <p className="mb-4 max-w-5xl text-base leading-relaxed text-ink-secondary">
        Press Run. In both panes the hollow circles are wakeups that have not happened yet, each
        sitting at the moment it is due. The left pane&apos;s marker walks to them one millisecond
        at a time; the right pane&apos;s jumps straight to the next one, because nothing in the gap
        is worth visiting. Then compare the two result tables:{" "}
        <strong>same requests, same tokens, same latencies</strong>. That is
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
            <button type="button" onClick={() => reset()}
              className="rounded border border-white/15 bg-surface-panel px-4 py-2 text-base font-semibold text-ink-secondary">
              Reset
            </button>
            <button type="button"
              onClick={() => { setSim((s) => (s.done ? s : stepSim(s, tasks))); setRunning(false); }}
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
          <div className="flex items-center gap-1.5">
            <span className="mr-1 text-base text-ink-tertiary">benchmark length</span>
            {STRETCHES.map((o) => (
              <button key={o.factor} type="button" onClick={() => setStretch(o.factor)}
                title="Same requests, same events — only the empty time between them changes"
                className={`rounded border px-3 py-1.5 text-sm font-semibold ${
                  stretch === o.factor ? "border-transparent bg-accent-primary text-black"
                    : "border-white/15 bg-surface-panel text-ink-secondary"}`}>
                {o.label}
              </button>
            ))}
          </div>
          <div className="ml-auto text-base tabular-nums">
            <strong>{humanNs(span)}</strong>
            <span className="text-ink-quaternary"> of benchmark · always {finished.events.length} events</span>
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
          hint="RealClock · current-thread tokio, real timers"
          tasks={tasks} span={span} instants={allInstants}
          cost={
            <span>
              <strong style={{ color: BLUE }}>{humanNs(span)}</strong> of elapsed time — the whole
              span, empty stretches and all, even though only <strong>{instants}</strong> moments in it
              contain anything. Cost is O(span), so stretching the benchmark stretches this.
            </span>
          } />
        <ClockPane state={sim} label="Simulated time — instant" accent={CYAN}
          hint="SimClock · advance_to(next_event_time), event by event"
          tasks={tasks} span={span} instants={allInstants}
          cost={
            <span>
              <strong style={{ color: CYAN }}>{instants} visits</strong> — one per moment, and
              nothing at all for the space between: <code>advance_to</code> steps over each in a
              single assignment. Cost is O(events), so this number never moves.
            </span>
          } />
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

/** One colour per request, so a lane, its parked circle, and its log line all match. */
const REQ_COLORS = [
  "var(--color-category-blue)",
  "var(--color-category-green)",
  "var(--color-category-purple)",
  "var(--color-category-orange)",
  "var(--color-category-cyan)",
  "var(--color-category-yellow)",
  "var(--color-category-red)",
  "var(--color-category-gray)",
];

/** `req 3` → 2. The lane index is the request's own number, so lanes never reshuffle. */
function laneOf(taskId: string): number {
  const n = Number.parseInt(taskId.replace(/\D+/g, ""), 10);
  return Number.isFinite(n) ? n - 1 : 0;
}

const LANE_H = 30;
const PAD_L = 62;
const PAD_R = 14;
const VIEW_W = 760;

/**
 * The run, drawn on its own clock.
 *
 * One lane per request; x is time. Everything left of the playhead has happened and is solid.
 * Everything right of it is still parked — a hollow circle sitting at the deadline it is waiting
 * for. That is the heap, drawn where its entries actually point rather than as a list of numbers.
 *
 * The two panes differ only in how the playhead gets across: the real one slides, and the
 * simulated one jumps to the next hollow circle, because there is nothing in between worth
 * visiting.
 */
function ClockTrack({
  state,
  accent,
  tasks,
  span,
  instants,
}: {
  state: ClockState;
  accent: string;
  tasks: readonly Task[];
  span: number;
  /** Every moment in which anything happens. Between them the run is empty. */
  instants: readonly number[];
}): React.JSX.Element {
  const lanes = tasks.length;
  const height = lanes * LANE_H + 26;
  const innerW = VIEW_W - PAD_L - PAD_R;
  const x = (ns: number) => PAD_L + (Math.min(ns, span) / Math.max(1, span)) * innerW;
  const y = (lane: number) => 20 + lane * LANE_H + LANE_H / 2;

  // Per lane: where it was sent, where its first token landed, and its most recent event. Those
  // three points are the whole shape of a request — wait, then stream.
  const sent = new Map<number, number>();
  const first = new Map<number, number>();
  const last = new Map<number, number>();
  const tokens = new Map<number, number[]>();
  for (const e of state.events) {
    const lane = laneOf(e.taskId);
    if (e.label === "sent") sent.set(lane, e.atNs);
    else {
      if (e.label === "first token") first.set(lane, e.atNs);
      else {
        const list = tokens.get(lane);
        if (list === undefined) tokens.set(lane, [e.atNs]);
        else list.push(e.atNs);
      }
      last.set(lane, e.atNs);
    }
  }

  const nowX = x(state.nowNs);
  const nextAt = state.heap[0]?.atNs;

  return (
    <svg viewBox={`0 0 ${VIEW_W} ${height}`} width="100%" height={height}
      role="img" aria-label="requests over time, with parked wakeups ahead of the playhead">
      {/* The span the whole run occupies, so both panes share one frame of reference. */}
      {/*
        The run, in full. Everything not on one of these lines is empty — and since events have no
        width, that is the entire rest of the span. Drawing the emptiness would be drawing the
        whole rectangle, which says nothing; drawing the moments says it exactly.
      */}
      {instants.map((t, i) => (
        <line key={i} x1={x(t)} x2={x(t)} y1={14} y2={height - 6}
          stroke="rgba(255,255,255,0.16)" strokeWidth={1} />
      ))}

      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={f}>
          <line x1={x(f * span)} x2={x(f * span)} y1={14} y2={height - 6}
            stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
          <text x={x(f * span)} y={10} fontSize={10} textAnchor="middle" fill="var(--color-ink-quaternary)">
            {humanNs(f * span)}
          </text>
        </g>
      ))}

      {tasks.map((task, lane) => {
        const color = REQ_COLORS[lane % REQ_COLORS.length]!;
        const sentAt = sent.get(lane);
        const firstAt = first.get(lane);
        const lastAt = last.get(lane);
        const parked = state.heap.find((s) => laneOf(s.taskId) === lane);
        const isNext = parked !== undefined && parked.atNs === nextAt;
        const cy = y(lane);
        return (
          <g key={task.id}>
            <text x={PAD_L - 8} y={cy + 4} fontSize={12} textAnchor="end"
              fill={sentAt === undefined ? "var(--color-ink-quaternary)" : color}
              fontFamily="var(--font-mono, monospace)">
              {task.id}
            </text>
            <line x1={PAD_L} x2={VIEW_W - PAD_R} y1={cy} y2={cy}
              stroke="rgba(255,255,255,0.06)" strokeWidth={1} />

            {/* Waiting for the first token: the request exists and nothing has come back yet. */}
            {sentAt !== undefined && (
              <line x1={x(sentAt)} x2={x(firstAt ?? Math.min(state.nowNs, span))} y1={cy} y2={cy}
                stroke={color} strokeWidth={3} strokeDasharray="3 3" opacity={0.5} />
            )}
            {/* Streaming: solid, because tokens are arriving. */}
            {firstAt !== undefined && lastAt !== undefined && (
              <line x1={x(firstAt)} x2={x(lastAt)} y1={cy} y2={cy}
                stroke={color} strokeWidth={5} strokeLinecap="round" opacity={0.85} />
            )}
            {sentAt !== undefined && (
              <rect x={x(sentAt) - 2.5} y={cy - 5} width={5} height={10} fill={color} />
            )}
            {(tokens.get(lane) ?? []).map((t, i) => (
              <circle key={i} cx={x(t)} cy={cy} r={2.5} fill="var(--color-surface-page)"
                stroke={color} strokeWidth={1.5} />
            ))}
            {firstAt !== undefined && <circle cx={x(firstAt)} cy={cy} r={4.5} fill={color} />}

            {/* The parked wakeup: a hollow circle sitting at the deadline it is waiting on. */}
            {parked !== undefined && (
              <g>
                <line x1={nowX} x2={x(parked.atNs)} y1={cy} y2={cy}
                  stroke={color} strokeWidth={1} strokeDasharray="2 4" opacity={0.35} />
                <circle cx={x(parked.atNs)} cy={cy} r={isNext ? 8 : 6}
                  fill="none" stroke={color} strokeWidth={isNext ? 2.5 : 1.5}
                  strokeDasharray={isNext ? undefined : "3 2"} opacity={isNext ? 1 : 0.6}>
                  {isNext && (
                    <animate attributeName="r" values="8;10;8" dur="1.1s" repeatCount="indefinite" />
                  )}
                </circle>
                <text x={x(parked.atNs)} y={cy - 12} fontSize={9} textAnchor="middle"
                  fill={color} opacity={isNext ? 0.95 : 0.5}>
                  seq {parked.seqNo}
                </text>
              </g>
            )}
          </g>
        );
      })}

      {/* Now. The whole difference between the two panes is how this line travels. */}
      <line x1={nowX} x2={nowX} y1={12} y2={height - 4} stroke={accent} strokeWidth={2} />
      <polygon points={`${nowX - 5},12 ${nowX + 5},12 ${nowX},19`} fill={accent} />
    </svg>
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
  tasks,
  span,
  instants,
  cost,
}: {
  state: ClockState;
  label: string;
  accent: string;
  hint: string;
  tasks: readonly Task[];
  span: number;
  instants: readonly number[];
  /** What this clock is billed for, in its own currency. */
  cost: React.ReactNode;
}): React.JSX.Element {
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = logRef.current;
    if (el !== null) el.scrollTop = el.scrollHeight;
  }, [state.events.length]);

  const progress = Math.min(1, state.nowNs / Math.max(1, span));
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
          <strong>{humanNs(state.nowNs)}</strong></span>
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
        THE RUN
      </div>
      <div className="mb-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-ink-quaternary">
        <span><span className="text-ink-secondary">▮</span> sent</span>
        <span><span className="text-ink-secondary">●</span> first token</span>
        <span><span className="text-ink-secondary">◦</span> each token</span>
        <span>┈ waiting</span>
        <span><span className="text-ink-secondary">│</span> a moment something happens</span>
        <span><span className="text-ink-secondary">━</span> streaming</span>
        <span><span className="text-ink-secondary">◯</span> parked — hasn&apos;t happened yet</span>
        <span style={{ color: accent }}>▎ now</span>
      </div>
      <div className="mb-3">
        <ClockTrack state={state} accent={accent} tasks={tasks} span={span} instants={instants} />
      </div>

      <div className="mb-3 rounded border px-3 py-2 text-[15px]"
        style={{ borderColor: "rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.02)" }}>
        <div className="mb-0.5 text-[12px] font-bold tracking-widest text-ink-secondary">
          BILLED FOR
        </div>
        {cost}
      </div>

      <div className="mb-1.5 text-[12px] font-bold tracking-widest text-ink-secondary">
        BENCHMARK RESULT
      </div>
      <div className="mb-3 grid grid-cols-2 gap-x-6 gap-y-2 text-[16px] tabular-nums">
        <Metric label="requests done" value={`${summarize(state).completed}`} />
        <Metric label="tokens" value={`${summarize(state).tokens}`} />
        <Metric label="mean TTFT" value={humanNs(summarize(state).meanTtftMs * NS_PER_MS)} accent={accent} />
        <Metric label="mean ITL" value={humanNs(summarize(state).meanItlMs * NS_PER_MS)} accent={accent} />
        <Metric label="slowest request" value={humanNs(summarize(state).p100LatencyMs * NS_PER_MS)} />
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
