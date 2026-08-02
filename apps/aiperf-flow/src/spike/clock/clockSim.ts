/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the same workload on two clocks.
//!
//! Modelled on `rust/runtime/src/clock/`:
//!
//! - `Clock` (clock.rs:12) is `now_ns`, `sleep`, `is_virtual`, and `drive`. Everything on the hot
//!   path takes time through it; nothing calls `Instant::now` directly.
//! - `SimClock` (sim_clock.rs:48) parks sleepers in a `BinaryHeap` ordered by `(at_ns, seq_no)`.
//!   The comparison is reversed so the max-heap yields the *earliest* deadline, and the sequence
//!   number breaks same-deadline ties — which is what makes simultaneous wakes deterministic
//!   rather than arbitrary.
//! - `next_event_time` (sim_clock.rs:92) is `max(earliest deadline, now)`, or none when idle.
//! - `advance_to` (sim_clock.rs:106) jumps virtual time and drains every sleeper due at or before
//!   it, so the empty space between events costs nothing.
//! - `RealClock` takes the default `drive`: a current-thread tokio runtime whose real timers make
//!   that same empty space cost exactly as long as it says it does.
//!
//! The point of running both is that the *outputs* are identical while the wall time is not.

/** One parked sleeper. `seqNo` is the registration order, and the tie-break. */
export type Sleeper = {
  taskId: string;
  atNs: number;
  seqNo: number;
  /** Index of the next step this task will run when woken. */
  step: number;
};

/** A task is a fixed list of sleep durations; it emits one event per wake. */
export type Task = {
  id: string;
  sleepsNs: number[];
  /** What each wake means, so the log reads as a benchmark rather than a scheduler dump. */
  labels?: string[];
};

/** One benchmarked request: it is sent, waits out its TTFT, then streams tokens. */
export type Request = {
  id: string;
  arrivalNs: number;
  ttftNs: number;
  itlNs: number;
  tokens: number;
};

/**
 * Turn requests into the sleep schedule a driver actually walks.
 *
 * The point of doing it this way is that the workload is ordinary async code — sleep, wake, do
 * something — with no knowledge of which clock is underneath it. That is what makes the same run
 * possible on both.
 */
export function requestsToTasks(requests: readonly Request[]): Task[] {
  return requests.map((r) => {
    const sleepsNs = [r.arrivalNs, r.ttftNs];
    const labels = ["sent", "first token"];
    for (let i = 1; i < r.tokens; i++) {
      sleepsNs.push(r.itlNs);
      labels.push(i === r.tokens - 1 ? "done" : `token ${i + 1}`);
    }
    return { id: r.id, sleepsNs, labels };
  });
}

/** A small benchmark: eight requests, staggered arrivals, realistic TTFT and ITL. */
export function defaultRequests(): Request[] {
  return Array.from({ length: 8 }, (_, i) => ({
    id: `req ${i + 1}`,
    // The first request is sent at t=0. A lead-in before anything happens is dead screen, and it
    // scales with the stretch factor — at a 3-minute benchmark a 120ms lead-in is 12 seconds of
    // watching an empty track.
    arrivalNs: i * 140 * NS_PER_MS,
    ttftNs: (380 + ((i * 37) % 190)) * NS_PER_MS,
    itlNs: (24 + ((i * 11) % 18)) * NS_PER_MS,
    tokens: 6 + (i % 4),
  }));
}

export type ClockKind = "real" | "sim";

export type ClockState = {
  kind: ClockKind;
  /** Virtual time for sim; monotonic time for real. Same units, same workload. */
  nowNs: number;
  /** Wall milliseconds actually spent driving. The whole difference lives here. */
  wallMs: number;
  seq: number;
  heap: Sleeper[];
  /** One entry per wake, in order. This is the run's output — what the two clocks must agree on. */
  events: ClockEvent[];
  /** Nothing parked and work outstanding: no virtual event can make progress. */
  deadlocked: boolean;
  done: boolean;
};

export const NS_PER_MS = 1_000_000;

/** One thing that happened, and when it happened on this clock. */
export type ClockEvent = {
  taskId: string;
  step: number;
  label: string;
  atNs: number;
};

/** The default workload: the benchmark above, as sleeps. */
export function defaultTasks(): Task[] {
  return requestsToTasks(defaultRequests());
}

export function createClock(kind: ClockKind, tasks: readonly Task[] = defaultTasks()): ClockState {
  const state: ClockState = {
    kind,
    nowNs: 0,
    wallMs: 0,
    seq: 0,
    heap: [],
    events: [],
    deadlocked: false,
    done: false,
  };
  // Every task registers its first sleep up front, in task order — so the initial sequence
  // numbers are deterministic and the tie-break has something to order.
  for (const task of tasks) {
    schedule(state, task.id, task.sleepsNs[0] ?? 0, 0);
  }
  return state;
}

/** Park a sleeper at `now + duration`, taking the next sequence number. */
function schedule(state: ClockState, taskId: string, durationNs: number, step: number): void {
  state.heap.push({ taskId, atNs: state.nowNs + durationNs, seqNo: state.seq, step });
  state.seq += 1;
  sortHeap(state.heap);
}

/**
 * Order by `(at_ns, seq_no)`.
 *
 * The Rust uses a max-heap with a reversed comparison to get the same result; a sorted array is
 * the same ordering, just without the amortization that matters at scale and not here.
 */
function sortHeap(heap: Sleeper[]): void {
  heap.sort((a, b) => a.atNs - b.atNs || a.seqNo - b.seqNo);
}

/** `max(earliest parked deadline, now)`, or null when nothing is parked. */
export function nextEventTime(state: ClockState): number | null {
  const top = state.heap[0];
  if (top === undefined) return null;
  return top.atNs > state.nowNs ? top.atNs : state.nowNs;
}

/**
 * Wake every sleeper due at or before `ns`, in heap order.
 *
 * The Rust collects the crossed wakers under the borrow and fires them outside it, so a wake that
 * re-schedules cannot re-borrow the heap mid-iteration. The same split is kept here: due sleepers
 * are drained first, then run.
 */
function drainDue(state: ClockState, ns: number, tasks: readonly Task[]): void {
  const due: Sleeper[] = [];
  while (state.heap.length > 0 && state.heap[0]!.atNs <= ns) {
    due.push(state.heap.shift()!);
  }
  for (const sleeper of due) {
    const task = tasks.find((t) => t.id === sleeper.taskId);
    state.events.push({
      taskId: sleeper.taskId,
      step: sleeper.step,
      label: task?.labels?.[sleeper.step] ?? `step ${sleeper.step}`,
      atNs: ns,
    });
    const next = task?.sleepsNs[sleeper.step + 1];
    if (next !== undefined) schedule(state, sleeper.taskId, next, sleeper.step + 1);
  }
  if (state.heap.length === 0) state.done = true;
}

/**
 * One step of the simulated driver: jump straight to the next event.
 *
 * The gap between events costs no wall time at all, which is the entire proposition. A step that
 * finds nothing parked with work outstanding is the `RunOutcome::deadlocked` case.
 */
export function stepSim(input: ClockState, tasks: readonly Task[] = defaultTasks()): ClockState {
  const state: ClockState = { ...input, heap: [...input.heap], events: [...input.events] };
  if (state.done) return state;
  const next = nextEventTime(state);
  if (next === null) {
    state.deadlocked = !state.done;
    state.done = true;
    return state;
  }
  state.nowNs = Math.max(state.nowNs, next);
  // A pump tick is real work, but it is bounded by the event count rather than by the span.
  state.wallMs += 0.02;
  drainDue(state, state.nowNs, tasks);
  return state;
}

/**
 * One step of the real driver: time advances only as fast as it actually passes.
 *
 * `dtNs` is elapsed wall time, so the empty space between events costs exactly what it says.
 */
export function stepReal(
  input: ClockState,
  dtNs: number,
  tasks: readonly Task[] = defaultTasks(),
): ClockState {
  const state: ClockState = { ...input, heap: [...input.heap], events: [...input.events] };
  if (state.done) return state;
  state.nowNs += Math.max(0, dtNs);
  state.wallMs += Math.max(0, dtNs) / NS_PER_MS;
  drainDue(state, state.nowNs, tasks);
  return state;
}

/** Run a clock to completion, for comparing outputs without waiting on wall time. */
export function runToEnd(kind: ClockKind, tasks: readonly Task[] = defaultTasks()): ClockState {
  let state = createClock(kind, tasks);
  let guard = 10_000;
  while (!state.done && guard-- > 0) {
    state = kind === "sim" ? stepSim(state, tasks) : stepReal(state, 5 * NS_PER_MS, tasks);
  }
  return state;
}

/** Virtual time the workload spans, for scaling a shared axis. */
export function spanNs(tasks: readonly Task[] = defaultTasks()): number {
  return runToEnd("sim", tasks).nowNs;
}

/** What a benchmark run actually reports. This is the comparison an end user cares about. */
export type BenchmarkResult = {
  completed: number;
  tokens: number;
  meanTtftMs: number;
  meanItlMs: number;
  /** Longest request, end to end. */
  p100LatencyMs: number;
};

/**
 * Reduce a run's events into the numbers a benchmark exists to produce.
 *
 * Derived purely from event timestamps on that clock, which is the whole argument: if the two
 * clocks emit the same events at the same virtual times, they report the same results — and one
 * of them got there without waiting.
 */
export function summarize(state: ClockState): BenchmarkResult {
  const sent = new Map<string, number>();
  const first = new Map<string, number>();
  const last = new Map<string, number>();
  let tokens = 0;

  for (const event of state.events) {
    if (event.label === "sent") sent.set(event.taskId, event.atNs);
    else {
      tokens += 1;
      if (event.label === "first token") first.set(event.taskId, event.atNs);
      last.set(event.taskId, event.atNs);
    }
  }

  const ttfts: number[] = [];
  const latencies: number[] = [];
  const itls: number[] = [];
  for (const [id, sentAt] of sent) {
    const firstAt = first.get(id);
    const lastAt = last.get(id);
    if (firstAt === undefined || lastAt === undefined) continue;
    ttfts.push((firstAt - sentAt) / NS_PER_MS);
    latencies.push((lastAt - sentAt) / NS_PER_MS);
    // Decode span divided by the gaps within it.
    const decodeEvents = state.events.filter(
      (e) => e.taskId === id && e.label !== "sent" && e.label !== "first token",
    ).length;
    if (decodeEvents > 0) itls.push((lastAt - firstAt) / NS_PER_MS / decodeEvents);
  }

  const mean = (xs: number[]) => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);
  return {
    completed: latencies.length,
    tokens,
    meanTtftMs: mean(ttfts),
    meanItlMs: mean(itls),
    p100LatencyMs: latencies.length === 0 ? 0 : Math.max(...latencies),
  };
}

/**
 * Stretch every sleep by `factor`, leaving the event count untouched.
 *
 * This is the asymmetry the two clocks are built on, made adjustable. `advance_to` costs one visit
 * per event regardless of how far apart they are, so simulation is O(events); a real timer must
 * actually wait out each gap, so reality is O(span). Stretching only the gaps drives those two
 * costs apart without changing a single thing about the run.
 */
export function stretchTasks(tasks: readonly Task[], factor: number): Task[] {
  return tasks.map((t) => ({ ...t, sleepsNs: t.sleepsNs.map((ns) => ns * factor) }));
}

/** A stretch of the timeline with no event in it. Real time is billed for these; virtual is not. */
export type IdleSpan = { fromNs: number; toNs: number };

/**
 * The gaps between consecutive event times.
 *
 * Events sharing a timestamp are one instant, not several, so the spans are built from distinct
 * times. The leading gap before the first event counts: nothing happens there either.
 */
export function idleSpans(state: ClockState): IdleSpan[] {
  const times = [...new Set(state.events.map((e) => e.atNs))].sort((a, b) => a - b);
  const spans: IdleSpan[] = [];
  let cursor = 0;
  for (const t of times) {
    if (t > cursor) spans.push({ fromNs: cursor, toNs: t });
    cursor = t;
  }
  return spans;
}

/**
 * Total time in which nothing at all happened.
 *
 * Events are instants with no width, so for a run that ends on an event this is the entire span —
 * which is the point, not a rounding artefact. A benchmark is a handful of moments with waiting
 * in between, and `countInstants` is the honest measure of how few of them there are.
 */
export function idleNs(state: ClockState): number {
  return idleSpans(state).reduce((sum, s) => sum + (s.toNs - s.fromNs), 0);
}

/** The distinct moments the run consists of. Simulation pays once for each and nothing between. */
export function instantsOf(state: ClockState): number[] {
  return [...new Set(state.events.map((e) => e.atNs))].sort((a, b) => a - b);
}

/** How many distinct moments the run actually consists of. Simulation pays once for each. */
export function countInstants(state: ClockState): number {
  return instantsOf(state).length;
}
