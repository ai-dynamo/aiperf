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
};

export type ClockKind = "real" | "sim";

export type ClockState = {
  kind: ClockKind;
  /** Virtual time for sim; monotonic time for real. Same units, same workload. */
  nowNs: number;
  /** Wall milliseconds actually spent driving. The whole difference lives here. */
  wallMs: number;
  seq: number;
  heap: Sleeper[];
  /** `taskId@step` per wake, in order. The comparable output. */
  events: string[];
  /** Nothing parked and work outstanding: no virtual event can make progress. */
  deadlocked: boolean;
  done: boolean;
};

export const NS_PER_MS = 1_000_000;

/** A workload with staggered starts and one deliberate same-deadline collision. */
export function defaultTasks(): Task[] {
  return [
    { id: "warmup", sleepsNs: [200 * NS_PER_MS, 300 * NS_PER_MS] },
    { id: "poll", sleepsNs: [500 * NS_PER_MS, 500 * NS_PER_MS, 500 * NS_PER_MS] },
    // These two land on the same deadline as `poll`'s first wake: 500ms.
    { id: "tie-a", sleepsNs: [500 * NS_PER_MS, 900 * NS_PER_MS] },
    { id: "tie-b", sleepsNs: [500 * NS_PER_MS] },
    { id: "drain", sleepsNs: [1_400 * NS_PER_MS] },
  ];
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
    state.events.push(`${sleeper.taskId}@${sleeper.step}`);
    const task = tasks.find((t) => t.id === sleeper.taskId);
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
