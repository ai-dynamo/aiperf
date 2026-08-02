/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — sharded versus global admission.
//!
//! Modelled on the runtime, read before writing:
//!
//! - `DispatchMode` (`engine/protocol.rs:14`): `Sharded` statically partitions concurrency and
//!   rate `1/workers` up front; `Global` (the default) admits from one shared per-cell pool so
//!   aggregate concurrency is byte-exact against a single global limiter.
//! - `slice_phase_for_thread` (`engine/sharded_scheduled.rs:128`) does the slicing. Two details
//!   matter and are easy to get wrong: the request *budget* is sliced in **every** mode, because
//!   the shared gate covers concurrency and rate only; and `owned_cap` applies `.max(1)`.
//! - `owned_positions` (`engine/cell_launcher.rs:272`) is the split itself:
//!   `k >= total ? 0 : ceil((total - k) / count)`, which tiles exactly.
//! - `VariableLatencyMock` (`engine/workers_characterization.rs:1325`) states the condition under
//!   which sharding is visibly wrong: *uneven completion times*. A thread stuck behind a slow
//!   request cannot borrow capacity from a thread whose fast requests already finished, so the
//!   wire-observed aggregate concurrency drifts below what a shared pool would allow.
//!
//! That drift is the point of the page. It is invisible per worker — every lane is obeying its own
//! cap correctly — and only exists in the sum.

export type Mode = "sharded" | "global";

/**
 * `owned_positions`: this worker's share of `total`.
 *
 * Ported exactly, including the early zero. The shares tile — they sum to `total` — which is what
 * makes the static partition defensible right up until completion times stop being even.
 */
export function ownedPositions(total: number, workerId: number, workers: number): number {
  if (workerId >= total) return 0;
  return Math.ceil((total - workerId) / workers);
}

/**
 * `owned_cap`: an admission cap, floored to one.
 *
 * The floor is deliberate — a cap below the thread count would otherwise starve a thread. The
 * consequence is that the shares stop tiling and start *over-subscribing*: four workers under a
 * concurrency of three admit four at once.
 */
export function ownedCap(total: number, workerId: number, workers: number): number {
  return Math.max(1, ownedPositions(total, workerId, workers));
}

/** Per-worker caps under a mode. Under `global` there is no per-worker cap at all. */
export function capsFor(mode: Mode, concurrency: number, workers: number): number[] {
  if (mode === "global") return Array.from({ length: workers }, () => concurrency);
  return Array.from({ length: workers }, (_, t) => ownedCap(concurrency, t, workers));
}

/** What the caps add up to. Equal to the target only when the split happens to tile. */
export function admissibleTotal(mode: Mode, concurrency: number, workers: number): number {
  if (mode === "global") return concurrency;
  return capsFor("sharded", concurrency, workers).reduce((a, b) => a + b, 0);
}

export type Request = {
  id: number;
  worker: number;
  startedAt: number;
  /** Ticks of service. Alternates short/long by arrival order, as the characterization mock does. */
  duration: number;
};

export type Config = {
  workers: number;
  concurrency: number;
  /** Total requests to dispatch, before slicing. */
  requests: number;
  shortTicks: number;
  longTicks: number;
  /** One in `slowEvery` requests is a long one. */
  slowEvery: number;
};

/**
 * Service time for a request, deterministic in its id.
 *
 * Draws are spread by a multiplicative hash rather than by `id % slowEvery`, because a strict
 * modulus deals the slow requests round-robin and every worker receives exactly the same mix. Real
 * unevenness is unequal *totals* per worker, which is what leaves one lane still working while
 * another has run out of budget.
 */
export function durationFor(id: number, config: Config): number {
  const mixed = (Math.imul(id + 1, 2654435761) >>> 0) % config.slowEvery;
  return mixed === 0 ? config.longTicks : config.shortTicks;
}

export const DEFAULT_CONFIG: Config = {
  workers: 4,
  concurrency: 8,
  requests: 96,
  shortTicks: 3,
  longTicks: 17,
  slowEvery: 4,
};

export type WorkerState = {
  id: number;
  cap: number;
  /** Requests this worker still has to dispatch. Sliced in every mode. */
  remaining: number;
  inFlight: Request[];
  completed: number;
};

export type RunState = {
  mode: Mode;
  tick: number;
  workers: WorkerState[];
  /** Shared pool occupancy under `global`; unused under `sharded`. */
  globalHeld: number;
  nextId: number;
  /** Aggregate in-flight at each tick — the only place the difference shows up. */
  curve: number[];
  /** Whether any worker still had budget to admit at each tick, parallel to `curve`. */
  admitting: boolean[];
  done: boolean;
};

export function createRun(mode: Mode, config: Config = DEFAULT_CONFIG): RunState {
  const caps = capsFor(mode, config.concurrency, config.workers);
  return {
    mode,
    tick: 0,
    workers: Array.from({ length: config.workers }, (_, id) => ({
      id,
      cap: caps[id]!,
      // The request budget is sliced under EVERY mode — the shared gate is a
      // concurrency/rate gate only, with no shared request counter.
      remaining: ownedPositions(config.requests, id, config.workers),
      inFlight: [],
      completed: 0,
    })),
    globalHeld: 0,
    nextId: 0,
    curve: [],
    admitting: [],
    done: false,
  };
}

export function inFlightTotal(state: RunState): number {
  return state.workers.reduce((sum, w) => sum + w.inFlight.length, 0);
}

/**
 * One tick: retire everything that finished, then admit whatever the mode allows.
 *
 * Retiring first is what gives a freed slot a chance to be reused in the same tick — under
 * `global` by any worker, under `sharded` only by the worker that freed it. That asymmetry is the
 * entire mechanism.
 */
export function step(state: RunState, config: Config = DEFAULT_CONFIG): RunState {
  if (state.done) return state;

  const next: RunState = {
    ...state,
    tick: state.tick + 1,
    workers: state.workers.map((w) => ({ ...w, inFlight: [...w.inFlight] })),
    curve: [...state.curve],
    admitting: [...state.admitting],
  };

  for (const worker of next.workers) {
    const finished = worker.inFlight.filter((r) => next.tick - r.startedAt >= r.duration);
    if (finished.length > 0) {
      worker.inFlight = worker.inFlight.filter((r) => next.tick - r.startedAt < r.duration);
      worker.completed += finished.length;
      if (next.mode === "global") next.globalHeld -= finished.length;
    }
  }

  for (const worker of next.workers) {
    while (worker.remaining > 0) {
      const admitted =
        next.mode === "global"
          ? next.globalHeld < config.concurrency
          : worker.inFlight.length < worker.cap;
      if (!admitted) break;
      const id = next.nextId++;
      worker.inFlight.push({
        id,
        worker: worker.id,
        startedAt: next.tick,
        duration: durationFor(id, config),
      });
      worker.remaining -= 1;
      if (next.mode === "global") next.globalHeld += 1;
    }
  }

  next.curve.push(inFlightTotal(next));
  next.admitting.push(next.workers.some((w) => w.remaining > 0));
  next.done = next.workers.every((w) => w.remaining === 0 && w.inFlight.length === 0);
  return next;
}

/** Run to completion, for comparing the two modes without waiting on animation. */
export function runToEnd(mode: Mode, config: Config = DEFAULT_CONFIG): RunState {
  let state = createRun(mode, config);
  let guard = 100_000;
  while (!state.done && guard-- > 0) state = step(state, config);
  return state;
}

export type Summary = {
  /** Mean aggregate in-flight while the run was actually dispatching. */
  meanInFlight: number;
  peakInFlight: number;
  ticks: number;
  completed: number;
  /** Mean in-flight as a fraction of the configured target. */
  utilisation: number;
};

/**
 * What the run achieved against what it was told to hold.
 *
 * Only ticks in which some worker still had budget are counted. The final drain is excluded — a
 * drain is not under-utilisation — but a tick where one lane is still working and another has run
 * dry very much is, and that is precisely where the static partition loses.
 */
export function summarize(state: RunState, config: Config = DEFAULT_CONFIG): Summary {
  const window = state.curve.filter((_, i) => state.admitting[i] === true);
  const mean = window.reduce((a, b) => a + b, 0) / Math.max(1, window.length);
  return {
    meanInFlight: mean,
    peakInFlight: Math.max(0, ...state.curve),
    ticks: window.length,
    completed: state.workers.reduce((sum, w) => sum + w.completed, 0),
    utilisation: mean / Math.max(1, config.concurrency),
  };
}


/**
 * Slots that exist, are free, and cannot be used.
 *
 * A worker that has finished its own budget still owns its cap; under `sharded` that capacity
 * cannot be lent to a worker that still has requests queued. Summing those free-but-unreachable
 * slots gives the exact size of the shortfall at each tick, and it is zero by construction under
 * `global`, where there is only one pool to be free in.
 */
export function strandedSlots(state: RunState): number {
  if (state.mode === "global") return 0;
  const anyoneWaiting = state.workers.some((w) => w.remaining > 0);
  if (!anyoneWaiting) return 0;
  return state.workers
    .filter((w) => w.remaining === 0)
    .reduce((sum, w) => sum + Math.max(0, w.cap - w.inFlight.length), 0);
}
