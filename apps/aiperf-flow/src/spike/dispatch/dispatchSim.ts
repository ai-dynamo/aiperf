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

export type Mode = "sharded" | "global" | "global-hop";

/**
 * Worker-assignment policy at the single `GlobalHop` pick site.
 *
 * Ported from `HopRouting` (`config/model/dispatch.rs:51`). The hop chooses only *which worker
 * executes an already-issued request*; every global-hop guarantee is coordinator-side and
 * unaffected, so the policy is free to trade placement determinism for connection reuse.
 */
export type HopRouting = "round-robin" | "sticky" | "least-loaded";

/**
 * `fnv1a64`, ported exactly from `turn_execution.rs:481`.
 *
 * Deliberately not `DefaultHasher`: seed-free and stable across processes and runs, so the same
 * correlation id always lands on the same worker.
 */
export function fnv1a64(text: string): bigint {
  let hash = 0xcbf2_9ce4_8422_2325n;
  const prime = 0x0000_0100_0000_01b3n;
  const mask = 0xffff_ffff_ffff_ffffn;
  for (const ch of text) hash = ((hash ^ BigInt(ch.codePointAt(0) ?? 0)) * prime) & mask;
  return hash;
}

/** Mutable state the pick site carries between decisions. */
export type PickState = {
  /** Advanced only when a round-robin pick is actually made. */
  rrCursor: number;
  /** correlation id to worker, bound by `least-loaded` on first sight. */
  sticky: Map<string, number>;
};

export function createPickState(): PickState {
  return { rrCursor: 0, sticky: new Map() };
}

/**
 * `pick_worker`, ported from `turn_execution.rs:438`.
 *
 * - `round-robin` takes `cursor % workers` and advances the cursor.
 * - `sticky` hashes the correlation id, falling back to round-robin when there is none.
 * - `least-loaded` honours an existing binding, else picks the shallowest in-flight worker with
 *   ties resolving to the lowest index, then binds the correlation id to it.
 */
export function pickWorker(
  routing: HopRouting,
  workers: number,
  correlation: string | null,
  inflight: readonly number[],
  state: PickState,
): number {
  const roundRobin = (): number => {
    const worker = state.rrCursor % workers;
    state.rrCursor += 1;
    return worker;
  };
  if (routing === "round-robin") return roundRobin();
  if (routing === "sticky") {
    if (correlation === null) return roundRobin();
    return Number(fnv1a64(correlation) % BigInt(workers));
  }
  if (correlation !== null) {
    const bound = state.sticky.get(correlation);
    if (bound !== undefined) return bound;
  }
  let worker = 0;
  for (let i = 1; i < workers; i++) {
    if ((inflight[i] ?? 0) < (inflight[worker] ?? 0)) worker = i;
  }
  if (correlation !== null) state.sticky.set(correlation, worker);
  return worker;
}

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

/**
 * Per-worker caps under a mode.
 *
 * Only `sharded` has per-worker caps at all. `global` admits from one shared pool; `global-hop`
 * runs one coordinator loop holding the full un-sliced cap, which — as `global_hop.rs:22` puts it
 * — *is* the global cap, so it needs no cross-thread gate either.
 */
export function capsFor(mode: Mode, concurrency: number, workers: number): number[] {
  if (mode !== "sharded") return Array.from({ length: workers }, () => concurrency);
  return Array.from({ length: workers }, (_, t) => ownedCap(concurrency, t, workers));
}

/** What the caps add up to. Equal to the target only when the split happens to tile. */
export function admissibleTotal(mode: Mode, concurrency: number, workers: number): number {
  if (mode !== "sharded") return concurrency;
  return capsFor("sharded", concurrency, workers).reduce((a, b) => a + b, 0);
}

export type Request = {
  id: number;
  worker: number;
  startedAt: number;
  /** Ticks of service. Uneven by design — that is the condition sharding loses under. */
  duration: number;
  /** The conversation this turn belongs to. Routing policies key on it. */
  correlation: string;
};

/** Which conversation a global request index belongs to. Stable across every mode. */
export function correlationOf(id: number, config: Config): string {
  return `session-${id % config.sessions}`;
}

export type Config = {
  workers: number;
  concurrency: number;
  /** Total requests to dispatch, before slicing. */
  requests: number;
  shortTicks: number;
  longTicks: number;
  /** One in `slowEvery` requests is a long one. */
  slowEvery: number;
  /** Distinct conversations the requests belong to. */
  sessions: number;
  /**
   * Ticks of extra service charged to a `global-hop` request for the cross-thread trip.
   *
   * Every hopped request crosses a bounded mpsc to a worker thread and awaits a oneshot reply
   * (`global_hop.rs:10`), so the cost is per-request and real. The magnitude here is illustrative;
   * the runtime's own measured figure is for a different comparison — `Global` inside a cellular
   * run was ~7-8x slower than `Sharded` on a c4-144 (`protocol_v2.rs:255`), which is why cellular
   * runs default to `Sharded`.
   */
  hopCostTicks: number;
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
  // A full avalanche, not a single multiply. `sharded`/`global` deal ids `k, k+W, …` to worker
  // `k`, so any mixer whose low bits still track `id % W` hands one worker every slow request and
  // manufactures a far larger gap than the real effect. This finalizer decorrelates them.
  let h = (id + 1) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 2246822507) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 3266489909) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h % config.slowEvery === 0 ? config.longTicks : config.shortTicks;
}

export const DEFAULT_CONFIG: Config = {
  workers: 4,
  concurrency: 8,
  requests: 96,
  shortTicks: 3,
  longTicks: 17,
  slowEvery: 4,
  sessions: 6,
  hopCostTicks: 1,
};

export type WorkerState = {
  id: number;
  cap: number;
  /**
   * Global request indices this worker still has to dispatch.
   *
   * Held as explicit ids rather than a count so that a request's duration and correlation are the
   * same in every mode — otherwise the comparison would be between different workloads.
   * `sharded`/`global` deal ids `k, k+W, k+2W, …` to worker `k`, matching the round-robin share
   * `owned_positions` produces. `global-hop` leaves every id on the coordinator instead.
   */
  queue: number[];
  inFlight: Request[];
  completed: number;
  /** Distinct workers this worker's sessions have also been served by. Populated by the hop. */
};

export type RunState = {
  mode: Mode;
  tick: number;
  workers: WorkerState[];
  /** Shared pool occupancy under `global`; unused under `sharded`. */
  globalHeld: number;
  /** Ids the single coordinator has yet to issue. Only `global-hop` uses it. */
  coordinatorQueue: number[];
  routing: HopRouting;
  pick: PickState;
  /** correlation id to the set of workers that has served it. One is ideal; more fragments pools. */
  touched: Map<string, Set<number>>;
  /** Ticks charged to cross-thread hops, which no other mode pays. */
  hopTicksCharged: number;
  /** Aggregate in-flight at each tick — the only place the difference shows up. */
  curve: number[];
  /** Whether any worker still had budget to admit at each tick, parallel to `curve`. */
  admitting: boolean[];
  done: boolean;
};

export function createRun(
  mode: Mode,
  config: Config = DEFAULT_CONFIG,
  routing: HopRouting = "round-robin",
): RunState {
  const caps = capsFor(mode, config.concurrency, config.workers);
  const hop = mode === "global-hop";
  return {
    mode,
    tick: 0,
    workers: Array.from({ length: config.workers }, (_, id) => ({
      id,
      cap: caps[id]!,
      // The request budget is sliced under `sharded` AND `global` — the shared gate is a
      // concurrency/rate gate only, with no shared request counter. `global-hop` has one
      // coordinator issuing in exact global order, so nothing is partitioned up front.
      queue: hop
        ? []
        : Array.from({ length: ownedPositions(config.requests, id, config.workers) },
            (_, k) => id + k * config.workers),
      inFlight: [],
      completed: 0,
    })),
    globalHeld: 0,
    coordinatorQueue: hop ? Array.from({ length: config.requests }, (_, i) => i) : [],
    routing,
    pick: createPickState(),
    touched: new Map(),
    hopTicksCharged: 0,
    curve: [],
    admitting: [],
    done: false,
  };
}

export function inFlightTotal(state: RunState): number {
  return state.workers.reduce((sum, w) => sum + w.inFlight.length, 0);
}

/** Requests not yet dispatched, wherever they are queued. */
export function pendingWork(state: RunState): number {
  return (
    state.coordinatorQueue.length + state.workers.reduce((sum, w) => sum + w.queue.length, 0)
  );
}

/** Note that `worker` served `correlation`, for the connection-fragmentation count. */
function recordTouch(state: RunState, correlation: string, worker: number): void {
  const seen = state.touched.get(correlation);
  if (seen === undefined) state.touched.set(correlation, new Set([worker]));
  else seen.add(worker);
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
    workers: state.workers.map((w) => ({ ...w, inFlight: [...w.inFlight], queue: [...w.queue] })),
    coordinatorQueue: [...state.coordinatorQueue],
    pick: { rrCursor: state.pick.rrCursor, sticky: new Map(state.pick.sticky) },
    touched: new Map([...state.touched].map(([k, v]) => [k, new Set(v)])),
    curve: [...state.curve],
    admitting: [...state.admitting],
  };

  for (const worker of next.workers) {
    const finished = worker.inFlight.filter((r) => next.tick - r.startedAt >= r.duration);
    if (finished.length > 0) {
      worker.inFlight = worker.inFlight.filter((r) => next.tick - r.startedAt < r.duration);
      worker.completed += finished.length;
      if (next.mode !== "sharded") next.globalHeld -= finished.length;
    }
  }

  if (next.mode === "global-hop") {
    // ONE coordinator loop, driven with the FULL un-sliced cap. `global_hop.rs:22`: one loop
    // holding the full cap *is* the global cap, so there is no cross-thread admission gate and
    // no race between scheduling loops. Issuance order is exact global order.
    const inflight = next.workers.map((w) => w.inFlight.length);
    while (next.coordinatorQueue.length > 0 && next.globalHeld < config.concurrency) {
      const id = next.coordinatorQueue.shift()!;
      const correlation = correlationOf(id, config);
      const target = pickWorker(next.routing, config.workers, correlation, inflight, next.pick);
      const worker = next.workers[target]!;
      worker.inFlight.push({
        id,
        worker: target,
        startedAt: next.tick,
        // The hop is a bounded mpsc plus a oneshot reply, charged per request.
        duration: durationFor(id, config) + config.hopCostTicks,
        correlation,
      });
      inflight[target] = (inflight[target] ?? 0) + 1;
      next.globalHeld += 1;
      next.hopTicksCharged += config.hopCostTicks;
      recordTouch(next, correlation, target);
    }
  } else {
    for (const worker of next.workers) {
      while (worker.queue.length > 0) {
        const admitted =
          next.mode === "global"
            ? next.globalHeld < config.concurrency
            : worker.inFlight.length < worker.cap;
        if (!admitted) break;
        const id = worker.queue.shift()!;
        const correlation = correlationOf(id, config);
        worker.inFlight.push({
          id,
          worker: worker.id,
          startedAt: next.tick,
          duration: durationFor(id, config),
          correlation,
        });
        if (next.mode === "global") next.globalHeld += 1;
        recordTouch(next, correlation, worker.id);
      }
    }
  }

  next.curve.push(inFlightTotal(next));
  next.admitting.push(pendingWork(next) > 0);
  next.done = pendingWork(next) === 0 && next.workers.every((w) => w.inFlight.length === 0);
  return next;
}

/** Run to completion, for comparing the two modes without waiting on animation. */
export function runToEnd(
  mode: Mode,
  config: Config = DEFAULT_CONFIG,
  routing: HopRouting = "round-robin",
): RunState {
  let state = createRun(mode, config, routing);
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
  const anyoneWaiting = state.workers.some((w) => w.queue.length > 0);
  if (!anyoneWaiting) return 0;
  return state.workers
    .filter((w) => w.queue.length === 0)
    .reduce((sum, w) => sum + Math.max(0, w.cap - w.inFlight.length), 0);
}

/** Distinct workers that served each session. One is ideal; more fragments the sticky pool. */
export function fragmentation(state: RunState): { mean: number; worst: number } {
  const counts = [...state.touched.values()].map((s) => s.size);
  if (counts.length === 0) return { mean: 0, worst: 0 };
  return {
    mean: counts.reduce((a, b) => a + b, 0) / counts.length,
    worst: Math.max(...counts),
  };
}
