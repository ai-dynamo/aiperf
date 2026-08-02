/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  admissibleTotal,
  capsFor,
  createPickState,
  createRun,
  fnv1a64,
  fragmentation,
  pickWorker,
  prefillCapsFor,
  strandedPrefill,
  DEFAULT_CONFIG,
  inFlightTotal,
  ownedCap,
  ownedPositions,
  runToEnd,
  step,
  strandedSlots,
  summarize,
  type Config,
} from "./dispatchSim.js";

describe("owned_positions", () => {
  it("tiles exactly, for every total and worker count", () => {
    // The property the Rust's own test asserts: the shares sum to the total.
    for (const total of [1, 7, 96, 100, 501]) {
      for (let workers = 1; workers <= 8; workers++) {
        const sum = Array.from({ length: workers }, (_, k) =>
          ownedPositions(total, k, workers),
        ).reduce((a, b) => a + b, 0);
        expect(sum, `total ${total} workers ${workers}`).toBe(total);
      }
    }
  });

  it("gives a worker beyond the total nothing at all", () => {
    expect(ownedPositions(3, 3, 4)).toBe(0);
    expect(ownedPositions(3, 5, 8)).toBe(0);
  });
});

describe("the floor on admission caps", () => {
  it("over-subscribes when the target is smaller than the worker count", () => {
    // owned_cap applies .max(1) so no thread is starved. The cost is that four workers under a
    // concurrency of three admit four at once — the caps no longer tile, they exceed.
    expect(capsFor("sharded", 3, 4)).toEqual([1, 1, 1, 1]);
    expect(admissibleTotal("sharded", 3, 4)).toBe(4);
    expect(admissibleTotal("global", 3, 4)).toBe(3);
  });

  it("tiles exactly when the target divides evenly", () => {
    expect(capsFor("sharded", 8, 4)).toEqual([2, 2, 2, 2]);
    expect(admissibleTotal("sharded", 8, 4)).toBe(8);
  });

  it("never returns a cap below one", () => {
    for (let workers = 1; workers <= 8; workers++) {
      for (let k = 0; k < workers; k++) expect(ownedCap(1, k, workers)).toBeGreaterThanOrEqual(1);
    }
  });
});

describe("the request budget", () => {
  it("is sliced under both modes", () => {
    // The shared gate covers concurrency and rate only. There is no shared request counter, so
    // leaving the budget unsliced under `global` would dispatch `workers` duplicate copies of it.
    const sharded = createRun("sharded");
    const global = createRun("global");
    expect(global.workers.map((w) => w.queue.length)).toEqual(sharded.workers.map((w) => w.queue.length));
    expect(global.workers.reduce((a, w) => a + w.queue.length, 0)).toBe(DEFAULT_CONFIG.requests);
  });

  it("dispatches every request under both modes", () => {
    expect(summarize(runToEnd("sharded")).completed).toBe(DEFAULT_CONFIG.requests);
    expect(summarize(runToEnd("global")).completed).toBe(DEFAULT_CONFIG.requests);
  });
});

describe("the drift that only exists in the sum", () => {
  it("holds the target under global and falls short under sharded", () => {
    // VariableLatencyMock's premise: with uneven completion times, a thread stuck behind a slow
    // request cannot lend its idle capacity to a thread that has work waiting.
    const global = summarize(runToEnd("global"));
    const sharded = summarize(runToEnd("sharded"));
    expect(global.utilisation).toBeGreaterThan(sharded.utilisation);
    expect(global.meanInFlight).toBeGreaterThan(sharded.meanInFlight);
  });

  it("never exceeds the target under global", () => {
    expect(summarize(runToEnd("global")).peakInFlight).toBeLessThanOrEqual(
      DEFAULT_CONFIG.concurrency,
    );
  });

  it("disappears when every request takes the same time", () => {
    // The control. Sharding is not wrong in itself — it is wrong when completions are uneven, so
    // removing the unevenness should remove the gap.
    const even: Config = { ...DEFAULT_CONFIG, longTicks: DEFAULT_CONFIG.shortTicks };
    const gap =
      summarize(runToEnd("global", even), even).meanInFlight -
      summarize(runToEnd("sharded", even), even).meanInFlight;
    expect(Math.abs(gap)).toBeLessThan(0.5);
  });

  it("keeps every individual worker inside its own cap", () => {
    // The reason this is hard to catch: no lane is misbehaving. Each obeys the cap it was given.
    let state = createRun("sharded");
    for (let i = 0; i < 400 && !state.done; i++) {
      state = step(state);
      for (const worker of state.workers) {
        expect(worker.inFlight.length).toBeLessThanOrEqual(worker.cap);
      }
    }
  });
});

describe("stepping", () => {
  it("frees a slot before admitting, so a completion can be reused in the same tick", () => {
    let state = createRun("global");
    state = step(state);
    expect(inFlightTotal(state)).toBe(DEFAULT_CONFIG.concurrency);
  });

  it("stops when every worker is empty and out of budget", () => {
    const state = runToEnd("global");
    expect(state.done).toBe(true);
    expect(inFlightTotal(state)).toBe(0);
  });
});

describe("stranded capacity", () => {
  it("is always zero under global", () => {
    let state = createRun("global");
    for (let i = 0; i < 400 && !state.done; i++) {
      state = step(state);
      expect(strandedSlots(state)).toBe(0);
    }
  });

  it("appears under sharded once a worker outruns its neighbours", () => {
    let state = createRun("sharded");
    let seen = 0;
    for (let i = 0; i < 400 && !state.done; i++) {
      state = step(state);
      seen = Math.max(seen, strandedSlots(state));
    }
    expect(seen).toBeGreaterThan(0);
  });

  it("counts nothing once every worker is out of budget", () => {
    // The drain is not a shortfall: there is no work left that the free slots could have taken.
    const state = runToEnd("sharded");
    expect(strandedSlots(state)).toBe(0);
  });
});

describe("pick_worker", () => {
  const inflight = (xs: number[]) => xs;

  it("round-robins in issuance order and advances only on a round-robin pick", () => {
    const state = createPickState();
    const picks = Array.from({ length: 6 }, (_, i) =>
      pickWorker("round-robin", 4, `s${i}`, inflight([0, 0, 0, 0]), state),
    );
    expect(picks).toEqual([0, 1, 2, 3, 0, 1]);
  });

  it("sends every turn of a session to one worker under sticky", () => {
    const state = createPickState();
    const first = pickWorker("sticky", 4, "session-3", inflight([0, 0, 0, 0]), state);
    // Load is irrelevant to a sticky pick; only the correlation id matters.
    const again = pickWorker("sticky", 4, "session-3", inflight([9, 0, 0, 0]), state);
    expect(again).toBe(first);
  });

  it("falls back to round-robin when a sticky turn has no correlation id", () => {
    const state = createPickState();
    expect(pickWorker("sticky", 4, null, inflight([0, 0, 0, 0]), state)).toBe(0);
    expect(pickWorker("sticky", 4, null, inflight([0, 0, 0, 0]), state)).toBe(1);
  });

  it("picks the shallowest worker, resolving ties to the lowest index", () => {
    const state = createPickState();
    expect(pickWorker("least-loaded", 4, null, inflight([2, 1, 5, 1]), state)).toBe(1);
    expect(pickWorker("least-loaded", 4, null, inflight([0, 0, 0, 0]), createPickState())).toBe(0);
  });

  it("binds a correlation id on first sight and honours the binding after", () => {
    const state = createPickState();
    const first = pickWorker("least-loaded", 4, "s", inflight([9, 0, 9, 9]), state);
    expect(first).toBe(1);
    // Even though worker 3 is now shallowest, the binding wins — continuations stay sticky.
    expect(pickWorker("least-loaded", 4, "s", inflight([0, 9, 0, 0]), state)).toBe(1);
  });

  it("hashes correlation ids with the runtime's seed-free fnv1a64", () => {
    // Ported constants, so the mapping is stable across processes and runs. Pinning one value
    // catches an accidental switch to a seeded hash.
    expect(fnv1a64("")).toBe(0xcbf2_9ce4_8422_2325n);
    expect(fnv1a64("session-0")).toBe(fnv1a64("session-0"));
    expect(fnv1a64("session-0")).not.toBe(fnv1a64("session-1"));
  });
});

describe("global-hop", () => {
  it("issues in exact global order, so turn i lands on worker i % W under round-robin", () => {
    // The property `global_hop.rs:16` claims. Sharded and global cannot state this at all: their
    // W loops race, and a request's worker is whichever loop happened to have room.
    let state = createRun("global-hop", DEFAULT_CONFIG, "round-robin");
    state = step(state, DEFAULT_CONFIG);
    const issued = state.workers.flatMap((w) => w.inFlight).sort((a, b) => a.id - b.id);
    for (const request of issued) {
      expect(request.worker).toBe(request.id % DEFAULT_CONFIG.workers);
    }
  });

  it("holds the full cap from one loop, with no shared gate", () => {
    // "One loop holding the full cap IS the global cap" — the cap is never sliced here.
    expect(capsFor("global-hop", 8, 4)).toEqual([8, 8, 8, 8]);
    expect(summarize(runToEnd("global-hop")).peakInFlight).toBeLessThanOrEqual(
      DEFAULT_CONFIG.concurrency,
    );
  });

  it("charges a cross-thread cost that the other modes do not pay", () => {
    const hop = runToEnd("global-hop");
    expect(hop.hopTicksCharged).toBe(DEFAULT_CONFIG.requests * DEFAULT_CONFIG.hopCostTicks);
    expect(runToEnd("global").hopTicksCharged).toBe(0);
  });

  it("fragments a session across workers under round-robin and not under sticky", () => {
    // The concrete cost of the default policy: the sticky connection pool is worker-local, so a
    // session spread over W workers opens a connection on each.
    const rr = fragmentation(runToEnd("global-hop", DEFAULT_CONFIG, "round-robin"));
    const sticky = fragmentation(runToEnd("global-hop", DEFAULT_CONFIG, "sticky"));
    expect(sticky.worst).toBe(1);
    expect(rr.mean).toBeGreaterThan(sticky.mean);
  });
});

describe("the second gate", () => {
  const partitioned: Config = { ...DEFAULT_CONFIG, gatePrefill: false };

  it("never strands prefill while the shared gate covers it", () => {
    let state = createRun("global", DEFAULT_CONFIG);
    for (let i = 0; i < 400 && !state.done; i++) {
      state = step(state, DEFAULT_CONFIG);
      expect(strandedPrefill(state, DEFAULT_CONFIG)).toBe(0);
    }
  });

  it("strands prefill once the cap is partitioned instead", () => {
    // The bug the shared prefill gate fixes: a worker blocked on its own prefill share cannot
    // borrow a free slot from a worker that has finished, exactly as concurrency used to behave.
    let state = createRun("global", partitioned);
    let worst = 0;
    for (let i = 0; i < 400 && !state.done; i++) {
      state = step(state, partitioned);
      worst = Math.max(worst, strandedPrefill(state, partitioned));
    }
    expect(worst).toBeGreaterThan(0);
  });

  it("over-subscribes a prefill cap smaller than the worker count", () => {
    // Same `.max(1)` floor as concurrency: four workers under an authored 3 admit 4.
    const small: Config = { ...partitioned, prefillConcurrency: 3, workers: 4 };
    const caps = prefillCapsFor("global", small, 4);
    expect(caps).toEqual([1, 1, 1, 1]);
    expect(caps.reduce((a, b) => a + b, 0)).toBe(4);
    expect(prefillCapsFor("global", { ...small, gatePrefill: true }, 4)).toEqual([3, 3, 3, 3]);
  });

  it("leaves the concurrency story unchanged when it is not the binding cap", () => {
    // Prefill defaults to the concurrency target, so adding the gate must not move the headline.
    expect(summarize(runToEnd("global")).utilisation).toBeGreaterThan(
      summarize(runToEnd("sharded")).utilisation,
    );
    expect(summarize(runToEnd("global")).completed).toBe(DEFAULT_CONFIG.requests);
  });
});
