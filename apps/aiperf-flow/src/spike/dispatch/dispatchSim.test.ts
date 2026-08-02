/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  admissibleTotal,
  capsFor,
  createRun,
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
    expect(global.workers.map((w) => w.remaining)).toEqual(sharded.workers.map((w) => w.remaining));
    expect(global.workers.reduce((a, w) => a + w.remaining, 0)).toBe(DEFAULT_CONFIG.requests);
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
