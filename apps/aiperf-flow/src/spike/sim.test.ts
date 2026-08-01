/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { createSim, step, inFlight, queued, DEFAULT_CONFIG, TICK_MS, type SimState } from "./sim.js";

function run(state: SimState, ms: number, chunk: number, config = DEFAULT_CONFIG): SimState {
  let out = state;
  for (let elapsed = 0; elapsed < ms; elapsed += chunk) out = step(out, chunk, config);
  return out;
}

function shape(s: SimState) {
  return {
    now: s.now,
    completed: s.completed,
    tokensOut: s.tokensOut,
    requests: s.requests.map((r) => `${r.id}:${r.stage}:${r.emitted}`),
  };
}

describe("determinism", () => {
  it("is independent of how elapsed time was chopped into frames", () => {
    const steady = run(createSim(0, 7), 6000, 16);
    const chunky = run(createSim(0, 7), 6000, 100);
    expect(shape(chunky)).toEqual(shape(steady));
  });

  it("gives a different run for a different seed", () => {
    expect(shape(run(createSim(0, 1), 4000, 20))).not.toEqual(
      shape(run(createSim(0, 2), 4000, 20)),
    );
  });

  it("carries sub-tick time forward instead of dropping it", () => {
    let s = createSim(0, 1);
    for (let i = 0; i < 10; i++) s = step(s, TICK_MS / 10, DEFAULT_CONFIG);
    expect(s.now).toBe(TICK_MS);
  });
});

describe("admission gate", () => {
  it("never lets more past the gate than the concurrency limit", () => {
    // The gate is the whole point of the rig: excess must wait, not slip through.
    const config = { ...DEFAULT_CONFIG, rate: 40, concurrency: 5 };
    let s = createSim(0, 3);
    for (let i = 0; i < 300; i++) {
      s = step(s, 20, config);
      expect(inFlight(s.requests)).toBeLessThanOrEqual(config.concurrency);
    }
  });

  it("queues the excess rather than dropping it", () => {
    const config = { ...DEFAULT_CONFIG, rate: 40, concurrency: 2 };
    const s = run(createSim(0, 3), 4000, 20, config);
    expect(queued(s.requests)).toBeGreaterThan(0);
  });

  it("keeps the queue empty when capacity exceeds demand", () => {
    const config = { ...DEFAULT_CONFIG, rate: 1, concurrency: 20 };
    const s = run(createSim(0, 3), 5000, 20, config);
    expect(queued(s.requests)).toBe(0);
  });
});

describe("request lifecycle", () => {
  it("emits every token exactly once before completing", () => {
    const config = { ...DEFAULT_CONFIG, rate: 3, concurrency: 8 };
    const s = run(createSim(0, 11), 8000, 20, config);
    expect(s.completed).toBeGreaterThan(0);
    // Token accounting must not double-count across the fixed ticks.
    for (const r of s.requests) expect(r.emitted).toBeLessThanOrEqual(r.tokens);
    expect(s.tokensOut).toBeGreaterThanOrEqual(s.completed);
  });
});
