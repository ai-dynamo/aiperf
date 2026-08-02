/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { arrivals, createIngest, ingestOne, traceExact, traceFold, traceSketch } from "./ingest.js";
import { exactPercentile, latencySamples, quantile } from "./sketchSim.js";

const run = (values: readonly number[], cells: number, compression: number) => {
  let state = createIngest(cells, compression);
  for (const arrival of arrivals(values, cells)) state = ingestOne(state, arrival);
  return state;
};

describe("stepping values in", () => {
  it("keeps the exact path sorted and complete", () => {
    const values = latencySamples(400, 3);
    const state = run(values, 3, 20);
    expect(state.arrived).toEqual([...values]);
    expect(state.sorted).toEqual([...values].sort((a, b) => a - b));
  });

  it("reports where each value landed in the sorted array", () => {
    let state = createIngest(1, 20);
    for (const arrival of arrivals([50, 10, 30], 1)) state = ingestOne(state, arrival);
    // 30 belongs between 10 and 50.
    expect(state.lastSortedIndex).toBe(1);
    expect(state.sorted).toEqual([10, 30, 50]);
  });

  it("routes each value to exactly one cell and leaves the others untouched", () => {
    const state = run(latencySamples(90, 5), 3, 20);
    expect(state.cells.map((c) => c.totalWeight)).toEqual([30, 30, 30]);
  });

  it("notes when the new centroid was absorbed rather than added", () => {
    // With a low compression the digest clusters aggressively, so absorption must be observable —
    // it is the event the page exists to show.
    let state = createIngest(1, 5);
    let absorbed = 0;
    for (const arrival of arrivals(latencySamples(200, 7), 1)) {
      state = ingestOne(state, arrival);
      absorbed += state.compressedCells.length;
    }
    expect(absorbed).toBeGreaterThan(0);
  });

  it("reaches the same digest as the pinned path", () => {
    // Stepping compresses per value where the runtime batches. Clustering sorts first, so both
    // arrive at the same place — asserting that here is what makes the visual trustworthy.
    const values = latencySamples(500, 11);
    const stepped = run(values, 1, 50).cells[0]!;
    const { folded } = traceFold([stepped], 50);
    for (const q of [0.5, 0.9, 0.99]) {
      expect(quantile(folded, q)).toBeCloseTo(quantile(stepped, q)!, 9);
    }
  });
});

describe("tracing the exact percentile", () => {
  it("exposes the two values it interpolates between", () => {
    const sorted = [10, 20, 30, 40, 50];
    const trace = traceExact(sorted, 75)!;
    // virtual index 0.75 * 4 = 3 exactly, so lo == hi == 3 and no blending happens.
    expect(trace.virtualIndex).toBe(3);
    expect(trace.loValue).toBe(40);
    expect(trace.frac).toBe(0);
    expect(trace.result).toBe(40);
  });

  it("blends when the virtual index falls between two values", () => {
    const trace = traceExact([0, 100], 25)!;
    expect(trace.lo).toBe(0);
    expect(trace.hi).toBe(1);
    expect(trace.frac).toBeCloseTo(0.25, 12);
    expect(trace.result).toBeCloseTo(25, 12);
  });

  it("agrees with the pinned implementation", () => {
    const sorted = [...latencySamples(1_000, 13)].sort((a, b) => a - b);
    for (const p of [10, 50, 90, 99]) {
      expect(traceExact(sorted, p)!.result).toBe(exactPercentile(sorted, p));
    }
  });
});

describe("tracing the sketch walk", () => {
  it("stops on the first centroid whose centre passes the target", () => {
    const state = run(latencySamples(600, 17), 1, 20);
    const trace = traceSketch(state.cells[0]!, 0.9)!;
    expect(trace.steps.at(-1)!.stopped).toBe(true);
    expect(trace.steps.filter((s) => s.stopped)).toHaveLength(1);
  });

  it("produces the same number the pinned quantile does", () => {
    const state = run(latencySamples(600, 19), 1, 20);
    for (const q of [0.25, 0.5, 0.9, 0.99]) {
      expect(traceSketch(state.cells[0]!, q)!.result).toBeCloseTo(quantile(state.cells[0]!, q)!, 9);
    }
  });

  it("marks q0 and q1 as anchored rather than interpolated", () => {
    const state = run(latencySamples(200, 23), 1, 20);
    for (const q of [0, 1]) {
      const trace = traceSketch(state.cells[0]!, q)!;
      expect(trace.anchored).toBe(true);
      expect(trace.steps).toHaveLength(0);
    }
    expect(traceSketch(state.cells[0]!, 0.5)!.anchored).toBe(false);
  });
});

describe("tracing the fold", () => {
  it("concatenates every cell's centroids before compressing", () => {
    const state = run(latencySamples(600, 29), 3, 20);
    const trace = traceFold(state.cells, 20);
    const contributed = trace.contributed.reduce((n, c) => n + c.centroids.length, 0);
    expect(trace.concatenated).toHaveLength(contributed);
    // Compression is what makes the fold bounded rather than cumulative.
    expect(trace.folded.centroids.length).toBeLessThanOrEqual(trace.concatenated.length);
  });

  it("preserves total weight and the exact extremes through the fold", () => {
    const values = latencySamples(600, 31);
    const state = run(values, 3, 20);
    const { folded } = traceFold(state.cells, 20);
    expect(folded.totalWeight).toBe(values.length);
    expect(folded.min).toBe(Math.min(...values));
    expect(folded.max).toBe(Math.max(...values));
  });
});
