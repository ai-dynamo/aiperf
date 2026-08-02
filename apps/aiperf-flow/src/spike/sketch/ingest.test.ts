/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  arrivals,
  createIngest,
  ingestOne,
  sortedAtSummarize,
  traceExact,
  traceFold,
  traceSketch,
} from "./ingest.js";
import { exactPercentile, latencySamples, quantile } from "./sketchSim.js";

const run = (values: readonly number[], cells: number, compression: number, threshold = 8) => {
  let state = createIngest(cells, compression, threshold);
  for (const arrival of arrivals(values, cells)) state = ingestOne(state, arrival);
  return state;
};

describe("stepping values in", () => {
  it("keeps the exact path in arrival order, unsorted, during the run", () => {
    // The runtime appends and sorts once at summarize; nothing maintains order as values land.
    const values = latencySamples(400, 3);
    const state = run(values, 3, 20);
    expect(state.arrived).toEqual([...values]);
    expect(sortedAtSummarize(state)).toEqual([...values].sort((a, b) => a - b));
  });

  it("only sorts when summarize asks for it", () => {
    let state = createIngest(1, 20, 8);
    for (const arrival of arrivals([50, 10, 30], 1)) state = ingestOne(state, arrival);
    expect(state.arrived).toEqual([50, 10, 30]);
    expect(sortedAtSummarize(state)).toEqual([10, 30, 50]);
  });

  it("leaves the new centroid in an unsorted pending tail until the threshold", () => {
    // Below the threshold nothing clusters, so the tail holds raw weight-1 centroids in the order
    // they arrived — the transient state the Rust's struct doc calls out.
    let state = createIngest(1, 20, 8);
    for (const arrival of arrivals([90, 10, 50], 1)) state = ingestOne(state, arrival);
    expect(state.settled[0]).toBe(0);
    expect(state.cells[0]!.centroids.map((c) => c.mean)).toEqual([90, 10, 50]);
    expect(state.cells[0]!.centroids.every((c) => c.weight === 1)).toBe(true);
  });

  it("compresses the whole buffer at once when the threshold is exceeded", () => {
    let state = createIngest(1, 20, 4);
    const seen: number[] = [];
    for (const arrival of arrivals(latencySamples(20, 9), 1)) {
      state = ingestOne(state, arrival);
      if (state.compressedCells.length > 0) seen.push(state.collapsedFrom!);
    }
    // A compress fires only once the count exceeds the threshold, never before. It is not always
    // threshold+1: when clustering cannot merge anything the count stays above the threshold and
    // the next append trips it again from a higher starting point.
    expect(seen.length).toBeGreaterThan(0);
    for (const from of seen) expect(from).toBeGreaterThan(4);
  });

  it("routes each value to exactly one cell and leaves the others untouched", () => {
    const state = run(latencySamples(90, 5), 3, 20);
    expect(state.cells.map((c) => c.totalWeight)).toEqual([30, 30, 30]);
  });

  it("keeps the settled prefix sorted after a compress", () => {
    let state = createIngest(1, 20, 6);
    for (const arrival of arrivals(latencySamples(60, 7), 1)) state = ingestOne(state, arrival);
    const settled = state.cells[0]!.centroids.slice(0, state.settled[0]);
    const means = settled.map((c) => c.mean);
    expect(means).toEqual([...means].sort((a, b) => a - b));
  });

  it("reaches the same digest as the pinned path", () => {
    // Stepping compresses per value where the runtime batches. Clustering sorts first, so both
    // arrive at the same place — asserting that here is what makes the visual trustworthy.
    const values = latencySamples(500, 11);
    const stepped = run(values, 1, 50, 40).cells[0]!;
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
