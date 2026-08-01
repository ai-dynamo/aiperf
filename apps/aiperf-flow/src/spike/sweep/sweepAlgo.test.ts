/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  buildColumnStore,
  coarseTokenEvents,
  collisionsIn,
  concurrencyEvents,
  detectSteadyWindow,
  iclTokenEvents,
  snapSmallResiduals,
  sortSweepEvents,
  sweepLineCumsum,
  valueAt,
  type Record,
} from "./sweepAlgo.js";

const R = (
  index: number,
  startNs: number,
  generationStartNs: number,
  endNs: number,
  inputTokens = 10,
  outputTokens = 8,
  iclNs: number[] = [],
): Record => ({ index, startNs, generationStartNs, endNs, inputTokens, outputTokens, iclNs });

describe("columnar storage", () => {
  it("keeps every column aligned by absolute request index", () => {
    const store = buildColumnStore([R(0, 0, 1, 5), R(1, 2, 3, 9)]);
    expect(store.rows).toBe(2);
    for (const column of store.columns) expect(column.values).toHaveLength(2);
    expect(store.columns.find((c) => c.name === "end_ns")!.values).toEqual([5, 9]);
  });

  it("gives an empty list an offset too, so ragged rows stay index-aligned", () => {
    // A row with no ICL values must still occupy a slot: the numeric columns are addressed by
    // absolute index, and the ragged series has to agree with them.
    const store = buildColumnStore([
      R(0, 0, 1, 5, 10, 8, [3, 4]),
      R(1, 2, 3, 9, 10, 8, []),
      R(2, 4, 5, 11, 10, 8, [7]),
    ]);
    expect(store.icl.offsets).toEqual([0, 2, 2]);
    expect(store.icl.lengths).toEqual([2, 0, 1]);
    expect(store.icl.valuesNs).toEqual([3, 4, 7]);
    // Only rows that contributed appear in append order.
    expect(store.icl.appendOrder).toEqual([0, 2]);
  });
});

describe("collision avoidance", () => {
  it("puts an end before a start at an equal timestamp", () => {
    // One request ends exactly as the next begins. Sorted by timestamp alone the order is
    // arbitrary; with the delta tie-break the -1 always lands first.
    const sorted = sortSweepEvents(concurrencyEvents([R(0, 0, 1, 5), R(1, 5, 6, 9)]));
    const atFive = sorted.filter((e) => e.timestampNs === 5);
    expect(atFive.map((e) => e.delta)).toEqual([-1, 1]);
  });

  it("never shows a phantom extra unit for touching intervals", () => {
    const { curve } = sweepLineCumsum(concurrencyEvents([R(0, 0, 1, 5), R(1, 5, 6, 9)]));
    // Concurrency must peak at 1, not 2: the two requests never actually overlapped.
    expect(Math.max(...curve.values)).toBe(1);
  });

  it("does show 2 when the intervals genuinely overlap", () => {
    const { curve } = sweepLineCumsum(concurrencyEvents([R(0, 0, 1, 6), R(1, 5, 6, 9)]));
    expect(Math.max(...curve.values)).toBe(2);
  });

  it("reports the colliding positions", () => {
    const sorted = sortSweepEvents(concurrencyEvents([R(0, 0, 1, 5), R(1, 5, 6, 9)]));
    expect(collisionsIn(sorted).length).toBeGreaterThan(0);
  });
});

describe("cumsum residual snapping", () => {
  it("snaps a residual below 1e-9 of the maximum to zero", () => {
    expect(snapSmallResiduals([100, 1e-8, -1e-9], 1e-9 * 100)).toEqual([100, 0, 0]);
  });

  it("leaves genuine values alone", () => {
    expect(snapSmallResiduals([100, 3, -2], 1e-9 * 100)).toEqual([100, 3, -2]);
  });

  it("returns the curve to exactly zero after every request completes", () => {
    const { curve } = sweepLineCumsum(concurrencyEvents([R(0, 0, 1, 5), R(1, 2, 3, 9)]));
    expect(curve.values[curve.values.length - 1]).toBe(0);
  });
});

describe("ICL awareness", () => {
  const record = R(0, 0, 100, 500, 10, 40, [50, 50, 50]);

  it("spreads output tokens across chunk arrivals rather than one lump", () => {
    const store = buildColumnStore([record]);
    const coarse = sweepLineCumsum(coarseTokenEvents([record]));
    const aware = sweepLineCumsum(iclTokenEvents([record], store));

    // Coarse: every output token is in flight the instant generation starts.
    expect(valueAt(coarse.curve, 100)).toBe(50);
    // ICL-aware: only the first chunk has arrived at that point.
    expect(valueAt(aware.curve, 100)).toBeLessThan(50);
    // By the last chunk both agree on the total.
    expect(valueAt(aware.curve, 260)).toBeCloseTo(50, 6);
  });

  it("places each chunk at generation_start plus the cumulative ICL", () => {
    const store = buildColumnStore([record]);
    const chunks = iclTokenEvents([record], store)
      .filter((e) => e.kind === "chunk")
      .map((e) => e.timestampNs);
    expect(chunks).toEqual([100, 150, 200, 250]);
  });

  it("clamps a chunk that would land past the record's end", () => {
    const late = R(0, 0, 100, 180, 10, 40, [50, 50, 50]);
    const store = buildColumnStore([late]);
    const chunks = iclTokenEvents([late], store)
      .filter((e) => e.kind === "chunk")
      .map((e) => e.timestampNs);
    expect(Math.max(...chunks)).toBe(180);
  });

  it("falls back to the coarse curve when no ICL was retained", () => {
    const plain = R(0, 0, 100, 500, 10, 40, []);
    const store = buildColumnStore([plain]);
    expect(valueAt(sweepLineCumsum(iclTokenEvents([plain], store)).curve, 100)).toBe(50);
  });
});

describe("steady-state detection", () => {
  /** Ten overlapping requests: a ramp up, a saturated middle, and a drain. */
  const ramp: Record[] = Array.from({ length: 10 }, (_, i) => R(i, i * 10, i * 10 + 1, 100 + i * 10));

  it("opens at the first crossing and closes at the last descent", () => {
    const { curve } = sweepLineCumsum(concurrencyEvents(ramp));
    const window = detectSteadyWindow(curve, 10, 0.8)!;
    expect(window.threshold).toBe(8);
    expect(window.startNs).toBeLessThan(window.endNs);
    // The window must exclude the ramp: concurrency is below 8 before it opens.
    expect(valueAt(curve, window.startNs - 1)).toBeLessThan(8);
  });

  it("uses ceil, so a fractional threshold rounds up", () => {
    const { curve } = sweepLineCumsum(concurrencyEvents(ramp));
    // 0.75 * 10 = 7.5 → 8.
    expect(detectSteadyWindow(curve, 10, 0.75)!.threshold).toBe(8);
  });

  it("never returns a threshold below one", () => {
    const { curve } = sweepLineCumsum(concurrencyEvents(ramp));
    expect(detectSteadyWindow(curve, 1, 0.01)!.threshold).toBe(1);
  });

  it("falls back to the default fraction when given a nonsense one", () => {
    const { curve } = sweepLineCumsum(concurrencyEvents(ramp));
    expect(detectSteadyWindow(curve, 10, 0)!.threshold).toBe(
      detectSteadyWindow(curve, 10, 0.8)!.threshold,
    );
  });

  it("returns nothing when there is no concurrency target", () => {
    const { curve } = sweepLineCumsum(concurrencyEvents(ramp));
    expect(detectSteadyWindow(curve, 0)).toBeNull();
  });

  it("returns nothing when the curve never reaches the threshold", () => {
    const { curve } = sweepLineCumsum(concurrencyEvents([R(0, 0, 1, 5)]));
    expect(detectSteadyWindow(curve, 50, 0.8)).toBeNull();
  });

  it("closes at the last event when the run ends while still saturated", () => {
    // Every request starts and none of them end before the curve does.
    const held: Record[] = Array.from({ length: 5 }, (_, i) => R(i, i, i + 1, 1000));
    const { curve } = sweepLineCumsum(concurrencyEvents(held));
    const window = detectSteadyWindow(curve, 5, 0.8)!;
    expect(window.endNs).toBe(curve.timestampsNs[curve.timestampsNs.length - 1]);
  });
});
