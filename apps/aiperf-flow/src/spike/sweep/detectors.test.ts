/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  batchMeansTrend,
  consensusWindow,
  cusumWindow,
  mser5,
  timeWeightedP95,
} from "./detectors.js";
import type { StepFn } from "./sweepAlgo.js";

/** A curve that ramps 0→10, holds, then drains back to 0. */
function rampHoldDrain(): StepFn {
  const timestampsNs: number[] = [];
  const values: number[] = [];
  let t = 0;
  for (let v = 1; v <= 10; v++) { timestampsNs.push((t += 10)); values.push(v); }
  for (let i = 0; i < 30; i++) { timestampsNs.push((t += 10)); values.push(10); }
  for (let v = 9; v >= 0; v--) { timestampsNs.push((t += 10)); values.push(v); }
  return { timestampsNs, values };
}

describe("timeWeightedP95", () => {
  it("weights by how long a level was held, not by how many events it had", () => {
    // Value 1 is held for a single nanosecond; value 5 is held for a thousand.
    const curve: StepFn = { timestampsNs: [0, 1, 1001], values: [1, 5, 5] };
    expect(timeWeightedP95(curve)).toBe(5);
  });

  it("returns zero for an empty curve", () => {
    expect(timeWeightedP95({ timestampsNs: [], values: [] })).toBe(0);
  });
});

describe("CUSUM window", () => {
  it("uses a time-weighted p95 as its target, not the peak", () => {
    expect(cusumWindow(rampHoldDrain()).target).toBe(10);
  });

  it("degenerates to the full range on a run with a drain", () => {
    // Not a porting error — the shipped Python does exactly this. Verified by running
    // ramp_detection.py's algorithm directly on the same shapes: a clean trapezoid gives
    // (ramp_up=48, ramp_down=0) and a jittery plateau gives (78, 0), both inverted.
    //
    // The cause is structural. The target is a p95, so by construction at most 5% of the
    // time-weighted mass sits above it and deviations are non-positive almost everywhere. The
    // forward cumulative sum therefore only ever falls, and its argmin lands at the last index
    // whenever a drain exists. Ordering then fails and the detector yields the whole range.
    const trace = cusumWindow(rampHoldDrain());
    expect(trace.method).toBe("cusum_inverted");
    expect(trace.rampUpIndex).toBeGreaterThanOrEqual(trace.rampDownIndex);
    expect(trace.window).toEqual({
      startNs: rampHoldDrain().timestampsNs[0],
      endNs: rampHoldDrain().timestampsNs[rampHoldDrain().timestampsNs.length - 1],
    });
  });

  it("still reports its trace, so the failure is visible rather than silent", () => {
    const trace = cusumWindow(rampHoldDrain());
    expect(trace.forward).toHaveLength(rampHoldDrain().values.length);
    expect(trace.deviations.every((d) => d <= 0)).toBe(true);
  });

  it("reports empty for an empty curve rather than throwing", () => {
    expect(cusumWindow({ timestampsNs: [], values: [] }).method).toBe("empty");
  });
});

describe("MSER-5", () => {
  it("truncates a warm-up transient", () => {
    // Twenty noisy-but-settled samples behind ten that are wildly high.
    const series = [...Array.from({ length: 10 }, () => 100), ...Array.from({ length: 40 }, (_, i) => 10 + (i % 2))];
    const trace = mser5(series);
    expect(trace.truncation).toBeGreaterThan(0);
    // Everything it deleted should come from the transient.
    expect(trace.truncation).toBeLessThanOrEqual(20);
  });

  it("never deletes more than half the batches", () => {
    const trace = mser5(Array.from({ length: 100 }, (_, i) => 100 - i));
    expect(trace.dStar).toBeLessThanOrEqual(trace.maxD);
    expect(trace.maxD).toBe(Math.floor(trace.batches.length / 2));
  });

  it("truncates nothing when the series is already flat", () => {
    expect(mser5(Array.from({ length: 50 }, () => 7)).truncation).toBe(0);
  });

  it("declines to run on too few samples", () => {
    expect(mser5([1, 2, 3]).truncation).toBe(0);
    // Fewer than four batches is also a decline, not a guess.
    expect(mser5(Array.from({ length: 15 }, () => 1)).batches).toEqual([]);
  });
});

describe("consensus", () => {
  const full = { startNs: 0, endNs: 1000 };

  it("takes the latest start and the earliest end", () => {
    // Deliberately the most conservative reading: every signal must agree it is steady.
    const result = consensusWindow(
      [
        { name: "cusum", window: { startNs: 100, endNs: 900 } },
        { name: "mser5_latency", window: { startNs: 250, endNs: 800 } },
      ],
      full,
    );
    expect(result.window).toEqual({ startNs: 250, endNs: 800 });
    expect(result.method).toBe("cusum_mser5_latency");
  });

  it("ignores a signal that produced nothing", () => {
    const result = consensusWindow(
      [
        { name: "cusum", window: { startNs: 100, endNs: 900 } },
        { name: "mser5_ttft", window: null },
      ],
      full,
    );
    expect(result.window).toEqual({ startNs: 100, endNs: 900 });
    expect(result.method).toBe("cusum");
  });

  it("falls back to the full range when the signals do not overlap", () => {
    const result = consensusWindow(
      [
        { name: "a", window: { startNs: 800, endNs: 900 } },
        { name: "b", window: { startNs: 100, endNs: 200 } },
      ],
      full,
    );
    expect(result.method).toBe("fallback_no_overlap");
    expect(result.window).toEqual(full);
  });

  it("falls back when the agreed window is below the minimum fraction", () => {
    const result = consensusWindow(
      [{ name: "a", window: { startNs: 500, endNs: 520 } }],
      full,
      10,
    );
    expect(result.method).toBe("fallback_min_window");
  });

  it("reports empty when no signal produced anything", () => {
    expect(consensusWindow([{ name: "a", window: null }], full).method).toBe("empty");
  });
});

describe("batch-means stationarity", () => {
  it("flags a window that is still trending", () => {
    const trending = Array.from({ length: 100 }, (_, i) => i);
    const result = batchMeansTrend(trending);
    expect(result.rho).toBeCloseTo(1, 6);
    expect(result.warning).toBe(true);
  });

  it("passes a flat window", () => {
    const result = batchMeansTrend(Array.from({ length: 100 }, () => 5));
    expect(result.rho).toBe(0);
    expect(result.warning).toBe(false);
  });

  it("flags a declining window too, since the sign does not matter", () => {
    expect(batchMeansTrend(Array.from({ length: 100 }, (_, i) => 100 - i)).warning).toBe(true);
  });

  it("declines on too few samples", () => {
    expect(batchMeansTrend([1, 2, 3])).toEqual({ rho: 0, batches: [], warning: false });
  });
});
