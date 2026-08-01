/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { buildEvents, niceMax, stepPathD, stepPoints, type SweepRequest } from "./sweepMath.js";

const REQUESTS: SweepRequest[] = [
  { id: "A", start: 0, gen: 6, end: 20, tokens: 120 },
  { id: "B", start: 3, gen: 10, end: 30, tokens: 200 },
  { id: "C", start: 8, gen: 12, end: 24, tokens: 90 },
];

describe("buildEvents", () => {
  it("emits a +weight event at each interval start and a -weight event at each end, sorted by time", () => {
    const evts = buildEvents(REQUESTS, "concurrency");
    expect(evts.map((e) => [e.t, e.d])).toEqual([
      [0, 1],
      [3, 1],
      [8, 1],
      [20, -1],
      [24, -1],
      [30, -1],
    ]);
  });

  it("weights the tokens curve by output_tokens instead of a flat 1", () => {
    const evts = buildEvents(REQUESTS, "tokens");
    expect(evts[0]).toMatchObject({ t: 0, d: 120, id: "A" });
    expect(evts.find((e) => e.id === "B" && e.d > 0)).toMatchObject({ t: 3, d: 200 });
  });

  it("weights the throughput curve over the decode window (gen -> end) as tokens/duration", () => {
    const evts = buildEvents(REQUESTS, "throughput");
    // A: gen=6, end=20, tokens=120 -> 120/14
    const start = evts.find((e) => e.id === "A" && e.d > 0)!;
    expect(start.t).toBe(6);
    expect(start.d).toBeCloseTo(120 / 14);
  });

  it("sorts end events before start events on a tie so touching intervals don't double-count", () => {
    const touching: SweepRequest[] = [
      { id: "X", start: 0, gen: 0, end: 10, tokens: 1 },
      { id: "Y", start: 10, gen: 10, end: 20, tokens: 1 },
    ];
    const evts = buildEvents(touching, "concurrency");
    const atTen = evts.filter((e) => e.t === 10);
    expect(atTen[0].d).toBeLessThan(0);
    expect(atTen[1].d).toBeGreaterThan(0);
  });
});

describe("stepPoints", () => {
  it("cumulative-sums the sorted event stream into a step function matching request overlap", () => {
    const evts = buildEvents(REQUESTS, "concurrency");
    const pts = stepPoints(evts);
    // At t=8 all three requests (A,B,C) are in flight -> concurrency 3.
    const atEight = pts.find((p) => p.t === 8);
    expect(atEight?.v).toBe(3);
    // Final point returns to 0 once every request has ended.
    expect(pts[pts.length - 1].v).toBe(0);
  });
});

describe("stepPathD", () => {
  it("returns a flat zero-height path when there are no points", () => {
    const d = stepPathD([], (t) => t, (v) => v, 0, 10);
    expect(d).toBe("M 0 0 L 10 0");
  });

  it("produces an SVG path string starting at tMin and ending at tMax", () => {
    const pts = stepPoints(buildEvents(REQUESTS, "concurrency"));
    const d = stepPathD(pts, (t) => t, (v) => 100 - v * 10, 0, 30);
    expect(d.startsWith("M 0 100")).toBe(true);
    expect(d.endsWith("L 30 100")).toBe(true);
  });
});

describe("niceMax", () => {
  it("returns 1 for values at or below 1", () => {
    expect(niceMax(0)).toBe(1);
    expect(niceMax(1)).toBe(1);
  });

  it("rounds up to the nearest 1/2/5 * 10^n", () => {
    expect(niceMax(3)).toBe(5);
    expect(niceMax(12)).toBe(20);
    expect(niceMax(45)).toBe(50);
  });
});
