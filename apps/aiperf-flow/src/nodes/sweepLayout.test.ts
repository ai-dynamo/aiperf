/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { layoutSweep, sweepNodeSize } from "./sweepLayout.js";
import type { SweepRequest } from "./sweepMath.js";

const REQUESTS: SweepRequest[] = [
  { id: "r0", start: 0, gen: 6, end: 20, tokens: 120 },
  { id: "r1", start: 3, gen: 10, end: 30, tokens: 200 },
  { id: "r2", start: 8, gen: 12, end: 24, tokens: 90 },
];

describe("layoutSweep", () => {
  it("scales the value axis to the curve, so tokens and concurrency do not share a maximum", () => {
    const base = { requests: REQUESTS, hasTitle: false };
    const concurrency = layoutSweep({ ...base, curve: "concurrency" });
    const tokens = layoutSweep({ ...base, curve: "tokens" });

    // Three overlapping requests peak at 3 concurrent, which `niceMax` rounds the axis up to 5;
    // the same trace carries hundreds of tokens in flight, so the axes must not be shared.
    expect(concurrency.vMax).toBe(5);
    expect(tokens.vMax).toBeGreaterThan(100);
  });

  it("puts the Gantt above the step plot on one shared x projection", () => {
    const layout = layoutSweep({ requests: REQUESTS, curve: "concurrency", hasTitle: false });

    expect(layout.stepTop).toBeGreaterThan(layout.top + layout.ganttHeight);
    expect(layout.x(0)).toBe(layout.xLeft);
    expect(layout.x(layout.tMax)).toBeCloseTo(layout.xRight);
  });

  it("grows a row per request without changing the plot height", () => {
    const two = layoutSweep({ requests: REQUESTS.slice(0, 2), curve: "concurrency", hasTitle: false });
    const three = layoutSweep({ requests: REQUESTS, curve: "concurrency", hasTitle: false });

    expect(three.svgHeight - two.svgHeight).toBe(three.rowHeight);
    expect(three.stepHeight).toBe(two.stepHeight);
  });

  it("stays finite with no requests", () => {
    const { width, height } = sweepNodeSize({ requests: [], curve: "concurrency", hasTitle: false });
    expect(Number.isFinite(width)).toBe(true);
    expect(Number.isFinite(height)).toBe(true);
  });
});
