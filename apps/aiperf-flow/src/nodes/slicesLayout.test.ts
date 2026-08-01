/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { binOf, buildSlices, layoutSlices, type SliceRequest } from "./slicesLayout.js";

const REQUESTS: SliceRequest[] = [
  { id: "r0", start: 0, end: 20 },
  { id: "r1", start: 3, end: 30 },
  { id: "r2", start: 28, end: 50 },
];

describe("buildSlices", () => {
  it("flags a trailing slice that runs past activity, and clips it", () => {
    const slices = buildSlices(0, 50, 15);

    expect(slices).toHaveLength(4);
    expect(slices[2]!.isComplete).toBe(true);
    expect(slices[3]!.isComplete).toBe(false);
    // A rate over slice 3 must divide by 5, not by the grid-defined 15.
    expect(slices[3]!.end).toBe(60);
    expect(slices[3]!.clippedEnd).toBe(50);
  });

  it("marks every slice complete when the grid divides the span exactly", () => {
    const slices = buildSlices(0, 50, 25);
    expect(slices).toHaveLength(2);
    expect(slices.every((s) => s.isComplete)).toBe(true);
  });

  it("returns nothing for a non-positive duration rather than looping forever", () => {
    expect(buildSlices(0, 50, 0)).toEqual([]);
  });
});

describe("binOf", () => {
  it("bins by start, so an interval spanning several slices still counts once", () => {
    // r1 runs 3..30 across three 15-wide slices; its start puts it in slice 0.
    expect(binOf(3, 0, 15, 4)).toBe(0);
    expect(binOf(28, 0, 15, 4)).toBe(1);
  });

  it("clamps a start on the right edge into the last slice", () => {
    expect(binOf(60, 0, 15, 4)).toBe(3);
  });
});

describe("layoutSlices", () => {
  it("extends the axis to the grid end, so an incomplete slice is still drawn", () => {
    const layout = layoutSlices({ requests: REQUESTS, duration: 15, hasTitle: false });

    expect(layout.spanEnd).toBe(50);
    expect(layout.tMax).toBe(60);
    expect(layout.x(60)).toBeCloseTo(layout.xRight);
  });
});
