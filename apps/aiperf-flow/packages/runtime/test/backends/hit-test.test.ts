// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import { hitTest } from "../../src/backends/canvas/hit-test.js";
import type { HitRegion } from "../../src/display-list.js";

const regions = [
  {
    id: "background-hit",
    semanticId: "background",
    order: 0,
    bounds: { x: 0, y: 0, width: 100, height: 100 },
  },
  {
    id: "foreground-hit",
    semanticId: "request-7",
    order: 1,
    bounds: { x: 20, y: 20, width: 40, height: 40 },
  },
] as const satisfies readonly HitRegion[];

describe("Canvas semantic hit testing", () => {
  test("returns the topmost semantic region containing the point", () => {
    expect(hitTest(regions, { x: 30, y: 30 })?.semanticId).toBe("request-7");
  });

  test("uses deterministic id ordering to break equal-order overlaps", () => {
    const tied = [
      { ...regions[0], id: "alpha", semanticId: "alpha", order: 2 },
      { ...regions[0], id: "zebra", semanticId: "zebra", order: 2 },
    ] satisfies readonly HitRegion[];

    expect(hitTest(tied, { x: 30, y: 30 })?.semanticId).toBe("zebra");
  });

  test("treats bounds edges as hittable and misses points outside them", () => {
    expect(hitTest(regions, { x: 60, y: 60 })?.semanticId).toBe("request-7");
    expect(hitTest(regions, { x: 101, y: 50 })).toBeUndefined();
  });
});
