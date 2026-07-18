// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import { layoutWaterfallNest } from "../../src/leaves/waterfall-nest-layout.js";

describe("leaf.viz.waterfall.nest-layout", () => {
  test("lays out RequestLifecycleWaterfall fixture across ordered lanes", () => {
    const layout = layoutWaterfallNest(
      [
        { id: "ev-arrival", lane: "arrival", start: 0, end: 0 },
        { id: "ev-admission", lane: "admission", start: 2, end: 2 },
        { id: "ev-connect", lane: "connect", start: 2, end: 18 },
        { id: "ev-first-token", lane: "first-token", start: 120, end: 120 },
      ],
      {
        laneOrder: ["arrival", "admission", "connect", "first-token"],
        originX: 0,
        originY: 0,
        laneHeight: 16,
        laneGap: 4,
        pxPerMs: 1,
      },
    );

    expect(layout.version).toBe(1);
    expect(layout.routes).toEqual([]);

    const arrival = layout.nodes.find((node) => node.nodeId === "ev-arrival");
    const admission = layout.nodes.find((node) => node.nodeId === "ev-admission");
    const connect = layout.nodes.find((node) => node.nodeId === "ev-connect");
    const firstToken = layout.nodes.find((node) => node.nodeId === "ev-first-token");

    expect(admission?.bounds.y).not.toBe(arrival?.bounds.y);
    expect(connect?.bounds.width).toBe(16);
    expect(firstToken?.bounds.x).toBe(120);
  });
});
