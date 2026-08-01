/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { laneRole, layoutTimeline, timelineNodeSize, type TimelineBar } from "./timelineLayout.js";

const BARS: TimelineBar[] = [
  { id: "m0", lane: "main", rawStart: 0, rawEnd: 4, warpStart: 0, warpEnd: 4 },
  { id: "s0", lane: "sub", rawStart: 20, rawEnd: 26, warpStart: 6, warpEnd: 12 },
];

describe("layoutTimeline", () => {
  it("reserves a second block only when the warped clock is drawn", () => {
    const base = { lanes: ["main", "sub"], bars: BARS, hasTitle: false };
    const withWarp = layoutTimeline({ ...base, showWarp: true });
    const withoutWarp = layoutTimeline({ ...base, showWarp: false });

    expect(withWarp.svgHeight).toBeGreaterThan(withoutWarp.svgHeight);
    expect(withWarp.svgHeight - withoutWarp.svgHeight).toBe(
      withWarp.warpTop - withoutWarp.rawBottom + withWarp.blockHeight,
    );
  });

  it("scales to the widest of either clock, so a warped bar is never clipped", () => {
    // The warped clock here runs past the raw one, which the raw-only max would miss.
    const bars: TimelineBar[] = [
      { id: "a", lane: "main", rawStart: 0, rawEnd: 5, warpStart: 0, warpEnd: 30 },
    ];
    const layout = layoutTimeline({ lanes: ["main"], bars, showWarp: true, hasTitle: false });

    expect(layout.maxEnd).toBe(30);
    expect(layout.x(30)).toBeLessThanOrEqual(layout.svgWidth);
  });

  it("stays finite with no bars, rather than dividing by a zero span", () => {
    const layout = layoutTimeline({ lanes: [], bars: [], showWarp: true, hasTitle: false });

    expect(layout.maxEnd).toBe(1);
    expect(Number.isFinite(layout.px)).toBe(true);
    expect(Number.isFinite(layout.svgWidth)).toBe(true);
    expect(layout.blockHeight).toBe(0);
  });
});

describe("timelineNodeSize", () => {
  it("adds title chrome to the height it reports", () => {
    const base = { lanes: ["main"], bars: BARS, showWarp: true };
    const titled = timelineNodeSize({ ...base, hasTitle: true });
    const untitled = timelineNodeSize({ ...base, hasTitle: false });

    expect(titled.width).toBe(untitled.width);
    expect(titled.height).toBeGreaterThan(untitled.height);
  });
});

describe("laneRole", () => {
  it("gives each lane its own hue and cycles past the palette", () => {
    const lanes = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];

    expect(laneRole("a", lanes)).not.toBe(laneRole("b", lanes));
    expect(laneRole("i", lanes)).toBe(laneRole("a", lanes));
  });

  it("falls back to the first hue for a lane not in the list", () => {
    expect(laneRole("absent", ["a", "b"])).toBe(laneRole("a", ["a", "b"]));
  });
});
