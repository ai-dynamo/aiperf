/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  intervalRanks,
  intervalsNodeSize,
  layoutIntervals,
  resolveRanks,
  type IntervalRow,
} from "./intervalsLayout.js";

const ROWS: IntervalRow[] = [
  { id: "P0", label: "parent", start: 0, end: 1, role: "blue" },
  { id: "B0", label: "Explore #2", start: 1.3, end: 5, role: "green" },
  { id: "A0", label: "Explore #1", start: 1.2, end: 4, role: "green" },
];

describe("intervalRanks", () => {
  it("ranks by start, not by authored row order", () => {
    const ranks = intervalRanks(ROWS);
    expect(ranks.get("P0")).toBe(0);
    expect(ranks.get("A0")).toBe(1);
    expect(ranks.get("B0")).toBe(2);
  });

  it("breaks a shared start by end, then by id", () => {
    const ranks = intervalRanks([
      { id: "Z", label: "z", start: 1, end: 2, role: "blue" },
      { id: "A", label: "a", start: 1, end: 2, role: "blue" },
      { id: "M", label: "m", start: 1, end: 1.5, role: "blue" },
    ]);
    expect(ranks.get("M")).toBe(0);
    expect(ranks.get("A")).toBe(1);
    expect(ranks.get("Z")).toBe(2);
  });
});

describe("resolveRanks", () => {
  it("lets a row override its derived rank", () => {
    const ranks = resolveRanks([{ ...ROWS[0]!, rank: 9 }, ROWS[1]!, ROWS[2]!]);
    expect(ranks.get("P0")).toBe(9);
    expect(ranks.get("A0")).toBe(1);
  });
});

describe("layoutIntervals", () => {
  it("leaves headroom past the last end so the final badge is not clipped", () => {
    const layout = layoutIntervals({ rows: ROWS, hasTitle: false });
    // The badge sits at x(end) with radius 8.
    expect(layout.x(5) + 8).toBeLessThan(layout.svgWidth);
  });

  it("grows by one row pitch per added interval", () => {
    const two = layoutIntervals({ rows: ROWS.slice(0, 2), hasTitle: false });
    const three = layoutIntervals({ rows: ROWS, hasTitle: false });
    expect(three.svgHeight - two.svgHeight).toBe(three.rowY(1) - three.rowY(0));
  });

  it("stays finite with no rows", () => {
    const { width, height } = intervalsNodeSize({ rows: [], hasTitle: false });
    expect(Number.isFinite(width)).toBe(true);
    expect(Number.isFinite(height)).toBe(true);
  });
});

describe("whole-pixel boxes", () => {
  it("keeps width integral even when a trace ends on a fractional second", () => {
    const { width } = intervalsNodeSize({
      rows: [{ id: "P1", label: "resume", start: 7.5, end: 8, role: "blue" }],
      hasTitle: true,
    });
    expect(Number.isInteger(width)).toBe(true);
  });
});
