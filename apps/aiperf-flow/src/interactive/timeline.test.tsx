/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  buildOffsetForOrder,
  eventOffsetMs,
  fractionForEvent,
  fractionForOrder,
  timelineBounds,
  type Lane,
  type RequestPath,
  type SeamFrame,
  type StageRegion,
  type TimelineEvent,
} from "./timeline.js";

const EVENTS: TimelineEvent[] = [
  { id: "a", label: "A", laneId: "dataset", atOrder: 0, realOffsetMs: 0 },
  { id: "b", label: "B", laneId: "scheduler", atOrder: 1, realOffsetMs: 10 },
  // A big wall-clock jump (like TTFT): order +1, but offset +190.
  { id: "c", label: "C", laneId: "server", atOrder: 2, realOffsetMs: 200 },
];

describe("interactive/timeline data model", () => {
  it("types a Lane / StageRegion / SeamFrame / RequestPath with the documented fields", () => {
    const lane: Lane = { id: "dataset", label: "Dataset" };
    const region: StageRegion = {
      id: "dataset",
      laneId: "dataset",
      label: "Dataset loading",
      startOrder: 0,
      endOrder: 2,
    };
    const frame: SeamFrame = { id: "transport", label: "Transport", spanLaneIds: ["server"], spanOrder: [1, 2] };
    const path: RequestPath = ["a", "b", "c"];

    expect(lane.label).toBe("Dataset");
    expect(region.laneId).toBe("dataset");
    expect(frame.spanOrder).toEqual([1, 2]);
    expect(path).toHaveLength(3);
  });

  it("falls back to atOrder when an event has no realOffsetMs", () => {
    expect(eventOffsetMs({ id: "x", label: "X", laneId: "l", atOrder: 4 })).toBe(4);
    expect(eventOffsetMs({ id: "y", label: "Y", laneId: "l", atOrder: 4, realOffsetMs: 99 })).toBe(99);
  });
});

describe("interactive/timeline layout math", () => {
  it("computes order + wall-ms bounds across events", () => {
    const bounds = timelineBounds(EVENTS);
    expect(bounds).toEqual({ minOrder: 0, maxOrder: 2, minOffsetMs: 0, maxOffsetMs: 200 });
  });

  it("returns all-zero bounds for an empty event set", () => {
    expect(timelineBounds([])).toEqual({ minOrder: 0, maxOrder: 0, minOffsetMs: 0, maxOffsetMs: 0 });
  });

  it("spaces events evenly on the virtual scale (by atOrder)", () => {
    const bounds = timelineBounds(EVENTS);
    expect(fractionForEvent(EVENTS[0]!, "virtual", bounds)).toBeCloseTo(0);
    expect(fractionForEvent(EVENTS[1]!, "virtual", bounds)).toBeCloseTo(0.5);
    expect(fractionForEvent(EVENTS[2]!, "virtual", bounds)).toBeCloseTo(1);
  });

  it("spaces events by wall-ms on the real scale — a latency gap opens a wider gap", () => {
    const bounds = timelineBounds(EVENTS);
    // On the real axis the middle event sits at 10/200 = 0.05, not 0.5 — the TTFT-like jump dominates.
    expect(fractionForEvent(EVENTS[1]!, "real", bounds)).toBeCloseTo(0.05);
    expect(fractionForEvent(EVENTS[2]!, "real", bounds)).toBeCloseTo(1);
    // The real fraction of the middle event is far smaller than its virtual fraction.
    expect(fractionForEvent(EVENTS[1]!, "real", bounds)).toBeLessThan(
      fractionForEvent(EVENTS[1]!, "virtual", bounds),
    );
  });

  it("interpolates order→offset for region/seam bounds on the real scale", () => {
    const offsetForOrder = buildOffsetForOrder(EVENTS);
    expect(offsetForOrder(0)).toBe(0);
    expect(offsetForOrder(1)).toBe(10);
    expect(offsetForOrder(2)).toBe(200);
    // Halfway between order 1 (10ms) and order 2 (200ms).
    expect(offsetForOrder(1.5)).toBeCloseTo(105);
    // Clamps beyond the ends.
    expect(offsetForOrder(-5)).toBe(0);
    expect(offsetForOrder(9)).toBe(200);
  });

  it("maps a bare order to a fraction under both scales", () => {
    const bounds = timelineBounds(EVENTS);
    const offsetForOrder = buildOffsetForOrder(EVENTS);
    expect(fractionForOrder(1, "virtual", bounds, offsetForOrder)).toBeCloseTo(0.5);
    expect(fractionForOrder(1, "real", bounds, offsetForOrder)).toBeCloseTo(0.05);
  });

  it("clamps a single-event (zero-span) domain to fraction 0", () => {
    const one: TimelineEvent[] = [{ id: "solo", label: "S", laneId: "l", atOrder: 3, realOffsetMs: 7 }];
    const bounds = timelineBounds(one);
    expect(fractionForEvent(one[0]!, "virtual", bounds)).toBe(0);
    expect(fractionForEvent(one[0]!, "real", bounds)).toBe(0);
  });
});
