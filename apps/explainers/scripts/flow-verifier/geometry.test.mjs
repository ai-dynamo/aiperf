/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { routeCurve } from "./geometry.mjs";

const BASE_OPTIONS = {
  clearance: 12,
  curvature: 0.45,
  avoidObstacles: true,
  preferredSide: "auto",
  bundle: false,
  parallelGap: 8,
};

describe("geometry.mjs routeCurve (Node flow-verifier mirror)", () => {
  it("keeps a near-endpoint obstacle after clearance inflation instead of dropping it globally", () => {
    // Regression for the drop-on-halo bug: an obstacle whose *inflated*
    // bounds cover an endpoint must stay in the search graph (shrunk around
    // that endpoint), not disappear for the whole path and let the fallback
    // straight chord cut through it far from the endpoint.
    const result = routeCurve({
      edgeId: "near-endpoint",
      start: { x: 100, y: 50 },
      end: { x: 400, y: 50 },
      fromAnchor: "e",
      toAnchor: "w",
      obstacles: [{ id: "near", bounds: { x: 106, y: 20, width: 80, height: 60 } }],
      siblings: [],
      options: { ...BASE_OPTIONS, clearance: 12 },
    });

    expect(result.usedFallback).toBe(false);
    expect(result.penetratedObstacleIds).toEqual([]);
    expect(result.waypoints.length).toBeGreaterThan(2);
  });

  it("routes around an obstacle whose inflated halo covers both endpoints from opposite sides", () => {
    const result = routeCurve({
      edgeId: "both-endpoints-in-halo",
      start: { x: 80, y: 100 },
      end: { x: 140, y: 100 },
      fromAnchor: "e",
      toAnchor: "w",
      obstacles: [{ id: "boxed", bounds: { x: 100, y: 0, width: 20, height: 200 } }],
      siblings: [],
      options: { ...BASE_OPTIONS, clearance: 30 },
    });

    expect(result.usedFallback).toBe(false);
    expect(result.penetratedObstacleIds).toEqual([]);
    expect(result.waypoints.length).toBeGreaterThan(2);
  });

  it("reports a fallback with penetrated ids when the obstacle truly cannot be avoided", () => {
    const result = routeCurve({
      edgeId: "forced-fallback",
      start: { x: 40, y: 100 },
      end: { x: 360, y: 100 },
      fromAnchor: "e",
      toAnchor: "w",
      obstacles: [{ id: "wall", bounds: { x: 0, y: 60, width: 400, height: 80 } }],
      siblings: [],
      options: { ...BASE_OPTIONS, clearance: 12 },
    });

    expect(result.usedFallback).toBe(true);
    expect(result.penetratedObstacleIds).toEqual(["wall"]);
  });
});
