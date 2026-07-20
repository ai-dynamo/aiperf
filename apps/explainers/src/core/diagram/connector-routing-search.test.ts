/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { segmentIntersectsBounds } from "./connector-routing-geometry.js";
import { resolveWaypoints, routeCurve } from "./connector-routing-search.js";
import {
  DEFAULT_CURVE_ROUTE_OPTIONS,
  type Bounds2,
  type Point2,
  type RouteObstacle,
} from "./connector-routing-types.js";

function polylineCrossesObstacleInterior(
  waypoints: readonly Point2[],
  bounds: Bounds2,
): boolean {
  for (let i = 0; i < waypoints.length - 1; i += 1) {
    if (segmentIntersectsBounds(waypoints[i]!, waypoints[i + 1]!, bounds, true)) {
      return true;
    }
  }
  return false;
}

describe("resolveWaypoints", () => {
  it("keeps a near-endpoint obstacle after clearance inflation instead of dropping it globally", () => {
    // Start sits in the clearance halo of `near`, not in its true interior.
    // The old filter dropped `near` for the whole path, letting the straight
    // chord cut through it far from the endpoint.
    const start: Point2 = { x: 100, y: 50 };
    const end: Point2 = { x: 400, y: 50 };
    const clearance = 12;
    const near: RouteObstacle = {
      id: "near",
      bounds: { x: 106, y: 20, width: 80, height: 60 },
    };

    const resolved = resolveWaypoints(start, end, [near], clearance);

    expect(resolved.feasible).toBe(true);
    expect(polylineCrossesObstacleInterior(resolved.waypoints, near.bounds)).toBe(false);
    expect(resolved.waypoints.length).toBeGreaterThan(2);
  });
});

describe("routeCurve self-loops", () => {
  it("uses perimeter loop candidates when source and target share an id", () => {
    const box: Bounds2 = { x: 0, y: 0, width: 100, height: 60 };
    const result = routeCurve({
      edgeId: "self",
      start: { x: 100, y: 20 },
      end: { x: 100, y: 40 },
      sourceId: "node",
      targetId: "node",
      sourceBounds: box,
      targetBounds: box,
      fromAnchor: "e",
      toAnchor: "e",
      obstacles: [],
      siblings: [],
      options: DEFAULT_CURVE_ROUTE_OPTIONS,
    });

    // Distinct same-side anchors are farther than the old distance gate;
    // a real self-loop still needs the perimeter candidates (not a straight chord).
    expect(result.waypoints.length).toBeGreaterThan(2);
    expect(
      result.waypoints.some(
        (point) =>
          point.x < box.x ||
          point.x > box.x + box.width ||
          point.y < box.y ||
          point.y > box.y + box.height,
      ),
    ).toBe(true);
  });
});
