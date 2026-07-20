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

  it("keeps an obstacle whose inflated halo covers BOTH endpoints from opposite sides", () => {
    // `boxed` is narrow and both start and end sit just outside its true
    // interior but inside its clearance halo on opposite sides. A filter that
    // drops the obstacle whenever an endpoint lands in the inflated bounds
    // (rather than shrinking around each endpoint) removes it entirely, and
    // the direct start→end chord then cuts straight through its interior.
    const clearance = 30;
    const boxed: RouteObstacle = {
      id: "boxed",
      bounds: { x: 100, y: 0, width: 20, height: 200 },
    };
    const start: Point2 = { x: 80, y: 100 };
    const end: Point2 = { x: 140, y: 100 };

    const resolved = resolveWaypoints(start, end, [boxed], clearance);

    expect(resolved.feasible).toBe(true);
    expect(polylineCrossesObstacleInterior(resolved.waypoints, boxed.bounds)).toBe(false);
    expect(resolved.waypoints.length).toBeGreaterThan(2);
  });

  it("marks a route infeasible instead of silently reporting success when an obstacle cannot be avoided", () => {
    // `wall` truly contains both endpoints, so it is unavoidable at the
    // endpoints themselves. `feasible` must reflect the resulting straight
    // penetrating fallback rather than reporting success just because A*
    // found *some* polyline over the (obstacle-free) search graph.
    const wall: RouteObstacle = {
      id: "wall",
      bounds: { x: 0, y: 60, width: 400, height: 80 },
    };
    const start: Point2 = { x: 40, y: 100 };
    const end: Point2 = { x: 360, y: 100 };

    const resolved = resolveWaypoints(start, end, [wall], 12);

    expect(resolved.feasible).toBe(false);
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

  it("tries the south perimeter candidate first when preferredSide is 's'", () => {
    // The default (unordered) candidate list is north, east, south, west, so an
    // obstacle-free self-loop always bows north regardless of authoring intent.
    // `preferredSide: "s"` must move the south candidate to the front so it
    // wins the (all-zero-penetration) tie instead of being ignored.
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
      options: { ...DEFAULT_CURVE_ROUTE_OPTIONS, preferredSide: "s" },
    });

    const ys = result.waypoints.map((point) => point.y);
    // A south loop bows below the box; it must not also reach above it.
    expect(Math.max(...ys)).toBeGreaterThan(box.y + box.height);
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(box.y);
  });

  it("still bows north (the default order's first candidate) for preferredSide 'auto'", () => {
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

    const ys = result.waypoints.map((point) => point.y);
    expect(Math.min(...ys)).toBeLessThan(box.y);
    expect(Math.max(...ys)).toBeLessThanOrEqual(box.y + box.height);
  });
});

function densePointInBounds(x: number, y: number, bounds: Bounds2): boolean {
  return (
    x > bounds.x && x < bounds.x + bounds.width && y > bounds.y && y < bounds.y + bounds.height
  );
}

function denseCubicEntersBounds(
  segments: readonly { start: Point2; control1: Point2; control2: Point2; end: Point2 }[],
  bounds: Bounds2,
): boolean {
  for (const segment of segments) {
    for (let i = 0; i <= 400; i += 1) {
      const t = i / 400;
      const u = 1 - t;
      const x =
        u * u * u * segment.start.x +
        3 * u * u * t * segment.control1.x +
        3 * u * t * t * segment.control2.x +
        t * t * t * segment.end.x;
      const y =
        u * u * u * segment.start.y +
        3 * u * u * t * segment.control1.y +
        3 * u * t * t * segment.control2.y +
        t * t * t * segment.end.y;
      if (densePointInBounds(x, y, bounds)) {
        return true;
      }
    }
  }
  return false;
}

describe("routeCurve clearance-halo penetration", () => {
  it("retries down the curvature ladder so the smoothed cubic clears an obstacle's clearance halo, not just its true bounds", () => {
    // `avoidObstacles: false` isolates the plain anchor-tangent bow from the
    // waypoint search so only the post-smoothing penetration/retry ladder is
    // under test. Both anchors exit south with the authored curvature 0.9
    // (handle capped at MAX_HANDLE = 180), so the *initial* anchor-tangent bow
    // is the symmetric cubic y(t) = 540*t*(1-t), peaking at (120, 135).
    // `slab`'s true bounds sit entirely above that peak (top edge y=150>135),
    // so the raw-bounds-only penetration check the bug describes would report
    // zero penetrations immediately and keep the curvature-0.9 bow — which
    // still dips into `slab`'s clearance-40 halo (effective top edge y=110).
    // Checking clearance-inflated bounds must trigger a retry down the
    // curvature ladder (to curvature 0.45, peak 81) that actually keeps clear
    // of the halo instead of merely the true rectangle.
    const clearance = 40;
    const options = {
      ...DEFAULT_CURVE_ROUTE_OPTIONS,
      avoidObstacles: false,
      curvature: 0.9,
      clearance,
    };
    const start: Point2 = { x: 0, y: 0 };
    const end: Point2 = { x: 240, y: 0 };
    const slab: RouteObstacle = { id: "slab", bounds: { x: 100, y: 150, width: 40, height: 30 } };
    const halo: Bounds2 = { x: 60, y: 110, width: 120, height: 110 };

    const result = routeCurve({
      edgeId: "e",
      start,
      end,
      fromAnchor: "s",
      toAnchor: "s",
      obstacles: [slab],
      siblings: [],
      options,
    });

    // Ground truth: the rendered curve never crosses `slab`'s true interior
    // (confirms the fix isn't just over-inflating into an already-blocked
    // route) ...
    expect(denseCubicEntersBounds(result.segments, slab.bounds)).toBe(false);
    // ... but it also stays clear of the clearance halo around `slab`, which
    // the raw-bounds-only check would have let it dip into.
    expect(denseCubicEntersBounds(result.segments, halo)).toBe(false);
  });
});
