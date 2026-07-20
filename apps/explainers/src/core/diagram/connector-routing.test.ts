/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { segmentIntersectsBounds } from "./connector-routing-geometry.js";
import { elbowPathData, type Point2 } from "./connector-routing.js";
import type { RouteObstacle } from "./connector-routing-types.js";

/** Parse an elbow `M<x> <y> H.../V...` path back into its ordered vertex list. */
function elbowVertices(d: string): Point2[] {
  const match = /^M(-?\d*\.?\d+) (-?\d*\.?\d+)/.exec(d);
  const start: Point2 = match ? { x: Number(match[1]), y: Number(match[2]) } : { x: 0, y: 0 };
  const rest = d.slice(match?.[0].length ?? 0);
  const tokens = rest.match(/[HV]-?\d*\.?\d+/g) ?? [];
  const points: Point2[] = [start];
  let cursor = start;
  for (const token of tokens) {
    const value = Number(token.slice(1));
    cursor = token.startsWith("H") ? { x: value, y: cursor.y } : { x: cursor.x, y: value };
    points.push(cursor);
  }
  return points;
}

function pathCrossesObstacle(points: readonly Point2[], obstacle: RouteObstacle): boolean {
  for (let index = 1; index < points.length; index += 1) {
    if (segmentIntersectsBounds(points[index - 1]!, points[index]!, obstacle.bounds, true)) {
      return true;
    }
  }
  return false;
}

describe("elbowPathData obstacle avoidance", () => {
  it("routes around a blocker directly between e/w anchors instead of falling back through it", () => {
    // Regression for a silent obstructed fallback: the escape stub used to
    // land inside the blocker (it sits only 10px from `start`, closer than
    // the 12px clearance escape), aborting the search; elbowPathData then
    // fell through to the direct L-route, which still crossed the blocker.
    const start: Point2 = { x: 100, y: 100 };
    const end: Point2 = { x: 200, y: 100 };
    const blocker: RouteObstacle = {
      id: "blocker",
      bounds: { x: 110, y: 0, width: 30, height: 200 },
    };

    const d = elbowPathData(start, end, undefined, undefined, "e", "w", [blocker], 12);
    const points = elbowVertices(d);

    expect(points[0]).toEqual(start);
    expect(points.at(-1)).toEqual(end);
    expect(pathCrossesObstacle(points, blocker)).toBe(false);
  });

  it("routes around a blocker directly between n/s anchors with same-x escape collision", () => {
    const start: Point2 = { x: 100, y: 100 };
    const end: Point2 = { x: 100, y: 200 };
    const blocker: RouteObstacle = {
      id: "blocker",
      bounds: { x: 0, y: 110, width: 200, height: 30 },
    };

    const d = elbowPathData(start, end, undefined, undefined, "s", "n", [blocker], 12);
    const points = elbowVertices(d);

    expect(points[0]).toEqual(start);
    expect(points.at(-1)).toEqual(end);
    expect(pathCrossesObstacle(points, blocker)).toBe(false);
  });

  it("still routes a normal straight e/w connection with no obstacles", () => {
    const start: Point2 = { x: 0, y: 50 };
    const end: Point2 = { x: 200, y: 50 };
    const d = elbowPathData(start, end, undefined, undefined, "e", "w", [], 12);
    const points = elbowVertices(d);
    expect(points[0]).toEqual(start);
    expect(points.at(-1)).toEqual(end);
    expect(points.every((point) => point.y === start.y)).toBe(true);
  });

  it("never penetrates a blocker when both the grid search and the plain 4-point detour fail", () => {
    // Construct case for the detour-fallthrough regression: `capTop`/`capBottom`
    // straddle the source's escape column (x=100) closely enough that both of
    // `resolveElbowEscapeStub`'s perpendicular bend candidates for the east
    // escape are themselves blocked, so `obstacleAwareElbowPoints` aborts
    // entirely. Because `capTop`/`capBottom` also sit on `start`'s straight
    // vertical escape and `wall` spans the full straight horizontal one, all
    // four plain U-shaped candidates in the old `detourAroundObstacles` are
    // blocked too. `elbowPathData` must still return a route that never
    // crosses any of the three obstacles' true bounds instead of silently
    // falling back through `wall`.
    const start: Point2 = { x: 100, y: 100 };
    const end: Point2 = { x: 300, y: 100 };
    const wall: RouteObstacle = {
      id: "wall",
      bounds: { x: 110, y: 0, width: 30, height: 200 },
    };
    const capTop: RouteObstacle = {
      id: "capTop",
      bounds: { x: 90, y: -20, width: 20, height: 60 },
    };
    const capBottom: RouteObstacle = {
      id: "capBottom",
      bounds: { x: 90, y: 160, width: 20, height: 60 },
    };
    const obstacles = [wall, capTop, capBottom];

    const d = elbowPathData(start, end, undefined, undefined, "e", "w", obstacles, 12);
    const points = elbowVertices(d);

    expect(points[0]).toEqual(start);
    expect(points.at(-1)).toEqual(end);
    for (const obstacle of obstacles) {
      expect(pathCrossesObstacle(points, obstacle)).toBe(false);
    }
  });

  it("drops an authored via corridor that cuts through a blocker instead of routing through it", () => {
    // Regression for via bypassing avoidance: the old code composed the
    // `via` polyline unconditionally, even when obstacles were supplied. Here
    // the via corridor detours through `blocker`, but the plain direct route
    // (a straight line between soft/center anchors) is already clear, so
    // dropping `via` and falling through to the direct check must recover a
    // clean route rather than keep the obstructed via path.
    const start: Point2 = { x: 0, y: 0 };
    const end: Point2 = { x: 200, y: 0 };
    const via: Point2 = { x: 100, y: -50 };
    const blocker: RouteObstacle = {
      id: "blocker",
      bounds: { x: 90, y: -70, width: 20, height: 50 },
    };

    const d = elbowPathData(start, end, via, undefined, undefined, undefined, [blocker], 12);
    const points = elbowVertices(d);

    expect(points[0]).toEqual(start);
    expect(points.at(-1)).toEqual(end);
    expect(pathCrossesObstacle(points, blocker)).toBe(false);
  });

  it("still honors a via corridor that avoidance confirms is clear", () => {
    const start: Point2 = { x: 0, y: 0 };
    const end: Point2 = { x: 200, y: 0 };
    const via: Point2 = { x: 100, y: -50 };
    // Far from the via corridor: avoidance must not disturb a genuinely
    // clear authored `via`.
    const distantBlocker: RouteObstacle = {
      id: "distant",
      bounds: { x: 500, y: 500, width: 20, height: 20 },
    };

    const d = elbowPathData(start, end, via, undefined, undefined, undefined, [distantBlocker], 12);
    const points = elbowVertices(d);

    expect(points).toContainEqual(via);
    expect(pathCrossesObstacle(points, distantBlocker)).toBe(false);
  });

  it("runs obstacle avoidance for a diagonal corner anchor, not only cardinal pairs", () => {
    // Regression: obstacle avoidance used to require BOTH anchors to be
    // cardinal (n/s/e/w), so a corner anchor like `se` skipped the check
    // entirely and could return the direct elbow straight through a
    // blocker. It must now run the same search/detour machinery, with escape
    // stubs bent to the diagonal exit direction's dominant axis (bug 4)
    // rather than treating any nonzero dx as horizontal.
    const start: Point2 = { x: 100, y: 100 };
    const end: Point2 = { x: 300, y: 180 };
    const blocker: RouteObstacle = {
      id: "blocker",
      bounds: { x: 190, y: 120, width: 20, height: 40 },
    };

    const d = elbowPathData(start, end, undefined, undefined, "se", "w", [blocker], 12);
    const points = elbowVertices(d);

    expect(points[0]).toEqual(start);
    expect(points.at(-1)).toEqual(end);
    expect(pathCrossesObstacle(points, blocker)).toBe(false);
  });
});

describe("elbowPathData same-side cardinal anchors", () => {
  it("keeps a proper vertical detour for e-to-e anchors that share an x coordinate", () => {
    // Regression: averaging start.x/end.x collapsed to start.x when they
    // were equal, producing a straight vertical line — the wrong axis for
    // an east-facing terminal leg on either end.
    const start: Point2 = { x: 100, y: 50 };
    const end: Point2 = { x: 100, y: 150 };

    const d = elbowPathData(start, end, undefined, undefined, "e", "e", [], 12);
    const points = elbowVertices(d);

    expect(points[0]).toEqual(start);
    expect(points.at(-1)).toEqual(end);
    // The first leg out of `start` must be horizontal (east anchor) rather
    // than an immediate vertical drop.
    expect(points[1]!.y).toBe(start.y);
    expect(points[1]!.x).not.toBe(start.x);
    // The last leg into `end` must also be horizontal.
    const secondToLast = points.at(-2)!;
    expect(secondToLast.y).toBe(end.y);
  });

  it("keeps a proper horizontal detour for n-to-n anchors that share a y coordinate", () => {
    // Regression: averaging start.y/end.y collapsed to start.y when they
    // were equal, producing a straight horizontal line — the wrong axis for
    // a north-facing terminal leg on either end.
    const start: Point2 = { x: 50, y: 100 };
    const end: Point2 = { x: 150, y: 100 };

    const d = elbowPathData(start, end, undefined, undefined, "n", "n", [], 12);
    const points = elbowVertices(d);

    expect(points[0]).toEqual(start);
    expect(points.at(-1)).toEqual(end);
    expect(points[1]!.x).toBe(start.x);
    expect(points[1]!.y).not.toBe(start.y);
    const secondToLast = points.at(-2)!;
    expect(secondToLast.x).toBe(end.x);
  });

  it("leaves a normal (non-degenerate) e-to-e route unchanged in shape", () => {
    const start: Point2 = { x: 0, y: 0 };
    const end: Point2 = { x: 100, y: 100 };
    const d = elbowPathData(start, end, undefined, undefined, "e", "e", [], 12);
    expect(d).toBe("M0 0 H50 V100 H100");
  });
});
