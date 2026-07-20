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
