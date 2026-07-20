/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Deterministic geometry primitives for the curved connector router.
 *
 * All helpers are pure and finite: they take plain points and rectangles and
 * never touch React, the DOM, or global state. The TypeScript renderer and the
 * Node/browser verifiers rely on byte-identical behavior, so every constant and
 * sampling count here is fixed rather than tunable.
 */

import type { Bounds2, CubicSegment, Point2, RouteObstacle } from "./connector-routing-types.js";

/** Fixed number of cubic samples used for collision checks (0/32 … 32/32). */
export const CUBIC_SAMPLE_COUNT = 33;

/** Coordinate rounding precision (three decimals) for canonical output. */
const CANONICAL_SCALE = 1000;

/** Round to canonical three-decimal precision, mapping -0 to 0. */
export function roundCanonical(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  const rounded = Math.round(value * CANONICAL_SCALE) / CANONICAL_SCALE;
  return rounded === 0 ? 0 : rounded;
}

/** Stable string key for a point, used for deterministic tie-breaking. */
export function canonicalPointKey(point: Point2): string {
  return `${roundCanonical(point.x)},${roundCanonical(point.y)}`;
}

/** Expand a rectangle by `amount` on every side (clamped to nonnegative). */
export function inflateBounds(bounds: Bounds2, amount: number): Bounds2 {
  const pad = Number.isFinite(amount) && amount > 0 ? amount : 0;
  return {
    x: bounds.x - pad,
    y: bounds.y - pad,
    width: bounds.width + pad * 2,
    height: bounds.height + pad * 2,
  };
}

/**
 * True when a point lies inside a rectangle. `strict` requires the point to be
 * in the open interior; otherwise boundary points count as inside.
 */
export function pointInBounds(point: Point2, bounds: Bounds2, strict = false): boolean {
  const left = bounds.x;
  const right = bounds.x + bounds.width;
  const top = bounds.y;
  const bottom = bounds.y + bounds.height;
  if (strict) {
    return point.x > left && point.x < right && point.y > top && point.y < bottom;
  }
  return point.x >= left && point.x <= right && point.y >= top && point.y <= bottom;
}

/**
 * Liang–Barsky segment/rectangle overlap.
 *
 * Returns true when the segment shares any interior with the rectangle. When
 * `allowBoundary` is true (the default), a segment that only grazes the
 * rectangle edge is not treated as intersecting: the overlap interval must have
 * a midpoint strictly inside the rectangle.
 */
export function segmentIntersectsBounds(
  start: Point2,
  end: Point2,
  bounds: Bounds2,
  allowBoundary = true,
): boolean {
  const left = bounds.x;
  const right = bounds.x + bounds.width;
  const top = bounds.y;
  const bottom = bounds.y + bounds.height;
  const dx = end.x - start.x;
  const dy = end.y - start.y;

  let t0 = 0;
  let t1 = 1;
  const p = [-dx, dx, -dy, dy];
  const q = [start.x - left, right - start.x, start.y - top, bottom - start.y];

  for (let i = 0; i < 4; i += 1) {
    const pi = p[i]!;
    const qi = q[i]!;
    if (pi === 0) {
      // Segment parallel to this edge; outside if it starts beyond the slab.
      if (qi < 0) {
        return false;
      }
    } else {
      const r = qi / pi;
      if (pi < 0) {
        if (r > t1) {
          return false;
        }
        if (r > t0) {
          t0 = r;
        }
      } else {
        if (r < t0) {
          return false;
        }
        if (r < t1) {
          t1 = r;
        }
      }
    }
  }

  if (t0 > t1) {
    return false;
  }
  if (!allowBoundary) {
    return true;
  }
  // Boundary-tangent segments (zero-length overlap) do not count; require the
  // overlap midpoint to sit strictly inside the rectangle interior.
  const mid = (t0 + t1) / 2;
  const midPoint = { x: start.x + dx * mid, y: start.y + dy * mid };
  return pointInBounds(midPoint, bounds, true);
}

/** True when a straight segment enters no inflated obstacle interior. */
export function segmentIsVisible(
  start: Point2,
  end: Point2,
  obstacles: readonly RouteObstacle[],
): boolean {
  if (!Number.isFinite(start.x) || !Number.isFinite(start.y)) {
    return false;
  }
  if (!Number.isFinite(end.x) || !Number.isFinite(end.y)) {
    return false;
  }
  for (const obstacle of obstacles) {
    if (segmentIntersectsBounds(start, end, obstacle.bounds, true)) {
      return false;
    }
  }
  return true;
}

/** Drop duplicate and collinear interior waypoints; endpoints are preserved. */
export function simplifyWaypoints(points: readonly Point2[]): readonly Point2[] {
  const deduped: Point2[] = [];
  for (const point of points) {
    const previous = deduped[deduped.length - 1];
    if (
      previous === undefined ||
      roundCanonical(previous.x) !== roundCanonical(point.x) ||
      roundCanonical(previous.y) !== roundCanonical(point.y)
    ) {
      deduped.push(point);
    }
  }
  if (deduped.length <= 2) {
    return deduped;
  }
  const simplified: Point2[] = [deduped[0]!];
  for (let i = 1; i < deduped.length - 1; i += 1) {
    const prev = simplified[simplified.length - 1]!;
    const curr = deduped[i]!;
    const next = deduped[i + 1]!;
    const cross =
      (curr.x - prev.x) * (next.y - prev.y) - (curr.y - prev.y) * (next.x - prev.x);
    if (Math.abs(cross) > 1e-6) {
      simplified.push(curr);
    }
  }
  simplified.push(deduped[deduped.length - 1]!);
  return simplified;
}

/** Point on a cubic Bézier segment at parameter `t` in [0, 1]. */
export function cubicPoint(segment: CubicSegment, t: number): Point2 {
  const u = 1 - t;
  const a = u * u * u;
  const b = 3 * u * u * t;
  const c = 3 * u * t * t;
  const d = t * t * t;
  return {
    x: a * segment.start.x + b * segment.control1.x + c * segment.control2.x + d * segment.end.x,
    y: a * segment.start.y + b * segment.control1.y + c * segment.control2.y + d * segment.end.y,
  };
}

/**
 * Ids of obstacles a rounded cubic segment penetrates, using fixed sampling.
 * Endpoint samples (t = 0 and t = 1) are allowed to rest on escape corridors,
 * so only strict-interior samples count as penetration.
 */
export function cubicPenetrations(
  segment: CubicSegment,
  obstacles: readonly RouteObstacle[],
): readonly string[] {
  const hits = new Set<string>();
  for (const obstacle of obstacles) {
    for (let i = 0; i < CUBIC_SAMPLE_COUNT; i += 1) {
      const t = i / (CUBIC_SAMPLE_COUNT - 1);
      if (pointInBounds(cubicPoint(segment, t), obstacle.bounds, true)) {
        hits.add(obstacle.id);
        break;
      }
    }
  }
  return [...hits].sort((left, right) => left.localeCompare(right));
}

/** Tight axis-aligned bounds of a point set (zero-size for empty input). */
export function routeBounds(points: readonly Point2[]): Bounds2 {
  if (points.length === 0) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  }
  return {
    x: roundCanonical(minX),
    y: roundCanonical(minY),
    width: roundCanonical(maxX - minX),
    height: roundCanonical(maxY - minY),
  };
}
