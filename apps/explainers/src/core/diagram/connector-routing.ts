/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Shared connector routing helpers for straight, elbow, and curved edges.
 *
 * Anchor vocabulary matches `ConnectorEndpointIr.anchor` and
 * `SceneRenderer.nodeAnchorPoint`: center plus the eight perimeter anchors
 * (`n`/`s`/`e`/`w` and corners `ne`/`nw`/`se`/`sw`, with directional aliases).
 */

import {
  inflateBounds,
  pointInBounds,
  roundCanonical,
  segmentIntersectsBounds,
  shrinkBoundsToExcludePoint,
} from "./connector-routing-geometry.js";
import type { Bounds2, RouteObstacle } from "./connector-routing-types.js";

/**
 * Public facade for the obstacle-aware curved router. The deterministic search,
 * smoothing, option normalization, and path rebasing live in sibling modules and
 * are re-exported here so renderers and verifiers depend on one entry point.
 */
export {
  normalizeCurveRouteOptions,
  routeCurve,
  segmentsToPathData,
  translateRoutePath,
} from "./connector-routing-search.js";
export {
  DEFAULT_CURVE_ROUTE_OPTIONS,
  type Bounds2,
  type CubicSegment,
  type CurveRouteInput,
  type CurveRouteOptions,
  type CurveRouteResult,
  type PreferredSide,
  type RouteObstacle,
  type RoutedSibling,
} from "./connector-routing-types.js";

/** 2D point in scene coordinates. */
export type Point2 = Readonly<{ x: number; y: number }>;

/** Minimal node shape for route-style detection in renderers and verifiers. */
export type RouteStyleNodeLike = Readonly<{
  kind?: string | undefined;
  capability?: string | undefined;
  capabilityId?: string | undefined;
  style?: Readonly<Record<string, unknown>> | undefined;
}>;

function formatPathNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  const rounded = Math.round(value * 1000) / 1000;
  return String(rounded);
}

function normalizeVector(vector: Point2): Point2 {
  const length = Math.hypot(vector.x, vector.y);
  if (length < 1e-6) {
    return { x: 1, y: 0 };
  }
  return { x: vector.x / length, y: vector.y / length };
}

function capabilityOf(node: RouteStyleNodeLike): string {
  const explicit = node.capabilityId ?? node.capability;
  return typeof explicit === "string" ? explicit : "";
}

/**
 * Unit vector pointing outward from a box at the given anchor.
 *
 * For soft `center` anchors, falls back to the direction from `self` toward
 * `peer` when both are provided.
 */
export function anchorExitDirection(
  anchor: string | undefined,
  peer?: Point2,
  self?: Point2,
): Point2 {
  switch ((anchor ?? "center").toLowerCase()) {
    case "left":
    case "west":
    case "w":
      return { x: -1, y: 0 };
    case "right":
    case "east":
    case "e":
      return { x: 1, y: 0 };
    case "top":
    case "north":
    case "n":
      return { x: 0, y: -1 };
    case "bottom":
    case "south":
    case "s":
      return { x: 0, y: 1 };
    case "ne":
      return normalizeVector({ x: 1, y: -1 });
    case "nw":
      return normalizeVector({ x: -1, y: -1 });
    case "se":
      return normalizeVector({ x: 1, y: 1 });
    case "sw":
      return normalizeVector({ x: -1, y: 1 });
    case "center":
    case "middle":
    case "c":
      if (peer !== undefined && self !== undefined) {
        return normalizeVector({ x: peer.x - self.x, y: peer.y - self.y });
      }
      return { x: 1, y: 0 };
    default:
      return { x: 1, y: 0 };
  }
}

/** True when a node should use orthogonal elbow routing. */
export function isElbowRoute(node: RouteStyleNodeLike): boolean {
  const capability = capabilityOf(node);
  if (capability === "core.elbow" || capability === "core.route") {
    return true;
  }
  if (node.kind === "elbow") {
    return true;
  }
  return node.style?.route === "elbow";
}

/** True when a node should use anchor-aware cubic curve routing. */
export function isCurveRoute(node: RouteStyleNodeLike): boolean {
  if (node.style?.route === "curve") {
    return true;
  }
  return capabilityOf(node) === "core.curve";
}

/**
 * Cubic-bezier connector between two resolved anchor points.
 *
 * Control points extend along each endpoint's outward anchor direction so
 * curves leave and enter boxes tangentially for all nine anchor positions.
 */
export function curvePathData(
  start: Point2,
  end: Point2,
  fromAnchor: string | undefined,
  toAnchor: string | undefined,
  curvature = 0.45,
): string {
  const distance = Math.hypot(end.x - start.x, end.y - start.y);
  const clampedCurvature = Number.isFinite(curvature)
    ? Math.min(Math.max(curvature, 0.05), 0.95)
    : 0.45;
  const tension = Math.min(Math.max(distance * clampedCurvature, 24), 180);
  const exitFrom = anchorExitDirection(fromAnchor, end, start);
  const exitTo = anchorExitDirection(toAnchor, start, end);
  const cp1 = {
    x: start.x + exitFrom.x * tension,
    y: start.y + exitFrom.y * tension,
  };
  const cp2 = {
    x: end.x - exitTo.x * tension,
    y: end.y - exitTo.y * tension,
  };
  return `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} C${formatPathNumber(cp1.x)} ${formatPathNumber(cp1.y)} ${formatPathNumber(cp2.x)} ${formatPathNumber(cp2.y)} ${formatPathNumber(end.x)} ${formatPathNumber(end.y)}`;
}

function cardinalAnchorAxis(anchor: string | undefined): "x" | "y" | undefined {
  switch ((anchor ?? "").toLowerCase()) {
    case "left":
    case "west":
    case "w":
    case "right":
    case "east":
    case "e":
      return "x";
    case "top":
    case "north":
    case "n":
    case "bottom":
    case "south":
    case "s":
      return "y";
    default:
      return undefined;
  }
}

function samePoint(a: Point2, b: Point2): boolean {
  return Math.abs(a.x - b.x) <= 1e-6 && Math.abs(a.y - b.y) <= 1e-6;
}

/**
 * Axis a vector points along predominantly. Used for corner/soft anchors
 * (whose exit direction is diagonal or peer-relative) so escape stubs bend
 * perpendicular to the leg that actually dominates travel instead of
 * defaulting to a fixed axis.
 */
function dominantAxis(vector: Point2): "x" | "y" {
  return Math.abs(vector.x) >= Math.abs(vector.y) ? "x" : "y";
}

/**
 * Snaps a possibly-diagonal exit direction to the cardinal direction along
 * its dominant axis, preserving sign. Escape stubs feed directly into
 * H/V-only elbow segments, so a corner or soft anchor's genuinely diagonal
 * exit vector (from `anchorExitDirection`) must be reduced to one axis
 * before it is used to build a stub — otherwise the stub's own endpoint is
 * off-axis from the anchor and produces a segment `elbowPathFromPoints`
 * cannot represent as pure H/V.
 */
function dominantCardinalDirection(direction: Point2): Point2 {
  return dominantAxis(direction) === "x"
    ? { x: direction.x >= 0 ? 1 : -1, y: 0 }
    : { x: 0, y: direction.y >= 0 ? 1 : -1 };
}

/** Deterministic offset used to keep a same-axis elbow bend from collapsing. */
const SAME_AXIS_ELBOW_OFFSET = 64.8;

/**
 * Midpoint of `a`/`b`, offset off of `a` when they coincide. Same-side
 * cardinal anchors (e.g. `e`→`e`, `n`→`n`) can share the coordinate that runs
 * along their own axis; averaging that coordinate would collapse the bend to
 * the anchor itself and leave the terminal leg on the wrong axis.
 */
function axisSafeMidpoint(a: number, b: number): number {
  if (Math.abs(a - b) > 1e-6) {
    return (a + b) / 2;
  }
  return a + SAME_AXIS_ELBOW_OFFSET;
}

function defaultElbowPoints(
  start: Point2,
  end: Point2,
  preferX: boolean,
  sourceAxis: "x" | "y" | undefined,
  targetAxis: "x" | "y" | undefined,
): readonly Point2[] {
  if (sourceAxis !== undefined && targetAxis !== undefined && sourceAxis !== targetAxis) {
    return sourceAxis === "x"
      ? [start, { x: end.x, y: start.y }, end]
      : [start, { x: start.x, y: end.y }, end];
  }
  if (preferX) {
    const midX = axisSafeMidpoint(start.x, end.x);
    return [start, { x: midX, y: start.y }, { x: midX, y: end.y }, end];
  }
  const midY = axisSafeMidpoint(start.y, end.y);
  return [start, { x: start.x, y: midY }, { x: end.x, y: midY }, end];
}

function orthogonalSegmentsVisible(
  points: readonly Point2[],
  obstacles: readonly RouteObstacle[],
): boolean {
  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1]!;
    const end = points[index]!;
    if (
      obstacles.some((obstacle) =>
        segmentIntersectsBounds(start, end, obstacle.bounds, true),
      )
    ) {
      return false;
    }
  }
  return true;
}

/**
 * Obstacles inflated by clearance for elbow routing, with each obstacle's
 * inflated rectangle shrunk away from `start`/`end` (mirrors the curved
 * router's `resolveWaypoints`). Clearance halos frequently overlap a nearby
 * endpoint without the endpoint's own node being the obstacle; shrinking
 * keeps that obstacle in play for the rest of the path instead of either
 * treating the endpoint as permanently blocked or dropping the obstacle
 * globally. An obstacle whose true (uninflated) interior contains an
 * endpoint is dropped entirely, since it cannot be an avoidable third party.
 */
function inflateElbowObstacles(
  obstacles: readonly RouteObstacle[],
  clearance: number,
  start: Point2,
  end: Point2,
): RouteObstacle[] {
  return obstacles.flatMap((obstacle): RouteObstacle[] => {
    if (pointInBounds(start, obstacle.bounds, true) || pointInBounds(end, obstacle.bounds, true)) {
      return [];
    }
    let bounds = inflateBounds(obstacle.bounds, clearance);
    bounds = shrinkBoundsToExcludePoint(bounds, start);
    bounds = shrinkBoundsToExcludePoint(bounds, end);
    if (bounds.width <= 0 || bounds.height <= 0) {
      return [];
    }
    return [{ id: obstacle.id, bounds }];
  });
}

/**
 * Escape stub leaving `anchor` along its cardinal `direction`, ordered from
 * the anchor outward. Returns a single point when the direct escape ray is
 * clear. When an obstacle blocks it — including when the anchor sits close
 * enough that no point along the ray escapes it — retries by bending once
 * perpendicular (above/below a blocked horizontal exit, left/right of a
 * blocked vertical one) far enough to clear every obstacle obstructing the
 * ray, then resumes travel toward the original escape coordinate from clear
 * ground. Returns undefined when no such stub is obstacle-free.
 */
function resolveElbowEscapeStub(
  anchor: Point2,
  direction: Point2,
  distance: number,
  inflated: readonly RouteObstacle[],
): readonly Point2[] | undefined {
  const direct: Point2 = {
    x: roundCanonical(anchor.x + direction.x * distance),
    y: roundCanonical(anchor.y + direction.y * distance),
  };
  const segmentBlocked = (from: Point2, to: Point2): boolean =>
    inflated.some(
      (obstacle) =>
        pointInBounds(to, obstacle.bounds, true) ||
        segmentIntersectsBounds(from, to, obstacle.bounds, true),
    );
  if (!segmentBlocked(anchor, direct)) {
    return [direct];
  }
  // Bend perpendicular to whichever axis dominates travel, not merely
  // whichever component is nonzero — corner/soft anchors exit diagonally,
  // so `direction.x !== 0` would misclassify a mostly-vertical escape as
  // horizontal and bend along the wrong axis.
  const horizontal = dominantAxis(direction) === "x";
  const blockers = inflated.filter(
    (obstacle) =>
      pointInBounds(direct, obstacle.bounds, true) ||
      segmentIntersectsBounds(anchor, direct, obstacle.bounds, true),
  );
  if (blockers.length === 0) {
    return undefined;
  }
  const margin = 1;
  const candidates: Point2[] = horizontal
    ? [
        { x: anchor.x, y: roundCanonical(Math.min(...blockers.map((o) => o.bounds.y)) - margin) },
        {
          x: anchor.x,
          y: roundCanonical(Math.max(...blockers.map((o) => o.bounds.y + o.bounds.height)) + margin),
        },
      ]
    : [
        { x: roundCanonical(Math.min(...blockers.map((o) => o.bounds.x)) - margin), y: anchor.y },
        {
          x: roundCanonical(Math.max(...blockers.map((o) => o.bounds.x + o.bounds.width)) + margin),
          y: anchor.y,
        },
      ];
  candidates.sort(
    (left, right) =>
      Math.hypot(left.x - anchor.x, left.y - anchor.y) -
      Math.hypot(right.x - anchor.x, right.y - anchor.y),
  );
  for (const bend of candidates) {
    if (segmentBlocked(anchor, bend)) {
      continue;
    }
    const resume: Point2 = horizontal ? { x: direct.x, y: bend.y } : { x: bend.x, y: direct.y };
    if (segmentBlocked(bend, resume)) {
      continue;
    }
    return [bend, resume];
  }
  return undefined;
}

/** Axis of a stub's final leg, or `fallback` for a single-point (direct) stub. */
function elbowStubArrivalAxis(stub: readonly Point2[], fallback: "x" | "y"): "x" | "y" {
  if (stub.length < 2) {
    return fallback;
  }
  const previous = stub[stub.length - 2]!;
  const last = stub[stub.length - 1]!;
  return previous.y === last.y ? "x" : "y";
}

/** Tight bounding rectangle enclosing every obstacle, or undefined when empty. */
function unionBounds(boxes: readonly RouteObstacle[]): Bounds2 | undefined {
  if (boxes.length === 0) {
    return undefined;
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const box of boxes) {
    minX = Math.min(minX, box.bounds.x);
    minY = Math.min(minY, box.bounds.y);
    maxX = Math.max(maxX, box.bounds.x + box.bounds.width);
    maxY = Math.max(maxY, box.bounds.y + box.bounds.height);
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/**
 * One rectilinear clear line strictly outside a union bounds rectangle: an
 * axis, the cardinal direction that reaches it from inside, and the
 * coordinate value of the line itself. Any point that lies on this line sits
 * outside every obstacle folded into the union, so a straight leg between two
 * such points — regardless of their other coordinate — can never cross an
 * obstacle.
 */
type ClearLine = Readonly<{ axis: "x" | "y"; direction: Point2; value: number }>;

function clearLinesAround(union: Bounds2): readonly ClearLine[] {
  const margin = 1;
  return [
    { axis: "y", direction: { x: 0, y: -1 }, value: roundCanonical(union.y - margin) },
    {
      axis: "y",
      direction: { x: 0, y: 1 },
      value: roundCanonical(union.y + union.height + margin),
    },
    { axis: "x", direction: { x: -1, y: 0 }, value: roundCanonical(union.x - margin) },
    {
      axis: "x",
      direction: { x: 1, y: 0 },
      value: roundCanonical(union.x + union.width + margin),
    },
  ];
}

/**
 * Rectilinear detour around the combined footprint of every obstacle,
 * against one particular (already inflated) obstacle set. Tries the plain
 * 4-point U through each clear line first; when a leg of every U still
 * crosses some other obstacle (e.g. one sits directly above `start` on the
 * "top" candidate), falls back to escaping `start` and `end` toward the same
 * clear line independently — each escape may bend once around a blocker,
 * mirroring `resolveElbowEscapeStub` — then connecting straight across the
 * line itself, which by construction cannot cross any obstacle in this set.
 */
function detourAroundObstaclesAt(
  start: Point2,
  end: Point2,
  inflated: readonly RouteObstacle[],
): readonly Point2[] | undefined {
  const union = unionBounds(inflated);
  if (union === undefined) {
    return undefined;
  }
  const lines = clearLinesAround(union);
  for (const line of lines) {
    const candidate: Point2[] =
      line.axis === "y"
        ? [start, { x: start.x, y: line.value }, { x: end.x, y: line.value }, end]
        : [start, { x: line.value, y: start.y }, { x: line.value, y: end.y }, end];
    if (orthogonalSegmentsVisible(candidate, inflated)) {
      return candidate;
    }
  }
  for (const line of lines) {
    const startDistance =
      line.axis === "y" ? Math.abs(start.y - line.value) : Math.abs(start.x - line.value);
    const endDistance =
      line.axis === "y" ? Math.abs(end.y - line.value) : Math.abs(end.x - line.value);
    const startStub = resolveElbowEscapeStub(start, line.direction, startDistance, inflated);
    const endStub = resolveElbowEscapeStub(end, line.direction, endDistance, inflated);
    if (startStub === undefined || endStub === undefined) {
      continue;
    }
    const raw = [start, ...startStub, ...[...endStub].reverse(), end];
    const candidate = raw.filter(
      (point, index) => index === 0 || !samePoint(point, raw[index - 1]!),
    );
    if (orthogonalSegmentsVisible(candidate, inflated)) {
      return candidate;
    }
  }
  return undefined;
}

/**
 * Best-effort rectilinear detour around every obstacle. Used only as a last
 * resort when the deterministic grid search cannot find a route at all, so a
 * search abort never silently falls back to a straight path that cuts
 * through a blocker. Retries against the clearance-relaxed (true-bounds)
 * obstacle set when the halo-inflated set leaves no room to maneuver, so the
 * result is always checked against — and guaranteed clear of — at least the
 * real obstacle geometry even in that fallback.
 */
function detourAroundObstacles(
  start: Point2,
  end: Point2,
  obstacles: readonly RouteObstacle[],
  clearance: number,
): readonly Point2[] | undefined {
  const primary = detourAroundObstaclesAt(
    start,
    end,
    inflateElbowObstacles(obstacles, clearance, start, end),
  );
  if (primary !== undefined) {
    return primary;
  }
  return detourAroundObstaclesAt(start, end, inflateElbowObstacles(obstacles, 0, start, end));
}

/**
 * Deterministic rectilinear shortest path over an obstacle-coordinate grid.
 * Endpoint escape stubs preserve cardinal tangents before grid search
 * begins; when a stub's direct ray is obstructed, it bends once perpendicular
 * around the blocker (see `resolveElbowEscapeStub`) before the search runs.
 */
function obstacleAwareElbowPoints(
  start: Point2,
  end: Point2,
  fromAnchor: string,
  toAnchor: string,
  obstacles: readonly RouteObstacle[],
  clearance: number,
): readonly Point2[] | undefined {
  const pad = Number.isFinite(clearance) ? Math.max(clearance, 0) : 12;
  const inflated = inflateElbowObstacles(obstacles, pad, start, end);
  // Corner/soft anchors exit diagonally; cardinalize to the dominant axis so
  // every escape stub — and the segment from `start`/`end` into it — stays
  // orthogonal (see `dominantCardinalDirection`).
  const sourceDirection = dominantCardinalDirection(anchorExitDirection(fromAnchor, end, start));
  const targetDirection = dominantCardinalDirection(anchorExitDirection(toAnchor, start, end));
  const escape = Math.max(pad, 1);
  const sourceStub = resolveElbowEscapeStub(start, sourceDirection, escape, inflated);
  const targetStub = resolveElbowEscapeStub(end, targetDirection, escape, inflated);
  if (sourceStub === undefined || targetStub === undefined) {
    return undefined;
  }
  const sourceAttach = sourceStub[sourceStub.length - 1]!;
  const targetAttach = targetStub[targetStub.length - 1]!;

  const xs = new Set<number>();
  const ys = new Set<number>();
  for (const point of [...sourceStub, ...targetStub]) {
    xs.add(point.x);
    ys.add(point.y);
  }
  for (const obstacle of inflated) {
    xs.add(roundCanonical(obstacle.bounds.x - 0.5));
    xs.add(roundCanonical(obstacle.bounds.x + obstacle.bounds.width + 0.5));
    ys.add(roundCanonical(obstacle.bounds.y - 0.5));
    ys.add(roundCanonical(obstacle.bounds.y + obstacle.bounds.height + 0.5));
  }
  const sortedX = [...xs].sort((left, right) => left - right);
  const sortedY = [...ys].sort((left, right) => left - right);
  const points: Point2[] = [];
  const indexByKey = new Map<string, number>();
  const pointKey = (point: Point2): string => `${roundCanonical(point.x)},${roundCanonical(point.y)}`;
  for (const x of sortedX) {
    for (const y of sortedY) {
      const point = { x, y };
      if (inflated.some((obstacle) => pointInBounds(point, obstacle.bounds, true))) {
        continue;
      }
      indexByKey.set(pointKey(point), points.length);
      points.push(point);
    }
  }
  const sourceIndex = indexByKey.get(pointKey(sourceAttach));
  const targetIndex = indexByKey.get(pointKey(targetAttach));
  if (sourceIndex === undefined || targetIndex === undefined) {
    return undefined;
  }

  const adjacency: Array<Array<{ index: number; axis: "x" | "y"; length: number }>> =
    points.map(() => []);
  const connectVisible = (indices: number[], axis: "x" | "y") => {
    indices.sort((left, right) =>
      axis === "x"
        ? points[left]!.x - points[right]!.x
        : points[left]!.y - points[right]!.y,
    );
    for (let index = 1; index < indices.length; index += 1) {
      const left = indices[index - 1]!;
      const right = indices[index]!;
      const a = points[left]!;
      const b = points[right]!;
      if (!orthogonalSegmentsVisible([a, b], inflated)) {
        continue;
      }
      const length = Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
      adjacency[left]!.push({ index: right, axis, length });
      adjacency[right]!.push({ index: left, axis, length });
    }
  };
  for (const y of sortedY) {
    connectVisible(
      points.flatMap((point, index) => (point.y === y ? [index] : [])),
      "x",
    );
  }
  for (const x of sortedX) {
    connectVisible(
      points.flatMap((point, index) => (point.x === x ? [index] : [])),
      "y",
    );
  }

  type SearchState = Readonly<{ index: number; axis: "x" | "y" }>;
  const stateKey = (state: SearchState): string => `${state.index}:${state.axis}`;
  // Fall back to the exit direction's dominant axis, not a hardcoded "x" —
  // corner/soft anchors have no cardinal axis but still travel predominantly
  // along one of the two when the escape stub landed with a single point.
  const sourceArrivalAxis = elbowStubArrivalAxis(sourceStub, dominantAxis(sourceDirection));
  const startState: SearchState = { index: sourceIndex, axis: sourceArrivalAxis };
  const distances = new Map<string, number>([[stateKey(startState), 0]]);
  const previous = new Map<string, string>();
  const states = new Map<string, SearchState>([[stateKey(startState), startState]]);
  const open = new Set<string>([stateKey(startState)]);
  let targetKey: string | undefined;
  while (open.size > 0) {
    let currentKey: string | undefined;
    let currentDistance = Infinity;
    for (const key of open) {
      const candidate = distances.get(key) ?? Infinity;
      if (
        candidate < currentDistance - 1e-9 ||
        (Math.abs(candidate - currentDistance) <= 1e-9 &&
          (currentKey === undefined || key.localeCompare(currentKey) < 0))
      ) {
        currentKey = key;
        currentDistance = candidate;
      }
    }
    if (currentKey === undefined) break;
    open.delete(currentKey);
    const current = states.get(currentKey)!;
    if (current.index === targetIndex) {
      targetKey = currentKey;
      break;
    }
    for (const edge of adjacency[current.index]!) {
      const next: SearchState = { index: edge.index, axis: edge.axis };
      const nextKey = stateKey(next);
      const bendCost = current.axis === edge.axis ? 0 : Math.max(pad * 2, 8);
      const candidate = currentDistance + edge.length + bendCost;
      const known = distances.get(nextKey) ?? Infinity;
      if (
        candidate < known - 1e-9 ||
        (Math.abs(candidate - known) <= 1e-9 &&
          currentKey.localeCompare(previous.get(nextKey) ?? "\uffff") < 0)
      ) {
        distances.set(nextKey, candidate);
        previous.set(nextKey, currentKey);
        states.set(nextKey, next);
        open.add(nextKey);
      }
    }
  }
  if (targetKey === undefined) {
    return undefined;
  }
  const routed: Point2[] = [];
  let cursor: string | undefined = targetKey;
  while (cursor !== undefined) {
    routed.push(points[states.get(cursor)!.index]!);
    cursor = previous.get(cursor);
  }
  routed.reverse();
  const result = [
    start,
    ...sourceStub.slice(0, -1),
    ...routed,
    ...targetStub.slice(0, -1).reverse(),
    end,
  ];
  return result.filter(
    (point, index) => index === 0 || !samePoint(point, result[index - 1]!),
  );
}

/**
 * Ordered vertices of the authored `via`-corridor elbow: the terminal legs
 * are perpendicular to cardinal component anchors when either end has one,
 * mirroring `defaultElbowPoints`; otherwise the first leg follows `preferX`.
 */
function viaElbowPoints(
  start: Point2,
  end: Point2,
  via: Point2,
  sourceAxis: "x" | "y" | undefined,
  targetAxis: "x" | "y" | undefined,
  preferX: boolean,
): readonly Point2[] {
  if (sourceAxis !== undefined || targetAxis !== undefined) {
    const firstAxis = sourceAxis ?? (preferX ? "x" : "y");
    const lastAxis = targetAxis ?? (firstAxis === "x" ? "y" : "x");
    const firstJoin = firstAxis === "x" ? { x: via.x, y: start.y } : { x: start.x, y: via.y };
    const lastJoin = lastAxis === "x" ? { x: via.x, y: end.y } : { x: end.x, y: via.y };
    return [start, firstJoin, via, lastJoin, end];
  }
  if (preferX) {
    return [start, { x: via.x, y: start.y }, via, { x: end.x, y: via.y }, end];
  }
  return [start, { x: start.x, y: via.y }, via, { x: via.x, y: end.y }, end];
}

/**
 * Orthogonal elbow path whose terminal legs are perpendicular to cardinal
 * component anchors. This keeps intermediate legs away from running directly
 * along a component side. `via` supplies an authored corridor coordinate;
 * otherwise the corridor is centered between endpoints. `axis` remains the
 * fallback first-segment preference for soft or corner anchors.
 *
 * When `obstacles` is non-empty, avoidance is mandatory: the candidate path
 * (the `via` corridor if authored, otherwise the direct elbow) is only used
 * once it is confirmed clear of every obstacle's clearance halo. A blocked
 * `via` is dropped — an authored corridor is a hint, not permission to cut
 * through a blocker — and every blocked candidate defers to the deterministic
 * grid search, then to a guaranteed-clear detour around the obstacles' union
 * footprint. The obstructed candidate itself is never returned.
 */
export function elbowPathData(
  start: Point2,
  end: Point2,
  via: Point2 | undefined,
  axis: "x" | "y" | undefined,
  fromAnchor?: string | undefined,
  toAnchor?: string | undefined,
  obstacles: readonly RouteObstacle[] = [],
  clearance = 12,
): string {
  const dx = Math.abs(end.x - start.x);
  const dy = Math.abs(end.y - start.y);
  const sourceAxis = cardinalAnchorAxis(fromAnchor);
  const targetAxis = cardinalAnchorAxis(toAnchor);
  const preferX =
    sourceAxis === "x" ||
    (sourceAxis === undefined && targetAxis === "x") ||
    (sourceAxis === undefined &&
      targetAxis === undefined &&
      (axis === "x" || (axis !== "y" && dx >= dy)));

  const avoiding = obstacles.length > 0;
  const inflated = avoiding ? inflateElbowObstacles(obstacles, clearance, start, end) : [];

  if (via !== undefined) {
    const viaPoints = viaElbowPoints(start, end, via, sourceAxis, targetAxis, preferX);
    if (!avoiding || orthogonalSegmentsVisible(viaPoints, inflated)) {
      return elbowPathFromPoints(viaPoints);
    }
    // The authored corridor cuts through an obstacle's clearance halo: drop
    // it and fall through to the same obstacle-aware search used below for
    // via-less routes, rather than silently routing through the blocker.
  }

  const direct = defaultElbowPoints(start, end, preferX, sourceAxis, targetAxis);
  if (!avoiding || orthogonalSegmentsVisible(direct, inflated)) {
    return elbowPathFromPoints(direct);
  }

  // Corner/soft anchors have no cardinal axis but still exit along a real
  // direction; `obstacleAwareElbowPoints` and its escape stubs derive
  // bend axes from that direction (see `dominantAxis`), so the search runs
  // regardless of whether either anchor is cardinal.
  const routed = obstacleAwareElbowPoints(
    start,
    end,
    fromAnchor ?? "center",
    toAnchor ?? "center",
    obstacles,
    clearance,
  );
  if (routed !== undefined) {
    return elbowPathFromPoints(routed);
  }
  // The deterministic grid search found no obstacle-free route (e.g. an
  // escape stub could not clear a blocker even after bending). Never fall
  // through to the obstructed `direct` path above: detour around every
  // obstacle's combined footprint instead, which always succeeds short of an
  // endpoint being fully enclosed by obstacles on every side.
  const detour = detourAroundObstacles(start, end, obstacles, clearance);
  if (detour !== undefined) {
    return elbowPathFromPoints(detour);
  }
  return elbowPathFromPoints(direct);
}

function elbowPathFromPoints(points: readonly Point2[]): string {
  const compact = points.filter(
    (point, index) =>
      index === 0 ||
      point.x !== points[index - 1]!.x ||
      point.y !== points[index - 1]!.y,
  );
  const first = compact[0] ?? { x: 0, y: 0 };
  let path = `M${formatPathNumber(first.x)} ${formatPathNumber(first.y)}`;
  for (let index = 1; index < compact.length; index += 1) {
    const previous = compact[index - 1]!;
    const point = compact[index]!;
    path +=
      previous.y === point.y
        ? ` H${formatPathNumber(point.x)}`
        : ` V${formatPathNumber(point.y)}`;
  }
  return path;
}
