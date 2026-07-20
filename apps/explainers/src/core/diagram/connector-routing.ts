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

/** Deterministic offset used to keep a same-axis elbow bend from collapsing. */
const SAME_AXIS_ELBOW_OFFSET = 24;

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
  const horizontal = direction.x !== 0;
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
 * Best-effort rectilinear detour around the combined footprint of every
 * obstacle. Used only as a last resort when the deterministic grid search
 * cannot find a route at all, so a search abort never silently falls back to
 * a straight path that cuts through the blocker. Tries clearing over the
 * top, under the bottom, and past either side of the union bounds and keeps
 * the first candidate whose legs cross no obstacle.
 */
function detourAroundObstacles(
  start: Point2,
  end: Point2,
  inflated: readonly RouteObstacle[],
): readonly Point2[] | undefined {
  const union = unionBounds(inflated);
  if (union === undefined) {
    return undefined;
  }
  const margin = 1;
  const top = roundCanonical(union.y - margin);
  const bottom = roundCanonical(union.y + union.height + margin);
  const left = roundCanonical(union.x - margin);
  const right = roundCanonical(union.x + union.width + margin);
  const candidates: readonly Point2[][] = [
    [start, { x: start.x, y: top }, { x: end.x, y: top }, end],
    [start, { x: start.x, y: bottom }, { x: end.x, y: bottom }, end],
    [start, { x: left, y: start.y }, { x: left, y: end.y }, end],
    [start, { x: right, y: start.y }, { x: right, y: end.y }, end],
  ];
  for (const candidate of candidates) {
    if (orthogonalSegmentsVisible(candidate, inflated)) {
      return candidate;
    }
  }
  return undefined;
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
  const sourceDirection = anchorExitDirection(fromAnchor, end, start);
  const targetDirection = anchorExitDirection(toAnchor, start, end);
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
  const sourceArrivalAxis = elbowStubArrivalAxis(sourceStub, cardinalAnchorAxis(fromAnchor) ?? "x");
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
 * Orthogonal elbow path whose terminal legs are perpendicular to cardinal
 * component anchors. This keeps intermediate legs away from running directly
 * along a component side. `via` supplies an authored corridor coordinate;
 * otherwise the corridor is centered between endpoints. `axis` remains the
 * fallback first-segment preference for soft or corner anchors.
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
  if (
    via === undefined &&
    sourceAxis !== undefined &&
    targetAxis !== undefined &&
    obstacles.length > 0
  ) {
    const inflated = inflateElbowObstacles(obstacles, clearance, start, end);
    const direct = defaultElbowPoints(start, end, preferX, sourceAxis, targetAxis);
    if (!orthogonalSegmentsVisible(direct, inflated)) {
      const routed = obstacleAwareElbowPoints(
        start,
        end,
        fromAnchor!,
        toAnchor!,
        obstacles,
        clearance,
      );
      if (routed !== undefined) {
        return elbowPathFromPoints(routed);
      }
      // The deterministic grid search found no obstacle-free route (e.g. an
      // escape stub could not clear a blocker even after bending). Never
      // fall through to the obstructed `direct` path below: detour around
      // every obstacle's combined footprint as a best-effort last resort.
      const detour = detourAroundObstacles(start, end, inflated);
      if (detour !== undefined) {
        return elbowPathFromPoints(detour);
      }
    }
  }
  if (via !== undefined) {
    if (sourceAxis !== undefined || targetAxis !== undefined) {
      const firstAxis = sourceAxis ?? (preferX ? "x" : "y");
      const lastAxis = targetAxis ?? (firstAxis === "x" ? "y" : "x");
      const firstJoin =
        firstAxis === "x"
          ? { x: via.x, y: start.y }
          : { x: start.x, y: via.y };
      const lastJoin =
        lastAxis === "x"
          ? { x: via.x, y: end.y }
          : { x: end.x, y: via.y };
      return elbowPathFromPoints([start, firstJoin, via, lastJoin, end]);
    }
    if (preferX) {
      return `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} H${formatPathNumber(via.x)} V${formatPathNumber(via.y)} H${formatPathNumber(end.x)} V${formatPathNumber(end.y)}`;
    }
    return `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} V${formatPathNumber(via.y)} H${formatPathNumber(via.x)} V${formatPathNumber(end.y)} H${formatPathNumber(end.x)}`;
  }
  return elbowPathFromPoints(defaultElbowPoints(start, end, preferX, sourceAxis, targetAxis));
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
