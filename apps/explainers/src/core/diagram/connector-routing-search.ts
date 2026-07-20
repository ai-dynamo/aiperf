/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Deterministic visibility-graph search and cubic smoothing for curved routes.
 *
 * The search takes resolved endpoints plus inflated obstacle rectangles, builds
 * a visibility graph over the endpoints and obstacle corners, runs an A* with
 * fully specified tie-breaking, and smooths the resulting polyline into
 * anchor-tangent cubic Bézier segments. All ordering is derived from canonical
 * coordinate keys so repeated runs and the Node/browser verifiers agree byte for
 * byte.
 */

import { anchorExitDirection } from "./connector-routing.js";
import {
  canonicalPointKey,
  cubicPenetrations,
  inflateBounds,
  pointInBounds,
  roundCanonical,
  routeBounds,
  segmentIsVisible,
  simplifyWaypoints,
} from "./connector-routing-geometry.js";
import {
  DEFAULT_CURVE_ROUTE_OPTIONS,
  type Bounds2,
  type CubicSegment,
  type CurveRouteInput,
  type CurveRouteOptions,
  type CurveRouteResult,
  type Point2,
  type PreferredSide,
  type RouteObstacle,
} from "./connector-routing-types.js";

/** Smallest control-handle length, in scene units. */
const MIN_HANDLE = 12;
/** Largest control-handle length, in scene units. */
const MAX_HANDLE = 180;
/** Outward nudge applied to obstacle corners so grazing edges stay visible. */
const CORNER_EPSILON = 0.5;

function unit(vector: Point2): Point2 {
  const length = Math.hypot(vector.x, vector.y);
  if (length < 1e-6) {
    return { x: 1, y: 0 };
  }
  return { x: vector.x / length, y: vector.y / length };
}

function distance(a: Point2, b: Point2): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function clampHandle(length: number, curvature: number): number {
  const raw = length * curvature;
  if (!Number.isFinite(raw)) {
    return MIN_HANDLE;
  }
  return Math.min(Math.max(raw, MIN_HANDLE), MAX_HANDLE);
}

/**
 * Candidate corner points for the visibility graph: the four corners of each
 * inflated obstacle, nudged outward so an edge that only touches the corner is
 * not rejected as an interior crossing. Corners that fall inside another
 * obstacle interior are dropped.
 */
function obstacleCorners(obstacles: readonly RouteObstacle[]): Point2[] {
  const seen = new Set<string>();
  const corners: Point2[] = [];
  for (const obstacle of obstacles) {
    const { x, y, width, height } = obstacle.bounds;
    const candidates: Point2[] = [
      { x: x - CORNER_EPSILON, y: y - CORNER_EPSILON },
      { x: x + width + CORNER_EPSILON, y: y - CORNER_EPSILON },
      { x: x - CORNER_EPSILON, y: y + height + CORNER_EPSILON },
      { x: x + width + CORNER_EPSILON, y: y + height + CORNER_EPSILON },
    ];
    for (const candidate of candidates) {
      if (obstacles.some((other) => pointInBounds(candidate, other.bounds, true))) {
        continue;
      }
      const key = canonicalPointKey(candidate);
      if (!seen.has(key)) {
        seen.add(key);
        corners.push({ x: roundCanonical(candidate.x), y: roundCanonical(candidate.y) });
      }
    }
  }
  return corners;
}

type SearchNode = Readonly<{ point: Point2; key: string }>;

/**
 * Deterministic A* over the visibility graph. Returns the ordered polyline from
 * `start` to `end`, or `undefined` when no obstacle-free polyline exists.
 */
function shortestVisiblePath(
  start: Point2,
  end: Point2,
  obstacles: readonly RouteObstacle[],
): readonly Point2[] | undefined {
  const startNode: SearchNode = { point: start, key: canonicalPointKey(start) };
  const endNode: SearchNode = { point: end, key: canonicalPointKey(end) };
  const nodes: SearchNode[] = [startNode];
  const seenKeys = new Set<string>([startNode.key]);
  for (const corner of obstacleCorners(obstacles)) {
    const key = canonicalPointKey(corner);
    if (!seenKeys.has(key)) {
      seenKeys.add(key);
      nodes.push({ point: corner, key });
    }
  }
  if (!seenKeys.has(endNode.key)) {
    seenKeys.add(endNode.key);
    nodes.push(endNode);
  }
  // Canonical, coordinate-sorted node order for stable neighbor iteration.
  nodes.sort((a, b) => a.key.localeCompare(b.key));

  const gScore = new Map<string, number>();
  const cameFrom = new Map<string, string>();
  const nodeByKey = new Map<string, SearchNode>();
  for (const node of nodes) {
    nodeByKey.set(node.key, node);
  }
  gScore.set(startNode.key, 0);

  const open = new Set<string>([startNode.key]);
  const heuristic = (node: SearchNode): number => distance(node.point, end);

  while (open.size > 0) {
    let currentKey: string | undefined;
    let bestF = Infinity;
    for (const key of open) {
      const g = gScore.get(key) ?? Infinity;
      const node = nodeByKey.get(key)!;
      const f = g + heuristic(node);
      if (f < bestF - 1e-9 || (Math.abs(f - bestF) <= 1e-9 && (currentKey === undefined || key.localeCompare(currentKey) < 0))) {
        bestF = f;
        currentKey = key;
      }
    }
    if (currentKey === undefined) {
      break;
    }
    if (currentKey === endNode.key) {
      const path: Point2[] = [];
      let cursor: string | undefined = currentKey;
      while (cursor !== undefined) {
        path.push(nodeByKey.get(cursor)!.point);
        cursor = cameFrom.get(cursor);
      }
      path.reverse();
      return path;
    }
    open.delete(currentKey);
    const current = nodeByKey.get(currentKey)!;
    const currentG = gScore.get(currentKey) ?? Infinity;
    for (const neighbor of nodes) {
      if (neighbor.key === currentKey) {
        continue;
      }
      if (!segmentIsVisible(current.point, neighbor.point, obstacles)) {
        continue;
      }
      const tentative = currentG + distance(current.point, neighbor.point);
      const known = gScore.get(neighbor.key) ?? Infinity;
      if (tentative < known - 1e-9) {
        cameFrom.set(neighbor.key, currentKey);
        gScore.set(neighbor.key, tentative);
        open.add(neighbor.key);
      }
    }
  }
  return undefined;
}

/**
 * Per-point unit travel tangents for a polyline. Endpoints use the outward
 * anchor directions (entry runs opposite to the target's outward direction);
 * interior points use the normalized chord across their neighbors.
 */
function travelTangents(
  points: readonly Point2[],
  fromDir: Point2,
  toDir: Point2,
): Point2[] {
  const tangents: Point2[] = [];
  for (let i = 0; i < points.length; i += 1) {
    if (i === 0) {
      tangents.push(unit(fromDir));
    } else if (i === points.length - 1) {
      tangents.push(unit({ x: -toDir.x, y: -toDir.y }));
    } else {
      tangents.push(unit({ x: points[i + 1]!.x - points[i - 1]!.x, y: points[i + 1]!.y - points[i - 1]!.y }));
    }
  }
  return tangents;
}

/** Smooth a polyline into anchor-tangent cubic segments. */
export function smoothPolyline(
  points: readonly Point2[],
  fromDir: Point2,
  toDir: Point2,
  curvature: number,
): CubicSegment[] {
  const segments: CubicSegment[] = [];
  if (points.length < 2) {
    return segments;
  }
  const tangents = travelTangents(points, fromDir, toDir);
  for (let i = 0; i < points.length - 1; i += 1) {
    const a = points[i]!;
    const b = points[i + 1]!;
    const handle = clampHandle(distance(a, b), curvature);
    segments.push({
      start: a,
      control1: { x: a.x + tangents[i]!.x * handle, y: a.y + tangents[i]!.y * handle },
      control2: { x: b.x - tangents[i + 1]!.x * handle, y: b.y - tangents[i + 1]!.y * handle },
      end: b,
    });
  }
  return segments;
}

/** Compute the outward anchor direction for an endpoint. */
export function endpointExitDirection(
  anchor: string | undefined,
  peer: Point2,
  self: Point2,
): Point2 {
  return unit(anchorExitDirection(anchor, peer, self));
}

/**
 * Resolve an obstacle-avoiding polyline between two endpoints. Returns the
 * simplified waypoint list (always including both endpoints). When no
 * obstacle-free polyline exists, returns just the endpoints.
 */
export function resolveWaypoints(
  start: Point2,
  end: Point2,
  obstacles: readonly RouteObstacle[],
  clearance: number,
): { waypoints: readonly Point2[]; feasible: boolean } {
  const inflated = obstacles
    .map((obstacle): RouteObstacle => ({ id: obstacle.id, bounds: inflateBounds(obstacle.bounds, clearance) }))
    .filter(
      (obstacle) =>
        !pointInBounds(start, obstacle.bounds, true) && !pointInBounds(end, obstacle.bounds, true),
    );
  if (inflated.length === 0 || segmentIsVisible(start, end, inflated)) {
    return { waypoints: [start, end], feasible: true };
  }
  const path = shortestVisiblePath(start, end, inflated);
  if (path === undefined || path.length < 2) {
    return { waypoints: [start, end], feasible: false };
  }
  return { waypoints: simplifyWaypoints(path), feasible: true };
}

function formatNumber(value: number): string {
  return String(roundCanonical(value));
}

/** Serialize a start point plus cubic segments into SVG path data. */
export function segmentsToPathData(start: Point2, segments: readonly CubicSegment[]): string {
  if (segments.length === 0) {
    return `M${formatNumber(start.x)} ${formatNumber(start.y)}`;
  }
  let d = `M${formatNumber(start.x)} ${formatNumber(start.y)}`;
  for (const segment of segments) {
    d +=
      ` C${formatNumber(segment.control1.x)} ${formatNumber(segment.control1.y)}` +
      ` ${formatNumber(segment.control2.x)} ${formatNumber(segment.control2.y)}` +
      ` ${formatNumber(segment.end.x)} ${formatNumber(segment.end.y)}`;
  }
  return d;
}

function readNumber(record: Readonly<Record<string, unknown>> | undefined, key: string): number | undefined {
  const value = record?.[key];
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

function readBoolean(
  record: Readonly<Record<string, unknown>> | undefined,
  key: string,
): boolean | undefined {
  const value = record?.[key];
  if (typeof value === "boolean") {
    return value;
  }
  if (value === "true") {
    return true;
  }
  if (value === "false") {
    return false;
  }
  return undefined;
}

function readPreferredSide(
  record: Readonly<Record<string, unknown>> | undefined,
  key: string,
): PreferredSide | undefined {
  const value = record?.[key];
  if (typeof value !== "string") {
    return undefined;
  }
  switch (value.toLowerCase()) {
    case "auto":
      return "auto";
    case "n":
    case "north":
    case "top":
      return "n";
    case "s":
    case "south":
    case "bottom":
      return "s";
    case "e":
    case "east":
    case "right":
      return "e";
    case "w":
    case "west":
    case "left":
      return "w";
    default:
      return undefined;
  }
}

/**
 * Derive finite, clamped routing options from an open style record. Missing or
 * malformed fields fall back to {@link DEFAULT_CURVE_ROUTE_OPTIONS}.
 */
export function normalizeCurveRouteOptions(
  style: Readonly<Record<string, unknown>> | undefined,
): CurveRouteOptions {
  const clearanceRaw = readNumber(style, "clearance");
  const curvatureRaw = readNumber(style, "curvature");
  const parallelGapRaw = readNumber(style, "parallelGap") ?? readNumber(style, "laneGap");
  return {
    clearance:
      clearanceRaw !== undefined && clearanceRaw >= 0
        ? clearanceRaw
        : DEFAULT_CURVE_ROUTE_OPTIONS.clearance,
    curvature:
      curvatureRaw !== undefined
        ? Math.min(Math.max(curvatureRaw, 0.05), 0.95)
        : DEFAULT_CURVE_ROUTE_OPTIONS.curvature,
    avoidObstacles: readBoolean(style, "avoidObstacles") ?? DEFAULT_CURVE_ROUTE_OPTIONS.avoidObstacles,
    preferredSide: readPreferredSide(style, "preferredSide") ?? DEFAULT_CURVE_ROUTE_OPTIONS.preferredSide,
    bundle: readBoolean(style, "bundle") ?? DEFAULT_CURVE_ROUTE_OPTIONS.bundle,
    parallelGap:
      parallelGapRaw !== undefined && parallelGapRaw >= 0
        ? parallelGapRaw
        : DEFAULT_CURVE_ROUTE_OPTIONS.parallelGap,
  };
}

function segmentBoundsPoints(start: Point2, segments: readonly CubicSegment[]): Point2[] {
  const points: Point2[] = [start];
  for (const segment of segments) {
    points.push(segment.control1, segment.control2, segment.end);
  }
  return points;
}

/** Deterministic descending curvature ladder for penetration-reducing retries. */
const CURVATURE_LADDER = [1, 0.5, 0.25, 0.05] as const;

/** Deterministic descending lane-offset ladder (100% → 0% in 25% steps). */
const LANE_OFFSET_LADDER = [1, 0.75, 0.5, 0.25] as const;

function penetratedIds(
  segments: readonly CubicSegment[],
  obstacles: readonly RouteObstacle[],
): string[] {
  const hits = new Set<string>();
  for (const segment of segments) {
    for (const id of cubicPenetrations(segment, obstacles)) {
      hits.add(id);
    }
  }
  return [...hits].sort((a, b) => a.localeCompare(b));
}

/** Perpendicular unit normal (left of travel) for a chord `a → b`. */
function chordNormal(a: Point2, b: Point2): Point2 {
  return unit({ x: -(b.y - a.y), y: b.x - a.x });
}

/**
 * Displace a polyline laterally by `offset` scene units. A straight two-point
 * route gains a bowed midpoint; multi-point routes push interior waypoints along
 * their local normal. Endpoints are always preserved exactly.
 */
function applyLaneOffset(points: readonly Point2[], offset: number): readonly Point2[] {
  if (!Number.isFinite(offset) || offset === 0 || points.length < 2) {
    return points;
  }
  const start = points[0]!;
  const end = points[points.length - 1]!;
  if (points.length === 2) {
    const normal = chordNormal(start, end);
    const mid = {
      x: (start.x + end.x) / 2 + normal.x * offset,
      y: (start.y + end.y) / 2 + normal.y * offset,
    };
    return [start, mid, end];
  }
  const shifted: Point2[] = [start];
  for (let i = 1; i < points.length - 1; i += 1) {
    const normal = chordNormal(points[i - 1]!, points[i + 1]!);
    shifted.push({ x: points[i]!.x + normal.x * offset, y: points[i]!.y + normal.y * offset });
  }
  shifted.push(end);
  return shifted;
}

/**
 * Perimeter loop candidates for a self-edge (source and target on one box).
 * Each candidate exits one side, runs a lane parallel to the box, and returns.
 */
function selfLoopCandidates(input: CurveRouteInput): readonly (readonly Point2[])[] {
  const bounds = input.sourceBounds ?? input.targetBounds;
  if (bounds === undefined) {
    return [];
  }
  const gap = input.options.clearance + input.options.parallelGap;
  const start = { x: roundCanonical(input.start.x), y: roundCanonical(input.start.y) };
  const end = { x: roundCanonical(input.end.x), y: roundCanonical(input.end.y) };
  const left = bounds.x - gap;
  const right = bounds.x + bounds.width + gap;
  const top = bounds.y - gap;
  const bottom = bounds.y + bounds.height + gap;
  return [
    [start, { x: start.x, y: top }, { x: end.x, y: top }, end],
    [start, { x: right, y: start.y }, { x: right, y: end.y }, end],
    [start, { x: start.x, y: bottom }, { x: end.x, y: bottom }, end],
    [start, { x: left, y: start.y }, { x: left, y: end.y }, end],
  ];
}

type RenderedPolyline = Readonly<{
  segments: readonly CubicSegment[];
  penetrations: readonly string[];
}>;

/** Smooth a polyline down the curvature ladder, keeping the least-penetrating result. */
function renderPolyline(
  waypoints: readonly Point2[],
  fromDir: Point2,
  toDir: Point2,
  obstacles: readonly RouteObstacle[],
  curvature: number,
): RenderedPolyline {
  let bestSegments = smoothPolyline(waypoints, fromDir, toDir, curvature);
  let bestPenetrations = penetratedIds(bestSegments, obstacles);
  if (bestPenetrations.length > 0) {
    for (const factor of CURVATURE_LADDER) {
      const candidate = smoothPolyline(waypoints, fromDir, toDir, Math.max(curvature * factor, 0.02));
      const penetrations = penetratedIds(candidate, obstacles);
      if (penetrations.length < bestPenetrations.length) {
        bestSegments = candidate;
        bestPenetrations = penetrations;
      }
      if (penetrations.length === 0) {
        break;
      }
    }
  }
  return { segments: bestSegments, penetrations: bestPenetrations };
}

/**
 * Resolve one curved edge into obstacle-aware cubic geometry.
 *
 * The result is deterministic for identical input: the polyline comes from the
 * visibility-graph A*, then smoothing is retried down a fixed curvature ladder
 * until no rounded segment penetrates an obstacle interior. When avoidance is
 * disabled or no penetration-free rounding exists, the least-penetrating finite
 * geometry is returned and `usedFallback` reports the degradation. The function
 * never throws and never emits non-finite coordinates.
 */
export function routeCurve(input: CurveRouteInput): CurveRouteResult {
  const options = input.options ?? DEFAULT_CURVE_ROUTE_OPTIONS;
  const start = { x: roundCanonical(input.start.x), y: roundCanonical(input.start.y) };
  const end = { x: roundCanonical(input.end.x), y: roundCanonical(input.end.y) };
  const fromDir = endpointExitDirection(input.fromAnchor, end, start);
  const toDir = endpointExitDirection(input.toAnchor, start, end);

  const isSelfLoop =
    input.sourceId !== undefined &&
    input.sourceId === input.targetId &&
    Math.hypot(end.x - start.x, end.y - start.y) < 1e-3 * (options.clearance + 1) + 4;

  let waypoints: readonly Point2[] = [start, end];
  let feasible = true;
  if (isSelfLoop) {
    const candidates = selfLoopCandidates(input);
    let bestLoop: RenderedPolyline | undefined;
    let bestLoopWaypoints: readonly Point2[] | undefined;
    for (const candidate of candidates) {
      const rendered = renderPolyline(candidate, fromDir, toDir, input.obstacles, options.curvature);
      if (rendered.penetrations.length === 0) {
        bestLoop = rendered;
        bestLoopWaypoints = candidate;
        break;
      }
      if (bestLoop === undefined || rendered.penetrations.length < bestLoop.penetrations.length) {
        bestLoop = rendered;
        bestLoopWaypoints = candidate;
      }
    }
    if (bestLoop !== undefined && bestLoopWaypoints !== undefined) {
      return {
        d: segmentsToPathData(start, bestLoop.segments),
        waypoints: bestLoopWaypoints,
        segments: bestLoop.segments,
        bounds: routeBounds(segmentBoundsPoints(start, bestLoop.segments)),
        usedFallback: bestLoop.penetrations.length > 0,
        penetratedObstacleIds: bestLoop.penetrations,
      };
    }
  }

  if (options.avoidObstacles && input.obstacles.length > 0) {
    const resolved = resolveWaypoints(start, end, input.obstacles, options.clearance);
    waypoints = resolved.waypoints;
    feasible = resolved.feasible;
  }

  let best = renderPolyline(waypoints, fromDir, toDir, input.obstacles, options.curvature);
  let bestWaypoints = waypoints;

  const laneOffset = input.laneOffset ?? 0;
  if (laneOffset !== 0) {
    for (const factor of LANE_OFFSET_LADDER) {
      const offsetWaypoints = applyLaneOffset(waypoints, laneOffset * factor);
      const rendered = renderPolyline(offsetWaypoints, fromDir, toDir, input.obstacles, options.curvature);
      if (rendered.penetrations.length <= best.penetrations.length) {
        best = rendered;
        bestWaypoints = offsetWaypoints;
        break;
      }
    }
  }

  return {
    d: segmentsToPathData(start, best.segments),
    waypoints: bestWaypoints,
    segments: best.segments,
    bounds: routeBounds(segmentBoundsPoints(start, best.segments)),
    usedFallback: !feasible || best.penetrations.length > 0,
    penetratedObstacleIds: best.penetrations,
  };
}

/** Translate a resolved route by `(dx, dy)`, e.g. to rebase into layout space. */
export function translateRoutePath(
  route: CurveRouteResult,
  dx: number,
  dy: number,
): CurveRouteResult {
  const shift = (point: Point2): Point2 => ({
    x: roundCanonical(point.x + dx),
    y: roundCanonical(point.y + dy),
  });
  const segments = route.segments.map(
    (segment): CubicSegment => ({
      start: shift(segment.start),
      control1: shift(segment.control1),
      control2: shift(segment.control2),
      end: shift(segment.end),
    }),
  );
  const waypoints = route.waypoints.map(shift);
  const start = waypoints[0] ?? shift(route.segments[0]?.start ?? { x: 0, y: 0 });
  return {
    d: segmentsToPathData(start, segments),
    waypoints,
    segments,
    bounds: routeBounds(segmentBoundsPoints(start, segments)),
    usedFallback: route.usedFallback,
    penetratedObstacleIds: route.penetratedObstacleIds,
  };
}

export type { Bounds2 };
