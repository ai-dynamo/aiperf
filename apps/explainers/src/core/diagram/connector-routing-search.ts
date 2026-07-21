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
  segmentIntersectsBounds,
  segmentIsVisible,
  shrinkBoundsToExcludePoint,
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
const MIN_HANDLE = 32.4;
/** Largest control-handle length, in scene units. */
const MAX_HANDLE = 486;
/** Outward nudge applied to obstacle corners so grazing edges stay visible. */
const CORNER_EPSILON = 1.35;

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
 * True when any leg of a polyline crosses an obstacle's true (uninflated)
 * interior. Used to downgrade `feasible` when the resolved waypoints — found
 * against the clearance-inflated search graph — still cut through real
 * obstacle geometry, e.g. an obstacle dropped from the graph because an
 * endpoint sits inside its true bounds.
 */
function polylinePenetratesTrueBounds(
  waypoints: readonly Point2[],
  obstacles: readonly RouteObstacle[],
): boolean {
  for (let i = 0; i < waypoints.length - 1; i += 1) {
    const a = waypoints[i]!;
    const b = waypoints[i + 1]!;
    for (const obstacle of obstacles) {
      if (segmentIntersectsBounds(a, b, obstacle.bounds, true)) {
        return true;
      }
    }
  }
  return false;
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
  // Source/target boxes are excluded upstream. Do not drop third-party
  // obstacles merely because clearance inflation covers an endpoint — that
  // removed them for the *entire* path and let curves cut through far away.
  // Keep the obstacle, but shrink the inflated rect so the endpoint itself is
  // not treated as blocked. Still skip obstacles whose true (uninflated)
  // interior contains an endpoint — that geometry is unavoidable from the
  // endpoint itself; `feasible` below still catches a resulting penetration
  // elsewhere on the path.
  const inflated = obstacles.flatMap((obstacle): RouteObstacle[] => {
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
  if (inflated.length === 0 || segmentIsVisible(start, end, inflated)) {
    const waypoints = [start, end];
    return { waypoints, feasible: !polylinePenetratesTrueBounds(waypoints, obstacles) };
  }
  const path = shortestVisiblePath(start, end, inflated);
  if (path === undefined || path.length < 2) {
    return { waypoints: [start, end], feasible: false };
  }
  const waypoints = simplifyWaypoints(path);
  return { waypoints, feasible: !polylinePenetratesTrueBounds(waypoints, obstacles) };
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

/**
 * Shrink `bounds` so no point within `runway` of `point` — measured along
 * whichever axis {@link shrinkBoundsToExcludePoint} would clear — counts as
 * interior, instead of clearing only the exact point. Leaves bounds unchanged
 * when `point` is outside the open interior.
 *
 * `smoothPolyline` guarantees every rounded segment travels at least
 * `MIN_HANDLE` from each of its own waypoints along the fixed anchor tangent
 * before the curve can bend back toward the other end, and the curvature
 * ladder cannot shrink that runway: `clampHandle` floors the handle length at
 * `MIN_HANDLE` no matter how small `curvature` gets. Clearing only the exact
 * endpoint (as `shrinkBoundsToExcludePoint` does) leaves that unavoidable
 * runway inside the clearance halo whenever the endpoint sits within
 * `clearance` of an obstacle edge, so every curvature the ladder tries keeps
 * registering the same unavoidable penetration and `usedFallback` fires for a
 * route that was already obstacle-free at the polyline level.
 */
function shrinkBoundsForRunway(bounds: Bounds2, point: Point2, runway: number): Bounds2 {
  if (!pointInBounds(point, bounds, true)) {
    return bounds;
  }
  const left = bounds.x;
  const right = bounds.x + bounds.width;
  const top = bounds.y;
  const bottom = bounds.y + bounds.height;
  const distLeft = point.x - left;
  const distRight = right - point.x;
  const distTop = point.y - top;
  const distBottom = bottom - point.y;
  const minDist = Math.min(distLeft, distRight, distTop, distBottom);
  if (minDist === distLeft) {
    const edge = Math.min(point.x + runway, right);
    return { x: edge, y: bounds.y, width: Math.max(right - edge, 0), height: bounds.height };
  }
  if (minDist === distRight) {
    const edge = Math.max(point.x - runway, left);
    return { x: bounds.x, y: bounds.y, width: Math.max(edge - left, 0), height: bounds.height };
  }
  if (minDist === distTop) {
    const edge = Math.min(point.y + runway, bottom);
    return { x: bounds.x, y: edge, width: bounds.width, height: Math.max(bottom - edge, 0) };
  }
  const edge = Math.max(point.y - runway, top);
  return { x: bounds.x, y: bounds.y, width: bounds.width, height: Math.max(edge - top, 0) };
}

/**
 * Obstacles widened by `clearance` for post-smoothing penetration checks, with
 * each inflated rectangle shrunk away from the route's own `start`/`end` by
 * the guaranteed `MIN_HANDLE` runway (see {@link shrinkBoundsForRunway}) —
 * the same endpoints `resolveWaypoints` gives the search graph, but with the
 * wider exemption a rounded curve actually needs. Without it, an obstacle
 * that legitimately sits inside its own clearance halo of a connector
 * endpoint — the exact case `resolveWaypoints` keeps routable — would make
 * the curve's unavoidable minimum-handle bulge near that endpoint register as
 * a permanent penetration no curvature setting can clear.
 *
 * The waypoint search already routes interior legs around this same halo;
 * checking rounded cubics against only the raw obstacle rectangle let a
 * smoothed curve bow into the clearance buffer between waypoints — visually
 * hugging an obstacle closer than authored — without ever tripping the
 * curvature/lane retry ladder below.
 */
function inflateForPenetrationCheck(
  obstacles: readonly RouteObstacle[],
  clearance: number,
  start: Point2,
  end: Point2,
): RouteObstacle[] {
  return obstacles.flatMap((obstacle): RouteObstacle[] => {
    let bounds = inflateBounds(obstacle.bounds, clearance);
    bounds = shrinkBoundsForRunway(bounds, start, MIN_HANDLE);
    bounds = shrinkBoundsForRunway(bounds, end, MIN_HANDLE);
    if (bounds.width <= 0 || bounds.height <= 0) {
      return [];
    }
    return [{ id: obstacle.id, bounds }];
  });
}

function penetratedIds(
  segments: readonly CubicSegment[],
  obstacles: readonly RouteObstacle[],
  clearance: number,
  start: Point2,
  end: Point2,
): string[] {
  const inflated = inflateForPenetrationCheck(obstacles, clearance, start, end);
  const hits = new Set<string>();
  for (const segment of segments) {
    for (const id of cubicPenetrations(segment, inflated)) {
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

/** Perimeter loop side order used when the author has no side preference. */
const SELF_LOOP_SIDE_ORDER: readonly Exclude<PreferredSide, "auto">[] = ["n", "e", "s", "w"];

/**
 * Perimeter loop candidates for a self-edge (source and target on one box).
 * Each candidate exits one side, runs a lane parallel to the box, and returns.
 *
 * Candidates are tried in order and the first zero-penetration result wins
 * (see `routeCurve`), so `preferredSide` is honored by moving the matching
 * side to the front rather than by filtering: an obstructed preferred side
 * still falls through to the remaining sides in their default order.
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
  const bySide: Record<Exclude<PreferredSide, "auto">, readonly Point2[]> = {
    n: [start, { x: start.x, y: top }, { x: end.x, y: top }, end],
    e: [start, { x: right, y: start.y }, { x: right, y: end.y }, end],
    s: [start, { x: start.x, y: bottom }, { x: end.x, y: bottom }, end],
    w: [start, { x: left, y: start.y }, { x: left, y: end.y }, end],
  };
  const preferred = input.options.preferredSide;
  const order: readonly Exclude<PreferredSide, "auto">[] =
    preferred === "auto"
      ? SELF_LOOP_SIDE_ORDER
      : [preferred, ...SELF_LOOP_SIDE_ORDER.filter((side) => side !== preferred)];
  return order.map((side) => bySide[side]);
}

type RenderedPolyline = Readonly<{
  segments: readonly CubicSegment[];
  penetrations: readonly string[];
}>;

/**
 * Smooth a polyline down the curvature ladder, keeping the least-penetrating
 * result. Penetration is checked against obstacles inflated by `clearance` —
 * the same halo the waypoint search avoids, minus the shrink around this
 * polyline's own endpoints — so a curve that bows into the buffer zone
 * (without touching the true obstacle rectangle) still triggers a retry
 * instead of being reported as clean.
 */
function renderPolyline(
  waypoints: readonly Point2[],
  fromDir: Point2,
  toDir: Point2,
  obstacles: readonly RouteObstacle[],
  curvature: number,
  clearance: number,
): RenderedPolyline {
  // `waypoints` always keeps its own first/last points exactly at the route's
  // start/end (resolveWaypoints, selfLoopCandidates, and applyLaneOffset all
  // preserve endpoints), so they double as the shrink anchors `penetratedIds`
  // needs to exempt a legitimately near-endpoint obstacle halo.
  const start = waypoints[0]!;
  const end = waypoints[waypoints.length - 1]!;
  let bestSegments = smoothPolyline(waypoints, fromDir, toDir, curvature);
  let bestPenetrations = penetratedIds(bestSegments, obstacles, clearance, start, end);
  if (bestPenetrations.length > 0) {
    for (const factor of CURVATURE_LADDER) {
      const candidate = smoothPolyline(waypoints, fromDir, toDir, Math.max(curvature * factor, 0.02));
      const penetrations = penetratedIds(candidate, obstacles, clearance, start, end);
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
 * until no rounded segment penetrates a clearance-inflated obstacle (not just
 * the obstacle's raw rectangle). When avoidance is disabled or no
 * penetration-free rounding exists, the least-penetrating finite geometry is
 * returned and `usedFallback` reports the degradation. The function never
 * throws and never emits non-finite coordinates.
 */
export function routeCurve(input: CurveRouteInput): CurveRouteResult {
  const options = input.options ?? DEFAULT_CURVE_ROUTE_OPTIONS;
  const start = { x: roundCanonical(input.start.x), y: roundCanonical(input.start.y) };
  const end = { x: roundCanonical(input.end.x), y: roundCanonical(input.end.y) };
  const fromDir = endpointExitDirection(input.fromAnchor, end, start);
  const toDir = endpointExitDirection(input.toAnchor, start, end);

  // Same-node edges are self-loops even when anchors are far apart; a distance
  // gate near 1e-3*clearance made perimeter candidates effectively unreachable.
  const isSelfLoop = input.sourceId !== undefined && input.sourceId === input.targetId;

  let waypoints: readonly Point2[] = [start, end];
  let feasible = true;
  if (isSelfLoop) {
    const candidates = selfLoopCandidates(input);
    let bestLoop: RenderedPolyline | undefined;
    let bestLoopWaypoints: readonly Point2[] | undefined;
    for (const candidate of candidates) {
      const rendered = renderPolyline(
        candidate,
        fromDir,
        toDir,
        input.obstacles,
        options.curvature,
        options.clearance,
      );
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

  let best = renderPolyline(waypoints, fromDir, toDir, input.obstacles, options.curvature, options.clearance);
  let bestWaypoints = waypoints;

  const laneOffset = input.laneOffset ?? 0;
  if (laneOffset !== 0) {
    for (const factor of LANE_OFFSET_LADDER) {
      const offsetWaypoints = applyLaneOffset(waypoints, laneOffset * factor);
      const rendered = renderPolyline(
        offsetWaypoints,
        fromDir,
        toDir,
        input.obstacles,
        options.curvature,
        options.clearance,
      );
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
