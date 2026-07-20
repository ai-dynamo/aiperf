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
} from "./connector-routing-geometry.js";
import type { RouteObstacle } from "./connector-routing-types.js";

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
    const midX = (start.x + end.x) / 2;
    return [start, { x: midX, y: start.y }, { x: midX, y: end.y }, end];
  }
  const midY = (start.y + end.y) / 2;
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
 * Deterministic rectilinear shortest path over an obstacle-coordinate grid.
 * Endpoint escape stubs preserve cardinal tangents before grid search begins.
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
  const inflated = obstacles.map((obstacle) => ({
    id: obstacle.id,
    bounds: inflateBounds(obstacle.bounds, pad),
  }));
  const sourceDirection = anchorExitDirection(fromAnchor, end, start);
  const targetDirection = anchorExitDirection(toAnchor, start, end);
  const escape = Math.max(pad, 1);
  const sourceEscape = {
    x: roundCanonical(start.x + sourceDirection.x * escape),
    y: roundCanonical(start.y + sourceDirection.y * escape),
  };
  const targetEscape = {
    x: roundCanonical(end.x + targetDirection.x * escape),
    y: roundCanonical(end.y + targetDirection.y * escape),
  };
  if (
    inflated.some(
      (obstacle) =>
        pointInBounds(sourceEscape, obstacle.bounds, true) ||
        pointInBounds(targetEscape, obstacle.bounds, true),
    )
  ) {
    return undefined;
  }

  const xs = new Set<number>([sourceEscape.x, targetEscape.x]);
  const ys = new Set<number>([sourceEscape.y, targetEscape.y]);
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
  const sourceIndex = indexByKey.get(pointKey(sourceEscape));
  const targetIndex = indexByKey.get(pointKey(targetEscape));
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
  const sourceAxis = cardinalAnchorAxis(fromAnchor) ?? "x";
  const startState: SearchState = { index: sourceIndex, axis: sourceAxis };
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
  const result = [start, ...routed, end];
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
    const inflated = obstacles.map((obstacle) => ({
      id: obstacle.id,
      bounds: inflateBounds(obstacle.bounds, clearance),
    }));
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
  if (sourceAxis !== undefined && targetAxis !== undefined && sourceAxis !== targetAxis) {
    return sourceAxis === "x"
      ? `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} H${formatPathNumber(end.x)} V${formatPathNumber(end.y)}`
      : `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} V${formatPathNumber(end.y)} H${formatPathNumber(end.x)}`;
  }
  if (preferX) {
    const midX = (start.x + end.x) / 2;
    return `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} H${formatPathNumber(midX)} V${formatPathNumber(end.y)} H${formatPathNumber(end.x)}`;
  }
  const midY = (start.y + end.y) / 2;
  return `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} V${formatPathNumber(midY)} H${formatPathNumber(end.x)} V${formatPathNumber(end.y)}`;
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
