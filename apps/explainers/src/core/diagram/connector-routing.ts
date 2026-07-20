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

/**
 * Orthogonal elbow path: `M x1 y1 H/V mid H/V x2 y2`.
 * `via` supplies the bend coordinate; otherwise midpoint. `axis` prefers the
 * first segment direction (`x` → horizontal first).
 */
export function elbowPathData(
  start: Point2,
  end: Point2,
  via: Point2 | undefined,
  axis: "x" | "y" | undefined,
): string {
  const dx = Math.abs(end.x - start.x);
  const dy = Math.abs(end.y - start.y);
  const preferX = axis === "y" ? false : axis === "x" ? true : dx >= dy;
  if (via !== undefined) {
    if (preferX) {
      return `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} H${formatPathNumber(via.x)} V${formatPathNumber(via.y)} H${formatPathNumber(end.x)} V${formatPathNumber(end.y)}`;
    }
    return `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} V${formatPathNumber(via.y)} H${formatPathNumber(via.x)} V${formatPathNumber(end.y)} H${formatPathNumber(end.x)}`;
  }
  if (preferX) {
    const midX = (start.x + end.x) / 2;
    return `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} H${formatPathNumber(midX)} V${formatPathNumber(end.y)} H${formatPathNumber(end.x)}`;
  }
  const midY = (start.y + end.y) / 2;
  return `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} V${formatPathNumber(midY)} H${formatPathNumber(end.x)} V${formatPathNumber(end.y)}`;
}
