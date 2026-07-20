/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Public data contracts for the deterministic curved connector router.
 *
 * These types are transport-neutral: the router, the React renderer, and the
 * Node/browser verifiers all exchange the same immutable value shapes. Nothing
 * here depends on React, the DOM, or scene traversal.
 */

/** 2D point in scene (world) coordinates. */
export type Point2 = Readonly<{ x: number; y: number }>;

/** Axis-aligned rectangle in scene (world) coordinates. */
export type Bounds2 = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
}>;

/** A routing obstacle: a stable id paired with its world-space bounds. */
export type RouteObstacle = Readonly<{ id: string; bounds: Bounds2 }>;

/** Author-selectable side preference for route emergence and loops. */
export type PreferredSide = "auto" | "n" | "s" | "e" | "w";

/** Normalized, always-finite routing options derived from the open style record. */
export type CurveRouteOptions = Readonly<{
  clearance: number;
  curvature: number;
  avoidObstacles: boolean;
  preferredSide: PreferredSide;
  bundle: boolean;
  parallelGap: number;
}>;

/** One rounded cubic Bézier segment of a resolved route. */
export type CubicSegment = Readonly<{
  start: Point2;
  control1: Point2;
  control2: Point2;
  end: Point2;
}>;

/**
 * A previously resolved sibling route, supplied as an immutable input so later
 * routes can separate into lanes, avoid crossings, or share bundle corridors
 * without mutating earlier results.
 */
export type RoutedSibling = Readonly<{
  id: string;
  sourceId?: string | undefined;
  targetId?: string | undefined;
  fromAnchor?: string | undefined;
  toAnchor?: string | undefined;
  waypoints: readonly Point2[];
  segments: readonly CubicSegment[];
}>;

/** Complete input required to resolve a single curved edge. */
export type CurveRouteInput = Readonly<{
  edgeId: string;
  start: Point2;
  end: Point2;
  fromAnchor?: string | undefined;
  toAnchor?: string | undefined;
  sourceId?: string | undefined;
  targetId?: string | undefined;
  sourceBounds?: Bounds2 | undefined;
  targetBounds?: Bounds2 | undefined;
  obstacles: readonly RouteObstacle[];
  siblings: readonly RoutedSibling[];
  options: CurveRouteOptions;
  /**
   * Signed lateral lane displacement in scene units, applied perpendicular to
   * the route so parallel edges between the same anchors separate into lanes.
   * Zero (the default) keeps the base route; the scene layer assigns symmetric
   * values, and bundling collapses them back to zero.
   */
  laneOffset?: number | undefined;
}>;

/** Resolved route geometry plus metadata needed by renderers and verifiers. */
export type CurveRouteResult = Readonly<{
  d: string;
  waypoints: readonly Point2[];
  segments: readonly CubicSegment[];
  bounds: Bounds2;
  usedFallback: boolean;
  penetratedObstacleIds: readonly string[];
}>;

/** Default routing options; every field matches the approved design spec. */
export const DEFAULT_CURVE_ROUTE_OPTIONS: CurveRouteOptions = Object.freeze({
  clearance: 12,
  curvature: 0.45,
  avoidObstacles: true,
  preferredSide: "auto",
  bundle: false,
  parallelGap: 8,
});
