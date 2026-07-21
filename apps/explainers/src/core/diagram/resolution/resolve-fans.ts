/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure canonical fan-out/fan-in topology resolution.
//!
//! Mirrors `resolve-connectors.ts`'s shape for a fan's trunk/branch geometry:
//! one authored `core.fan-out`/`core.fan-in` (or `kind:"fan"`) node resolves
//! to one junction plus N branch trajectories, all in final world-space scene
//! bounds. SceneRenderer and the flow verifier both consume this output
//! instead of reconstructing fan geometry independently.

import { capabilityOf, isFanNode } from "../node-classification.js";
import type {
  SceneGeometryLike,
  SceneNodeLike,
  ScenePointLike,
} from "../scene-types.js";
import type {
  FanSegment,
  FanTrajectory,
  ResolvedFanGeometry,
  ResolvedPoint,
} from "./types.js";

/** Inputs required to resolve every fan in one canonical scene. */
export type ResolveFansInput = Readonly<{
  nodesById: ReadonlyMap<string, SceneNodeLike>;
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>;
}>;

function geometryOf(node: SceneNodeLike): SceneGeometryLike {
  const geometry = node.geometry ?? node.layout;
  return {
    x: typeof geometry?.x === "number" ? geometry.x : 0,
    y: typeof geometry?.y === "number" ? geometry.y : 0,
    width: typeof geometry?.width === "number" ? geometry.width : 0,
    height: typeof geometry?.height === "number" ? geometry.height : 0,
  };
}

function nodeCenter(geometry: SceneGeometryLike): ResolvedPoint {
  return {
    x: geometry.x + geometry.width / 2,
    y: geometry.y + geometry.height / 2,
  };
}

function nodeAnchorPoint(
  geometry: SceneGeometryLike,
  anchor: string | undefined,
): ResolvedPoint {
  const center = nodeCenter(geometry);
  const left = geometry.x;
  const right = geometry.x + geometry.width;
  const top = geometry.y;
  const bottom = geometry.y + geometry.height;
  switch ((anchor ?? "center").toLowerCase()) {
    case "left":
    case "west":
    case "w":
      return { x: left, y: center.y };
    case "right":
    case "east":
    case "e":
      return { x: right, y: center.y };
    case "top":
    case "north":
    case "n":
      return { x: center.x, y: top };
    case "bottom":
    case "south":
    case "s":
      return { x: center.x, y: bottom };
    case "ne":
      return { x: right, y: top };
    case "nw":
      return { x: left, y: top };
    case "se":
      return { x: right, y: bottom };
    case "sw":
      return { x: left, y: bottom };
    default:
      return center;
  }
}

/** Soft / missing anchors that should be upgraded to facing edges for a fan. */
function isSoftFanAnchor(anchor: string | undefined): boolean {
  if (anchor === undefined || anchor.length === 0) {
    return true;
  }
  const token = anchor.toLowerCase();
  return token === "center" || token === "middle" || token === "c";
}

function facingAnchorToPoint(
  geometry: SceneGeometryLike,
  peer: ResolvedPoint,
): "e" | "w" | "n" | "s" {
  const center = nodeCenter(geometry);
  const dx = peer.x - center.x;
  const dy = peer.y - center.y;
  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0 ? "e" : "w";
  }
  return dy >= 0 ? "s" : "n";
}

/** Resolve a fan endpoint in world space, upgrading soft anchors to facing edges. */
function resolveFanEndpoint(
  endpoint: ScenePointLike,
  peer: ResolvedPoint,
  input: ResolveFansInput,
): ResolvedPoint {
  const hasX = typeof endpoint.x === "number" && Number.isFinite(endpoint.x);
  const hasY = typeof endpoint.y === "number" && Number.isFinite(endpoint.y);
  if (hasX && hasY) {
    return { x: endpoint.x as number, y: endpoint.y as number };
  }
  if (typeof endpoint.nodeId !== "string" || endpoint.nodeId.length === 0) {
    return { x: 0, y: 0 };
  }
  const world =
    input.worldGeometryById.get(endpoint.nodeId) ??
    (input.nodesById.has(endpoint.nodeId)
      ? geometryOf(input.nodesById.get(endpoint.nodeId)!)
      : undefined);
  if (world === undefined) {
    return { x: 0, y: 0 };
  }
  const anchor = isSoftFanAnchor(endpoint.anchor)
    ? facingAnchorToPoint(world, peer)
    : endpoint.anchor;
  return nodeAnchorPoint(world, anchor);
}

function isScenePointArray(
  value: ScenePointLike | readonly ScenePointLike[] | undefined,
): value is readonly ScenePointLike[] {
  return Array.isArray(value);
}

function singleScenePoint(
  value: ScenePointLike | readonly ScenePointLike[] | undefined,
): ScenePointLike | undefined {
  return isScenePointArray(value) ? undefined : value;
}

function scenePoints(
  value: ScenePointLike | readonly ScenePointLike[] | undefined,
): readonly ScenePointLike[] {
  if (value === undefined) {
    return [];
  }
  return isScenePointArray(value) ? value : [value];
}

function pointCentroid(points: readonly ResolvedPoint[]): ResolvedPoint {
  if (points.length === 0) {
    return { x: 0, y: 0 };
  }
  const sum = points.reduce(
    (total, point) => ({ x: total.x + point.x, y: total.y + point.y }),
    { x: 0, y: 0 },
  );
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function connectorAxisOf(node: SceneNodeLike): "x" | "y" | undefined {
  if (node.axis === "x" || node.axis === "y") {
    return node.axis;
  }
  const styled = node.style?.axis;
  if (styled === "x" || styled === "y") {
    return styled;
  }
  return undefined;
}

function formatNumber(value: number): string {
  const rounded = Math.round(value * 1000) / 1000;
  return String(rounded === 0 ? 0 : rounded);
}

function fanPath(points: readonly ResolvedPoint[]): string {
  const compact: ResolvedPoint[] = [];
  for (const point of points) {
    const previous = compact.at(-1);
    if (
      previous === undefined ||
      Math.abs(previous.x - point.x) > 0.001 ||
      Math.abs(previous.y - point.y) > 0.001
    ) {
      compact.push(point);
    }
  }
  const first = compact[0] ?? { x: 0, y: 0 };
  return compact
    .slice(1)
    .reduce(
      (d, point) => `${d} L${formatNumber(point.x)} ${formatNumber(point.y)}`,
      `M${formatNumber(first.x)} ${formatNumber(first.y)}`,
    );
}

function fanBranchPoints(
  start: ResolvedPoint,
  junction: ResolvedPoint,
  axis: "x" | "y",
  incoming: boolean,
): readonly ResolvedPoint[] {
  if (axis === "x") {
    return incoming
      ? [start, { x: junction.x, y: start.y }, junction]
      : [junction, { x: junction.x, y: start.y }, start];
  }
  return incoming
    ? [start, { x: start.x, y: junction.y }, junction]
    : [junction, { x: start.x, y: junction.y }, start];
}

function orthogonalFanPoints(
  start: ResolvedPoint,
  end: ResolvedPoint,
  axis: "x" | "y",
): readonly ResolvedPoint[] {
  return axis === "x"
    ? [start, { x: end.x, y: start.y }, end]
    : [start, { x: start.x, y: end.y }, end];
}

function automaticFanJunction(
  singleton: ResolvedPoint,
  many: readonly ResolvedPoint[],
  axis: "x" | "y",
): ResolvedPoint {
  if (many.length === 0) {
    // No "many"-side endpoints to route through — Math.min/max(...[]) would
    // collapse to +/-Infinity. Defensive fallback; callers guard this case
    // before routing a fan at all (see resolveFanGeometryForNode).
    return singleton;
  }
  const centroid = pointCentroid(many);
  if (axis === "x") {
    const towardPositive = centroid.x >= singleton.x;
    const corridorEdge = towardPositive
      ? Math.min(...many.map((point) => point.x))
      : Math.max(...many.map((point) => point.x));
    return { x: (singleton.x + corridorEdge) / 2, y: singleton.y };
  }
  const towardPositive = centroid.y >= singleton.y;
  const corridorEdge = towardPositive
    ? Math.min(...many.map((point) => point.y))
    : Math.max(...many.map((point) => point.y));
  return { x: singleton.x, y: (singleton.y + corridorEdge) / 2 };
}

function pointsNear(a: ResolvedPoint, b: ResolvedPoint, eps = 0.001): boolean {
  return Math.abs(a.x - b.x) <= eps && Math.abs(a.y - b.y) <= eps;
}

/** Collapse a polyline into atomic horizontal / vertical spans. */
function atomicOrthogonalSpans(
  points: readonly ResolvedPoint[],
): readonly Readonly<{ start: ResolvedPoint; end: ResolvedPoint }>[] {
  const compact: ResolvedPoint[] = [];
  for (const point of points) {
    const previous = compact.at(-1);
    if (previous === undefined || !pointsNear(previous, point)) {
      compact.push(point);
    }
  }
  const spans: Array<Readonly<{ start: ResolvedPoint; end: ResolvedPoint }>> = [];
  for (let i = 0; i < compact.length - 1; i++) {
    const start = compact[i]!;
    const end = compact[i + 1]!;
    const horizontal = Math.abs(start.y - end.y) <= 0.001;
    const vertical = Math.abs(start.x - end.x) <= 0.001;
    if (!horizontal && !vertical) {
      // Keep authored orthogonal intent: reject diagonals from paint.
      continue;
    }
    if (pointsNear(start, end)) {
      continue;
    }
    spans.push({ start, end });
  }
  return spans;
}

export type FanAtomicSpan = Readonly<{
  axis: "h" | "v";
  fixed: number;
  from: number;
  to: number;
  role: "trunk" | "branch" | "merge-trunk";
  destination?: ResolvedPoint;
}>;

function toFanAtomicSpan(
  start: ResolvedPoint,
  end: ResolvedPoint,
  role: "trunk" | "branch" | "merge-trunk",
  destination: ResolvedPoint | undefined,
): FanAtomicSpan | undefined {
  const horizontal = Math.abs(start.y - end.y) <= 0.001;
  const vertical = Math.abs(start.x - end.x) <= 0.001;
  if (horizontal === vertical) {
    return undefined;
  }
  if (horizontal) {
    return {
      axis: "h",
      fixed: start.y,
      from: Math.min(start.x, end.x),
      to: Math.max(start.x, end.x),
      role,
      ...(destination !== undefined ? { destination } : {}),
    };
  }
  return {
    axis: "v",
    fixed: start.x,
    from: Math.min(start.y, end.y),
    to: Math.max(start.y, end.y),
    role,
    ...(destination !== undefined ? { destination } : {}),
  };
}

function mergeCollinearFanSpans(
  spans: readonly FanAtomicSpan[],
): readonly FanAtomicSpan[] {
  type Bucket = {
    axis: "h" | "v";
    fixed: number;
    role: "trunk" | "branch" | "merge-trunk";
    intervals: Array<{ from: number; to: number }>;
  };
  const buckets = new Map<string, Bucket>();
  for (const span of spans) {
    const key = `${span.axis}:${formatNumber(span.fixed)}`;
    const existing = buckets.get(key);
    if (existing === undefined) {
      buckets.set(key, {
        axis: span.axis,
        fixed: span.fixed,
        role: span.role,
        intervals: [{ from: span.from, to: span.to }],
      });
      continue;
    }
    existing.intervals.push({ from: span.from, to: span.to });
    if (span.role === "trunk" || span.role === "merge-trunk") {
      existing.role = span.role;
    }
  }

  const merged: FanAtomicSpan[] = [];
  for (const bucket of buckets.values()) {
    const sorted = bucket.intervals
      .slice()
      .sort((left, right) => left.from - right.from || left.to - right.to);
    const collapsed: Array<{ from: number; to: number }> = [];
    for (const interval of sorted) {
      const last = collapsed.at(-1);
      if (last === undefined || interval.from > last.to + 0.001) {
        collapsed.push({ from: interval.from, to: interval.to });
      } else {
        last.to = Math.max(last.to, interval.to);
      }
    }
    for (const interval of collapsed) {
      if (interval.to - interval.from <= 0.001) {
        continue;
      }
      merged.push({
        axis: bucket.axis,
        fixed: bucket.fixed,
        from: interval.from,
        to: interval.to,
        role: bucket.role,
      });
    }
  }
  return merged;
}

function subtractInterval(
  host: Readonly<{ from: number; to: number }>,
  cut: Readonly<{ from: number; to: number }>,
): readonly Readonly<{ from: number; to: number }>[] {
  if (cut.to <= host.from + 0.001 || cut.from >= host.to - 0.001) {
    return [host];
  }
  const parts: Array<{ from: number; to: number }> = [];
  if (cut.from > host.from + 0.001) {
    parts.push({ from: host.from, to: Math.min(host.to, cut.from) });
  }
  if (cut.to < host.to - 0.001) {
    parts.push({ from: Math.max(host.from, cut.to), to: host.to });
  }
  return parts.filter((part) => part.to - part.from > 0.001);
}

/** Build one painted fan segment from an atomic H/V span (exported for unit tests). */
export function fanSegmentFromAtomic(id: string, span: FanAtomicSpan): FanSegment {
  const start =
    span.axis === "h"
      ? { x: span.from, y: span.fixed }
      : { x: span.fixed, y: span.from };
  const end =
    span.axis === "h"
      ? { x: span.to, y: span.fixed }
      : { x: span.fixed, y: span.to };
  const destination = span.destination;
  let directed = { start, end };
  if (destination !== undefined) {
    if (pointsNear(end, destination)) {
      directed = { start, end };
    } else if (pointsNear(start, destination)) {
      directed = { start: end, end: start };
    } else {
      // Neither endpoint sits on the destination within epsilon (a merged
      // corridor whose destination is inset from both span ends) — orient
      // toward whichever end is actually closer so the marker still points
      // at the destination instead of always defaulting to `end`.
      const distanceToEnd = Math.hypot(
        end.x - destination.x,
        end.y - destination.y,
      );
      const distanceToStart = Math.hypot(
        start.x - destination.x,
        start.y - destination.y,
      );
      directed =
        distanceToStart < distanceToEnd
          ? { start: end, end: start }
          : { start, end };
    }
  }
  return {
    id,
    d: fanPath([directed.start, directed.end]),
    directed: true,
    showMarker: destination !== undefined,
    role: span.role,
  };
}

/**
 * Paint spans are normalized atomic H/V segments with collinear overlaps
 * merged so shared corridors stroke once. Destination markers stay only on
 * terminal segments that end at a semantic destination.
 */
function paintFanSegments(
  nodeId: string,
  trunkPoints: readonly ResolvedPoint[],
  trunkRole: "trunk" | "merge-trunk",
  branchPointSets: readonly (readonly ResolvedPoint[])[],
  destinations: readonly ResolvedPoint[],
): readonly FanSegment[] {
  const corridors: FanAtomicSpan[] = [];
  const terminals: FanAtomicSpan[] = [];
  const pushPoints = (
    points: readonly ResolvedPoint[],
    role: "trunk" | "branch" | "merge-trunk",
    destination: ResolvedPoint | undefined,
  ) => {
    const spans = atomicOrthogonalSpans(points);
    spans.forEach((span, index) => {
      const isLast = index === spans.length - 1;
      const atomic = toFanAtomicSpan(
        span.start,
        span.end,
        role,
        isLast ? destination : undefined,
      );
      if (atomic === undefined) {
        return;
      }
      if (atomic.destination !== undefined) {
        terminals.push(atomic);
      } else {
        corridors.push(atomic);
      }
    });
  };

  pushPoints(
    trunkPoints,
    trunkRole,
    trunkRole === "merge-trunk" ? destinations[0] : undefined,
  );
  branchPointSets.forEach((points, branchIndex) => {
    pushPoints(
      points,
      "branch",
      trunkRole === "trunk" ? destinations[branchIndex] : undefined,
    );
  });

  // Merge shared corridors, then subtract terminal stubs so destinations
  // keep a dedicated marked segment and are never double-stroked.
  let mergedCorridors = mergeCollinearFanSpans(corridors);
  const uniqueTerminals: FanAtomicSpan[] = [];
  const terminalKeys = new Set<string>();
  for (const terminal of terminals) {
    const key = `${terminal.axis}:${formatNumber(terminal.fixed)}:${formatNumber(terminal.from)}:${formatNumber(terminal.to)}`;
    if (terminalKeys.has(key)) {
      continue;
    }
    terminalKeys.add(key);
    uniqueTerminals.push(terminal);
  }
  for (const terminal of uniqueTerminals) {
    mergedCorridors = mergedCorridors.flatMap((corridor) => {
      if (
        corridor.axis !== terminal.axis ||
        Math.abs(corridor.fixed - terminal.fixed) > 0.001
      ) {
        return [corridor];
      }
      return subtractInterval(corridor, terminal).map((part) => ({
        ...corridor,
        from: part.from,
        to: part.to,
      }));
    });
  }

  const painted = [...mergedCorridors, ...uniqueTerminals];
  return painted.map((span, index) =>
    fanSegmentFromAtomic(`${nodeId}-span-${index}`, span),
  );
}

/**
 * Resolve one fan node's trunk/branch topology from final world-space scene
 * bounds. Returns `undefined` when the fan lacks the two-or-more "many"-side
 * endpoints a junction requires — `automaticFanJunction`'s Math.min/max over
 * an empty array would otherwise collapse to +/-Infinity.
 */
function resolveFanGeometryForNode(
  node: SceneNodeLike,
  input: ResolveFansInput,
): ResolvedFanGeometry | undefined {
  const capability =
    capabilityOf(node) === "core.fan-in" ? "core.fan-in" : "core.fan-out";
  const fanOut = capability === "core.fan-out";
  const from = scenePoints(node.from);
  const to = scenePoints(node.to);
  const singletonEndpoint = (fanOut ? from[0] : to[0]) ?? {};
  const manyEndpoints = fanOut ? to : from;
  if (manyEndpoints.length < 2) {
    return undefined;
  }

  const roughSingleton = resolveFanEndpoint(singletonEndpoint, { x: 0, y: 0 }, input);
  const roughMany = manyEndpoints.map((endpoint) =>
    resolveFanEndpoint(endpoint, roughSingleton, input),
  );
  const roughManyCentroid = pointCentroid(roughMany);
  const singleton = resolveFanEndpoint(singletonEndpoint, roughManyCentroid, input);
  const many = manyEndpoints.map((endpoint) =>
    resolveFanEndpoint(endpoint, singleton, input),
  );
  const manyCentroid = pointCentroid(many);
  const axis =
    connectorAxisOf(node) ??
    (Math.abs(manyCentroid.x - singleton.x) >=
    Math.abs(manyCentroid.y - singleton.y)
      ? "x"
      : "y");
  const authoredJunction = singleScenePoint(node.junction);
  const junction =
    authoredJunction === undefined
      ? automaticFanJunction(singleton, many, axis)
      : resolveFanEndpoint(
          authoredJunction,
          fanOut ? manyCentroid : singleton,
          input,
        );

  const trunkPoints = fanOut
    ? orthogonalFanPoints(singleton, junction, axis)
    : orthogonalFanPoints(junction, singleton, axis);
  const trunkRole = fanOut ? "trunk" : "merge-trunk";
  const branchPointSets = many.map((endpoint) =>
    fanBranchPoints(endpoint, junction, axis, !fanOut),
  );
  const destinations = fanOut ? many : [singleton];
  const trajectories = many.map((_endpoint, branchIndex): FanTrajectory => {
    const branchPoints = branchPointSets[branchIndex]!;
    // Fan-in must keep the orthogonal merge-trunk (not a diagonal jump).
    const points = fanOut
      ? [...trunkPoints, ...branchPoints.slice(1)]
      : [...branchPoints, ...trunkPoints.slice(1)];
    return {
      id: `${node.id}-trajectory-${branchIndex}`,
      d: fanPath(points),
      role: fanOut ? "branch" : "merge-trunk",
    };
  });
  return {
    id: node.id,
    capability,
    segments: paintFanSegments(
      node.id,
      trunkPoints,
      trunkRole,
      branchPointSets,
      destinations,
    ),
    junction,
    trajectories,
  };
}

/** Resolve every fan-out/fan-in node once from final world-space scene bounds. */
export function resolveFans(
  input: ResolveFansInput,
): ReadonlyMap<string, ResolvedFanGeometry> {
  const fanGeometryById = new Map<string, ResolvedFanGeometry>();
  for (const node of input.nodesById.values()) {
    const capability = capabilityOf(node);
    if (!isFanNode(node, capability)) {
      continue;
    }
    const geometry = resolveFanGeometryForNode(node, input);
    if (geometry !== undefined) {
      fanGeometryById.set(node.id, geometry);
    }
  }
  return fanGeometryById;
}
