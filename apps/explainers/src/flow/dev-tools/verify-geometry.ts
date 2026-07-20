/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Browser-safe geometry helpers mirrored from the Node flow verifier.

import type {
  ConnectorEndpointIr,
  RenderNodeIr,
  SceneIr,
  TimelineCueIr,
} from "../schema/index.js";
import {
  elbowPathData,
  isCurveRoute,
  isElbowRoute,
  normalizeCurveRouteOptions,
  routeCurve,
  type CurveRouteResult,
  type RouteObstacle,
} from "../../core/diagram/connector-routing.js";
export { resolveSceneWorldGeometry as indexResolvedWorldGeometry } from "../../core/diagram/capabilities/resolved-geometry.js";

/** Default SceneRenderer viewport. */
export const DEFAULT_VIEWPORT = Object.freeze({
  width: 700,
  height: 400,
  margin: 24,
});

/** Snap distance (px) for connector endpoint / dot proximity. */
export const SNAP_PX = 36;

export type Point = Readonly<{ x: number; y: number }>;
export type Geometry = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
}>;
export type Viewport = Readonly<{
  width: number;
  height: number;
  margin: number;
}>;
export type FanGeometry = Readonly<{
  axis: "x" | "y";
  junction: Point;
  trunk: readonly Point[];
  branches: readonly (readonly Point[])[];
  trajectories: readonly (readonly Point[])[];
}>;

type UnknownRecord = Readonly<Record<string, unknown>>;
type NodesById = ReadonlyMap<string, RenderNodeIr>;

const ARROW_CAPS = new Set([
  "core.line",
  "core.path",
  "core.arrow",
  "core.connector",
  "core.elbow",
  "core.route",
  "core.fan-out",
  "core.fan-in",
]);
const ARROW_KINDS = new Set([
  "line",
  "path",
  "arrow",
  "connector",
  "elbow",
  "fan",
]);
const DOT_CAPS = new Set(["core.dot", "core.circle"]);
const DOT_KINDS = new Set(["dot", "circle"]);
const BOX_CAPS = new Set(["core.rect", "core.text"]);
const MOTION_CAPS = new Set([
  "motion.signal",
  "motion.dot",
  "core.motion",
  "motion.motion-signal",
]);

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

function styleOf(node: unknown): UnknownRecord {
  return record(record(node).style);
}

/** Returns a node's canonical or authoring-alias capability. */
export function capabilityOf(node: unknown): string {
  const value = record(node);
  return String(value.capabilityId ?? value.capability ?? "");
}

/** Returns a node's structural kind. */
export function kindOf(node: unknown): string {
  return String(record(node).kind ?? "");
}

/** Whether a node describes a connector or path. */
export function isArrowLike(node: unknown): boolean {
  return ARROW_CAPS.has(capabilityOf(node)) || ARROW_KINDS.has(kindOf(node));
}

/** Whether a node describes fan-in or fan-out topology. */
export function isFanNode(node: unknown): boolean {
  const capability = capabilityOf(node);
  return (
    capability === "core.fan-out" ||
    capability === "core.fan-in" ||
    kindOf(node) === "fan"
  );
}

/**
 * Mirror SceneRenderer motion-signal classification so guide strokes are not
 * treated as orphan connectors. Dots are never motion guides.
 */
export function isMotionSignalNode(node: unknown): boolean {
  if (isDotLike(node)) return false;
  const capability = capabilityOf(node);
  if (MOTION_CAPS.has(capability)) return true;
  const value = record(node);
  const id = String(value.id ?? "");
  if (/motion[-_]?sig/i.test(id) || /^motion\d+$/i.test(id)) return true;
  if (/motion/i.test(id) && isArrowLike(node)) return true;
  const label = String(record(value.accessibility).label ?? "").toLowerCase();
  if (label.includes("motion signal")) return true;
  const style = styleOf(node);
  const motion = style.motion;
  const role = style.role;
  return (
    motion === true ||
    motion === 1 ||
    motion === "signal" ||
    motion === "dot" ||
    role === "motion" ||
    role === "motion-signal"
  );
}

/** True when the author disabled arrowheads (undirected divider / guide). */
export function markerEndDisabled(node: unknown): boolean {
  const markerEnd = styleOf(node).markerEnd;
  if (markerEnd === undefined || markerEnd === null) return false;
  if (markerEnd === false || markerEnd === 0) return true;
  if (typeof markerEnd === "string") {
    const token = markerEnd.trim().toLowerCase();
    return token === "none" || token === "false" || token === "0";
  }
  const kind = record(markerEnd).kind;
  if (typeof kind === "string") {
    const token = kind.trim().toLowerCase();
    return token === "none" || token === "false";
  }
  return false;
}

/** Directed connectors that should snap to boxes (excludes motion + headless). */
export function isDirectedConnector(node: unknown): boolean {
  if (!isArrowLike(node) || isMotionSignalNode(node) || markerEndDisabled(node)) {
    return false;
  }
  const arrowhead = styleOf(node).arrowhead;
  if (arrowhead === false || arrowhead === 0 || arrowhead === "false") {
    return false;
  }
  const id = String(record(node).id ?? "").toLowerCase();
  if (/^(split|divider|rule|sep|guide)([-_]|$)/.test(id)) return false;
  return capabilityOf(node) !== "core.bracket";
}

/** Whether a node describes a dot or small circle. */
export function isDotLike(node: unknown): boolean {
  if (DOT_CAPS.has(capabilityOf(node)) || DOT_KINDS.has(kindOf(node))) {
    return true;
  }
  const radius = styleOf(node).r;
  return typeof radius === "number" && radius > 0 && radius <= 12;
}

/** Whether a dot is an obsolete companion visual for a motion signal. */
export function isMotionCompanionDot(node: unknown): boolean {
  if (!isDotLike(node)) return false;
  const role = String(styleOf(node).role ?? "").toLowerCase();
  if (role === "motion-signal" || role === "motion-dot") return true;
  const id = String(record(node).id ?? "");
  return /motion[-_]?sig/i.test(id) && /-dot$/i.test(id);
}

/** Id of the motion path a companion dot is paired with (`…-dot` → stem). */
export function motionCompanionPathId(dotId: unknown): string | null {
  const id = String(dotId ?? "");
  const match = /^(.*)-dot$/i.exec(id);
  if (!match || !/motion[-_]?sig/i.test(match[1])) return null;
  return match[1];
}

/** Static legend chips are exempt from orphan-dot proximity checks. */
export function isLegendDot(node: unknown): boolean {
  if (!isDotLike(node) || isMotionCompanionDot(node)) return false;
  const style = styleOf(node);
  const role = String(style.role ?? "").toLowerCase();
  if (role === "legend" || role === "legend-chip" || style.legend === true) {
    return true;
  }
  const value = record(node);
  const id = String(value.id ?? "").toLowerCase();
  const label = String(record(value.accessibility).label ?? "").toLowerCase();
  return id.includes("legend") || label.includes("legend");
}

/** Whether a node contributes box geometry. */
export function isBoxLike(node: unknown): boolean {
  if (BOX_CAPS.has(capabilityOf(node))) return true;
  if (isArrowLike(node) || isDotLike(node)) return false;
  const value = record(node);
  return Boolean(value.geometry || value.layout);
}

/** Traverses scene roots in source order. */
export function walkNodes(roots: readonly RenderNodeIr[]): RenderNodeIr[] {
  const out: RenderNodeIr[] = [];
  const visit = (node: RenderNodeIr): void => {
    out.push(node);
    if (node.kind === "group" || node.kind === "component") {
      node.children.forEach(visit);
    }
  };
  roots.forEach(visit);
  return out;
}

/** Collects non-empty node ids under scene roots. */
export function nodeIds(roots: readonly RenderNodeIr[]): Set<string> {
  return new Set(walkNodes(roots).map(({ id }) => id).filter(Boolean));
}

/** Resolves finite geometry from either `geometry` or the legacy `layout`. */
export function geomOf(node: unknown): Geometry | null {
  const value = record(node);
  const geometry = record(value.geometry ?? value.layout);
  const x = Number(geometry.x);
  const y = Number(geometry.y);
  const width = Number(geometry.width);
  const height = Number(geometry.height);
  if (![x, y, width, height].every(Number.isFinite)) return null;
  return { x, y, width, height };
}

/** Returns the center of a box. */
export function boxCenter(geometry: Geometry): Point {
  return {
    x: geometry.x + geometry.width / 2,
    y: geometry.y + geometry.height / 2,
  };
}

/** Edge / corner / center point on a box (SceneRenderer anchor parity). */
export function nodeAnchorPoint(geometry: Geometry, anchor: unknown): Point {
  const center = boxCenter(geometry);
  const left = geometry.x;
  const right = geometry.x + geometry.width;
  const top = geometry.y;
  const bottom = geometry.y + geometry.height;
  switch (String(anchor ?? "center").toLowerCase()) {
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

/** Resolves an absolute or node-anchored connector endpoint. */
export function resolveEndpoint(
  endpoint: ConnectorEndpointIr | unknown,
  nodesById?: NodesById,
): Point | null {
  const value = record(endpoint);
  const nodeId = value.nodeId;
  if (typeof nodeId === "string" && nodeId.length > 0 && nodesById) {
    const target = nodesById.get(nodeId);
    const geometry = target ? geomOf(target) : null;
    if (geometry) return nodeAnchorPoint(geometry, value.anchor);
  }
  if (Number.isFinite(value.x) && Number.isFinite(value.y)) {
    return { x: value.x as number, y: value.y as number };
  }
  return null;
}

function isSoftAnchor(anchor: unknown): boolean {
  if (anchor === undefined || anchor === null || String(anchor).length === 0) {
    return true;
  }
  const token = String(anchor).toLowerCase();
  return token === "center" || token === "middle" || token === "c";
}

function facingAnchor(geometry: Geometry, peer: Point): "e" | "w" | "s" | "n" {
  const center = boxCenter(geometry);
  const dx = peer.x - center.x;
  const dy = peer.y - center.y;
  if (Math.abs(dx) >= Math.abs(dy)) return dx >= 0 ? "e" : "w";
  return dy >= 0 ? "s" : "n";
}

function resolveFanEndpoint(
  endpoint: unknown,
  peer: Point,
  nodesById?: NodesById,
): Point | null {
  const value = record(endpoint);
  if (Number.isFinite(value.x) && Number.isFinite(value.y)) {
    return { x: value.x as number, y: value.y as number };
  }
  if (typeof value.nodeId !== "string" || value.nodeId.length === 0) return null;
  const target = nodesById?.get(value.nodeId);
  const geometry = target ? geomOf(target) : null;
  if (!geometry) return null;
  const anchor = isSoftAnchor(value.anchor)
    ? facingAnchor(geometry, peer)
    : value.anchor;
  return nodeAnchorPoint(geometry, anchor);
}

function centroid(points: readonly Point[]): Point | null {
  if (points.length === 0) return null;
  const sum = points.reduce(
    (total, point) => ({ x: total.x + point.x, y: total.y + point.y }),
    { x: 0, y: 0 },
  );
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function compactPoints(points: readonly Point[]): Point[] {
  return points.filter((point, index) => {
    const previous = points[index - 1];
    return (
      previous === undefined ||
      Math.abs(previous.x - point.x) > 0.001 ||
      Math.abs(previous.y - point.y) > 0.001
    );
  });
}

function fanBranchPoints(
  endpoint: Point,
  junction: Point,
  axis: "x" | "y",
  incoming: boolean,
): Point[] {
  if (axis === "x") {
    return incoming
      ? [endpoint, { x: junction.x, y: endpoint.y }, junction]
      : [junction, { x: junction.x, y: endpoint.y }, endpoint];
  }
  return incoming
    ? [endpoint, { x: endpoint.x, y: junction.y }, junction]
    : [junction, { x: endpoint.x, y: junction.y }, endpoint];
}

function orthogonalFanPoints(
  start: Point,
  end: Point,
  axis: "x" | "y",
): Point[] {
  return axis === "x"
    ? [start, { x: end.x, y: start.y }, end]
    : [start, { x: start.x, y: end.y }, end];
}

function automaticFanJunction(
  singleton: Point,
  many: readonly Point[],
  axis: "x" | "y",
): Point | null {
  const manyCentroid = centroid(many);
  if (!manyCentroid) return null;
  if (axis === "x") {
    const towardPositive = manyCentroid.x >= singleton.x;
    const corridorEdge = towardPositive
      ? Math.min(...many.map(({ x }) => x))
      : Math.max(...many.map(({ x }) => x));
    return { x: (singleton.x + corridorEdge) / 2, y: singleton.y };
  }
  const towardPositive = manyCentroid.y >= singleton.y;
  const corridorEdge = towardPositive
    ? Math.min(...many.map(({ y }) => y))
    : Math.max(...many.map(({ y }) => y));
  return { x: singleton.x, y: (singleton.y + corridorEdge) / 2 };
}

/**
 * Mirrors SceneRenderer endpoint, junction, and trajectory resolution.
 * Returns null when malformed endpoints cannot produce connected finite paths.
 */
export function resolveFanGeometry(
  node: RenderNodeIr,
  nodesById: NodesById,
): FanGeometry | null {
  const value = record(node);
  const fanOut = capabilityOf(node) !== "core.fan-in";
  const from = Array.isArray(value.from) ? value.from : [value.from];
  const to = Array.isArray(value.to) ? value.to : [value.to];
  const singletonEndpoint = (fanOut ? from : to)[0];
  const manyEndpoints = fanOut ? to : from;
  if (!singletonEndpoint || manyEndpoints.some((endpoint) => !endpoint)) {
    return null;
  }

  const origin = { x: 0, y: 0 };
  const roughSingleton = resolveFanEndpoint(
    singletonEndpoint,
    origin,
    nodesById,
  );
  if (!roughSingleton) return null;
  const roughManyNullable = manyEndpoints.map((endpoint) =>
    resolveFanEndpoint(endpoint, roughSingleton, nodesById),
  );
  if (roughManyNullable.some((point) => point === null)) return null;
  const roughMany = roughManyNullable as Point[];
  const roughManyCentroid = centroid(roughMany);
  if (!roughManyCentroid) return null;
  const singleton = resolveFanEndpoint(
    singletonEndpoint,
    roughManyCentroid,
    nodesById,
  );
  if (!singleton) return null;
  const manyNullable = manyEndpoints.map((endpoint) =>
    resolveFanEndpoint(endpoint, singleton, nodesById),
  );
  if (manyNullable.some((point) => point === null)) return null;
  const many = manyNullable as Point[];
  const manyCentroid = centroid(many);
  if (!manyCentroid) return null;

  const styleAxis = styleOf(node).axis;
  const axis =
    value.axis === "x" || value.axis === "y"
      ? value.axis
      : styleAxis === "x" || styleAxis === "y"
        ? styleAxis
        : Math.abs(manyCentroid.x - singleton.x) >=
            Math.abs(manyCentroid.y - singleton.y)
          ? "x"
          : "y";
  const junction =
    value.junction === undefined
      ? automaticFanJunction(singleton, many, axis)
      : resolveFanEndpoint(
          value.junction,
          fanOut ? manyCentroid : singleton,
          nodesById,
        );
  if (
    !junction ||
    !Number.isFinite(junction.x) ||
    !Number.isFinite(junction.y)
  ) {
    return null;
  }

  const trunk = compactPoints(
    fanOut
      ? orthogonalFanPoints(singleton, junction, axis)
      : orthogonalFanPoints(junction, singleton, axis),
  );
  const branches = many.map((endpoint) =>
    compactPoints(fanBranchPoints(endpoint, junction, axis, !fanOut)),
  );
  const trajectories = branches.map((branch) =>
    compactPoints(
      fanOut
        ? [...trunk, ...branch.slice(1)]
        : [...branch, ...trunk.slice(1)],
    ),
  );
  return { axis, junction, trunk, branches, trajectories };
}

/** Euclidean distance between two points. */
export function dist(left: Point, right: Point): number {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

/** Whether a point lies within snap distance of a box. */
export function pointNearBox(
  point: Point,
  geometry: Geometry,
  snap = SNAP_PX,
): boolean {
  const x = Math.min(
    Math.max(point.x, geometry.x),
    geometry.x + geometry.width,
  );
  const y = Math.min(
    Math.max(point.y, geometry.y),
    geometry.y + geometry.height,
  );
  return dist(point, { x, y }) <= snap;
}

/**
 * Parses SVG path commands into polyline points. M/L/H/V are exact; endpoint
 * positions are retained for C/S/Q/T/A commands.
 */
export function pathPoints(pathData: unknown): Point[] {
  if (typeof pathData !== "string" || pathData.trim() === "") return [];
  const tokens = pathData.match(
    /[MLHVCSQTAZmlhvcsqtaz]|-?\d*\.?\d+(?:e[-+]?\d+)?/gi,
  );
  if (!tokens) return [];
  const points: Point[] = [];
  let index = 0;
  let x = 0;
  let y = 0;
  let command = "M";
  const number = (): number => Number(tokens[index++]);
  const push = (nextX: number, nextY: number): void => {
    if (Number.isFinite(nextX) && Number.isFinite(nextY)) {
      points.push({ x: nextX, y: nextY });
    }
  };
  while (index < tokens.length) {
    const token = tokens[index];
    if (/^[MLHVCSQTAZmlhvcsqtaz]$/.test(token)) {
      command = token;
      index += 1;
      if (command === "Z" || command === "z") continue;
    }
    if (command === "M" || command === "L") {
      x = number();
      y = number();
      push(x, y);
      command = command === "M" ? "L" : command;
    } else if (command === "m" || command === "l") {
      x += number();
      y += number();
      push(x, y);
      command = command === "m" ? "l" : command;
    } else if (command === "H") {
      x = number();
      push(x, y);
    } else if (command === "h") {
      x += number();
      push(x, y);
    } else if (command === "V") {
      y = number();
      push(x, y);
    } else if (command === "v") {
      y += number();
      push(x, y);
    } else if (command === "C") {
      number();
      number();
      number();
      number();
      x = number();
      y = number();
      push(x, y);
    } else if (command === "c") {
      number();
      number();
      number();
      number();
      x += number();
      y += number();
      push(x, y);
    } else if (command === "S" || command === "Q") {
      number();
      number();
      x = number();
      y = number();
      push(x, y);
    } else if (command === "s" || command === "q") {
      number();
      number();
      x += number();
      y += number();
      push(x, y);
    } else if (command === "T") {
      x = number();
      y = number();
      push(x, y);
    } else if (command === "t") {
      x += number();
      y += number();
      push(x, y);
    } else if (command === "A") {
      number();
      number();
      number();
      number();
      number();
      x = number();
      y = number();
      push(x, y);
    } else if (command === "a") {
      number();
      number();
      number();
      number();
      number();
      x += number();
      y += number();
      push(x, y);
    } else {
      index += 1;
    }
  }
  return points;
}

function pathDataFromPoints(
  points: unknown,
  nodesById?: NodesById,
): string | null {
  if (!Array.isArray(points) || points.length === 0) return null;
  const resolved = points.map((point) => resolveEndpoint(point, nodesById));
  if (resolved.some((point) => point === null)) return null;
  const finite = resolved as Point[];
  return finite
    .slice(1)
    .reduce(
      (path, point) => `${path} L${point.x} ${point.y}`,
      `M${finite[0].x} ${finite[0].y}`,
    );
}

function endpointAnchor(endpoint: unknown): string | undefined {
  const value = record(endpoint);
  return typeof value.anchor === "string" && value.anchor.length > 0
    ? value.anchor
    : undefined;
}

/** Route-metadata finding raised when a curved edge misbehaves. */
export type CurveRouteFinding = Readonly<{
  severity: "error" | "warn";
  code: "CURVE_OBSTACLE_PENETRATION" | "CURVE_FALLBACK";
  edgeId: string;
  obstacleIds: readonly string[];
}>;

/**
 * Classify a resolved route: an error when it pierces an obstacle without
 * declaring a fallback, and a warning whenever it degrades to a deterministic
 * fallback route.
 */
export function verifyCurveRouteResult(
  edgeId: string,
  result: CurveRouteResult,
  _obstacles: readonly RouteObstacle[],
): readonly CurveRouteFinding[] {
  const findings: CurveRouteFinding[] = [];
  if (result.penetratedObstacleIds.length > 0 && !result.usedFallback) {
    findings.push({
      severity: "error",
      code: "CURVE_OBSTACLE_PENETRATION",
      edgeId,
      obstacleIds: result.penetratedObstacleIds,
    });
  }
  if (result.usedFallback) {
    findings.push({
      severity: "warn",
      code: "CURVE_FALLBACK",
      edgeId,
      obstacleIds: result.penetratedObstacleIds,
    });
  }
  return findings;
}

/** Resolves authored connector forms into SVG path data. */
export function arrowPathData(
  node: RenderNodeIr,
  nodesById?: NodesById,
): string | null {
  const value = record(node);
  if (typeof value.d === "string" && value.d.trim()) return value.d;
  if (typeof value.path === "string" && value.path.trim()) return value.path;
  const pointsPath = pathDataFromPoints(value.points, nodesById);
  if (pointsPath) return pointsPath;
  const start = resolveEndpoint(value.from, nodesById);
  const end = resolveEndpoint(value.to, nodesById);
  if (start && end) {
    if (isCurveRoute(node)) {
      const fromValue = record(value.from);
      const toValue = record(value.to);
      return routeCurve({
        edgeId: typeof value.id === "string" ? value.id : "",
        start,
        end,
        fromAnchor: endpointAnchor(value.from),
        toAnchor: endpointAnchor(value.to),
        sourceId: typeof fromValue.nodeId === "string" ? fromValue.nodeId : undefined,
        targetId: typeof toValue.nodeId === "string" ? toValue.nodeId : undefined,
        obstacles: [],
        siblings: [],
        options: normalizeCurveRouteOptions(styleOf(node)),
      }).d;
    }
    if (isElbowRoute(node)) {
      const via = resolveEndpoint(value.via, nodesById);
      const style = styleOf(node);
      const axis =
        value.axis === "x" || value.axis === "y"
          ? value.axis
          : style.axis === "x" || style.axis === "y"
            ? style.axis
            : undefined;
      const options = normalizeCurveRouteOptions(style);
      const fromValue = record(value.from);
      const toValue = record(value.to);
      const sourceId =
        typeof fromValue.nodeId === "string" ? fromValue.nodeId : undefined;
      const targetId =
        typeof toValue.nodeId === "string" ? toValue.nodeId : undefined;
      const obstacles =
        options.avoidObstacles && nodesById !== undefined
          ? [...nodesById.entries()].flatMap(([id, candidate]) => {
              if (id === sourceId || id === targetId || !isBoxLike(candidate)) {
                return [];
              }
              const geometry = geomOf(candidate);
              return geometry !== null &&
                geometry.width > 0 &&
                geometry.height > 0
                ? [{ id, bounds: geometry }]
                : [];
            })
          : [];
      return elbowPathData(
        start,
        end,
        via ?? undefined,
        axis,
        endpointAnchor(value.from),
        endpointAnchor(value.to),
        obstacles,
        options.clearance,
      );
    }
    return `M${start.x} ${start.y} L${end.x} ${end.y}`;
  }
  return null;
}

/** Returns the latest cue end in milliseconds. */
export function timelineDurationMs(
  timeline: readonly TimelineCueIr[],
): number {
  let maximum = 0;
  for (const cue of timeline) {
    const at = Number(cue.at) || 0;
    const duration = Number(cue.duration) || 0;
    maximum = Math.max(maximum, at + duration);
  }
  return maximum;
}

/** Whether an action progressively draws a path. */
export function isDrawAction(action: unknown): boolean {
  const value = String(action ?? "").toLowerCase();
  return value === "draw" || value === "trace" || value === "reveal-stroke";
}

/** Returns a path's latest draw-cue progress at one timeline time. */
export function drawProgress(
  timeline: readonly TimelineCueIr[],
  nodeId: string,
  timeMs: number,
): number | undefined {
  const cue = timeline
    .filter(
      (candidate) =>
        candidate.target === nodeId && isDrawAction(candidate.action),
    )
    .at(-1);
  if (!cue) return undefined;
  const at = Number(cue.at) || 0;
  const duration = Number(cue.duration) || 0;
  if (timeMs <= at) return 0;
  if (duration <= 0) return 1;
  return Math.min(1, Math.max(0, (timeMs - at) / duration));
}

/** Resolves scene viewport dimensions with the renderer defaults. */
export function sceneViewport(
  scene: SceneIr,
  override?: Partial<Viewport>,
): Viewport {
  const authored = scene.viewport;
  const width =
    typeof authored?.width === "number" && Number.isFinite(authored.width)
      ? authored.width
      : DEFAULT_VIEWPORT.width;
  const height =
    typeof authored?.height === "number" && Number.isFinite(authored.height)
      ? authored.height
      : DEFAULT_VIEWPORT.height;
  return {
    width: override?.width ?? width,
    height: override?.height ?? height,
    margin:
      typeof override?.margin === "number"
        ? override.margin
        : DEFAULT_VIEWPORT.margin,
  };
}

/** Whether a box intersects the viewport plus its accepted margin. */
export function inViewport(
  geometry: Geometry,
  viewport: Viewport = DEFAULT_VIEWPORT,
): boolean {
  const { width, height, margin } = viewport;
  return (
    geometry.x + geometry.width >= -margin &&
    geometry.y + geometry.height >= -margin &&
    geometry.x <= width + margin &&
    geometry.y <= height + margin
  );
}
