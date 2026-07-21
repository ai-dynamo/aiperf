/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure canonical connector resolution and source-mapped geometry diagnostics.
//!
//! Fan-in/fan-out topology (`core.fan-*`, `kind:"fan"`) resolves separately in
//! `resolve-fans.ts` and is exposed alongside this module's connectors on
//! `ResolvedScene.fanGeometryById` (and `ResolvedSceneSnapshot.fans`): a fan
//! authors one node with N branch trajectories rather than N independent
//! connector nodes, so it needs its own junction/trajectory resolution instead
//! of `resolveConnectors`'s one-source/one-target shape. Both resolvers are
//! canonical — SceneRenderer and the flow verifier consume the same resolved
//! connector and fan geometry rather than recomputing either locally.

import {
  elbowPathData,
  isCurveRoute,
  isElbowRoute,
  normalizeCurveRouteOptions,
  routeCurve,
} from "../connector-routing.js";
import {
  inflateBounds,
  segmentIntersectsBounds,
} from "../connector-routing-geometry.js";
import type {
  RouteObstacle,
  RoutedSibling,
} from "../connector-routing-types.js";
import {
  capabilityOf,
  isMotionSignalNode,
} from "../node-classification.js";
import type {
  SceneGeometryLike,
  SceneNodeLike,
  ScenePointLike,
  SceneSourceRangeLike,
} from "../scene-types.js";
import type {
  ResolvedConnector,
  ResolvedPoint,
  SceneResolutionDiagnostic,
} from "./types.js";

const ENDPOINT_TOLERANCE = 5.4;
const DEFAULT_CLEARANCE = 32.4;
const UNKNOWN_RANGE: SceneSourceRangeLike = Object.freeze({
  source: "<scene>",
  start: Object.freeze({ offset: 0, line: 1, column: 1 }),
  end: Object.freeze({ offset: 0, line: 1, column: 1 }),
});

/** Maximum resolved height for annotation text excluded from routing obstacles. */
const ANNOTATION_TEXT_MAX_HEIGHT = 86.4;

/** Decorative capabilities that must not block automatic connector routing. */
const NON_ROUTING_CAPABILITIES = new Set([
  "core.band",
  "core.bracket",
  "core.divider",
  "core.legend",
]);

/** Inputs required to resolve all connectors in one canonical scene. */
export type ResolveConnectorsInput = Readonly<{
  nodesById: ReadonlyMap<string, SceneNodeLike>;
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>;
  ancestorIdsById: ReadonlyMap<string, readonly string[]>;
  generatedPartIds?: ReadonlySet<string>;
}>;

/** Canonical connector paths plus deterministic ordered diagnostics. */
export type ResolveConnectorsResult = Readonly<{
  connectorsById: ReadonlyMap<string, ResolvedConnector>;
  diagnostics: readonly SceneResolutionDiagnostic[];
}>;

type ParsedPath = Readonly<{
  start: ResolvedPoint;
  end: ResolvedPoint;
  vertices: readonly ResolvedPoint[];
}>;

type ConnectorCandidate = Readonly<{
  node: SceneNodeLike;
  source: ResolvedPoint;
  target: ResolvedPoint;
  from: ScenePointLike | undefined;
  to: ScenePointLike | undefined;
  sourceId?: string;
  targetId?: string;
  obstacles: readonly RouteObstacle[];
}>;

function finitePoint(point: ScenePointLike | undefined): ResolvedPoint | undefined {
  return typeof point?.x === "number" &&
    Number.isFinite(point.x) &&
    typeof point.y === "number" &&
    Number.isFinite(point.y)
    ? { x: point.x, y: point.y }
    : undefined;
}

function singlePoint(
  point: ScenePointLike | readonly ScenePointLike[] | undefined,
): ScenePointLike | undefined {
  if (point === undefined || Array.isArray(point)) {
    return undefined;
  }
  return point as ScenePointLike;
}

function anchorPoint(
  geometry: SceneGeometryLike,
  anchor: string | undefined,
): ResolvedPoint {
  const center = {
    x: geometry.x + geometry.width / 2,
    y: geometry.y + geometry.height / 2,
  };
  switch ((anchor ?? "center").toLowerCase()) {
    case "left":
    case "west":
    case "w":
      return { x: geometry.x, y: center.y };
    case "right":
    case "east":
    case "e":
      return { x: geometry.x + geometry.width, y: center.y };
    case "top":
    case "north":
    case "n":
      return { x: center.x, y: geometry.y };
    case "bottom":
    case "south":
    case "s":
      return { x: center.x, y: geometry.y + geometry.height };
    case "ne":
      return { x: geometry.x + geometry.width, y: geometry.y };
    case "nw":
      return { x: geometry.x, y: geometry.y };
    case "se":
      return {
        x: geometry.x + geometry.width,
        y: geometry.y + geometry.height,
      };
    case "sw":
      return { x: geometry.x, y: geometry.y + geometry.height };
    default:
      return center;
  }
}

function resolveEndpoint(
  endpoint: ScenePointLike | undefined,
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>,
  node: SceneNodeLike,
  diagnostics: SceneResolutionDiagnostic[],
): ResolvedPoint {
  const authored = finitePoint(endpoint);
  if (authored !== undefined) {
    return authored;
  }
  const nodeId = endpoint?.nodeId;
  if (typeof nodeId === "string" && nodeId.length > 0) {
    const geometry = worldGeometryById.get(nodeId);
    if (geometry !== undefined) {
      return anchorPoint(geometry, endpoint?.anchor);
    }
    diagnostics.push(
      makeDiagnostic(
        node,
        "SCENE_CONNECTOR_ENDPOINT_MISSING_GEOMETRY",
        "error",
        `Connector "${node.id}" references node "${nodeId}", which has no resolved world geometry.`,
        [node.id, nodeId],
        "Reference a node id present in the scene, or author explicit x/y coordinates for this endpoint.",
      ),
    );
  }
  return { x: 0, y: 0 };
}

function isDecorativeConnector(node: SceneNodeLike): boolean {
  return capabilityOf(node) === "core.divider" || node.kind === "divider";
}

function isEdgeBoundMotionSignal(node: SceneNodeLike): boolean {
  return (
    isMotionSignalNode(node) &&
    typeof node.edgeRef === "string" &&
    node.edgeRef.length > 0
  );
}

function isRoutedConnector(node: SceneNodeLike): boolean {
  return isConnector(node) && !isEdgeBoundMotionSignal(node);
}

function isConnector(node: SceneNodeLike): boolean {
  const capability = capabilityOf(node);
  // Glyph icons and brace geometry reuse path IR but are not routed edges.
  if (
    capability === "core.path" ||
    capability === "core.bracket" ||
    node.kind === "path" ||
    node.kind === "bracket"
  ) {
    return false;
  }
  return (
    node.kind === "connector" ||
    node.kind === "arrow" ||
    node.kind === "elbow" ||
    node.kind === "line" ||
    capability === "core.connector" ||
    capability === "core.arrow" ||
    capability === "core.elbow" ||
    capability === "core.route" ||
    capability === "core.line" ||
    isMotionSignalNode(node)
  );
}

function markerDisabled(node: SceneNodeLike): boolean {
  const arrowhead = node.style?.arrowhead;
  const markerEnd = node.style?.markerEnd;
  return (
    arrowhead === false ||
    arrowhead === 0 ||
    arrowhead === "false" ||
    markerEnd === "none" ||
    markerEnd === false ||
    markerEnd === 0
  );
}

function explicitlyDirected(node: SceneNodeLike): boolean {
  const markerEnd = node.style?.markerEnd;
  return (
    node.style?.arrowhead === true ||
    node.style?.arrowhead === 1 ||
    node.style?.arrowhead === "true" ||
    (markerEnd !== undefined && !markerDisabled(node))
  );
}

function directedPolicy(
  node: SceneNodeLike,
): Readonly<{ directed: boolean; showArrowhead: boolean; defaulted: boolean }> {
  const capability = capabilityOf(node);
  if (
    markerDisabled(node) ||
    isMotionSignalNode(node) ||
    capability === "core.bracket" ||
    node.kind === "bracket"
  ) {
    return { directed: false, showArrowhead: false, defaulted: false };
  }
  return {
    directed: true,
    showArrowhead: true,
    defaulted: !explicitlyDirected(node),
  };
}

function numberTokens(input: string): readonly (string | number)[] | undefined {
  const tokens: Array<string | number> = [];
  const tokenPattern =
    /([AaCcHhLlMmQqSsTtVvZz])|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)/g;
  let cursor = 0;
  for (const match of input.matchAll(tokenPattern)) {
    const index = match.index ?? 0;
    if (!/^[\s,]*$/.test(input.slice(cursor, index))) {
      return undefined;
    }
    if (match[1] !== undefined) {
      tokens.push(match[1]);
    } else {
      const value = Number(match[2]);
      if (!Number.isFinite(value)) {
        return undefined;
      }
      tokens.push(value);
    }
    cursor = index + match[0].length;
  }
  return /^[\s,]*$/.test(input.slice(cursor)) && tokens.length > 0
    ? tokens
    : undefined;
}

function parseSvgPath(d: string): ParsedPath | undefined {
  const tokens = numberTokens(d);
  if (tokens === undefined) {
    return undefined;
  }
  let index = 0;
  let command: string | undefined;
  let current: ResolvedPoint = { x: 0, y: 0 };
  let subpathStart: ResolvedPoint | undefined;
  let start: ResolvedPoint | undefined;
  const vertices: ResolvedPoint[] = [];
  const arity: Readonly<Record<string, number>> = {
    M: 2,
    L: 2,
    H: 1,
    V: 1,
    C: 6,
    S: 4,
    Q: 4,
    T: 2,
    A: 7,
  };
  let movePair = false;

  while (index < tokens.length) {
    const token = tokens[index];
    if (typeof token === "string") {
      command = token;
      index += 1;
      if (command.toUpperCase() === "Z") {
        if (subpathStart === undefined) return undefined;
        current = subpathStart;
        vertices.push(current);
        command = undefined;
        continue;
      }
      movePair = command.toUpperCase() === "M";
    }
    if (command === undefined) {
      return undefined;
    }
    const operation = command.toUpperCase();
    const count = arity[operation];
    if (count === undefined || index + count > tokens.length) {
      return undefined;
    }
    const values = tokens.slice(index, index + count);
    if (values.some((value) => typeof value !== "number")) {
      return undefined;
    }
    const numbers = values as readonly number[];
    index += count;
    const relative = command === command.toLowerCase();
    const base = current;
    let next: ResolvedPoint;
    if (operation === "H") {
      next = { x: relative ? base.x + numbers[0]! : numbers[0]!, y: base.y };
    } else if (operation === "V") {
      next = { x: base.x, y: relative ? base.y + numbers[0]! : numbers[0]! };
    } else {
      const xIndex =
        operation === "C"
          ? 4
          : operation === "S" || operation === "Q"
            ? 2
            : operation === "A"
              ? 5
              : 0;
      const yIndex = xIndex + 1;
      next = {
        x: relative ? base.x + numbers[xIndex]! : numbers[xIndex]!,
        y: relative ? base.y + numbers[yIndex]! : numbers[yIndex]!,
      };
    }
    if (!Number.isFinite(next.x) || !Number.isFinite(next.y)) {
      return undefined;
    }
    current = next;
    if (operation === "M" && movePair) {
      subpathStart = current;
      start ??= current;
      movePair = false;
      command = relative ? "l" : "L";
    }
    vertices.push(current);
  }
  return start === undefined
    ? undefined
    : { start, end: current, vertices: Object.freeze(vertices) };
}

/**
 * Extract the first and final points from an SVG path.
 *
 * Absolute and relative M/L/H/V/C/S/Q/T/A/Z endpoint semantics are supported.
 * Malformed and non-finite path data returns `undefined`.
 */
export function svgPathEndpoints(
  d: string,
): Readonly<{ start: ResolvedPoint; end: ResolvedPoint }> | undefined {
  const parsed = parseSvgPath(d);
  return parsed === undefined ? undefined : { start: parsed.start, end: parsed.end };
}

function formatNumber(value: number): string {
  const rounded = Math.round(value * 1000) / 1000;
  return String(rounded === 0 ? 0 : rounded);
}

function polylinePath(points: readonly ResolvedPoint[]): string {
  return points
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"}${formatNumber(point.x)} ${formatNumber(point.y)}`,
    )
    .join(" ");
}

function distance(left: ResolvedPoint, right: ResolvedPoint): number {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

/**
 * Whether a resolved node should participate in automatic route obstacle sets.
 * Decorative chrome, generated semantic parts, and thin annotation labels are
 * excluded so routes are not forced around backdrop geometry they intentionally
 * cross.
 */
export function isRoutingObstacle(
  node: SceneNodeLike,
  geometry: SceneGeometryLike,
  generatedPartIds?: ReadonlySet<string>,
): boolean {
  if (generatedPartIds?.has(node.id) === true) {
    return false;
  }
  const capability = capabilityOf(node);
  if (NON_ROUTING_CAPABILITIES.has(capability)) {
    return false;
  }
  if (
    capability === "core.text" &&
    geometry.height > 0 &&
    geometry.height <= ANNOTATION_TEXT_MAX_HEIGHT
  ) {
    return false;
  }
  return true;
}

function routeObstacles(
  candidate: SceneNodeLike,
  sourceId: string | undefined,
  targetId: string | undefined,
  input: ResolveConnectorsInput,
): readonly RouteObstacle[] {
  const excluded = new Set<string>([candidate.id]);
  for (const id of [sourceId, targetId]) {
    if (id === undefined) continue;
    excluded.add(id);
    for (const ancestor of input.ancestorIdsById.get(id) ?? []) {
      excluded.add(ancestor);
    }
  }
  const obstacles: RouteObstacle[] = [];
  for (const [id, geometry] of input.worldGeometryById) {
    const node = input.nodesById.get(id);
    if (
      excluded.has(id) ||
      node === undefined ||
      isConnector(node) ||
      !isRoutingObstacle(node, geometry, input.generatedPartIds) ||
      !(geometry.width > 0) ||
      !(geometry.height > 0)
    ) {
      continue;
    }
    obstacles.push({ id, bounds: geometry });
  }
  return obstacles.sort((left, right) => left.id.localeCompare(right.id));
}

function makeDiagnostic(
  node: SceneNodeLike,
  code: string,
  severity: SceneResolutionDiagnostic["severity"],
  message: string,
  nodeIds: readonly string[],
  repair?: string,
): SceneResolutionDiagnostic {
  return {
    code,
    severity,
    message,
    range: node.sourceMap ?? UNKNOWN_RANGE,
    nodeIds: Object.freeze([...nodeIds]),
    ...(repair === undefined ? {} : { repair }),
  };
}

function routeCandidate(
  node: SceneNodeLike,
  input: ResolveConnectorsInput,
  diagnostics: SceneResolutionDiagnostic[],
): ConnectorCandidate {
  const from = singlePoint(node.from);
  const to = singlePoint(node.to);
  const sourceId =
    typeof from?.nodeId === "string" && from.nodeId.length > 0
      ? from.nodeId
      : undefined;
  const targetId =
    typeof to?.nodeId === "string" && to.nodeId.length > 0
      ? to.nodeId
      : undefined;
  return {
    node,
    source: resolveEndpoint(from, input.worldGeometryById, node, diagnostics),
    target: resolveEndpoint(to, input.worldGeometryById, node, diagnostics),
    from,
    to,
    ...(sourceId === undefined ? {} : { sourceId }),
    ...(targetId === undefined ? {} : { targetId }),
    obstacles: routeObstacles(node, sourceId, targetId, input),
  };
}

function candidatePath(
  candidate: ConnectorCandidate,
  input: ResolveConnectorsInput,
  diagnostics: SceneResolutionDiagnostic[],
  siblings: readonly RoutedSibling[],
  laneOffset: number,
): Readonly<{
  d: string;
  usedFallback: boolean;
  penetratedObstacleIds: readonly string[];
  vertices: readonly ResolvedPoint[];
}> {
  const { node, source, target } = candidate;
  const authored = node.d ?? node.path;
  if (typeof authored === "string" && authored.length > 0) {
    return {
      d: authored,
      usedFallback: false,
      penetratedObstacleIds: Object.freeze([]),
      vertices: parseSvgPath(authored)?.vertices ?? Object.freeze([]),
    };
  }
  if (Array.isArray(node.points) && node.points.length > 0) {
    const points = node.points.map((point) =>
      resolveEndpoint(point, input.worldGeometryById, node, diagnostics),
    );
    return {
      d: polylinePath(points),
      usedFallback: false,
      penetratedObstacleIds: Object.freeze([]),
      vertices: Object.freeze(points),
    };
  }
  const options = normalizeCurveRouteOptions(node.style);
  if (isCurveRoute(node) || (isElbowRoute(node) && node.via === undefined)) {
    const route = routeCurve({
      edgeId: node.id,
      start: source,
      end: target,
      fromAnchor: candidate.from?.anchor,
      toAnchor: candidate.to?.anchor,
      sourceId: candidate.sourceId,
      targetId: candidate.targetId,
      sourceBounds:
        candidate.sourceId === undefined
          ? undefined
          : input.worldGeometryById.get(candidate.sourceId),
      targetBounds:
        candidate.targetId === undefined
          ? undefined
          : input.worldGeometryById.get(candidate.targetId),
      obstacles: options.avoidObstacles ? candidate.obstacles : [],
      siblings,
      options,
      laneOffset,
    });
    return {
      d: isElbowRoute(node) ? polylinePath(route.waypoints) : route.d,
      usedFallback: route.usedFallback,
      penetratedObstacleIds: Object.freeze([...route.penetratedObstacleIds]),
      vertices: Object.freeze([...route.waypoints]),
    };
  }
  if (isElbowRoute(node) || node.via !== undefined || node.axis !== undefined) {
    const via =
      node.via === undefined
        ? undefined
        : resolveEndpoint(node.via, input.worldGeometryById, node, diagnostics);
    const d = elbowPathData(
      source,
      target,
      via,
      node.axis === "x" || node.axis === "y" ? node.axis : undefined,
      candidate.from?.anchor,
      candidate.to?.anchor,
      options.avoidObstacles ? candidate.obstacles : [],
      options.clearance,
    );
    return {
      d,
      usedFallback: false,
      penetratedObstacleIds: Object.freeze([]),
      vertices: parseSvgPath(d)?.vertices ?? Object.freeze([]),
    };
  }
  return {
    d: `M${formatNumber(source.x)} ${formatNumber(source.y)} L${formatNumber(target.x)} ${formatNumber(target.y)}`,
    usedFallback: false,
    penetratedObstacleIds: Object.freeze([]),
    vertices: Object.freeze([source, target]),
  };
}

function validatePath(
  candidate: ConnectorCandidate,
  path: ReturnType<typeof candidatePath>,
  diagnostics: SceneResolutionDiagnostic[],
): void {
  // Motion signals often bind to an edge and carry companion path geometry that
  // is not meant to attach to the signal's own from/to ports.
  if (isMotionSignalNode(candidate.node)) {
    return;
  }
  const authored =
    typeof candidate.node.d === "string" || typeof candidate.node.path === "string";
  const parsed = parseSvgPath(path.d);
  if (parsed === undefined) {
    return;
  }
  const normal =
    distance(parsed.start, candidate.source) <= ENDPOINT_TOLERANCE &&
    distance(parsed.end, candidate.target) <= ENDPOINT_TOLERANCE;
  const reversed =
    distance(parsed.start, candidate.target) <= ENDPOINT_TOLERANCE &&
    distance(parsed.end, candidate.source) <= ENDPOINT_TOLERANCE;
  if (authored && reversed && !normal) {
    diagnostics.push(
      makeDiagnostic(
        candidate.node,
        "SCENE_AUTHORED_PATH_REVERSED",
        "warning",
        `Authored path for "${candidate.node.id}" runs from target to source.`,
        [candidate.node.id],
        "Reverse the authored path commands or swap the declared endpoints.",
      ),
    );
  } else if (authored && !normal) {
    diagnostics.push(
      makeDiagnostic(
        candidate.node,
        "SCENE_CONNECTOR_ENDPOINT_DETACHED",
        "error",
        `Authored path for "${candidate.node.id}" does not attach to its declared endpoints.`,
        [candidate.node.id],
        "Make the path start and end at the declared source and target ports.",
      ),
    );
  }
  if (!authored || parsed.vertices.length < 2) {
    return;
  }
  if (isDecorativeConnector(candidate.node)) {
    return;
  }
  for (const obstacle of candidate.obstacles) {
    const intersects = parsed.vertices.slice(1).some((point, index) =>
      segmentIntersectsBounds(
        parsed.vertices[index]!,
        point,
        obstacle.bounds,
        true,
      ),
    );
    if (intersects) {
      diagnostics.push(
        makeDiagnostic(
          candidate.node,
          "SCENE_CONNECTOR_INTERSECTION",
          "warning",
          `Authored path for "${candidate.node.id}" crosses "${obstacle.id}".`,
          [candidate.node.id, obstacle.id],
          "Adjust the authored path or use automatic routing.",
        ),
      );
    }
  }
  const clearance = normalizeCurveRouteOptions(candidate.node.style).clearance;
  const terminalSegments =
    parsed.vertices.length === 2
      ? [[parsed.vertices[0]!, parsed.vertices[1]!] as const]
      : [
          [parsed.vertices[0]!, parsed.vertices[1]!] as const,
          [
            parsed.vertices[parsed.vertices.length - 2]!,
            parsed.vertices[parsed.vertices.length - 1]!,
          ] as const,
        ];
  for (const obstacle of candidate.obstacles) {
    if (
      terminalSegments.some(([start, end]) =>
        segmentIntersectsBounds(
          start,
          end,
          inflateBounds(obstacle.bounds, clearance ?? DEFAULT_CLEARANCE),
          true,
        ),
      )
    ) {
      diagnostics.push(
        makeDiagnostic(
          candidate.node,
          "SCENE_CONNECTOR_VISUALLY_AMBIGUOUS",
          "warning",
          `Path for "${candidate.node.id}" passes close to "${obstacle.id}" near an endpoint.`,
          [candidate.node.id, obstacle.id],
          "Increase clearance or author an unambiguous path.",
        ),
      );
    }
  }
}

function diagnosticOrder(
  left: SceneResolutionDiagnostic,
  right: SceneResolutionDiagnostic,
): number {
  return (
    left.range.source.localeCompare(right.range.source) ||
    left.range.start.offset - right.range.start.offset ||
    left.code.localeCompare(right.code) ||
    (left.nodeIds[0] ?? "").localeCompare(right.nodeIds[0] ?? "")
  );
}

/** Resolve every ordinary connector once from final world-space scene bounds. */
export function resolveConnectors(
  input: ResolveConnectorsInput,
): ResolveConnectorsResult {
  const connectorsById = new Map<string, ResolvedConnector>();
  const diagnostics: SceneResolutionDiagnostic[] = [];
  const candidates = [...input.nodesById.values()]
    .filter(isRoutedConnector)
    .map((node) => routeCandidate(node, input, diagnostics));

  const laneOffsets = new Map<string, number>();
  const curveGroups = new Map<string, ConnectorCandidate[]>();
  for (const candidate of candidates.filter(({ node }) => isCurveRoute(node))) {
    const key = `${candidate.sourceId ?? ""}\u0000${candidate.targetId ?? ""}\u0000${candidate.from?.anchor ?? ""}\u0000${candidate.to?.anchor ?? ""}`;
    const group = curveGroups.get(key) ?? [];
    group.push(candidate);
    curveGroups.set(key, group);
  }
  for (const group of curveGroups.values()) {
    const ordered = [...group].sort((left, right) =>
      left.node.id.localeCompare(right.node.id),
    );
    ordered.forEach((candidate, index) => {
      const options = normalizeCurveRouteOptions(candidate.node.style);
      const lane = index - (ordered.length - 1) / 2;
      laneOffsets.set(
        candidate.node.id,
        options.bundle ? 0 : lane * options.parallelGap,
      );
    });
  }

  const siblings: RoutedSibling[] = [];
  for (const candidate of candidates) {
    const policy = directedPolicy(candidate.node);
    if (policy.defaulted) {
      diagnostics.push(
        makeDiagnostic(
          candidate.node,
          "SCENE_DIRECTED_ARROWHEAD_DEFAULTED",
          "info",
          `Connector "${candidate.node.id}" was defaulted to a visible arrowhead.`,
          [candidate.node.id],
        ),
      );
    }
    const path = candidatePath(
      candidate,
      input,
      diagnostics,
      siblings,
      laneOffsets.get(candidate.node.id) ?? 0,
    );
    validatePath(candidate, path, diagnostics);
    if (path.penetratedObstacleIds.length > 0) {
      diagnostics.push(
        makeDiagnostic(
          candidate.node,
          "SCENE_ROUTE_FALLBACK",
          "warning",
          `Route for "${candidate.node.id}" penetrates obstacles after deterministic fallback.`,
          [candidate.node.id, ...path.penetratedObstacleIds],
          "Move obstacles, increase available space, or author an explicit path.",
        ),
      );
    }
    connectorsById.set(
      candidate.node.id,
      Object.freeze({
        id: candidate.node.id,
        source: candidate.source,
        target: candidate.target,
        ...(candidate.sourceId === undefined
          ? {}
          : { sourceId: candidate.sourceId }),
        ...(candidate.targetId === undefined
          ? {}
          : { targetId: candidate.targetId }),
        d: path.d,
        directed: policy.directed,
        showArrowhead: policy.showArrowhead,
        usedFallback: path.usedFallback,
        penetratedObstacleIds: path.penetratedObstacleIds,
      }),
    );
    if (isCurveRoute(candidate.node)) {
      const route = routeCurve({
        edgeId: candidate.node.id,
        start: candidate.source,
        end: candidate.target,
        fromAnchor: candidate.from?.anchor,
        toAnchor: candidate.to?.anchor,
        sourceId: candidate.sourceId,
        targetId: candidate.targetId,
        sourceBounds:
          candidate.sourceId === undefined
            ? undefined
            : input.worldGeometryById.get(candidate.sourceId),
        targetBounds:
          candidate.targetId === undefined
            ? undefined
            : input.worldGeometryById.get(candidate.targetId),
        obstacles: candidate.obstacles,
        siblings,
        options: normalizeCurveRouteOptions(candidate.node.style),
        laneOffset: laneOffsets.get(candidate.node.id) ?? 0,
      });
      siblings.push({
        id: candidate.node.id,
        sourceId: candidate.sourceId,
        targetId: candidate.targetId,
        fromAnchor: candidate.from?.anchor,
        toAnchor: candidate.to?.anchor,
        waypoints: route.waypoints,
        segments: route.segments,
      });
    }
  }

  resolveEdgeBoundMotionSignals(input, connectorsById, diagnostics);
  detectDuplicateStandaloneSignals(input, connectorsById, diagnostics);

  diagnostics.sort(diagnosticOrder);
  return Object.freeze({
    connectorsById,
    diagnostics: Object.freeze(diagnostics),
  });
}

function resolveEdgeBoundMotionSignals(
  input: ResolveConnectorsInput,
  connectorsById: Map<string, ResolvedConnector>,
  diagnostics: SceneResolutionDiagnostic[],
): void {
  const motionNodes = [...input.nodesById.values()]
    .filter(isEdgeBoundMotionSignal)
    .sort((left, right) => left.id.localeCompare(right.id));
  for (const node of motionNodes) {
    const edgeRef = node.edgeRef!;
    if (edgeRef === node.id) {
      diagnostics.push(
        makeDiagnostic(
          node,
          "SCENE_SIGNAL_EDGE_NOT_FOUND",
          "error",
          `Motion signal "${node.id}" cannot reference itself.`,
          [node.id, edgeRef],
          `Reference an ordinary connector id with edge = "${edgeRef}".`,
        ),
      );
      continue;
    }
    const referencedNode = input.nodesById.get(edgeRef);
    if (referencedNode === undefined || isMotionSignalNode(referencedNode)) {
      diagnostics.push(
        makeDiagnostic(
          node,
          "SCENE_SIGNAL_EDGE_NOT_FOUND",
          "error",
          `Motion signal "${node.id}" references unknown edge "${edgeRef}".`,
          [node.id, edgeRef],
          `Reference a resolved connector id with edge = "${edgeRef}".`,
        ),
      );
      continue;
    }
    const referenced = connectorsById.get(edgeRef);
    if (referenced === undefined) {
      diagnostics.push(
        makeDiagnostic(
          node,
          "SCENE_SIGNAL_EDGE_NOT_FOUND",
          "error",
          `Motion signal "${node.id}" references unresolved edge "${edgeRef}".`,
          [node.id, edgeRef],
          `Reference a resolved connector id with edge = "${edgeRef}".`,
        ),
      );
      continue;
    }
    connectorsById.set(
      node.id,
      Object.freeze({
        id: node.id,
        source: referenced.source,
        target: referenced.target,
        ...(referenced.sourceId === undefined
          ? {}
          : { sourceId: referenced.sourceId }),
        ...(referenced.targetId === undefined
          ? {}
          : { targetId: referenced.targetId }),
        d: referenced.d,
        directed: false,
        showArrowhead: false,
        usedFallback: referenced.usedFallback,
        penetratedObstacleIds: referenced.penetratedObstacleIds,
      }),
    );
  }
}

function connectorsMatch(
  left: ResolvedConnector,
  right: ResolvedConnector,
): boolean {
  return (
    left.d === right.d &&
    left.source.x === right.source.x &&
    left.source.y === right.source.y &&
    left.target.x === right.target.x &&
    left.target.y === right.target.y
  );
}

function detectDuplicateStandaloneSignals(
  input: ResolveConnectorsInput,
  connectorsById: Map<string, ResolvedConnector>,
  diagnostics: SceneResolutionDiagnostic[],
): void {
  const motionNodes = [...input.nodesById.values()]
    .filter((node) => isMotionSignalNode(node) && !isEdgeBoundMotionSignal(node))
    .sort((left, right) => left.id.localeCompare(right.id));
  for (const node of motionNodes) {
    const motion = connectorsById.get(node.id);
    if (motion === undefined) {
      continue;
    }
    for (const [otherId, other] of connectorsById) {
      if (otherId === node.id) {
        continue;
      }
      const otherNode = input.nodesById.get(otherId);
      if (otherNode === undefined || isMotionSignalNode(otherNode)) {
        continue;
      }
      if (connectorsMatch(motion, other)) {
        diagnostics.push(
          makeDiagnostic(
            node,
            "SCENE_SIGNAL_DUPLICATES_EDGE",
            "error",
            `Motion signal "${node.id}" duplicates connector "${otherId}".`,
            [node.id, otherId],
            `Reference the existing edge with edge = "${otherId}".`,
          ),
        );
        break;
      }
    }
  }
}
