/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lower package-form scene nodes into native semantic Scene IR nodes.

import type {
  ConnectorAxisIr,
  ConnectorEndpointIr,
  GeometryIr,
  PointIr,
  RenderNodeIr,
  SourceRange,
  StyleValueIr,
  ThemeRoleReferenceIr,
} from "../schema/index.js";

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

/** Return an object value as a record. */
export function asRecord(
  value: unknown,
): Record<string, unknown> | undefined {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return undefined;
}

function finiteOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/** Coerce an authored numeric field to a finite number, else `fallback`. */
function finiteNumber(value: unknown, fallback: number): number {
  const n = Number(value ?? fallback);
  return Number.isFinite(n) ? n : fallback;
}

/** Coerce an extent field to a finite nonnegative number, else `fallback`. */
function nonnegativeExtent(value: unknown, fallback: number): number {
  const n = finiteNumber(value, fallback);
  return n >= 0 ? n : fallback;
}

function geometryOf(node: Record<string, unknown>): GeometryIr {
  const geometry = asRecord(node.geometry) ?? asRecord(node.layout) ?? {};
  return {
    x: finiteNumber(geometry.x, 0),
    y: finiteNumber(geometry.y, 0),
    width: nonnegativeExtent(geometry.width, 0),
    height: nonnegativeExtent(geometry.height, 0),
  };
}

function isThemeRoleStyleValue(value: unknown): value is ThemeRoleReferenceIr {
  const record = asRecord(value);
  return (
    record !== undefined &&
    record.kind === "theme-role" &&
    typeof record.role === "string" &&
    record.role.length > 0
  );
}

function styleOf(
  node: Record<string, unknown>,
): Record<string, StyleValueIr> {
  const style = asRecord(node.style) ?? {};
  const out: Record<string, StyleValueIr> = {};
  for (const [key, value] of Object.entries(style)) {
    if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      out[key] = value;
    } else if (isThemeRoleStyleValue(value)) {
      out[key] = { kind: "theme-role", role: value.role };
    }
  }
  for (const key of [
    "direction",
    "cols",
    "gap",
    "r",
    "rx",
    "ry",
    "radius",
    "fill",
    "stroke",
  ] as const) {
    const value = node[key];
    if (out[key] !== undefined) {
      continue;
    }
    if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      out[key] = value;
    } else if (isThemeRoleStyleValue(value)) {
      out[key] = { kind: "theme-role", role: value.role };
    }
  }
  return out;
}

/** Map a package capability id onto its Scene IR node kind. */
export function capabilityKind(capability: string): RenderNodeIr["kind"] {
  const leaf = capability.includes(".")
    ? capability.slice(capability.lastIndexOf(".") + 1)
    : capability;
  switch (leaf) {
    case "text":
      return "text";
    case "connector":
    case "line":
    case "arrow":
    case "path":
    case "elbow":
    case "bracket":
    case "signal":
    case "divider":
    case "route":
      return "connector";
    case "fan-out":
    case "fan-in":
      return "fan";
    case "group":
    case "stack":
    case "grid":
    case "rail":
    case "panel":
    case "header":
    case "pad":
    case "callout":
    case "chip":
    case "note":
    case "lane":
    case "band":
    case "swimlane":
    case "stepper":
      return "group";
    case "pulse":
    default:
      return "rect";
  }
}

const SEMANTIC_PACKAGE_CAPABILITIES = new Set<string>([
  "core.circle",
  "core.ellipse",
  "core.panel",
  "core.header",
  "core.bracket",
  "core.callout",
  "core.chip",
  "core.note",
  "core.divider",
  "core.lane",
  "core.band",
  "core.swimlane",
  "core.stepper",
  "layout.pad",
]);

const FIRST_CLASS_PACKAGE_CAPABILITIES = new Set<string>([
  "core.rect",
  "core.text",
  "core.connector",
  "core.group",
  "core.path",
  "core.line",
  "core.arrow",
  "core.elbow",
  "core.route",
  "core.fan-out",
  "core.fan-in",
  "layout.stack",
  "layout.grid",
  "layout.rail",
  "motion.signal",
  "motion.pulse",
]);

/** Whether package-scene lowering recognizes a capability without a manifest. */
export function isSupportedPackageCapability(capability: string): boolean {
  return (
    SEMANTIC_PACKAGE_CAPABILITIES.has(capability) ||
    FIRST_CLASS_PACKAGE_CAPABILITIES.has(capability)
  );
}

function pathOf(node: Record<string, unknown>): string | undefined {
  if (typeof node.path === "string" && node.path.length > 0) {
    return node.path;
  }
  if (typeof node.d === "string" && node.d.length > 0) {
    return node.d;
  }
  return undefined;
}

/**
 * Resolve a `from` / `to` prop value into a `ConnectorEndpointIr`.
 *
 * Returns `undefined` when the value is absent or cannot be resolved to a
 * `nodeId` or an `x`/`y` coordinate pair; callers must omit the property
 * rather than invent a `{x: 0, y: 0}` origin endpoint, so the downstream
 * scene schema's `superRefine` fails closed on malformed authoring instead
 * of silently rendering a coordinate-only endpoint.
 */
function connectorEndpointOf(value: unknown): ConnectorEndpointIr | undefined {
  const record = asRecord(value);
  if (record === undefined) {
    return undefined;
  }
  const x = finiteOrUndefined(record.x);
  const y = finiteOrUndefined(record.y);
  const nodeId =
    typeof record.nodeId === "string" && record.nodeId.length > 0
      ? record.nodeId
      : typeof record.id === "string" && record.id.length > 0
        ? record.id
        : undefined;
  const anchor =
    typeof record.anchor === "string" && record.anchor.length > 0
      ? record.anchor
      : undefined;
  if (x !== undefined && y !== undefined) {
    return {
      x,
      y,
      ...(nodeId !== undefined ? { nodeId } : {}),
      ...(anchor !== undefined ? { anchor } : {}),
    };
  }
  if (nodeId !== undefined) {
    return {
      nodeId,
      ...(anchor !== undefined ? { anchor } : {}),
    };
  }
  return undefined;
}

function pointsOf(
  node: Record<string, unknown>,
): ReadonlyArray<PointIr | ConnectorEndpointIr> | undefined {
  if (!Array.isArray(node.points) || node.points.length === 0) {
    return undefined;
  }
  const points: Array<PointIr | ConnectorEndpointIr> = [];
  for (const point of node.points) {
    const record = asRecord(point);
    if (record === undefined) {
      continue;
    }
    const endpoint = connectorEndpointOf(record);
    if (endpoint === undefined) {
      continue;
    }
    const hasNode =
      typeof endpoint.nodeId === "string" && endpoint.nodeId.length > 0;
    if (
      typeof endpoint.x === "number" &&
      Number.isFinite(endpoint.x) &&
      typeof endpoint.y === "number" &&
      Number.isFinite(endpoint.y) &&
      !hasNode
    ) {
      points.push({ x: endpoint.x, y: endpoint.y });
      continue;
    }
    points.push(endpoint);
  }
  return points.length > 0 ? points : undefined;
}

function pointOf(value: unknown): PointIr | undefined {
  const record = asRecord(value);
  if (record === undefined) {
    return undefined;
  }
  const x = finiteOrUndefined(record.x);
  const y = finiteOrUndefined(record.y);
  if (x === undefined || y === undefined) {
    return undefined;
  }
  return { x, y };
}

function axisOf(value: unknown): ConnectorAxisIr | undefined {
  return value === "x" || value === "y" ? value : undefined;
}

function fanLabel(node: Record<string, unknown>, id: string): string {
  const sourceMap = asRecord(node.sourceMap);
  const start = sourceMap !== undefined ? asRecord(sourceMap.start) : undefined;
  const source =
    sourceMap !== undefined && typeof sourceMap.source === "string"
      ? sourceMap.source
      : undefined;
  const line = start !== undefined ? finiteOrUndefined(start.line) : undefined;
  return source !== undefined && line !== undefined
    ? `Fan "${id}" (${source}:${line})`
    : `Fan "${id}"`;
}

function requireFanEndpoint(
  value: unknown,
  label: string,
  side: string,
): ConnectorEndpointIr {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.length === 0) {
      throw new Error(`${label} ${side} endpoint is empty`);
    }
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try {
        return requireFanEndpoint(JSON.parse(trimmed) as unknown, label, side);
      } catch (error) {
        if (error instanceof Error && error.message.startsWith("Fan ")) {
          throw error;
        }
        throw new Error(`${label} ${side} endpoint is not resolvable`);
      }
    }
    return { nodeId: trimmed };
  }

  const record = asRecord(value);
  if (record === undefined) {
    throw new Error(`${label} ${side} endpoint is missing or unresolvable`);
  }
  const x = finiteOrUndefined(record.x);
  const y = finiteOrUndefined(record.y);
  const nodeId =
    typeof record.nodeId === "string" && record.nodeId.length > 0
      ? record.nodeId
      : typeof record.id === "string" && record.id.length > 0
        ? record.id
        : undefined;
  const anchor =
    typeof record.anchor === "string" && record.anchor.length > 0
      ? record.anchor
      : undefined;
  if (x !== undefined && y !== undefined) {
    return {
      x,
      y,
      ...(nodeId !== undefined ? { nodeId } : {}),
      ...(anchor !== undefined ? { anchor } : {}),
    };
  }
  if (x !== undefined || y !== undefined) {
    throw new Error(
      `${label} ${side} endpoint requires both x and y coordinates`,
    );
  }
  if (nodeId !== undefined) {
    return {
      nodeId,
      ...(anchor !== undefined ? { anchor } : {}),
    };
  }
  throw new Error(
    `${label} ${side} endpoint requires nodeId or x/y coordinates`,
  );
}

function requireFanEndpointSide(
  value: unknown,
  label: string,
  side: "from" | "to",
): ConnectorEndpointIr | readonly ConnectorEndpointIr[] {
  if (value === undefined) {
    throw new Error(`${label} ${side} is required`);
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.startsWith("[")) {
      try {
        return requireFanEndpointSide(
          JSON.parse(trimmed) as unknown,
          label,
          side,
        );
      } catch (error) {
        if (error instanceof Error && error.message.startsWith("Fan ")) {
          throw error;
        }
        throw new Error(`${label} ${side} endpoints are not resolvable`);
      }
    }
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      throw new Error(`${label} ${side} requires at least one endpoint`);
    }
    return value.map((endpoint, index) =>
      requireFanEndpoint(endpoint, label, `${side}[${index}]`),
    );
  }
  return requireFanEndpoint(value, label, side);
}

function requireFanAxis(
  node: Record<string, unknown>,
  style: Record<string, StyleValueIr>,
  label: string,
): ConnectorAxisIr | undefined {
  const raw = node.axis !== undefined ? node.axis : style.axis;
  if (raw === undefined) {
    return undefined;
  }
  if (raw === "x" || raw === "y") {
    return raw;
  }
  throw new Error(`${label} axis must be "x" or "y"`);
}

function geometryFromEndpoints(
  from: ConnectorEndpointIr | undefined,
  to: ConnectorEndpointIr | undefined,
  fallback: GeometryIr,
): GeometryIr {
  if (
    from === undefined ||
    to === undefined ||
    typeof from.x !== "number" ||
    typeof from.y !== "number" ||
    typeof to.x !== "number" ||
    typeof to.y !== "number"
  ) {
    return fallback;
  }
  return {
    x: Math.min(from.x, to.x),
    y: Math.min(from.y, to.y),
    width: Math.max(Math.abs(to.x - from.x), 0),
    height: Math.max(Math.abs(to.y - from.y), 0),
  };
}

/** Build a native RenderNodeIr from a normalized package authoring node. */
export function lowerFirstClassPackageNode(
  node: Record<string, unknown>,
  args: {
    id: string;
    capability: string;
    kind: RenderNodeIr["kind"];
    children: readonly RenderNodeIr[];
    label: string;
    description?: string;
    fallback: string;
  },
): RenderNodeIr {
  const { id, capability, children, label, description, fallback } = args;
  let kind = args.kind;
  const accessibility = {
    label,
    ...(description !== undefined ? { description } : {}),
  };
  const path = pathOf(node);
  const points = pointsOf(node);
  let geometry = geometryOf(node);
  let style = styleOf(node);

  // Default arrowheads for every core.arrow, including nodeId↔nodeId edges.
  // Absolute-geometry gating previously dropped markerEnd on anchored arrows.
  if (capability === "core.arrow") {
    kind = "connector";
    if (style.markerEnd === undefined && style.arrowhead === undefined) {
      style = { ...style, arrowhead: true, markerEnd: "arrow" };
    }
  }
  if (children.length > 0 && kind !== "group") {
    kind = "group";
  }

  const base = {
    id,
    capabilityId: capability,
    geometry,
    style,
    accessibility,
    fallback,
    sourceMap: unknownRange,
    ...(path !== undefined ? { path } : {}),
    ...(points !== undefined ? { points } : {}),
  };

  if (kind === "group") {
    return { ...base, kind: "group", children };
  }
  if (kind === "text") {
    return {
      ...base,
      kind: "text",
      text: typeof node.text === "string" ? node.text : label,
    };
  }
  if (kind === "fan") {
    const fan = fanLabel(node, id);
    if (capability !== "core.fan-out" && capability !== "core.fan-in") {
      throw new Error(
        `${fan} capability must be "core.fan-out" or "core.fan-in"`,
      );
    }
    const from = requireFanEndpointSide(node.from, fan, "from");
    const to = requireFanEndpointSide(node.to, fan, "to");
    const axis = requireFanAxis(node, style, fan);
    const junction = pointOf(node.junction);
    return {
      ...base,
      kind: "fan",
      capability,
      from,
      to,
      ...(axis !== undefined ? { axis } : {}),
      ...(junction !== undefined ? { junction } : {}),
    };
  }
  if (kind === "connector") {
    const from = connectorEndpointOf(node.from);
    const to = connectorEndpointOf(node.to);
    const via = pointOf(node.via);
    const axis = axisOf(node.axis) ?? axisOf(style.axis);
    if (
      geometry.width === 0 &&
      geometry.height === 0 &&
      geometry.x === 0 &&
      geometry.y === 0
    ) {
      geometry = geometryFromEndpoints(from, to, geometry);
    }
    if (
      (capability === "core.elbow" || capability === "core.route") &&
      style.route === undefined
    ) {
      style = { ...style, route: "elbow" };
    }
    return {
      ...base,
      geometry,
      style,
      kind: "connector",
      ...(from !== undefined ? { from } : {}),
      ...(to !== undefined ? { to } : {}),
      ...(via !== undefined ? { via } : {}),
      ...(axis !== undefined ? { axis } : {}),
    };
  }
  return { ...base, kind: "rect" };
}
