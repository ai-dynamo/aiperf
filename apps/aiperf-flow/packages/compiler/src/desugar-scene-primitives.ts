/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Desugar Flow core geometry macros into familiar SceneIr nodes, and map
//! first-class layout / motion / connector capabilities onto IR kinds.
//!
//! Macros expand here; runtime-needed concepts (`layout.stack` / `grid`,
//! `core.connector` / `elbow`, `motion.signal` / `pulse`) keep their
//! capability ids and pass through as group / connector / rect nodes.

import type {
  ConnectorAxisIr,
  ConnectorEndpointIr,
  GeometryIr,
  PointIr,
  RenderNodeIr,
  SourceRange,
} from "@aiperf/flow-schema";

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

const TITLE_HEIGHT = 22;
const DETAIL_HEIGHT = 20;
const HEADER_TEXT_HEIGHT = 24;
const INSET = 8;
const DEFAULT_PAD = 12;

export function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return undefined;
}

export function finiteOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function geometryOf(node: Record<string, unknown>): GeometryIr {
  const geometry = asRecord(node.geometry) ?? asRecord(node.layout) ?? {};
  return {
    x: Number(geometry.x ?? 0),
    y: Number(geometry.y ?? 0),
    width: Number(geometry.width ?? 0),
    height: Number(geometry.height ?? 0),
  };
}

export function styleOf(
  node: Record<string, unknown>,
): Record<string, string | number | boolean> {
  const style = asRecord(node.style) ?? {};
  const out: Record<string, string | number | boolean> = {};
  for (const [key, value] of Object.entries(style)) {
    if (
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean"
    ) {
      out[key] = value;
    }
  }
  // Lift common package authoring props into style for first-class layout.
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
    if (
      (typeof value === "string" ||
        typeof value === "number" ||
        typeof value === "boolean") &&
      out[key] === undefined
    ) {
      out[key] = value;
    }
  }
  return out;
}

/** Map a capability id onto the SceneIr node kind before / without desugar. */
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
    case "divider":
    case "route":
      return "connector";
    case "pulse":
    default:
      return "rect";
  }
}

/** True when the capability is a compiler macro that expands before IR emit. */
export function isDesugarCapability(capability: string): boolean {
  switch (capability) {
    case "core.circle":
    case "core.ellipse":
    case "core.panel":
    case "core.header":
    case "core.bracket":
    case "core.callout":
    case "core.chip":
    case "core.note":
    case "core.divider":
    case "core.lane":
    case "core.band":
    case "core.swimlane":
    case "core.stepper":
    case "layout.pad":
      return true;
    case "core.arrow":
      // Absolute geometry desugars; node-anchored arrows stay first-class.
      return false;
    default:
      return false;
  }
}

export function pathOf(node: Record<string, unknown>): string | undefined {
  if (typeof node.path === "string" && node.path.length > 0) {
    return node.path;
  }
  if (typeof node.d === "string" && node.d.length > 0) {
    return node.d;
  }
  return undefined;
}

export function pointsOf(
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
    const hasCoord =
      (typeof endpoint.x === "number" && Number.isFinite(endpoint.x)) ||
      (typeof endpoint.y === "number" && Number.isFinite(endpoint.y));
    const hasNode =
      typeof endpoint.nodeId === "string" && endpoint.nodeId.length > 0;
    if (!hasCoord && !hasNode) {
      continue;
    }
    // Prefer full absolute points when both coords are present.
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

export function connectorEndpointOf(
  value: unknown,
): ConnectorEndpointIr {
  const record = asRecord(value);
  if (record === undefined) {
    return { x: 0, y: 0 };
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
  return { x: 0, y: 0 };
}

function fanLabel(node: Record<string, unknown>, id: string): string {
  const sourceMap = asRecord(node.sourceMap);
  const start = sourceMap !== undefined ? asRecord(sourceMap.start) : undefined;
  const source =
    sourceMap !== undefined && typeof sourceMap.source === "string"
      ? sourceMap.source
      : undefined;
  const line = start !== undefined ? finiteOrUndefined(start.line) : undefined;
  if (source !== undefined && line !== undefined) {
    return `Fan "${id}" (${source}:${line})`;
  }
  return `Fan "${id}"`;
}

/**
 * Resolve one fan endpoint without the connector `{x:0,y:0}` fallback.
 * Bare strings are treated as node ids (native identifier authoring).
 */
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
  style: Record<string, string | number | boolean>,
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

export function pointOf(value: unknown): PointIr | undefined {
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

export function axisOf(value: unknown): ConnectorAxisIr | undefined {
  return value === "x" || value === "y" ? value : undefined;
}

function geometryFromEndpoints(
  from: ConnectorEndpointIr,
  to: ConnectorEndpointIr,
  fallback: GeometryIr,
): GeometryIr {
  if (
    typeof from.x !== "number" ||
    typeof from.y !== "number" ||
    typeof to.x !== "number" ||
    typeof to.y !== "number"
  ) {
    return fallback;
  }
  const x = Math.min(from.x, to.x);
  const y = Math.min(from.y, to.y);
  return {
    x,
    y,
    width: Math.max(Math.abs(to.x - from.x), 0),
    height: Math.max(Math.abs(to.y - from.y), 0),
  };
}

function stringProp(
  node: Record<string, unknown>,
  key: string,
): string | undefined {
  const value = node[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function stringArrayProp(
  node: Record<string, unknown>,
  key: string,
): readonly string[] {
  const value = node[key];
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter(
    (entry): entry is string => typeof entry === "string" && entry.length > 0,
  );
}

function textNode(args: {
  id: string;
  text: string;
  geometry: GeometryIr;
  style?: Record<string, string | number | boolean>;
}): RenderNodeIr {
  return {
    kind: "text",
    id: args.id,
    capabilityId: "core.text",
    geometry: args.geometry,
    style: args.style ?? {},
    accessibility: { label: args.text },
    fallback: args.text,
    sourceMap: unknownRange,
    text: args.text,
  };
}

function circleEllipseGeometry(node: Record<string, unknown>): {
  geometry: GeometryIr;
  style: Record<string, string | number | boolean>;
} {
  const style = styleOf(node);
  const center = pointOf(node.center);
  const r = finiteOrUndefined(node.r) ?? finiteOrUndefined(style.r);
  const rx =
    finiteOrUndefined(node.rx) ??
    finiteOrUndefined(style.rx) ??
    r ??
    (center !== undefined ? 0 : undefined);
  const ry =
    finiteOrUndefined(node.ry) ??
    finiteOrUndefined(style.ry) ??
    r ??
    rx;
  let geometry = geometryOf(node);
  if (center !== undefined && rx !== undefined && ry !== undefined) {
    geometry = {
      x: center.x - rx,
      y: center.y - ry,
      width: rx * 2,
      height: ry * 2,
    };
  } else if (
    geometry.width === 0 &&
    geometry.height === 0 &&
    rx !== undefined &&
    ry !== undefined
  ) {
    geometry = {
      ...geometry,
      width: rx * 2,
      height: ry * 2,
    };
  }
  if (r !== undefined) {
    style.r = r;
    style.radius = r;
  }
  if (rx !== undefined) {
    style.rx = rx;
  }
  if (ry !== undefined) {
    style.ry = ry;
  }
  return { geometry, style };
}

function bracePath(geometry: GeometryIr, side: string): string {
  const { x, y, width, height } = geometry;
  // Approximate a curly brace along the left / right / top / bottom span.
  switch (side) {
    case "right": {
      const mid = y + height / 2;
      return `M${x} ${y} C${x + width} ${y}, ${x + width} ${mid - height * 0.15}, ${x + width * 0.35} ${mid} C${x + width} ${mid + height * 0.15}, ${x + width} ${y + height}, ${x} ${y + height}`;
    }
    case "top": {
      const mid = x + width / 2;
      return `M${x} ${y + height} C${x} ${y}, ${mid - width * 0.15} ${y}, ${mid} ${y + height * 0.65} C${mid + width * 0.15} ${y}, ${x + width} ${y}, ${x + width} ${y + height}`;
    }
    case "bottom": {
      const mid = x + width / 2;
      return `M${x} ${y} C${x} ${y + height}, ${mid - width * 0.15} ${y + height}, ${mid} ${y + height * 0.35} C${mid + width * 0.15} ${y + height}, ${x + width} ${y + height}, ${x + width} ${y}`;
    }
    case "left":
    default: {
      const mid = y + height / 2;
      return `M${x + width} ${y} C${x} ${y}, ${x} ${mid - height * 0.15}, ${x + width * 0.65} ${mid} C${x} ${mid + height * 0.15}, ${x} ${y + height}, ${x + width} ${y + height}`;
    }
  }
}

function arrowHasAbsoluteGeometry(node: Record<string, unknown>): boolean {
  if (pathOf(node) !== undefined || pointsOf(node) !== undefined) {
    return true;
  }
  const from = asRecord(node.from);
  const to = asRecord(node.to);
  if (from === undefined || to === undefined) {
    return false;
  }
  const fromAbsolute =
    finiteOrUndefined(from.x) !== undefined &&
    finiteOrUndefined(from.y) !== undefined &&
    !(typeof from.nodeId === "string" && from.nodeId.length > 0);
  const toAbsolute =
    finiteOrUndefined(to.x) !== undefined &&
    finiteOrUndefined(to.y) !== undefined &&
    !(typeof to.nodeId === "string" && to.nodeId.length > 0);
  return fromAbsolute && toAbsolute;
}

/**
 * Desugar a package-form node when its capability is a macro, or return
 * `undefined` so the caller emits a first-class node.
 */
export function desugarPackageNode(
  node: Record<string, unknown>,
  args: {
    id: string;
    capability: string;
    children: readonly RenderNodeIr[];
    label: string;
    description?: string;
    fallback: string;
  },
): RenderNodeIr | undefined {
  const { id, capability, children, label, description, fallback } = args;
  const accessibility = {
    label,
    ...(description !== undefined ? { description } : {}),
  };

  switch (capability) {
    case "core.circle":
    case "core.ellipse": {
      const { geometry, style } = circleEllipseGeometry(node);
      return {
        kind: "rect",
        id,
        capabilityId: capability,
        geometry,
        style,
        accessibility,
        fallback,
        sourceMap: unknownRange,
      };
    }

    case "core.panel": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const title = stringProp(node, "title");
      const detail = stringProp(node, "detail");
      const desugared: RenderNodeIr[] = [];
      if (title !== undefined) {
        desugared.push(
          textNode({
            id: `${id}-title`,
            text: title,
            geometry: {
              x: INSET,
              y: INSET + 2,
              width: Math.max(geometry.width - INSET * 2, 0),
              height: TITLE_HEIGHT,
            },
            style: {
              fontSize: 13,
              fontWeight: "bold",
              textAnchor: "middle",
            },
          }),
        );
      }
      if (detail !== undefined) {
        desugared.push(
          textNode({
            id: `${id}-detail`,
            text: detail,
            geometry: {
              x: INSET,
              y: INSET + 2 + TITLE_HEIGHT + 2,
              width: Math.max(geometry.width - INSET * 2, 0),
              height: DETAIL_HEIGHT,
            },
            style: {
              fontSize: 11,
              textAnchor: "middle",
            },
          }),
        );
      }
      return {
        kind: "group",
        id,
        capabilityId: capability,
        geometry,
        style,
        accessibility,
        fallback,
        sourceMap: unknownRange,
        children: [...desugared, ...children],
      };
    }

    case "core.header": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const title = stringProp(node, "title");
      const caption = stringProp(node, "caption");
      const desugared: RenderNodeIr[] = [];
      const half = Math.max(geometry.width / 2 - INSET, 0);
      if (title !== undefined) {
        desugared.push(
          textNode({
            id: `${id}-title`,
            text: title,
            geometry: {
              x: INSET,
              y: Math.max((geometry.height - HEADER_TEXT_HEIGHT) / 2, 0),
              width: half,
              height: HEADER_TEXT_HEIGHT,
            },
            style: {
              fontSize: 14,
              fontWeight: "bold",
              textAnchor: "start",
            },
          }),
        );
      }
      if (caption !== undefined) {
        desugared.push(
          textNode({
            id: `${id}-caption`,
            text: caption,
            geometry: {
              // Right-align into the trailing inset so long captions stay inside
              // the header instead of spilling past the viewport edge.
              x: Math.max(geometry.width - INSET - half, geometry.width / 2),
              y: Math.max((geometry.height - HEADER_TEXT_HEIGHT) / 2, 0),
              width: half,
              height: HEADER_TEXT_HEIGHT,
            },
            style: {
              fontSize: 11,
              textAnchor: "end",
            },
          }),
        );
      }
      return {
        kind: "group",
        id,
        capabilityId: capability,
        geometry,
        style,
        accessibility,
        fallback,
        sourceMap: unknownRange,
        children: [...desugared, ...children],
      };
    }

    case "core.bracket": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const side =
        stringProp(node, "side") ??
        (typeof style.side === "string" ? style.side : "left");
      const path = pathOf(node) ?? bracePath(geometry, side);
      return {
        kind: "connector",
        id,
        capabilityId: capability,
        geometry,
        style: {
          ...style,
          fill: "none",
          // Braces are undirected geometry — never inherit connector tips.
          markerEnd: "none",
          ...(style.stroke === undefined ? { strokeWidth: 1.5 } : {}),
        },
        accessibility,
        fallback,
        sourceMap: unknownRange,
        path,
        from: { x: geometry.x, y: geometry.y },
        to: {
          x: geometry.x + geometry.width,
          y: geometry.y + geometry.height,
        },
      };
    }

    case "core.callout": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const text =
        stringProp(node, "text") ??
        stringProp(node, "title") ??
        label;
      const target =
        pointOf(node.target) ??
        pointOf(node.to) ?? {
          x: geometry.x + geometry.width / 2,
          y: geometry.y + geometry.height + 24,
        };
      // Stem is a child under local layout — coordinates must be parent-local.
      const localAnchor = {
        x: geometry.width / 2,
        y: geometry.height,
      };
      const localTarget = {
        x: target.x - geometry.x,
        y: target.y - geometry.y,
      };
      const stemPath = `M${localAnchor.x} ${localAnchor.y} L${localTarget.x} ${localTarget.y}`;
      const textChild = textNode({
        id: `${id}-label`,
        text,
        geometry: {
          x: 0,
          y: 0,
          width: geometry.width,
          height: geometry.height,
        },
        style: {
          fontSize: 12,
          textAnchor: "middle",
        },
      });
      const stem: RenderNodeIr = {
        kind: "connector",
        id: `${id}-stem`,
        capabilityId: "core.path",
        geometry: {
          x: Math.min(localAnchor.x, localTarget.x),
          y: Math.min(localAnchor.y, localTarget.y),
          width: Math.abs(localTarget.x - localAnchor.x),
          height: Math.abs(localTarget.y - localAnchor.y),
        },
        style: {
          strokeWidth: 1.25,
          fill: "none",
          markerEnd: "none",
          ...(typeof style.stroke === "string" || typeof style.stroke === "number"
            ? { stroke: style.stroke }
            : {}),
        },
        accessibility: { label: `${text} stem` },
        fallback: `${text} stem`,
        sourceMap: unknownRange,
        path: stemPath,
        from: localAnchor,
        to: localTarget,
      };
      return {
        kind: "group",
        id,
        capabilityId: capability,
        geometry,
        style,
        accessibility,
        fallback,
        sourceMap: unknownRange,
        children: [textChild, stem, ...children],
      };
    }

    case "core.chip": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const text =
        stringProp(node, "text") ??
        stringProp(node, "title") ??
        label;
      const radius =
        finiteOrUndefined(style.radius) ??
        finiteOrUndefined(style.rx) ??
        9;
      const chrome: RenderNodeIr = {
        kind: "rect",
        id: `${id}-chrome`,
        capabilityId: "core.rect",
        geometry: {
          x: 0,
          y: 0,
          width: geometry.width,
          height: geometry.height,
        },
        style: {
          ...style,
          radius,
          rx: radius,
        },
        accessibility: { label: text },
        fallback: text,
        sourceMap: unknownRange,
      };
      const textChild = textNode({
        id: `${id}-label`,
        text,
        geometry: {
          x: 0,
          y: Math.max((geometry.height - 16) / 2, 0),
          width: geometry.width,
          height: 16,
        },
        style: {
          fontSize: 11,
          fontWeight: "bold",
          textAnchor: "middle",
        },
      });
      return {
        kind: "group",
        id,
        capabilityId: "core.group",
        geometry,
        style: {},
        accessibility,
        fallback,
        sourceMap: unknownRange,
        children: [chrome, textChild, ...children],
      };
    }

    case "core.note": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const text =
        stringProp(node, "text") ??
        stringProp(node, "caption") ??
        stringProp(node, "title") ??
        label;
      const chrome: RenderNodeIr = {
        kind: "rect",
        id: `${id}-chrome`,
        capabilityId: "core.rect",
        geometry: {
          x: 0,
          y: 0,
          width: geometry.width,
          height: geometry.height,
        },
        style: {
          fill: "@theme.surface.elevated",
          stroke: "@theme.ink.secondary",
          strokeWidth: 1,
          radius: 6,
          ...style,
        },
        accessibility: { label: text },
        fallback: text,
        sourceMap: unknownRange,
      };
      const textChild = textNode({
        id: `${id}-caption`,
        text,
        geometry: {
          x: 8,
          y: Math.max((geometry.height - 14) / 2, 0),
          width: Math.max(geometry.width - 16, 0),
          height: 14,
        },
        style: {
          fontSize: 11,
          textAnchor: "middle",
          fill: "@theme.ink.secondary",
        },
      });
      return {
        kind: "group",
        id,
        capabilityId: "core.group",
        geometry,
        style: {},
        accessibility,
        fallback,
        sourceMap: unknownRange,
        children: [chrome, textChild, ...children],
      };
    }

    case "core.divider": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const axis =
        stringProp(node, "axis") ??
        (typeof style.axis === "string" ? style.axis : undefined) ??
        (geometry.width >= geometry.height ? "x" : "y");
      const path =
        pathOf(node) ??
        (axis === "y"
          ? `M${geometry.x + geometry.width / 2} ${geometry.y} V${geometry.y + geometry.height}`
          : `M${geometry.x} ${geometry.y + geometry.height / 2} H${geometry.x + geometry.width}`);
      return {
        kind: "connector",
        id,
        capabilityId: capability,
        geometry,
        style: {
          ...style,
          fill: "none",
          markerEnd: "none",
          strokeWidth: style.strokeWidth ?? 1.2,
        },
        accessibility,
        fallback,
        sourceMap: unknownRange,
        path,
        from: {
          x: geometry.x,
          y: geometry.y + geometry.height / 2,
        },
        to: {
          x: geometry.x + geometry.width,
          y: geometry.y + geometry.height / 2,
        },
      };
    }

    case "core.lane": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const title = stringProp(node, "title") ?? stringProp(node, "text") ?? label;
      const desugared: RenderNodeIr[] = [
        {
          kind: "rect",
          id: `${id}-chrome`,
          capabilityId: "core.rect",
          geometry: {
            x: 0,
            y: 0,
            width: geometry.width,
            height: geometry.height,
          },
          style: {
            fill: "@theme.surface.secondary",
            stroke: "@theme.ink.tertiary",
            strokeWidth: 1,
            radius: 8,
            ...style,
          },
          accessibility: { label: title },
          fallback: title,
          sourceMap: unknownRange,
        },
      ];
      if (title.length > 0) {
        desugared.push(
          textNode({
            id: `${id}-title`,
            text: title,
            geometry: {
              x: INSET,
              y: INSET,
              width: Math.max(geometry.width - INSET * 2, 0),
              height: TITLE_HEIGHT,
            },
            style: {
              fontSize: 12,
              fontWeight: "bold",
              textAnchor: "start",
            },
          }),
        );
      }
      return {
        kind: "group",
        id,
        capabilityId: capability,
        geometry,
        style: {},
        accessibility,
        fallback,
        sourceMap: unknownRange,
        children: [...desugared, ...children],
      };
    }

    case "core.band": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const title = stringProp(node, "title") ?? stringProp(node, "text");
      const chrome: RenderNodeIr = {
        kind: "rect",
        id: title !== undefined ? `${id}-chrome` : id,
        capabilityId: title !== undefined ? "core.rect" : capability,
        geometry:
          title !== undefined
            ? {
                x: 0,
                y: 0,
                width: geometry.width,
                height: geometry.height,
              }
            : geometry,
        style: {
          fill: "@theme.surface.secondary",
          stroke: "none",
          radius: 6,
          ...style,
        },
        accessibility: { label: title ?? label },
        fallback: title ?? fallback,
        sourceMap: unknownRange,
      };
      if (title === undefined) {
        return chrome;
      }
      return {
        kind: "group",
        id,
        capabilityId: capability,
        geometry,
        style: {},
        accessibility,
        fallback,
        sourceMap: unknownRange,
        children: [
          chrome,
          textNode({
            id: `${id}-title`,
            text: title,
            geometry: {
              x: INSET,
              y: INSET,
              width: Math.max(geometry.width - INSET * 2, 0),
              height: TITLE_HEIGHT,
            },
            style: {
              fontSize: 11,
              fontWeight: "bold",
              fill: "@theme.ink.secondary",
              textAnchor: "start",
            },
          }),
          ...children,
        ],
      };
    }

    case "core.swimlane": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const gap =
        finiteOrUndefined(node.gap) ??
        finiteOrUndefined(style.gap) ??
        8;
      const labelWidth =
        finiteOrUndefined(node.labelWidth) ??
        finiteOrUndefined(style.labelWidth) ??
        72;
      const labels = stringArrayProp(node, "labels");
      const rows = children.length > 0 ? children : [];
      const rowCount = Math.max(rows.length, labels.length, 1);
      const usableHeight = Math.max(geometry.height - gap * (rowCount - 1), 0);
      const rowHeight = rowCount > 0 ? usableHeight / rowCount : geometry.height;
      const contentWidth = Math.max(geometry.width - labelWidth - INSET, 0);
      const laidOut: RenderNodeIr[] = [];
      for (let index = 0; index < rowCount; index += 1) {
        const y = index * (rowHeight + gap);
        const labelText = labels[index];
        if (typeof labelText === "string" && labelText.length > 0) {
          laidOut.push(
            textNode({
              id: `${id}-label-${index}`,
              text: labelText,
              geometry: {
                x: 0,
                y: y + Math.max((rowHeight - 14) / 2, 0),
                width: labelWidth,
                height: 14,
              },
              style: {
                fontSize: 11,
                fontWeight: "bold",
                textAnchor: "end",
                fill: "@theme.ink.secondary",
              },
            }),
          );
        }
        const child = rows[index];
        if (child !== undefined) {
          laidOut.push({
            ...child,
            geometry: {
              ...child.geometry,
              x: labelWidth + INSET,
              y,
              width:
                child.geometry.width > 0 ? child.geometry.width : contentWidth,
              height:
                child.geometry.height > 0 ? child.geometry.height : rowHeight,
            },
          });
        } else {
          laidOut.push({
            kind: "rect",
            id: `${id}-row-${index}`,
            capabilityId: "core.rect",
            geometry: {
              x: labelWidth + INSET,
              y,
              width: contentWidth,
              height: rowHeight,
            },
            style: {
              fill: "@theme.surface.elevated",
              stroke: "@theme.ink.tertiary",
              strokeWidth: 1,
              radius: 6,
            },
            accessibility: { label: labelText ?? `${label} row ${index + 1}` },
            fallback: labelText ?? `${label} row ${index + 1}`,
            sourceMap: unknownRange,
          });
        }
      }
      return {
        kind: "group",
        id,
        capabilityId: capability,
        geometry,
        style,
        accessibility,
        fallback,
        sourceMap: unknownRange,
        children: laidOut,
      };
    }

    case "core.stepper": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const linked =
        node.linked === true ||
        node.linked === 1 ||
        node.linked === "true" ||
        style.linked === true ||
        style.linked === 1 ||
        style.linked === "true";
      const stepTexts = stringArrayProp(node, "steps");
      const chipHeight = 26;
      const gap = finiteOrUndefined(style.gap) ?? 12;
      const stepCount =
        stepTexts.length > 0 ? stepTexts.length : Math.max(children.length, 1);
      const railWidth =
        geometry.width > 0
          ? geometry.width
          : stepCount * 90 + gap * Math.max(stepCount - 1, 0);
      const slotWidth =
        stepCount > 0
          ? Math.max((railWidth - gap * Math.max(stepCount - 1, 0)) / stepCount, 0)
          : 0;
      const chips: RenderNodeIr[] =
        stepTexts.length > 0
          ? stepTexts.map((text, index) => {
              const chipId = `${id}-step-${index}`;
              return {
                kind: "group",
                id: chipId,
                capabilityId: "core.group",
                geometry: {
                  x: 0,
                  y: 0,
                  width: slotWidth,
                  height: chipHeight,
                },
                style: {},
                accessibility: { label: text },
                fallback: text,
                sourceMap: unknownRange,
                children: [
                  {
                    kind: "rect",
                    id: `${chipId}-chrome`,
                    capabilityId: "core.rect",
                    geometry: {
                      x: 0,
                      y: 0,
                      width: slotWidth,
                      height: chipHeight,
                    },
                    style: {
                      fill: "@theme.surface.elevated",
                      stroke: "@theme.ink.tertiary",
                      strokeWidth: 1.2,
                      radius: 9,
                    },
                    accessibility: { label: text },
                    fallback: text,
                    sourceMap: unknownRange,
                  },
                  textNode({
                    id: `${chipId}-label`,
                    text: `${index + 1}. ${text}`,
                    geometry: {
                      x: 0,
                      y: Math.max((chipHeight - 16) / 2, 0),
                      width: slotWidth,
                      height: 16,
                    },
                    style: {
                      fontSize: 11,
                      fontWeight: "bold",
                      textAnchor: "middle",
                    },
                  }),
                ],
              };
            })
          : children.map((child) => ({
              ...child,
              geometry: {
                ...child.geometry,
                width:
                  child.geometry.width > 0 ? child.geometry.width : slotWidth,
                height:
                  child.geometry.height > 0
                    ? child.geometry.height
                    : chipHeight,
              },
            }));
      const rail: RenderNodeIr = {
        kind: "group",
        id: `${id}-rail`,
        capabilityId: "layout.rail",
        geometry: {
          x: 0,
          y: 0,
          width: railWidth,
          height: geometry.height > 0 ? geometry.height : chipHeight,
        },
        style: {
          direction: "row",
          gap,
        },
        accessibility: { label: `${label} rail` },
        fallback: `${label} rail`,
        sourceMap: unknownRange,
        children: chips,
      };
      const links: RenderNodeIr[] = [];
      if (linked && chips.length > 1) {
        for (let index = 0; index < chips.length - 1; index += 1) {
          const fromId = chips[index]?.id;
          const toId = chips[index + 1]?.id;
          if (fromId === undefined || toId === undefined) {
            continue;
          }
          links.push({
            kind: "connector",
            id: `${id}-link-${index}`,
            capabilityId: "core.route",
            geometry: { x: 0, y: 0, width: 0, height: 0 },
            style: {
              stroke: "@theme.ink.tertiary",
              strokeWidth: 1.5,
              fill: "none",
              route: "elbow",
              markerEnd: "none",
            },
            accessibility: { label: `${label} link ${index + 1}` },
            fallback: `${label} link ${index + 1}`,
            sourceMap: unknownRange,
            from: { nodeId: fromId, anchor: "e" },
            to: { nodeId: toId, anchor: "w" },
          });
        }
      }
      return {
        kind: "group",
        id,
        capabilityId: capability,
        geometry: {
          ...geometry,
          width: geometry.width > 0 ? geometry.width : railWidth,
          height: geometry.height > 0 ? geometry.height : chipHeight,
        },
        style,
        accessibility,
        fallback,
        sourceMap: unknownRange,
        children: [rail, ...links],
      };
    }

    case "layout.pad": {
      const geometry = geometryOf(node);
      const style = styleOf(node);
      const inset =
        finiteOrUndefined(node.inset) ??
        finiteOrUndefined(node.pad) ??
        finiteOrUndefined(style.inset) ??
        finiteOrUndefined(style.pad) ??
        finiteOrUndefined(style.gap) ??
        DEFAULT_PAD;
      const paddedChildren = children.map((child) => ({
        ...child,
        geometry: {
          ...child.geometry,
          x: child.geometry.x + inset,
          y: child.geometry.y + inset,
        },
      }));
      return {
        kind: "group",
        id,
        capabilityId: capability,
        geometry,
        style: { ...style, pad: inset },
        accessibility,
        fallback,
        sourceMap: unknownRange,
        children: paddedChildren,
      };
    }

    default:
      return undefined;
  }
}

/**
 * Build a first-class (non-macro) RenderNodeIr from a package authoring node.
 * Handles absolute `core.arrow` desugar into connector + arrowhead defaults.
 */
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

  // Absolute arrow → connector with arrowhead style defaults.
  if (capability === "core.arrow" && arrowHasAbsoluteGeometry(node)) {
    kind = "connector";
    if (style.markerEnd === undefined && style.arrowhead === undefined) {
      style = { ...style, arrowhead: true, markerEnd: "arrow" };
    }
  }

  // Groups forced when children present (except connectors keep connector kind
  // only when no children — stack/grid/panel already handled as group).
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
    const label = fanLabel(node, id);
    if (capability !== "core.fan-out" && capability !== "core.fan-in") {
      throw new Error(
        `${label} capability must be "core.fan-out" or "core.fan-in"`,
      );
    }
    const from = requireFanEndpointSide(node.from, label, "from");
    const to = requireFanEndpointSide(node.to, label, "to");
    const axis = requireFanAxis(node, style, label);
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
    // Elbow / route distinguishable via capabilityId; also stamp style.route.
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
      from,
      to,
      ...(via !== undefined ? { via } : {}),
      ...(axis !== undefined ? { axis } : {}),
    };
  }
  return { ...base, kind: "rect" };
}
