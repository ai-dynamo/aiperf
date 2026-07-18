/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import { createElement } from "react";

/** Tip geometry kinds supported by SceneRenderer markers. */
export type MarkerTipKind =
  | "triangle"
  | "vee"
  | "circle"
  | "square"
  | "diamond"
  | "tee";

export type MarkerTipFill = "filled" | "open";
export type MarkerTipSize = "sm" | "md" | "lg";

/** Resolved tip used for SVG marker defs and path-end inset. */
export type ResolvedMarkerTip = Readonly<{
  kind: MarkerTipKind;
  fill: MarkerTipFill;
  size: MarkerTipSize;
  /** Length along the stroke axis in markerUnits=strokeWidth space. */
  insetUnits: number;
  /** Stable key for marker id / defs dedupe. */
  key: string;
}>;

/** Object form of `style.markerEnd`. */
export type MarkerEndObject = Readonly<{
  kind?: string;
  fill?: string;
  size?: string;
}>;

/** Authored `style.markerEnd` value. */
export type MarkerEndAuthored =
  | string
  | number
  | boolean
  | MarkerEndObject
  | null
  | undefined;

/** Size → tip length in strokeWidth units (md matches the legacy triangle). */
export const TIP_SIZE_UNITS: Readonly<Record<MarkerTipSize, number>> = {
  sm: 3.5,
  md: 6,
  lg: 9,
};

const KIND_ALIASES: Readonly<Record<string, MarkerTipKind>> = {
  triangle: "triangle",
  arrow: "triangle",
  vee: "vee",
  chevron: "vee",
  circle: "circle",
  square: "square",
  diamond: "diamond",
  tee: "tee",
  bar: "tee",
};

/**
 * Default tip when a directed edge shows a head but authors omit markerEnd.
 * Smaller than the legacy md triangle.
 */
export const DEFAULT_MARKER_TIP: ResolvedMarkerTip = resolveTipParts(
  "triangle",
  "filled",
  "sm",
);

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeKind(raw: string | undefined): MarkerTipKind | undefined {
  if (raw === undefined) {
    return undefined;
  }
  return KIND_ALIASES[raw.trim().toLowerCase()];
}

function normalizeFill(
  raw: string | undefined,
  kind: MarkerTipKind,
): MarkerTipFill {
  if (raw === undefined) {
    return kind === "vee" ? "open" : "filled";
  }
  const token = raw.trim().toLowerCase();
  if (
    token === "open" ||
    token === "none" ||
    token === "stroke" ||
    token === "hollow"
  ) {
    return "open";
  }
  return "filled";
}

function normalizeSize(raw: string | undefined): MarkerTipSize {
  if (raw === undefined) {
    return "sm";
  }
  const token = raw.trim().toLowerCase();
  if (token === "md" || token === "medium") {
    return "md";
  }
  if (token === "lg" || token === "large") {
    return "lg";
  }
  return "sm";
}

function insetUnitsFor(kind: MarkerTipKind, size: MarkerTipSize): number {
  const base = TIP_SIZE_UNITS[size];
  switch (kind) {
    case "tee":
      // Thin bar; only a small stroke-end reserve is needed.
      return Math.max(1, base * 0.2);
    case "circle":
    case "square":
    case "diamond":
      // Diameter / side length along the path.
      return base;
    case "triangle":
    case "vee":
    default:
      return base;
  }
}

function tipKey(
  kind: MarkerTipKind,
  fill: MarkerTipFill,
  size: MarkerTipSize,
): string {
  return `${kind}-${fill}-${size}`;
}

function resolveTipParts(
  kind: MarkerTipKind,
  fill: MarkerTipFill,
  size: MarkerTipSize,
): ResolvedMarkerTip {
  return {
    kind,
    fill,
    size,
    insetUnits: insetUnitsFor(kind, size),
    key: tipKey(kind, fill, size),
  };
}

/** True when markerEnd explicitly disables a tip. */
export function isMarkerEndNone(markerEnd: MarkerEndAuthored): boolean {
  if (markerEnd === undefined || markerEnd === null) {
    return false;
  }
  if (markerEnd === false || markerEnd === 0) {
    return true;
  }
  if (typeof markerEnd === "string") {
    const token = markerEnd.trim().toLowerCase();
    return token === "none" || token === "false" || token === "0";
  }
  if (isPlainObject(markerEnd)) {
    const kind = markerEnd.kind;
    if (typeof kind === "string") {
      const token = kind.trim().toLowerCase();
      return token === "none" || token === "false";
    }
  }
  return false;
}

/**
 * Parse authored `markerEnd` into a tip, or `null` when the tip is disabled.
 * When `markerEnd` is absent, returns `fallback` (default triangle/sm).
 */
export function resolveMarkerTip(
  markerEnd: MarkerEndAuthored,
  fallback: ResolvedMarkerTip | null = DEFAULT_MARKER_TIP,
): ResolvedMarkerTip | null {
  if (markerEnd === undefined || markerEnd === null) {
    return fallback;
  }
  if (isMarkerEndNone(markerEnd)) {
    return null;
  }

  if (typeof markerEnd === "string") {
    const token = markerEnd.trim().toLowerCase();
    // Compound tokens: "circle-open", "square-open", "diamond-open"
    const openMatch =
      /^(circle|square|diamond)-(open|filled)$/.exec(token);
    if (openMatch?.[1] !== undefined && openMatch[2] !== undefined) {
      const kind = normalizeKind(openMatch[1]);
      if (kind !== undefined) {
        return resolveTipParts(
          kind,
          openMatch[2] === "open" ? "open" : "filled",
          "sm",
        );
      }
    }
    // Plan aliases: triangle-open / vee / chevron → open chevron.
    if (
      token === "vee" ||
      token === "chevron" ||
      token === "triangle-open"
    ) {
      return resolveTipParts("vee", "open", "sm");
    }
    if (token === "triangle-filled" || token === "arrow-filled") {
      return resolveTipParts("triangle", "filled", "sm");
    }
    const kind = normalizeKind(token);
    if (kind !== undefined) {
      return resolveTipParts(kind, normalizeFill(undefined, kind), "sm");
    }
    // Unknown string → treat as default tip (legacy / forward-compat).
    return fallback;
  }

  if (typeof markerEnd === "number" || typeof markerEnd === "boolean") {
    return markerEnd ? fallback : null;
  }

  if (isPlainObject(markerEnd)) {
    const kindRaw =
      typeof markerEnd.kind === "string" ? markerEnd.kind : undefined;
    const fillRaw =
      typeof markerEnd.fill === "string" ? markerEnd.fill : undefined;
    const sizeRaw =
      typeof markerEnd.size === "string" ? markerEnd.size : undefined;
    const kind = normalizeKind(kindRaw) ?? "triangle";
    return resolveTipParts(
      kind,
      normalizeFill(fillRaw, kind),
      normalizeSize(sizeRaw),
    );
  }

  return fallback;
}

/** User-space inset for path shortening: tip units × stroke width. */
export function tipInsetUserUnits(
  tip: ResolvedMarkerTip,
  strokeWidth: number,
): number {
  const width =
    typeof strokeWidth === "number" && Number.isFinite(strokeWidth)
      ? strokeWidth
      : 2.2;
  return tip.insetUnits * width;
}

function tipPaint(
  fill: MarkerTipFill,
): Readonly<{ fill: string; stroke: string; strokeWidth?: number }> {
  if (fill === "open") {
    return {
      fill: "none",
      stroke: "context-stroke",
      strokeWidth: 1,
    };
  }
  return {
    fill: "context-stroke",
    stroke: "none",
  };
}

/**
 * Build the SVG marker element contents for a resolved tip.
 * Geometry sits in markerUnits=strokeWidth space with tip tip at +insetUnits.
 */
export function markerGeometry(
  tip: ResolvedMarkerTip,
): Readonly<{
  markerWidth: number;
  markerHeight: number;
  refX: number;
  refY: number;
  children: ReactNode;
}> {
  const u = tip.insetUnits;
  const paint = tipPaint(tip.fill);
  const half = u / 2;
  const midY = half;

  switch (tip.kind) {
    case "circle": {
      const r = half;
      return {
        markerWidth: u + 1,
        markerHeight: u + 1,
        refX: 0,
        refY: midY,
        children: createElement("circle", {
          cx: r,
          cy: midY,
          r: Math.max(0.5, r - (tip.fill === "open" ? 0.5 : 0)),
          fill: paint.fill,
          stroke: paint.stroke,
          strokeWidth: paint.strokeWidth,
          focusable: false,
        }),
      };
    }
    case "square": {
      const pad = tip.fill === "open" ? 0.5 : 0;
      return {
        markerWidth: u + 1,
        markerHeight: u + 1,
        refX: 0,
        refY: midY,
        children: createElement("rect", {
          x: pad,
          y: pad,
          width: Math.max(0.5, u - pad * 2),
          height: Math.max(0.5, u - pad * 2),
          fill: paint.fill,
          stroke: paint.stroke,
          strokeWidth: paint.strokeWidth,
          focusable: false,
        }),
      };
    }
    case "diamond": {
      const d = `M0,${midY} L${half},0 L${u},${midY} L${half},${u} Z`;
      return {
        markerWidth: u + 1,
        markerHeight: u + 1,
        refX: 0,
        refY: midY,
        children: createElement("path", {
          d,
          fill: paint.fill,
          stroke: paint.stroke,
          strokeWidth: paint.strokeWidth,
          focusable: false,
        }),
      };
    }
    case "tee": {
      // Vertical bar at the path end (perpendicular stop).
      const thickness = Math.max(0.6, u * 0.2);
      return {
        markerWidth: thickness + 1,
        markerHeight: u + 1,
        refX: 0,
        refY: midY,
        children: createElement("rect", {
          x: 0,
          y: 0,
          width: thickness,
          height: u,
          fill: paint.fill === "none" ? "context-stroke" : paint.fill,
          stroke: "none",
          focusable: false,
        }),
      };
    }
    case "vee": {
      // Open chevron pointing along +x; base at x=0.
      const d = `M0,0 L${u},${midY} L0,${u}`;
      return {
        markerWidth: u + 1,
        markerHeight: u + 1,
        refX: 0,
        refY: midY,
        children: createElement("path", {
          d,
          fill: "none",
          stroke: "context-stroke",
          strokeWidth: 1,
          strokeLinejoin: "miter",
          focusable: false,
        }),
      };
    }
    case "triangle":
    default: {
      const d = `M0,0 L${u},${midY} L0,${u} Z`;
      return {
        markerWidth: u + 1,
        markerHeight: u + 1,
        refX: 0,
        refY: midY,
        children: createElement("path", {
          d,
          fill: paint.fill,
          stroke: paint.stroke,
          strokeWidth: paint.strokeWidth,
          focusable: false,
        }),
      };
    }
  }
}

/** Stable SVG marker id for a tip within a scene. */
export function markerDomId(prefix: string, tip: ResolvedMarkerTip): string {
  return `${prefix}-${tip.key}`;
}
