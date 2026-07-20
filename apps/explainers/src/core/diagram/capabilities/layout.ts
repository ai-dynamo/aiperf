/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure deterministic layout for native semantic Scene IR capabilities.

import type {
  SceneGeometryLike,
  SceneNodeLike,
  SceneStyleValue,
} from "../SceneRenderer.js";
import type { CapabilityLayout, NativeSceneCapability } from "./types.js";

const LANE_TITLE_BAND = 28;
const LANE_INSET = 10;
const DEFAULT_CHILD_HEIGHT = 64;
const DEFAULT_GAP = 8;
const SWIMLANE_LABEL_WIDTH = 72;
const STEPPER_CHIP_HEIGHT = 26;
const STEPPER_MIN_CHIP_WIDTH = 72;
const STEPPER_CHAR_WIDTH = 6.2;
const STEPPER_CHIP_PAD = 24;

function geometryOf(node: SceneNodeLike): SceneGeometryLike {
  const geometry = node.geometry ?? node.layout;
  return {
    x: finite(geometry?.x),
    y: finite(geometry?.y),
    width: nonnegative(geometry?.width),
    height: nonnegative(geometry?.height),
  };
}

function finite(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function nonnegative(value: unknown, fallback = 0): number {
  return Math.max(0, finite(value, fallback));
}

function styleNumber(
  style: Readonly<Record<string, SceneStyleValue>> | undefined,
  key: string,
  fallback: number,
): number {
  return nonnegative(style?.[key], fallback);
}

function propNumber(node: SceneNodeLike, key: string, fallback: number): number {
  return nonnegative(node.props?.[key], fallback);
}

function recordProp(
  node: SceneNodeLike,
  key: string,
): Readonly<Record<string, unknown>> | undefined {
  const value = node.props?.[key];
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Readonly<Record<string, unknown>>)
    : undefined;
}

function directionOf(node: SceneNodeLike): "row" | "column" {
  return node.style?.direction === "row" ? "row" : "column";
}

function childBounds(children: readonly SceneGeometryLike[]): Readonly<{
  width: number;
  height: number;
}> {
  return children.reduce(
    (extent, child) => ({
      width: Math.max(extent.width, child.x + child.width),
      height: Math.max(extent.height, child.y + child.height),
    }),
    { width: 0, height: 0 },
  );
}

/** Identity layout for leaf capabilities and already-positioned groups. */
export function resolveIdentityLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const bounds = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  if (node.style?.overflow === "hidden" || node.style?.clip === true) {
    return { bounds, childGeometries };
  }
  const extent = childBounds(childGeometries);
  return {
    bounds: {
      ...bounds,
      width: Math.max(bounds.width, extent.width),
      height: Math.max(bounds.height, extent.height),
    },
    childGeometries,
  };
}

/** Ordered stack layout with authored dimensions treated as minimums. */
export function resolveStackLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const direction = directionOf(node);
  const gap = styleNumber(node.style, "gap", 0);
  let cursor = 0;
  let cross = 0;
  const childGeometries = children.map((child) => {
    const geometry = geometryOf(child);
    const placed =
      direction === "row"
        ? { ...geometry, x: cursor, y: 0 }
        : { ...geometry, x: 0, y: cursor };
    cursor += (direction === "row" ? geometry.width : geometry.height) + gap;
    cross = Math.max(
      cross,
      direction === "row" ? geometry.height : geometry.width,
    );
    return placed;
  });
  if (childGeometries.length > 0) {
    cursor = Math.max(0, cursor - gap);
  }
  const contentWidth = direction === "row" ? cursor : cross;
  const contentHeight = direction === "column" ? cursor : cross;
  return {
    bounds: {
      ...authored,
      width: Math.max(authored.width, contentWidth),
      height: Math.max(authored.height, contentHeight),
    },
    childGeometries,
  };
}

/** Row-major grid layout with per-column and per-row intrinsic dimensions. */
export function resolveGridLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const cols = Math.max(1, Math.floor(styleNumber(node.style, "cols", 1)));
  const gap = styleNumber(node.style, "gap", 0);
  const rows = Math.max(1, Math.ceil(children.length / cols));
  const widths = Array.from({ length: cols }, () => 0);
  const heights = Array.from({ length: rows }, () => 0);
  const childAuthored = children.map(geometryOf);
  childAuthored.forEach((geometry, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    widths[col] = Math.max(widths[col]!, geometry.width);
    heights[row] = Math.max(heights[row]!, geometry.height);
  });
  const xOffsets: number[] = [];
  const yOffsets: number[] = [];
  let x = 0;
  let y = 0;
  widths.forEach((width) => {
    xOffsets.push(x);
    x += width + gap;
  });
  heights.forEach((height) => {
    yOffsets.push(y);
    y += height + gap;
  });
  const childGeometries = childAuthored.map((geometry, index) => ({
    ...geometry,
    x: xOffsets[index % cols]!,
    y: yOffsets[Math.floor(index / cols)]!,
  }));
  const contentWidth = Math.max(0, x - (children.length > 0 ? gap : 0));
  const contentHeight = Math.max(0, y - (children.length > 0 ? gap : 0));
  return {
    bounds: {
      ...authored,
      width: Math.max(authored.width, contentWidth),
      height: Math.max(authored.height, contentHeight),
    },
    childGeometries,
  };
}

/** Equal-slot rail that first expands to the children's intrinsic minimum. */
export function resolveRailLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  if (children.length === 0) {
    return { bounds: authored, childGeometries: [] };
  }
  const direction = directionOf(node);
  const gap = styleNumber(node.style, "gap", 0);
  const childAuthored = children.map(geometryOf);
  const totalGap = gap * Math.max(children.length - 1, 0);
  const minWidth =
    childAuthored.reduce((sum, geometry) => sum + geometry.width, 0) +
    (direction === "row" ? totalGap : 0);
  const minHeight =
    childAuthored.reduce((sum, geometry) => sum + geometry.height, 0) +
    (direction === "column" ? totalGap : 0);
  const maxWidth = Math.max(...childAuthored.map((geometry) => geometry.width));
  const maxHeight = Math.max(...childAuthored.map((geometry) => geometry.height));
  const width = Math.max(
    authored.width,
    direction === "row" ? minWidth : maxWidth,
  );
  const height = Math.max(
    authored.height,
    direction === "column" ? minHeight : maxHeight,
  );
  const slot =
    direction === "row"
      ? Math.max((width - totalGap) / children.length, 0)
      : Math.max((height - totalGap) / children.length, 0);
  const childGeometries = childAuthored.map((geometry, index) =>
    direction === "row"
      ? {
          x: index * (slot + gap),
          y: 0,
          width: slot,
          height: geometry.height > 0 ? geometry.height : height,
        }
      : {
          x: 0,
          y: index * (slot + gap),
          width: geometry.width > 0 ? geometry.width : width,
          height: slot,
        },
  );
  return {
    bounds: { ...authored, width, height },
    childGeometries,
  };
}

/** Titled lane layout with content-aware vertical expansion. */
export function resolveLaneLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const gap = styleNumber(node.style, "gap", DEFAULT_GAP);
  let cursorY = LANE_TITLE_BAND;
  const childGeometries = children.map((child) => {
    const geometry = geometryOf(child);
    const width =
      geometry.width > 0
        ? geometry.width
        : Math.max(authored.width - LANE_INSET * 2, 0);
    const height = geometry.height > 0 ? geometry.height : DEFAULT_CHILD_HEIGHT;
    const placed = {
      x: LANE_INSET,
      y: cursorY,
      width,
      height,
    };
    cursorY += height + gap;
    return placed;
  });
  const contentHeight =
    childGeometries.length > 0 ? cursorY - gap + LANE_INSET : LANE_TITLE_BAND;
  const extent = childBounds(childGeometries);
  return {
    bounds: {
      ...authored,
      width: Math.max(authored.width, extent.width + LANE_INSET),
      height: Math.max(authored.height, contentHeight),
    },
    childGeometries,
  };
}

/** Swimlane rows reserve a label gutter and expand around intrinsic rows. */
export function resolveSwimlaneLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const gap = styleNumber(node.style, "gap", DEFAULT_GAP);
  const labelWidth = propNumber(
    node,
    "labelWidth",
    styleNumber(node.style, "labelWidth", SWIMLANE_LABEL_WIDTH),
  );
  let cursorY = 0;
  const childGeometries = children.map((child) => {
    const geometry = geometryOf(child);
    const height = geometry.height > 0 ? geometry.height : DEFAULT_CHILD_HEIGHT;
    const width =
      geometry.width > 0
        ? geometry.width
        : Math.max(authored.width - labelWidth - LANE_INSET, 0);
    const placed = {
      x: labelWidth + LANE_INSET,
      y: cursorY,
      width,
      height,
    };
    cursorY += height + gap;
    return placed;
  });
  const contentHeight =
    childGeometries.length > 0 ? cursorY - gap : DEFAULT_CHILD_HEIGHT;
  const extent = childBounds(childGeometries);
  return {
    bounds: {
      ...authored,
      width: Math.max(authored.width, extent.width),
      height: Math.max(authored.height, contentHeight),
    },
    childGeometries,
  };
}

function stringArrayProp(node: SceneNodeLike, key: string): readonly string[] {
  const value = node.props?.[key];
  return Array.isArray(value)
    ? value.filter(
        (entry): entry is string =>
          typeof entry === "string" && entry.length > 0,
      )
    : [];
}

function stepperChipWidth(label: string, index: number): number {
  const text = `${index + 1}. ${label}`;
  return Math.max(
    STEPPER_MIN_CHIP_WIDTH,
    Math.ceil(text.length * STEPPER_CHAR_WIDTH) + STEPPER_CHIP_PAD,
  );
}

/** Semantic stepper intrinsic width derived from numbered label content. */
export function resolveStepperLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const gap = styleNumber(node.style, "gap", 12);
  const labels = stringArrayProp(node, "steps");
  const widths =
    labels.length > 0
      ? labels.map(stepperChipWidth)
      : children.map((child, index) => {
          const geometry = geometryOf(child);
          return geometry.width > 0
            ? geometry.width
            : stepperChipWidth(
                child.accessibility?.label ?? `step ${index + 1}`,
                index,
              );
        });
  let cursorX = 0;
  const childGeometries = children.map((child, index) => {
    const geometry = geometryOf(child);
    const width = widths[index] ?? STEPPER_MIN_CHIP_WIDTH;
    const placed = {
      x: cursorX,
      y: 0,
      width,
      height: geometry.height > 0 ? geometry.height : STEPPER_CHIP_HEIGHT,
    };
    cursorX += width + (index < children.length - 1 ? gap : 0);
    return placed;
  });
  const intrinsicWidth =
    widths.reduce((sum, width) => sum + width, 0) +
    gap * Math.max(widths.length - 1, 0);
  return {
    bounds: {
      ...authored,
      width: Math.max(authored.width, intrinsicWidth),
      height: Math.max(authored.height, STEPPER_CHIP_HEIGHT),
    },
    childGeometries,
  };
}

/** Insets authored children and expands around the padded content. */
export function resolvePadLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const inset = propNumber(
    node,
    "inset",
    styleNumber(node.style, "inset", styleNumber(node.style, "pad", 12)),
  );
  const childGeometries = children.map((child) => {
    const geometry = geometryOf(child);
    return { ...geometry, x: geometry.x + inset, y: geometry.y + inset };
  });
  const extent = childBounds(childGeometries);
  return {
    bounds: {
      ...authored,
      width: Math.max(authored.width, extent.width + inset),
      height: Math.max(authored.height, extent.height + inset),
    },
    childGeometries,
  };
}

/** Resolve native circle/ellipse center and radii into an SVG layout box. */
export function resolveEllipseLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  if (authored.width > 0 || authored.height > 0) {
    return resolveIdentityLayout(node, children);
  }
  const center = recordProp(node, "center");
  const cx = finite(center?.x, authored.x);
  const cy = finite(center?.y, authored.y);
  const radius = propNumber(
    node,
    "r",
    styleNumber(node.style, "r", 0),
  );
  const rx = propNumber(
    node,
    "rx",
    styleNumber(node.style, "rx", radius),
  );
  const ry = propNumber(
    node,
    "ry",
    styleNumber(node.style, "ry", rx),
  );
  return {
    bounds: {
      x: cx - rx,
      y: cy - ry,
      width: rx * 2,
      height: ry * 2,
    },
    childGeometries: children.map(geometryOf),
  };
}

export const LAYOUT_CAPABILITIES: readonly NativeSceneCapability[] = [
  { capabilityId: "layout.stack", resolveLayout: resolveStackLayout },
  { capabilityId: "layout.grid", resolveLayout: resolveGridLayout },
  { capabilityId: "layout.rail", resolveLayout: resolveRailLayout },
  { capabilityId: "layout.pad", resolveLayout: resolvePadLayout },
  { capabilityId: "core.lane", resolveLayout: resolveLaneLayout },
  { capabilityId: "core.band", resolveLayout: resolveIdentityLayout },
  { capabilityId: "core.swimlane", resolveLayout: resolveSwimlaneLayout },
  { capabilityId: "core.stepper", resolveLayout: resolveStepperLayout },
  { capabilityId: "core.circle", resolveLayout: resolveEllipseLayout },
  { capabilityId: "core.ellipse", resolveLayout: resolveEllipseLayout },
];

