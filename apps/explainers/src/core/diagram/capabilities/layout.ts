/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure deterministic layout for native semantic Scene IR capabilities.

import type {
  SceneGeometryLike,
  SceneNodeLike,
  SceneStyleValue,
} from "../scene-types.js";
import {
  CHIP_PAD_X,
  DEFAULT_SCENE_FONT_SIZE,
  DETAIL_HEIGHT,
  INSET,
  STEPPER_CHIP_HEIGHT,
  STEPPER_MIN_CHIP_WIDTH,
  SUBTITLE_HEIGHT,
  TITLE_HEIGHT,
  estimateTextWidth,
  scaledSceneFontSize,
  stepperChipWidth,
} from "../text-metrics.js";
import type {
  CapabilityLayout,
  CapabilityLayoutDiagnostic,
  NativeSceneCapability,
} from "./types.js";

const LANE_TITLE_BAND = 32;
const LANE_INSET = 10;
const DEFAULT_CHILD_HEIGHT = 64;
const DEFAULT_GAP = 8;
const SWIMLANE_LABEL_WIDTH = 72;
const FRAME_TITLE_BAND = 32;
const FRAME_DETAIL_BAND = 52;
const TEXT_BAND_GAP = 4;
const DIAGRAM_GLYPH_GUTTER = 46;
const DIAGRAM_END_INSET = 10;
const DIAGRAM_BOUNDARY_INSET = 12;
const PRESENTATION_TEXT_INSET_X = 12;
const PRESENTATION_TEXT_INSET_Y = 10;
const ICON_LABEL_TEXT_OFFSET = 40;
const ICON_LABEL_END_INSET = 8;
const AVATAR_ICON_INSET = 8;
const AVATAR_ICON_SIZE = 24;
const AVATAR_MIN_SIDE = AVATAR_ICON_INSET * 2 + AVATAR_ICON_SIZE;

function capabilityLayout(
  bounds: SceneGeometryLike,
  childGeometries: readonly SceneGeometryLike[],
  contentBounds: SceneGeometryLike = bounds,
  diagnostics: readonly CapabilityLayoutDiagnostic[] = [],
): CapabilityLayout {
  return { bounds, contentBounds, childGeometries, diagnostics };
}

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

/** Cross-axis alignment shared by opt-in managed containers. */
export type ManagedAxisAlignment = "start" | "center" | "end" | "stretch";

/** Main-axis distribution shared by opt-in managed containers. */
export type ManagedMainAlignment =
  | "start"
  | "center"
  | "end"
  | "space-between";

/** Normalized package-compatible managed layout options. */
export type ManagedLayoutOptions = Readonly<{
  padding: number;
  align: ManagedAxisAlignment;
  justify: ManagedMainAlignment;
  fixedWidth: boolean;
  fixedHeight: boolean;
}>;

function managedValue(node: SceneNodeLike, key: string): unknown {
  return node.style?.[key] ?? node.props?.[key];
}

/** Normalize shared managed inputs while clamping compatibility packages. */
export function managedLayoutOptions(
  node: SceneNodeLike,
): ManagedLayoutOptions {
  const align = managedValue(node, "align");
  const justify = managedValue(node, "justify");
  return {
    padding: nonnegative(managedValue(node, "padding")),
    align:
      align === "center" || align === "end" || align === "stretch"
        ? align
        : "start",
    justify:
      justify === "center" ||
      justify === "end" ||
      justify === "space-between"
        ? justify
        : "start",
    fixedWidth: managedValue(node, "fixedWidth") === true,
    fixedHeight: managedValue(node, "fixedHeight") === true,
  };
}

function isAbsolute(child: SceneNodeLike): boolean {
  return child.style?.position === "absolute";
}

function axisSize(
  geometry: SceneGeometryLike,
  direction: "row" | "column",
): number {
  return direction === "row" ? geometry.width : geometry.height;
}

function crossSize(
  geometry: SceneGeometryLike,
  direction: "row" | "column",
): number {
  return direction === "row" ? geometry.height : geometry.width;
}

function placeOnAxes(
  geometry: SceneGeometryLike,
  direction: "row" | "column",
  main: number,
  cross: number,
  mainSize = axisSize(geometry, direction),
  placedCrossSize = crossSize(geometry, direction),
): SceneGeometryLike {
  return direction === "row"
    ? {
        x: main,
        y: cross,
        width: mainSize,
        height: placedCrossSize,
      }
    : {
        x: cross,
        y: main,
        width: placedCrossSize,
        height: mainSize,
      };
}

function alignedCross(
  options: ManagedLayoutOptions,
  contentCross: number,
  childCross: number,
): Readonly<{ offset: number; size: number }> {
  if (options.align === "stretch") {
    return { offset: 0, size: contentCross };
  }
  if (options.align === "center") {
    return { offset: Math.max((contentCross - childCross) / 2, 0), size: childCross };
  }
  if (options.align === "end") {
    return { offset: Math.max(contentCross - childCross, 0), size: childCross };
  }
  return { offset: 0, size: childCross };
}

function intersects(a: SceneGeometryLike, b: SceneGeometryLike): boolean {
  return (
    a.x < b.x + b.width &&
    a.x + a.width > b.x &&
    a.y < b.y + b.height &&
    a.y + a.height > b.y
  );
}

function managedDiagnostics(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
  geometries: readonly SceneGeometryLike[],
  contentBounds: SceneGeometryLike,
  allowOverlap: boolean,
): readonly CapabilityLayoutDiagnostic[] {
  const diagnostics: CapabilityLayoutDiagnostic[] = [];
  const overflowing = geometries
    .map((geometry, index) => ({ geometry, child: children[index] }))
    .filter(
      ({ geometry }) =>
        geometry.x < contentBounds.x ||
        geometry.y < contentBounds.y ||
        geometry.x + geometry.width > contentBounds.x + contentBounds.width ||
        geometry.y + geometry.height > contentBounds.y + contentBounds.height,
    )
    .map(({ child }) => child?.id)
    .filter((id): id is string => id !== undefined);
  if (overflowing.length > 0) {
    diagnostics.push({
      code: "SCENE_MANAGED_CONTENT_OVERFLOW",
      severity: "error",
      message: `Managed content exceeds the bounds of "${node.id}".`,
      nodeIds: [node.id, ...overflowing],
    });
  }
  if (!allowOverlap) {
    for (let left = 0; left < geometries.length; left += 1) {
      for (let right = left + 1; right < geometries.length; right += 1) {
        if (intersects(geometries[left]!, geometries[right]!)) {
          diagnostics.push({
            code: "SCENE_MANAGED_CHILD_OVERLAP",
            severity: "error",
            message: `Managed children overlap inside "${node.id}".`,
            nodeIds: [
              node.id,
              children[left]?.id ?? `${left}`,
              children[right]?.id ?? `${right}`,
            ],
          });
        }
      }
    }
  }
  return diagnostics;
}

function managedBounds(
  authored: SceneGeometryLike,
  intrinsicWidth: number,
  intrinsicHeight: number,
  options: ManagedLayoutOptions,
): SceneGeometryLike {
  return {
    ...authored,
    width: options.fixedWidth
      ? authored.width
      : Math.max(authored.width, intrinsicWidth),
    height: options.fixedHeight
      ? authored.height
      : Math.max(authored.height, intrinsicHeight),
  };
}

function paddedContentBounds(
  bounds: SceneGeometryLike,
  padding: number,
  topInset = padding,
): SceneGeometryLike {
  return {
    x: padding,
    y: topInset,
    width: Math.max(bounds.width - padding * 2, 0),
    height: Math.max(bounds.height - topInset - padding, 0),
  };
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

function clipsOverflow(node: SceneNodeLike): boolean {
  return node.style?.overflow === "hidden" || node.style?.clip === true;
}

/** Identity layout for leaf capabilities and already-positioned groups. */
export function resolveIdentityLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const bounds = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  if (clipsOverflow(node)) {
    return capabilityLayout(bounds, childGeometries);
  }
  const extent = childBounds(childGeometries);
  return capabilityLayout(
    {
      ...bounds,
      width: Math.max(bounds.width, extent.width),
      height: Math.max(bounds.height, extent.height),
    },
    childGeometries,
  );
}

/** Intrinsic chip layout with authored dimensions treated as minimums. */
export function resolveChipLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  if (clipsOverflow(node)) {
    return capabilityLayout(authored, childGeometries);
  }
  const label =
    (typeof node.props?.label === "string" && node.props.label) ||
    (typeof node.props?.text === "string" && node.props.text) ||
    node.accessibility?.label ||
    "";
  const width = Math.max(
    authored.width,
    label.length > 0
      ? estimateTextWidth(label, 11, "bold") + CHIP_PAD_X
      : authored.width,
  );
  return capabilityLayout(
    {
      ...authored,
      width,
      height: Math.max(authored.height, STEPPER_CHIP_HEIGHT),
    },
    childGeometries,
  );
}

/** Intrinsic title/detail chrome layout with authored dimensions as minimums. */
export function resolvePanelLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  if (clipsOverflow(node)) {
    return capabilityLayout(authored, childGeometries);
  }
  const title =
    (typeof node.props?.title === "string" && node.props.title) ||
    (typeof node.props?.label === "string" && node.props.label) ||
    (typeof node.props?.text === "string" && node.props.text) ||
    "";
  const detail =
    (typeof node.props?.detail === "string" && node.props.detail) ||
    (typeof node.props?.caption === "string" && node.props.caption) ||
    "";
  const subtitle =
    typeof node.props?.subtitle === "string" ? node.props.subtitle : "";
  const titleWidth =
    title.length > 0 ? estimateTextWidth(title, 14, "bold") + INSET * 2 : 0;
  const detailWidth =
    detail.length > 0
      ? estimateTextWidth(detail, 11.5, "normal") + INSET * 2
      : 0;
  const subtitleWidth =
    subtitle.length > 0
      ? estimateTextWidth(subtitle, 10, "normal") + INSET * 2
      : 0;
  const contentHeight =
    INSET * 2 +
    (title.length > 0 ? TITLE_HEIGHT : 0) +
    (detail.length > 0 ? DETAIL_HEIGHT + TEXT_BAND_GAP : 0) +
    (subtitle.length > 0 ? SUBTITLE_HEIGHT + TEXT_BAND_GAP : 0);
  return capabilityLayout(
    {
      ...authored,
      width: Math.max(authored.width, titleWidth, detailWidth, subtitleWidth),
      height: Math.max(authored.height, contentHeight),
    },
    childGeometries,
  );
}

/** Intrinsic diagram chrome layout with authored dimensions as minimums. */
export function resolveDiagramLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  if (clipsOverflow(node)) {
    return capabilityLayout(authored, childGeometries);
  }
  const title =
    (typeof node.props?.title === "string" && node.props.title) ||
    (typeof node.props?.label === "string" && node.props.label) ||
    (typeof node.props?.text === "string" && node.props.text) ||
    "";
  const detail =
    (typeof node.props?.detail === "string" && node.props.detail) ||
    (typeof node.props?.caption === "string" && node.props.caption) ||
    "";
  const isBoundary = node.capabilityId === "diagram.boundary";
  const horizontalChrome = isBoundary
    ? DIAGRAM_BOUNDARY_INSET * 2
    : DIAGRAM_GLYPH_GUTTER + DIAGRAM_END_INSET;
  const titleWidth =
    title.length > 0
      ? estimateTextWidth(title, isBoundary ? 12 : 13, "bold") +
        horizontalChrome
      : 0;
  const detailWidth =
    detail.length > 0
      ? estimateTextWidth(detail, 10, "normal") + horizontalChrome
      : 0;
  const contentHeight =
    title.length > 0
      ? detail.length > 0
        ? 38 + DETAIL_HEIGHT + TEXT_BAND_GAP
        : 20 + TITLE_HEIGHT
      : 0;
  return capabilityLayout(
    {
      ...authored,
      width: Math.max(authored.width, titleWidth, detailWidth),
      height: Math.max(authored.height, contentHeight),
    },
    childGeometries,
  );
}

/** Intrinsic square sizing for avatar presentation chrome and nested icon glyph. */
export function resolveAvatarLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  if (clipsOverflow(node)) {
    return capabilityLayout(authored, childGeometries);
  }
  const side = Math.max(authored.width, authored.height, AVATAR_MIN_SIDE);
  return capabilityLayout(
    {
      ...authored,
      width: side,
      height: side,
    },
    childGeometries,
  );
}

/** Intrinsic sizing for renderer-owned presentation chrome. */
export function resolvePresentationLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const presentation = node.props?.presentation;
  if (presentation === "avatar") {
    return resolveAvatarLayout(node, children);
  }
  if (
    presentation !== "code-block" &&
    presentation !== "quote" &&
    presentation !== "icon-label"
  ) {
    return resolveIdentityLayout(node, children);
  }
  const authored = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  if (clipsOverflow(node)) {
    return capabilityLayout(authored, childGeometries);
  }
  const text =
    presentation === "icon-label"
      ? (typeof node.props?.label === "string" && node.props.label) ||
        (typeof node.props?.text === "string" && node.props.text) ||
        ""
      : typeof node.props?.text === "string"
        ? node.props.text
        : "";
  const lines = presentation === "code-block" ? text.split("\n") : [text];
  const textWidth = lines.reduce(
    (width, line) => Math.max(width, estimateTextWidth(line, 12)),
    0,
  );
  const horizontalChrome =
    presentation === "icon-label"
      ? ICON_LABEL_TEXT_OFFSET + ICON_LABEL_END_INSET
      : PRESENTATION_TEXT_INSET_X * 2;
  const verticalChrome =
    presentation === "icon-label"
      ? 16
      : PRESENTATION_TEXT_INSET_Y * 2;
  return capabilityLayout(
    {
      ...authored,
      width: Math.max(authored.width, textWidth + horizontalChrome),
      height: Math.max(
        authored.height,
        Math.ceil(scaledSceneFontSize(12) * lines.length + verticalChrome),
      ),
    },
    childGeometries,
  );
}

/** Intrinsic callout layout with enough padded room for its centered label. */
export function resolveCalloutLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  if (clipsOverflow(node)) {
    return capabilityLayout(authored, childGeometries);
  }
  const label =
    (typeof node.props?.text === "string" && node.props.text) ||
    (typeof node.props?.label === "string" && node.props.label) ||
    node.accessibility?.label ||
    "";
  return capabilityLayout(
    {
      ...authored,
      width: Math.max(
        authored.width,
        label.length > 0
          ? estimateTextWidth(label, 12, "normal") + INSET * 2
          : authored.width,
      ),
      height: Math.max(
        authored.height,
        scaledSceneFontSize(12) + INSET * 2,
      ),
    },
    childGeometries,
  );
}

/** Intrinsic header chrome layout matching its rendered title/caption bands. */
export function resolveHeaderLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  if (clipsOverflow(node)) {
    return capabilityLayout(authored, childGeometries);
  }
  const title =
    typeof node.props?.title === "string" ? node.props.title : "";
  const caption =
    typeof node.props?.caption === "string" ? node.props.caption : "";
  const titleWidth =
    title.length > 0 ? estimateTextWidth(title, 13, "bold") + INSET * 2 : 0;
  const captionWidth =
    caption.length > 0
      ? estimateTextWidth(caption, 11.5, "normal") + INSET * 2
      : 0;
  const contentHeight =
    INSET * 2 +
    (title.length > 0 ? TITLE_HEIGHT : 0) +
    (caption.length > 0 ? DETAIL_HEIGHT + 4 : 0);
  return capabilityLayout(
    {
      ...authored,
      width: Math.max(authored.width, titleWidth, captionWidth),
      height: Math.max(authored.height, contentHeight),
    },
    childGeometries,
  );
}

/** Intrinsic text layout using the same scale-aware metrics as rendering. */
export function resolveTextLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const childGeometries = children.map(geometryOf);
  if (clipsOverflow(node)) {
    return capabilityLayout(authored, childGeometries);
  }
  const text = node.text ?? "";
  const fontSize = styleNumber(
    node.style,
    "fontSize",
    DEFAULT_SCENE_FONT_SIZE,
  );
  const weight = node.style?.fontWeight === "bold" ? "bold" : "normal";
  return capabilityLayout(
    {
      ...authored,
      width: Math.max(authored.width, estimateTextWidth(text, fontSize, weight)),
      height: Math.max(authored.height, scaledSceneFontSize(fontSize)),
    },
    childGeometries,
  );
}

/** Ordered stack layout with authored dimensions treated as minimums. */
export function resolveStackLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const direction = directionOf(node);
  const gap = styleNumber(node.style, "gap", 0);
  const options = managedLayoutOptions(node);
  const normal = children
    .map((child, index) => ({ child, index, geometry: geometryOf(child) }))
    .filter(({ child }) => !isAbsolute(child));
  const totalMain =
    normal.reduce(
      (sum, { geometry }) => sum + axisSize(geometry, direction),
      0,
    ) + gap * Math.max(normal.length - 1, 0);
  const maxCross = normal.reduce(
    (maximum, { geometry }) =>
      Math.max(maximum, crossSize(geometry, direction)),
    0,
  );
  const intrinsicWidth =
    options.padding * 2 + (direction === "row" ? totalMain : maxCross);
  const intrinsicHeight =
    options.padding * 2 + (direction === "column" ? totalMain : maxCross);
  const bounds = managedBounds(
    authored,
    intrinsicWidth,
    intrinsicHeight,
    options,
  );
  const contentBounds = paddedContentBounds(bounds, options.padding);
  const contentMain =
    direction === "row" ? contentBounds.width : contentBounds.height;
  const contentCross =
    direction === "row" ? contentBounds.height : contentBounds.width;
  const freeMain = Math.max(contentMain - totalMain, 0);
  const extraGap =
    options.justify === "space-between" && normal.length > 1
      ? freeMain / (normal.length - 1)
      : 0;
  let cursor =
    (direction === "row" ? contentBounds.x : contentBounds.y) +
    (options.justify === "center"
      ? freeMain / 2
      : options.justify === "end"
        ? freeMain
        : 0);
  const placed = new Map<number, SceneGeometryLike>();
  for (const { index, geometry } of normal) {
    const aligned = alignedCross(options, contentCross, crossSize(geometry, direction));
    const crossOrigin =
      (direction === "row" ? contentBounds.y : contentBounds.x) + aligned.offset;
    placed.set(
      index,
      placeOnAxes(
        geometry,
        direction,
        cursor,
        crossOrigin,
        axisSize(geometry, direction),
        aligned.size,
      ),
    );
    cursor += axisSize(geometry, direction) + gap + extraGap;
  }
  const childGeometries = children.map((child, index) =>
    isAbsolute(child) ? geometryOf(child) : placed.get(index)!,
  );
  return capabilityLayout(
    bounds,
    childGeometries,
    contentBounds,
    managedDiagnostics(node, children, childGeometries, contentBounds, false),
  );
}

/** Row-major grid layout with per-column and per-row intrinsic dimensions. */
export function resolveGridLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const cols = Math.max(1, Math.floor(styleNumber(node.style, "cols", 1)));
  const gap = styleNumber(node.style, "gap", 0);
  const options = managedLayoutOptions(node);
  const normal = children
    .map((child, index) => ({ child, index, geometry: geometryOf(child) }))
    .filter(({ child }) => !isAbsolute(child));
  const rows = Math.max(1, Math.ceil(normal.length / cols));
  const widths = Array.from({ length: cols }, () => 0);
  const heights = Array.from({ length: rows }, () => 0);
  normal.forEach(({ geometry }, flowIndex) => {
    const col = flowIndex % cols;
    const row = Math.floor(flowIndex / cols);
    widths[col] = Math.max(widths[col]!, geometry.width);
    heights[row] = Math.max(heights[row]!, geometry.height);
  });
  const intrinsicContentWidth =
    widths.reduce((sum, width) => sum + width, 0) +
    gap * Math.max(cols - 1, 0);
  const intrinsicContentHeight =
    heights.reduce((sum, height) => sum + height, 0) +
    gap * Math.max(rows - 1, 0);
  const bounds = managedBounds(
    authored,
    intrinsicContentWidth + options.padding * 2,
    intrinsicContentHeight + options.padding * 2,
    options,
  );
  const contentBounds = paddedContentBounds(bounds, options.padding);
  const freeX = Math.max(contentBounds.width - intrinsicContentWidth, 0);
  const freeY = Math.max(contentBounds.height - intrinsicContentHeight, 0);
  const justifyOffset =
    options.justify === "center"
      ? freeY / 2
      : options.justify === "end"
        ? freeY
        : 0;
  const rowGap =
    gap +
    (options.justify === "space-between" && rows > 1
      ? freeY / (rows - 1)
      : 0);
  const columnExtra = options.align === "stretch" ? freeX / cols : 0;
  const xOffsets: number[] = [];
  const yOffsets: number[] = [];
  let x = contentBounds.x;
  let y = contentBounds.y + justifyOffset;
  widths.forEach((width) => {
    xOffsets.push(x);
    x += width + columnExtra + gap;
  });
  heights.forEach((height) => {
    yOffsets.push(y);
    y += height + rowGap;
  });
  const placed = new Map<number, SceneGeometryLike>();
  normal.forEach(({ index, geometry }, flowIndex) => {
    const col = flowIndex % cols;
    const row = Math.floor(flowIndex / cols);
    const cellWidth = widths[col]! + columnExtra;
    const aligned = alignedCross(options, cellWidth, geometry.width);
    placed.set(index, {
      ...geometry,
      x: xOffsets[col]! + aligned.offset,
      y: yOffsets[row]!,
      width: aligned.size,
    });
  });
  const childGeometries = children.map((child, index) =>
    isAbsolute(child) ? geometryOf(child) : placed.get(index)!,
  );
  return capabilityLayout(
    bounds,
    childGeometries,
    contentBounds,
    managedDiagnostics(node, children, childGeometries, contentBounds, false),
  );
}

/** Equal-slot rail that first expands to the children's intrinsic minimum. */
export function resolveRailLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const direction = directionOf(node);
  const gap = styleNumber(node.style, "gap", 0);
  const options = managedLayoutOptions(node);
  const normal = children
    .map((child, index) => ({ child, index, geometry: geometryOf(child) }))
    .filter(({ child }) => !isAbsolute(child));
  if (normal.length === 0) {
    const bounds = managedBounds(
      authored,
      options.padding * 2,
      options.padding * 2,
      options,
    );
    const contentBounds = paddedContentBounds(bounds, options.padding);
    const childGeometries = children.map(geometryOf);
    return capabilityLayout(
      bounds,
      childGeometries,
      contentBounds,
      managedDiagnostics(node, children, childGeometries, contentBounds, false),
    );
  }
  const totalGap = gap * Math.max(normal.length - 1, 0);
  const totalMain =
    normal.reduce(
      (sum, { geometry }) => sum + axisSize(geometry, direction),
      0,
    ) + totalGap;
  const maxCross = Math.max(
    ...normal.map(({ geometry }) => crossSize(geometry, direction)),
  );
  const bounds = managedBounds(
    authored,
    options.padding * 2 + (direction === "row" ? totalMain : maxCross),
    options.padding * 2 + (direction === "column" ? totalMain : maxCross),
    options,
  );
  const contentBounds = paddedContentBounds(bounds, options.padding);
  const contentMain =
    direction === "row" ? contentBounds.width : contentBounds.height;
  const contentCross =
    direction === "row" ? contentBounds.height : contentBounds.width;
  const freeMain = Math.max(contentMain - totalMain, 0);
  const extraPerChild =
    options.justify === "start" ? freeMain / normal.length : 0;
  const remainingMain = freeMain - extraPerChild * normal.length;
  const extraGap =
    options.justify === "space-between" && normal.length > 1
      ? remainingMain / (normal.length - 1)
      : 0;
  let cursor =
    (direction === "row" ? contentBounds.x : contentBounds.y) +
    (options.justify === "center"
      ? remainingMain / 2
      : options.justify === "end"
        ? remainingMain
        : 0);
  const placed = new Map<number, SceneGeometryLike>();
  for (const { index, geometry } of normal) {
    const aligned = alignedCross(options, contentCross, crossSize(geometry, direction));
    const mainSize = axisSize(geometry, direction) + extraPerChild;
    placed.set(
      index,
      placeOnAxes(
        geometry,
        direction,
        cursor,
        (direction === "row" ? contentBounds.y : contentBounds.x) +
          aligned.offset,
        mainSize,
        aligned.size,
      ),
    );
    cursor += mainSize + gap + extraGap;
  }
  const childGeometries = children.map((child, index) =>
    isAbsolute(child) ? geometryOf(child) : placed.get(index)!,
  );
  return capabilityLayout(
    bounds,
    childGeometries,
    contentBounds,
    managedDiagnostics(node, children, childGeometries, contentBounds, false),
  );
}

/** Intentional overlap layout with shared padded alignment. */
export function resolveOverlayLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const options = managedLayoutOptions(node);
  const normal = children
    .map((child, index) => ({ child, index, geometry: geometryOf(child) }))
    .filter(({ child }) => !isAbsolute(child));
  const intrinsicWidth =
    options.padding * 2 +
    normal.reduce(
      (maximum, { geometry }) => Math.max(maximum, geometry.width),
      0,
    );
  const intrinsicHeight =
    options.padding * 2 +
    normal.reduce(
      (maximum, { geometry }) => Math.max(maximum, geometry.height),
      0,
    );
  const bounds = managedBounds(
    authored,
    intrinsicWidth,
    intrinsicHeight,
    options,
  );
  const contentBounds = paddedContentBounds(bounds, options.padding);
  const placed = new Map<number, SceneGeometryLike>();
  for (const { index, geometry } of normal) {
    const horizontal = alignedCross(options, contentBounds.width, geometry.width);
    const vertical =
      options.align === "stretch"
        ? { offset: 0, size: contentBounds.height }
        : options.justify === "center"
          ? {
              offset: Math.max((contentBounds.height - geometry.height) / 2, 0),
              size: geometry.height,
            }
          : options.justify === "end"
            ? {
                offset: Math.max(contentBounds.height - geometry.height, 0),
                size: geometry.height,
              }
            : { offset: 0, size: geometry.height };
    placed.set(index, {
      x: contentBounds.x + horizontal.offset,
      y: contentBounds.y + vertical.offset,
      width: horizontal.size,
      height: vertical.size,
    });
  }
  const childGeometries = children.map((child, index) =>
    isAbsolute(child) ? geometryOf(child) : placed.get(index)!,
  );
  return capabilityLayout(
    bounds,
    childGeometries,
    contentBounds,
    managedDiagnostics(node, children, childGeometries, contentBounds, true),
  );
}

/** Titled frame that reserves semantic chrome before managed child content. */
export function resolveFrameLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const authored = geometryOf(node);
  const options = managedLayoutOptions(node);
  const gap = styleNumber(node.style, "gap", 0);
  const title =
    typeof node.props?.title === "string" ? node.props.title : "";
  const detail =
    typeof node.props?.detail === "string" ? node.props.detail : "";
  const subtitle =
    typeof node.props?.subtitle === "string" ? node.props.subtitle : "";
  // Match chrome subtitle bottom: INSET + TITLE + DETAIL + gap + SUBTITLE.
  const subtitleBand =
    subtitle.length > 0
      ? INSET + TITLE_HEIGHT + DETAIL_HEIGHT + 6 + SUBTITLE_HEIGHT
      : 0;
  const titleBand = Math.max(
    detail.length > 0 ? FRAME_DETAIL_BAND : FRAME_TITLE_BAND,
    subtitleBand,
  );
  const normal = children
    .map((child, index) => ({ child, index, geometry: geometryOf(child) }))
    .filter(({ child }) => !isAbsolute(child));
  const contentMain =
    normal.reduce((sum, { geometry }) => sum + geometry.height, 0) +
    gap * Math.max(normal.length - 1, 0);
  const contentCross = normal.reduce(
    (maximum, { geometry }) => Math.max(maximum, geometry.width),
    0,
  );
  const titleWidth =
    title.length > 0 ? estimateTextWidth(title, 14, "bold") : 0;
  const detailWidth =
    detail.length > 0 ? estimateTextWidth(detail, 11.5, "normal") : 0;
  const subtitleWidth =
    subtitle.length > 0 ? estimateTextWidth(subtitle, 10, "normal") : 0;
  const bounds = managedBounds(
    authored,
    Math.max(contentCross, titleWidth, detailWidth, subtitleWidth) +
      options.padding * 2,
    titleBand + contentMain + options.padding * 2,
    options,
  );
  const contentBounds = paddedContentBounds(
    bounds,
    options.padding,
    titleBand + options.padding,
  );
  const freeMain = Math.max(contentBounds.height - contentMain, 0);
  const extraGap =
    options.justify === "space-between" && normal.length > 1
      ? freeMain / (normal.length - 1)
      : 0;
  let cursor =
    contentBounds.y +
    (options.justify === "center"
      ? freeMain / 2
      : options.justify === "end"
        ? freeMain
        : 0);
  const placed = new Map<number, SceneGeometryLike>();
  for (const { index, geometry } of normal) {
    const aligned = alignedCross(options, contentBounds.width, geometry.width);
    placed.set(index, {
      x: contentBounds.x + aligned.offset,
      y: cursor,
      width: aligned.size,
      height: geometry.height,
    });
    cursor += geometry.height + gap + extraGap;
  }
  const childGeometries = children.map((child, index) =>
    isAbsolute(child) ? geometryOf(child) : placed.get(index)!,
  );
  return capabilityLayout(
    bounds,
    childGeometries,
    contentBounds,
    managedDiagnostics(node, children, childGeometries, contentBounds, false),
  );
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
  return capabilityLayout(
    {
      ...authored,
      width: Math.max(authored.width, extent.width + LANE_INSET),
      height: Math.max(authored.height, contentHeight),
    },
    childGeometries,
  );
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
  return capabilityLayout(
    {
      ...authored,
      width: Math.max(authored.width, extent.width),
      height: Math.max(authored.height, contentHeight),
    },
    childGeometries,
  );
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
  return capabilityLayout(
    {
      ...authored,
      width: Math.max(authored.width, intrinsicWidth),
      height: Math.max(authored.height, STEPPER_CHIP_HEIGHT),
    },
    childGeometries,
  );
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
  return capabilityLayout(
    {
      ...authored,
      width: Math.max(authored.width, extent.width + inset),
      height: Math.max(authored.height, extent.height + inset),
    },
    childGeometries,
  );
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
  return capabilityLayout(
    {
      x: cx - rx,
      y: cy - ry,
      width: rx * 2,
      height: ry * 2,
    },
    children.map(geometryOf),
  );
}

export const LAYOUT_CAPABILITIES: readonly NativeSceneCapability[] = [
  { capabilityId: "layout.stack", resolveLayout: resolveStackLayout },
  { capabilityId: "layout.grid", resolveLayout: resolveGridLayout },
  { capabilityId: "layout.rail", resolveLayout: resolveRailLayout },
  { capabilityId: "layout.overlay", resolveLayout: resolveOverlayLayout },
  { capabilityId: "layout.frame", resolveLayout: resolveFrameLayout },
  { capabilityId: "layout.pad", resolveLayout: resolvePadLayout },
  { capabilityId: "core.chip", resolveLayout: resolveChipLayout },
  { capabilityId: "core.panel", resolveLayout: resolvePanelLayout },
  { capabilityId: "core.note", resolveLayout: resolvePanelLayout },
  { capabilityId: "core.callout", resolveLayout: resolveCalloutLayout },
  { capabilityId: "core.header", resolveLayout: resolveHeaderLayout },
  { capabilityId: "core.text", resolveLayout: resolveTextLayout },
  { capabilityId: "core.lane", resolveLayout: resolveLaneLayout },
  { capabilityId: "core.band", resolveLayout: resolveIdentityLayout },
  { capabilityId: "core.swimlane", resolveLayout: resolveSwimlaneLayout },
  { capabilityId: "core.stepper", resolveLayout: resolveStepperLayout },
  { capabilityId: "core.circle", resolveLayout: resolveEllipseLayout },
  { capabilityId: "core.ellipse", resolveLayout: resolveEllipseLayout },
  { capabilityId: "core.group", resolveLayout: resolvePresentationLayout },
  { capabilityId: "diagram.actor", resolveLayout: resolveDiagramLayout },
  { capabilityId: "diagram.compute", resolveLayout: resolveDiagramLayout },
  { capabilityId: "diagram.storage", resolveLayout: resolveDiagramLayout },
  { capabilityId: "diagram.messaging", resolveLayout: resolveDiagramLayout },
  { capabilityId: "diagram.network", resolveLayout: resolveDiagramLayout },
  { capabilityId: "diagram.control", resolveLayout: resolveDiagramLayout },
  { capabilityId: "diagram.boundary", resolveLayout: resolveDiagramLayout },
  { capabilityId: "diagram.symbol", resolveLayout: resolveDiagramLayout },
];

