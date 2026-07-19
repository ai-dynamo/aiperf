/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { useHostTheme, type Theme } from "../ui";
import { tokens } from "../tokens";
import {
  DEFAULT_MARKER_TIP,
  isMarkerEndNone,
  markerDomId,
  markerGeometry,
  resolveMarkerTip,
  type ResolvedMarkerTip,
} from "./arrow-tips";
import { FlowArrow } from "./FlowArrow";
import { MotionSignal } from "./MotionSignal";

/** Minimal geometry for a scene node. */
export type SceneGeometryLike = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
}>;

/**
 * Point or connector endpoint.
 * Explicit `x`/`y` win; otherwise `nodeId` resolves to an anchored point on
 * the target node (`anchor`: center|n/s/e/w/ne/nw/se/sw|top/bottom/left/right).
 */
export type ScenePointLike = Readonly<{
  x?: number;
  y?: number;
  nodeId?: string;
  anchor?: string;
}>;

/** Minimal accessibility metadata for a scene node. */
export type SceneNodeAccessibilityLike = Readonly<{
  label?: string;
  description?: string;
}>;

/** Style values may be scalars or nested objects (e.g. `markerEnd: { kind }`). */
export type SceneStyleValue =
  | string
  | number
  | boolean
  | Readonly<Record<string, string | number | boolean>>;

/** Minimal render node supporting rect / text / path / line-like shapes. */
export type SceneNodeLike = Readonly<{
  id: string;
  kind?: string;
  capabilityId?: string;
  capability?: string;
  geometry?: SceneGeometryLike;
  layout?: SceneGeometryLike;
  /**
   * Render-time x/y override resolved against an already-declared sibling's
   * world geometry (document order); width/height stay as authored.
   */
  relativePosition?: Readonly<{
    nodeId: string;
    anchor?: string;
    dx?: number;
    dy?: number;
  }>;
  style?: Readonly<Record<string, SceneStyleValue>>;
  /** Glyph content for `core.text` / `kind: "text"` nodes. */
  text?: string;
  accessibility?: SceneNodeAccessibilityLike;
  children?: readonly SceneNodeLike[];
  /** SVG path data for path / arrow nodes (FlowArrow `d`). */
  d?: string;
  path?: string;
  /** Polyline waypoints for path / connector-like nodes. */
  points?: readonly ScenePointLike[];
  /** Endpoint coordinates or node refs; fans use an array on their many side. */
  from?: ScenePointLike | readonly ScenePointLike[];
  to?: ScenePointLike | readonly ScenePointLike[];
  /** Optional elbow bend / waypoint (`core.elbow`). */
  via?: ScenePointLike;
  /** Preferred first-segment axis for orthogonal elbows (`"x"` | `"y"`). */
  axis?: "x" | "y" | string;
  /** Optional stable split / merge point for `core.fan-*`. */
  junction?: ScenePointLike;
}>;

/** Minimal timeline cue (enter/reveal/draw/emphasize/pulse/fade/exit/stagger). */
export type SceneTimelineCueLike = Readonly<{
  id: string;
  at: number;
  duration: number;
  action: string;
  /** Primary / group id; may be empty when `targets` identifies members. */
  target: string;
  /** Stagger member ids when `action` is `stagger` / `enter-children`. */
  targets?: readonly string[];
  /** Delay between successive stagger targets. */
  step?: number;
  /** Progress easing for this cue's envelope. */
  easing?: string;
}>;

/** Optional logical SVG bounds (defaults to 700×400). */
export type SceneViewportLike = Readonly<{
  width: number;
  height: number;
}>;

/** Authored camera keyframe: focus point `(x, y)` and zoom. */
export type SceneCameraKeyframeLike = Readonly<{
  id?: string;
  at: number;
  x: number;
  y: number;
  zoom: number;
}>;

/** Minimal Scene IR shape consumed by ExplainerShell diagrams. */
export type SceneIrLike = Readonly<{
  id?: string;
  title?: string;
  summary?: string;
  /** Optional diagram viewport; defaults to 700×400 when omitted. */
  viewport?: SceneViewportLike;
  roots: readonly SceneNodeLike[];
  /** Optional camera track; empty/absent keeps a static `0 0 W H` viewBox. */
  camera?: readonly SceneCameraKeyframeLike[];
  timeline: readonly SceneTimelineCueLike[];
  accessibility?: Readonly<{ label?: string }>;
}>;

export type SceneRendererProps = Readonly<{
  scene: SceneIrLike;
  playing: boolean;
  restartKey: number;
  reducedMotion?: boolean;
  /** Wall-clock multiplier for timeline advance (1 = realtime). */
  playbackRate?: number;
}>;

const VIEWPORT_WIDTH = 700;
const VIEWPORT_HEIGHT = 400;
const DEFAULT_ARROW_STROKE_WIDTH: number = tokens.diagram.strokeWidth;
const SVG_NS = "http://www.w3.org/2000/svg";
const DEFAULT_DOT_RADIUS = 5;
const PULSE_CYCLE_MS = 2200;
const PULSE_DELAY_MS = 800;
/** SMIL loop pace for idle motion dots — matched to typical edge-draw speed. */
const MOTION_DOT_DURATION_S = 1.6;
const MOTION_DOT_DELAY_S = 0.55;
const DASHED_STROKE: string = tokens.diagram.dashed;
const DOTTED_STROKE = "2 3";

function smilSeconds(seconds: number, playbackRate: number): string {
  const rate = playbackRate > 0 ? playbackRate : 1;
  const scaled = seconds / rate;
  return `${Number(scaled.toFixed(3))}s`;
}

const ARROW_CAPABILITIES = new Set([
  "core.line",
  "core.path",
  "core.arrow",
  "core.connector",
  "core.elbow",
  "core.route",
  "core.bracket",
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

/** Small filled dots only — circles/ellipses render as shapes. */
const DOT_CAPABILITIES = new Set(["core.dot"]);

const DOT_KINDS = new Set(["dot"]);

/** Groups whose children are authored in the parent local frame by default. */
const LOCAL_LAYOUT_CAPABILITIES = new Set([
  "core.group",
  "core.panel",
  "core.header",
  "core.callout",
  "core.lane",
  "core.band",
  "core.swimlane",
  "core.stepper",
  "layout.stack",
  "layout.grid",
  "layout.pad",
  "layout.rail",
]);

/** Container groups that paint a chrome rect behind nested children. */
const CHROME_GROUP_CAPABILITIES = new Set(["core.panel", "core.header"]);

const MOTION_SIGNAL_CAPABILITIES = new Set([
  "motion.signal",
  "motion.dot",
  "core.motion",
]);

type CameraTransform = Readonly<{ x: number; y: number; zoom: number }>;

type TimelineState = "hidden" | "entering" | "revealed" | "unchanged";

type TimelineAppearance = Readonly<{
  state: TimelineState;
  opacity: number;
}>;

/** Transient emphasize/emphasis/pulse envelope applied during an active cue. */
type EmphasisAppearance = Readonly<{
  intensity: number;
  strokeScale: number;
  opacityScale: number;
  filter: string;
}>;

/** Continuous box-pulse envelope (legacy MentalModel CSS keyframes). */
type PulseAppearance = Readonly<{
  intensity: number;
  opacity: number;
}>;

type PlaybackContext = Readonly<{
  playing: boolean;
  reducedMotion: boolean;
  restartKey: number;
  playbackRate: number;
}>;

type LayoutOrigin = Readonly<{ x: number; y: number }>;

const ZERO_ORIGIN: LayoutOrigin = { x: 0, y: 0 };

/** Scene node index plus world-space geometry (after group/container offsets). */
type SceneNodeIndex = Readonly<{
  nodesById: ReadonlyMap<string, SceneNodeLike>;
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>;
}>;

type ScenePoint = Readonly<{ x: number; y: number }>;

export type FanSegment = Readonly<{
  id: string;
  d: string;
  directed: true;
  showMarker: boolean;
  role: "trunk" | "branch" | "merge-trunk";
}>;

export type FanTrajectory = Readonly<{
  id: string;
  d: string;
  role: "trunk" | "branch" | "merge-trunk";
}>;

export type ResolvedFanGeometry = Readonly<{
  capability: "core.fan-out" | "core.fan-in";
  segments: readonly FanSegment[];
  junction: ScenePoint;
  trajectories: readonly FanTrajectory[];
}>;

function finiteNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clamp01(value: number): number {
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
}

/** Map linear progress through a simple cubic easing curve. */
function applyCueEasing(progress: number, easing: string | undefined): number {
  const t = clamp01(progress);
  switch ((easing ?? "linear").toLowerCase()) {
    case "ease-in":
      return t * t * t;
    case "ease-out": {
      const u = 1 - t;
      return 1 - u * u * u;
    }
    case "ease-in-out":
      return t < 0.5
        ? 4 * t * t * t
        : 1 - Math.pow(-2 * t + 2, 3) / 2;
    case "linear":
    default:
      return t;
  }
}

/** Linear cue window progress in [0, 1], then optionally eased. */
function cueProgress(
  cue: SceneTimelineCueLike,
  playbackTimeMs: number,
): number {
  const atMs = Math.max(0, finiteNumber(cue.at));
  const durationMs = Math.max(0, finiteNumber(cue.duration));
  let linear = 0;
  if (playbackTimeMs < atMs) {
    linear = 0;
  } else if (durationMs === 0) {
    linear = 1;
  } else {
    linear = clamp01((playbackTimeMs - atMs) / durationMs);
  }
  return applyCueEasing(linear, cue.easing);
}

function capabilityOf(node: SceneNodeLike): string {
  if (typeof node.capabilityId === "string" && node.capabilityId.length > 0) {
    return node.capabilityId;
  }
  if (typeof node.capability === "string" && node.capability.length > 0) {
    return node.capability;
  }
  if (typeof node.kind === "string" && node.kind.length > 0) {
    return `core.${node.kind}`;
  }
  return "";
}

function geometryOf(node: SceneNodeLike): SceneGeometryLike {
  const geometry = node.geometry ?? node.layout;
  return {
    x: finiteNumber(geometry?.x),
    y: finiteNumber(geometry?.y),
    width: finiteNumber(geometry?.width),
    height: finiteNumber(geometry?.height),
  };
}

/** Resolve authored viewport size, falling back to the ExplainerShell default. */
function resolveViewportSize(
  viewport: SceneViewportLike | undefined,
): Readonly<{ width: number; height: number }> {
  const width = finiteNumber(viewport?.width);
  const height = finiteNumber(viewport?.height);
  return {
    width: width > 0 ? width : VIEWPORT_WIDTH,
    height: height > 0 ? height : VIEWPORT_HEIGHT,
  };
}

/**
 * Sample the authored camera track at `timeMs`.
 * Returns `undefined` when the track is empty so the renderer keeps a static viewBox.
 */
function authoredCameraAt(
  camera: readonly SceneCameraKeyframeLike[] | undefined,
  timeMs: number,
): CameraTransform | undefined {
  if (!Array.isArray(camera) || camera.length === 0) {
    return undefined;
  }

  const time = Math.max(0, Math.trunc(finiteNumber(timeMs)));
  const keyframes = camera
    .map((keyframe, index) => ({ keyframe, index }))
    .sort(
      (left, right) =>
        finiteNumber(left.keyframe.at) - finiteNumber(right.keyframe.at) ||
        left.index - right.index,
    )
    .map(({ keyframe }) => keyframe);
  const first = keyframes[0]!;
  const last = keyframes[keyframes.length - 1]!;

  if (time <= finiteNumber(first.at)) {
    return {
      x: finiteNumber(first.x),
      y: finiteNumber(first.y),
      zoom: Math.max(finiteNumber(first.zoom), Number.EPSILON),
    };
  }
  if (time >= finiteNumber(last.at)) {
    return {
      x: finiteNumber(last.x),
      y: finiteNumber(last.y),
      zoom: Math.max(finiteNumber(last.zoom), Number.EPSILON),
    };
  }

  const endIndex = keyframes.findIndex(
    (keyframe) => finiteNumber(keyframe.at) > time,
  );
  const start = keyframes[endIndex - 1]!;
  const end = keyframes[endIndex]!;
  const startAt = finiteNumber(start.at);
  const endAt = finiteNumber(end.at);
  const progress = (time - startAt) / (endAt - startAt);
  const startZoom = Math.max(finiteNumber(start.zoom), Number.EPSILON);
  const endZoom = Math.max(finiteNumber(end.zoom), Number.EPSILON);
  return {
    x: finiteNumber(start.x) + (finiteNumber(end.x) - finiteNumber(start.x)) * progress,
    y: finiteNumber(start.y) + (finiteNumber(end.y) - finiteNumber(start.y)) * progress,
    zoom: startZoom + (endZoom - startZoom) * progress,
  };
}

/**
 * Build an SVG viewBox string from viewport size and an optional camera focus.
 * Camera `(x, y)` is the focus point centered in the viewport; `zoom` scales in.
 */
function sceneViewBox(
  width: number,
  height: number,
  camera: CameraTransform | undefined,
): string {
  if (camera === undefined) {
    return `0 0 ${width} ${height}`;
  }
  const zoom = Math.max(camera.zoom, Number.EPSILON);
  const visibleWidth = width / zoom;
  const visibleHeight = height / zoom;
  const minX = camera.x - visibleWidth / 2;
  const minY = camera.y - visibleHeight / 2;
  return `${minX} ${minY} ${visibleWidth} ${visibleHeight}`;
}

/** Pure group container: nests children in a `<g>` with no leaf body of its own. */
function isGroupLike(node: SceneNodeLike, capability: string): boolean {
  if (LOCAL_LAYOUT_CAPABILITIES.has(capability) || capability.startsWith("layout.")) {
    return true;
  }
  if (capability === "core.group") {
    return true;
  }
  return node.kind === "group" || node.kind === "component";
}

function styleCoordinateSpace(
  style: SceneNodeLike["style"],
): "local" | "absolute" | undefined {
  const value = style?.coordinateSpace;
  if (value === "absolute" || value === "local") {
    return value;
  }
  return undefined;
}

function styleGap(style: SceneNodeLike["style"]): number {
  const gap = style?.gap;
  return typeof gap === "number" && Number.isFinite(gap) ? Math.max(0, gap) : 0;
}

function styleCols(style: SceneNodeLike["style"]): number {
  const cols = style?.cols;
  if (typeof cols === "number" && Number.isFinite(cols) && cols >= 1) {
    return Math.floor(cols);
  }
  return 1;
}

function styleDirection(style: SceneNodeLike["style"]): "row" | "column" {
  return style?.direction === "row" ? "row" : "column";
}

/**
 * True when every geometried child fits in the parent's local [0,w]×[0,h] box.
 * Absolute scene children (e.g. y past parent height) return false.
 */
function childrenFitParentLocalBox(
  parentGeom: SceneGeometryLike,
  children: readonly SceneNodeLike[],
): boolean {
  if (parentGeom.width <= 0 || parentGeom.height <= 0) {
    return false;
  }
  let sawGeometry = false;
  for (const child of children) {
    if (child.geometry === undefined && child.layout === undefined) {
      continue;
    }
    sawGeometry = true;
    const childGeom = geometryOf(child);
    if (childGeom.x < -0.5 || childGeom.y < -0.5) {
      return false;
    }
    if (childGeom.x + childGeom.width > parentGeom.width + 0.5) {
      return false;
    }
    if (childGeom.y + childGeom.height > parentGeom.height + 0.5) {
      return false;
    }
  }
  return sawGeometry;
}

/**
 * Whether nested children are authored in the parent's local frame so the
 * renderer should apply `translate(parent.x, parent.y)` around them.
 *
 * Rule: children of `core.panel` / `core.header` / `layout.*` / `core.callout`
 * use local coordinates unless `style.coordinateSpace === "absolute"`.
 * Plain `core.group` prefers local when children fit the parent box; otherwise
 * legacy world-absolute nesting is preserved.
 */
function childrenUseLocalLayout(
  node: SceneNodeLike,
  parentGeom: SceneGeometryLike,
  children: readonly SceneNodeLike[] | undefined,
): boolean {
  if (!Array.isArray(children) || children.length === 0) {
    return false;
  }
  const space = styleCoordinateSpace(node.style);
  if (space === "absolute") {
    return false;
  }
  if (space === "local") {
    return true;
  }
  const capability = capabilityOf(node);
  if (
    capability === "core.panel" ||
    capability === "core.header" ||
    capability === "core.callout" ||
    capability === "core.lane" ||
    capability === "core.band" ||
    capability === "core.swimlane" ||
    capability === "core.stepper" ||
    capability === "layout.stack" ||
    capability === "layout.grid" ||
    capability === "layout.pad" ||
    capability === "layout.rail" ||
    capability.startsWith("layout.")
  ) {
    return true;
  }
  if (childrenFitParentLocalBox(parentGeom, children)) {
    return true;
  }
  if (capability !== "core.group" && node.kind !== "group") {
    return false;
  }
  if (parentGeom.width > 0 && parentGeom.height > 0) {
    return false;
  }
  if (parentGeom.x === 0 && parentGeom.y === 0) {
    return false;
  }
  for (const child of children) {
    if (child.geometry === undefined && child.layout === undefined) {
      continue;
    }
    const childGeom = geometryOf(child);
    if (
      childGeom.x >= parentGeom.x - 0.5 &&
      childGeom.y >= parentGeom.y - 0.5
    ) {
      return false;
    }
  }
  return true;
}

/** Compute stack child local geometries; enlarge parent when width/height are 0. */
function computeStackLayout(
  parentGeom: SceneGeometryLike,
  children: readonly SceneNodeLike[],
  style: SceneNodeLike["style"],
): Readonly<{
  parentGeom: SceneGeometryLike;
  childGeoms: readonly SceneGeometryLike[];
}> {
  const direction = styleDirection(style);
  const gap = styleGap(style);
  const childGeoms: SceneGeometryLike[] = [];
  let cursor = 0;
  let cross = 0;
  for (const child of children) {
    const g = geometryOf(child);
    if (direction === "row") {
      childGeoms.push({
        x: cursor,
        y: 0,
        width: g.width,
        height: g.height,
      });
      cursor += g.width + gap;
      cross = Math.max(cross, g.height);
    } else {
      childGeoms.push({
        x: 0,
        y: cursor,
        width: g.width,
        height: g.height,
      });
      cursor += g.height + gap;
      cross = Math.max(cross, g.width);
    }
  }
  if (childGeoms.length > 0) {
    cursor = Math.max(0, cursor - gap);
  }
  const width =
    parentGeom.width > 0
      ? parentGeom.width
      : direction === "row"
        ? cursor
        : cross;
  const height =
    parentGeom.height > 0
      ? parentGeom.height
      : direction === "column"
        ? cursor
        : cross;
  return {
    parentGeom: { ...parentGeom, width, height },
    childGeoms,
  };
}

/** Row-major grid child local geometries; enlarge parent when needed. */
function computeGridLayout(
  parentGeom: SceneGeometryLike,
  children: readonly SceneNodeLike[],
  style: SceneNodeLike["style"],
): Readonly<{
  parentGeom: SceneGeometryLike;
  childGeoms: readonly SceneGeometryLike[];
}> {
  const cols = styleCols(style);
  const gap = styleGap(style);
  const cellWidths: number[] = Array.from({ length: cols }, () => 0);
  const rowCount = Math.max(1, Math.ceil(children.length / cols));
  const rowHeights: number[] = Array.from({ length: rowCount }, () => 0);
  children.forEach((child, index) => {
    const g = geometryOf(child);
    const col = index % cols;
    const row = Math.floor(index / cols);
    cellWidths[col] = Math.max(cellWidths[col]!, g.width);
    rowHeights[row] = Math.max(rowHeights[row]!, g.height);
  });
  const colOffsets: number[] = [];
  let xCursor = 0;
  for (let col = 0; col < cols; col++) {
    colOffsets.push(xCursor);
    xCursor += cellWidths[col]! + gap;
  }
  const rowOffsets: number[] = [];
  let yCursor = 0;
  for (let row = 0; row < rowCount; row++) {
    rowOffsets.push(yCursor);
    yCursor += rowHeights[row]! + gap;
  }
  const childGeoms = children.map((child, index) => {
    const g = geometryOf(child);
    const col = index % cols;
    const row = Math.floor(index / cols);
    return {
      x: colOffsets[col]!,
      y: rowOffsets[row]!,
      width: g.width,
      height: g.height,
    };
  });
  const contentWidth = Math.max(0, xCursor - (children.length > 0 ? gap : 0));
  const contentHeight = Math.max(0, yCursor - (children.length > 0 ? gap : 0));
  return {
    parentGeom: {
      ...parentGeom,
      width: parentGeom.width > 0 ? parentGeom.width : contentWidth,
      height: parentGeom.height > 0 ? parentGeom.height : contentHeight,
    },
    childGeoms,
  };
}

/** Equal-slot rail child local geometries; enlarge parent when needed. */
function computeRailLayout(
  parentGeom: SceneGeometryLike,
  children: readonly SceneNodeLike[],
  style: SceneNodeLike["style"],
): Readonly<{
  parentGeom: SceneGeometryLike;
  childGeoms: readonly SceneGeometryLike[];
}> {
  const direction = styleDirection(style);
  const gap = styleGap(style);
  const count = children.length;
  if (count === 0) {
    return { parentGeom, childGeoms: [] };
  }
  const authored = children.map((child) => geometryOf(child));
  const totalGap = gap * Math.max(count - 1, 0);
  let parentWidth = parentGeom.width;
  let parentHeight = parentGeom.height;
  if (direction === "row") {
    if (parentWidth <= 0) {
      const maxChild = Math.max(...authored.map((g) => g.width), 1);
      parentWidth = maxChild * count + totalGap;
    }
    if (parentHeight <= 0) {
      parentHeight = Math.max(...authored.map((g) => g.height), 0);
    }
    const slot = Math.max((parentWidth - totalGap) / count, 0);
    const childGeoms = authored.map((g, index) => ({
      x: index * (slot + gap),
      y: 0,
      width: slot,
      height: g.height > 0 ? g.height : parentHeight,
    }));
    return {
      parentGeom: { ...parentGeom, width: parentWidth, height: parentHeight },
      childGeoms,
    };
  }
  if (parentHeight <= 0) {
    const maxChild = Math.max(...authored.map((g) => g.height), 1);
    parentHeight = maxChild * count + totalGap;
  }
  if (parentWidth <= 0) {
    parentWidth = Math.max(...authored.map((g) => g.width), 0);
  }
  const slot = Math.max((parentHeight - totalGap) / count, 0);
  const childGeoms = authored.map((g, index) => ({
    x: 0,
    y: index * (slot + gap),
    width: g.width > 0 ? g.width : parentWidth,
    height: slot,
  }));
  return {
    parentGeom: { ...parentGeom, width: parentWidth, height: parentHeight },
    childGeoms,
  };
}

/**
 * Resolve local child geometries for stack/grid/rail parents; otherwise authored.
 * Also returns a possibly auto-sized parent geometry.
 */
function resolveContainerLayout(
  node: SceneNodeLike,
  parentGeom: SceneGeometryLike,
  children: readonly SceneNodeLike[] | undefined,
): Readonly<{
  parentGeom: SceneGeometryLike;
  childGeoms: readonly SceneGeometryLike[] | undefined;
}> {
  if (!Array.isArray(children) || children.length === 0) {
    return { parentGeom, childGeoms: undefined };
  }
  const capability = capabilityOf(node);
  if (capability === "layout.stack") {
    return computeStackLayout(parentGeom, children, node.style);
  }
  if (capability === "layout.grid") {
    return computeGridLayout(parentGeom, children, node.style);
  }
  if (capability === "layout.rail") {
    return computeRailLayout(parentGeom, children, node.style);
  }
  return {
    parentGeom,
    childGeoms: children.map((child) => geometryOf(child)),
  };
}

/** Flatten scene roots (and nested children) into id → node / world-geometry maps. */
function indexSceneNodes(roots: readonly SceneNodeLike[]): SceneNodeIndex {
  const nodesById = new Map<string, SceneNodeLike>();
  const worldGeometryById = new Map<string, SceneGeometryLike>();

  const visit = (
    node: SceneNodeLike,
    originX: number,
    originY: number,
    coordsAreLocal: boolean,
    geometryOverride: SceneGeometryLike | undefined,
  ): void => {
    let authored = geometryOverride ?? geometryOf(node);
    const relative = node.relativePosition;
    if (geometryOverride === undefined && relative !== undefined) {
      const targetWorldGeom = worldGeometryById.get(relative.nodeId);
      if (targetWorldGeom !== undefined) {
        const anchorPoint = nodeAnchorPoint(targetWorldGeom, relative.anchor);
        const worldX = anchorPoint.x + finiteNumber(relative.dx);
        const worldY = anchorPoint.y + finiteNumber(relative.dy);
        authored = {
          ...authored,
          x: coordsAreLocal ? worldX - originX : worldX,
          y: coordsAreLocal ? worldY - originY : worldY,
        };
      }
    }
    const kids = node.children;
    const { parentGeom: laidOutParent, childGeoms } = resolveContainerLayout(
      node,
      authored,
      kids,
    );
    const worldGeom: SceneGeometryLike = coordsAreLocal
      ? {
          x: originX + laidOutParent.x,
          y: originY + laidOutParent.y,
          width: laidOutParent.width,
          height: laidOutParent.height,
        }
      : laidOutParent;
    nodesById.set(node.id, node);
    worldGeometryById.set(node.id, worldGeom);

    if (!Array.isArray(kids) || kids.length === 0) {
      return;
    }
    const local = childrenUseLocalLayout(node, laidOutParent, kids);
    kids.forEach((child, index) => {
      const childOverride = childGeoms?.[index];
      if (local) {
        visit(child, worldGeom.x, worldGeom.y, true, childOverride);
      } else {
        visit(child, 0, 0, false, childOverride);
      }
    });
  };

  for (const root of roots) {
    visit(root, 0, 0, false, undefined);
  }
  return { nodesById, worldGeometryById };
}

/** Center point of world-space geometry. */
function nodeCenter(
  worldGeom: SceneGeometryLike,
): Readonly<{ x: number; y: number }> {
  return {
    x: worldGeom.x + worldGeom.width / 2,
    y: worldGeom.y + worldGeom.height / 2,
  };
}

/**
 * Anchored point on world-space geometry.
 * Edge anchors land on mid-sides so connectors stop at box borders instead of
 * driving center-to-center strokes through fills. Corners use box corners.
 */
function nodeAnchorPoint(
  worldGeom: SceneGeometryLike,
  anchor: string | undefined,
): Readonly<{ x: number; y: number }> {
  const center = nodeCenter(worldGeom);
  const left = worldGeom.x;
  const right = worldGeom.x + worldGeom.width;
  const top = worldGeom.y;
  const bottom = worldGeom.y + worldGeom.height;
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
    case "center":
    default:
      return center;
  }
}

/** Soft / missing anchors that should be upgraded to facing edges for motion. */
function isSoftMotionAnchor(anchor: string | undefined): boolean {
  if (anchor === undefined || anchor.length === 0) {
    return true;
  }
  const token = anchor.toLowerCase();
  return token === "center" || token === "middle" || token === "c";
}

/**
 * Pick perimeter anchors so a traveler runs in the gap between boxes instead
 * of center-to-center through fills.
 */
function facingMotionAnchors(
  fromGeom: SceneGeometryLike,
  toGeom: SceneGeometryLike,
): Readonly<{ from: string; to: string }> {
  const fromCenter = nodeCenter(fromGeom);
  const toCenter = nodeCenter(toGeom);
  const dx = toCenter.x - fromCenter.x;
  const dy = toCenter.y - fromCenter.y;
  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0 ? { from: "e", to: "w" } : { from: "w", to: "e" };
  }
  return dy >= 0 ? { from: "s", to: "n" } : { from: "n", to: "s" };
}

/**
 * Resolve a motion endpoint. Soft `center` anchors on node refs become the
 * facing edge so balls travel along connectors, not through panel fills.
 */
function resolveMotionEndpoint(
  endpoint: ScenePointLike | undefined,
  peer: ScenePointLike | undefined,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
  role: "from" | "to",
): Readonly<{ x: number; y: number }> {
  if (endpoint === undefined) {
    return { x: 0, y: 0 };
  }
  const hasX = typeof endpoint.x === "number" && Number.isFinite(endpoint.x);
  const hasY = typeof endpoint.y === "number" && Number.isFinite(endpoint.y);
  if (hasX && hasY) {
    return {
      x: endpoint.x as number,
      y: endpoint.y as number,
    };
  }
  const nodeId = endpoint.nodeId;
  if (typeof nodeId !== "string" || nodeId.length === 0) {
    return resolveEndpoint(endpoint, index, layoutOrigin);
  }
  const world =
    index.worldGeometryById.get(nodeId) ??
    (index.nodesById.has(nodeId)
      ? geometryOf(index.nodesById.get(nodeId)!)
      : undefined);
  if (world === undefined) {
    return { x: 0, y: 0 };
  }

  let anchor = endpoint.anchor;
  if (isSoftMotionAnchor(anchor) && peer !== undefined) {
    const peerId = peer.nodeId;
    if (typeof peerId === "string" && peerId.length > 0) {
      const peerWorld =
        index.worldGeometryById.get(peerId) ??
        (index.nodesById.has(peerId)
          ? geometryOf(index.nodesById.get(peerId)!)
          : undefined);
      if (peerWorld !== undefined) {
        const facing = facingMotionAnchors(world, peerWorld);
        anchor = role === "from" ? facing.from : facing.to;
      }
    }
  }

  const point = nodeAnchorPoint(world, anchor);
  return {
    x: point.x - layoutOrigin.x,
    y: point.y - layoutOrigin.y,
  };
}

/**
 * Path for a traveling motion signal.
 * Node-anchored endpoints use facing edges; authored paths are clipped to the
 * gaps between boxes so balls only travel on line boundaries — never across
 * the top of the diagram and never through panel fills.
 */
function endpointsMatch(
  left: ScenePointLike | undefined,
  right: ScenePointLike | undefined,
): boolean {
  if (left === undefined || right === undefined) {
    return false;
  }
  if (
    typeof left.nodeId === "string" &&
    left.nodeId.length > 0 &&
    typeof right.nodeId === "string" &&
    right.nodeId.length > 0
  ) {
    return left.nodeId === right.nodeId;
  }
  return (
    typeof left.x === "number" &&
    typeof left.y === "number" &&
    typeof right.x === "number" &&
    typeof right.y === "number" &&
    Math.abs(left.x - right.x) <= 0.001 &&
    Math.abs(left.y - right.y) <= 0.001
  );
}

function visiblePathForMotion(
  motionNode: SceneNodeLike,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): string | undefined {
  const motionFrom = singleScenePoint(motionNode.from);
  const motionTo = singleScenePoint(motionNode.to);
  if (motionFrom === undefined || motionTo === undefined) {
    return undefined;
  }
  for (const candidate of index.nodesById.values()) {
    if (candidate.id === motionNode.id) {
      continue;
    }
    const capability = capabilityOf(candidate);
    if (capability === "core.fan-out" || capability === "core.fan-in") {
      const fanOut = capability === "core.fan-out";
      const singleton = singleScenePoint(fanOut ? candidate.from : candidate.to);
      const many = scenePoints(fanOut ? candidate.to : candidate.from);
      const branchIndex = many.findIndex((endpoint) =>
        fanOut
          ? endpointsMatch(singleton, motionFrom) &&
            endpointsMatch(endpoint, motionTo)
          : endpointsMatch(endpoint, motionFrom) &&
            endpointsMatch(singleton, motionTo),
      );
      if (branchIndex >= 0) {
        return resolveFanGeometry(candidate, index, layoutOrigin).trajectories[
          branchIndex
        ]?.d;
      }
      continue;
    }
    if (
      isMotionSignalNode(candidate, capability) ||
      !isArrowLike(candidate, capability)
    ) {
      continue;
    }
    const candidateFrom = singleScenePoint(candidate.from);
    const candidateTo = singleScenePoint(candidate.to);
    if (
      !endpointsMatch(candidateFrom, motionFrom) ||
      !endpointsMatch(candidateTo, motionTo)
    ) {
      continue;
    }
    const raw = arrowPathData(candidate, index, layoutOrigin);
    if (raw === undefined) {
      continue;
    }
    const tip = tipForArrowNode(candidate, capability);
    return tip === null
      ? raw
      : shortenPathForArrowhead(
          raw,
          strokeWidthFromStyle(candidate.style),
          tip.insetUnits,
        );
  }
  return undefined;
}

function motionSignalPathData(
  node: SceneNodeLike,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): string | undefined {
  const from = singleScenePoint(node.from);
  const to = singleScenePoint(node.to);
  const hasNodeEndpoint =
    (typeof from?.nodeId === "string" && from.nodeId.length > 0) ||
    (typeof to?.nodeId === "string" && to.nodeId.length > 0);

  let raw: string | undefined;
  const visible = visiblePathForMotion(node, index, layoutOrigin);
  if (visible !== undefined) {
    return visible;
  }
  if (hasNodeEndpoint) {
    const start = resolveMotionEndpoint(from, to, index, layoutOrigin, "from");
    const end = resolveMotionEndpoint(to, from, index, layoutOrigin, "to");
    raw =
      node.via !== undefined || connectorAxisOf(node) !== undefined
        ? elbowPathData(
            start,
            end,
            node.via === undefined
              ? undefined
              : resolveEndpoint(node.via, index, layoutOrigin),
            connectorAxisOf(node),
          )
        : `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} L${formatPathNumber(end.x)} ${formatPathNumber(end.y)}`;
  } else {
    const authored =
      typeof node.d === "string" && node.d.length > 0
        ? node.d
        : typeof node.path === "string" && node.path.length > 0
          ? node.path
          : undefined;
    if (authored !== undefined) {
      raw = authored;
    } else if (Array.isArray(node.points) && node.points.length > 0) {
      raw = polylinePathData(node.points, index, layoutOrigin);
    } else if (from !== undefined || to !== undefined) {
      const start = resolveEndpoint(from, index, layoutOrigin);
      const end = resolveEndpoint(to, index, layoutOrigin);
      if (
        (Math.abs(start.x) > 0.5 || Math.abs(start.y) > 0.5) &&
        (Math.abs(end.x) > 0.5 || Math.abs(end.y) > 0.5)
      ) {
        raw = `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} L${formatPathNumber(end.x)} ${formatPathNumber(end.y)}`;
      }
    }
  }

  if (raw === undefined) {
    return undefined;
  }
  // Keep travelers on inter-box corridors only (hide if nothing remains).
  return boundaryOnlyMotionPath(raw, index, layoutOrigin);
}

/** Minimum clear gap between boxes to treat as a connector corridor. */
const MOTION_BOUNDARY_GAP_MIN = 8;

function boxGeometriesForMotion(
  index: SceneNodeIndex,
): readonly SceneGeometryLike[] {
  const boxes: SceneGeometryLike[] = [];
  for (const [id, geom] of index.worldGeometryById) {
    if (!(geom.width >= 24 && geom.height >= 24)) {
      continue;
    }
    const node = index.nodesById.get(id);
    if (node === undefined) {
      continue;
    }
    const cap = capabilityOf(node);
    if (
      isArrowLike(node, cap) ||
      isMotionSignalNode(node, cap) ||
      isDotLike(node, cap) ||
      isPulseNode(node, cap) ||
      cap === "core.text" ||
      cap === "core.band" ||
      node.kind === "text"
    ) {
      continue;
    }
    // Stroke-only overlays / lanes are not solid obstacles.
    const fill = node.style?.fill;
    if (
      fill === "none" ||
      fill === "transparent" ||
      (typeof fill === "string" && fill.toLowerCase() === "rgba(0, 0, 0, 0)")
    ) {
      continue;
    }
    const role = node.style?.role;
    if (
      role === "band" ||
      role === "lane" ||
      role === "overlay" ||
      role === "pulse"
    ) {
      continue;
    }
    const lowerId = id.toLowerCase();
    if (
      lowerId.includes("band") ||
      lowerId.includes("-lane") ||
      lowerId.startsWith("lane-")
    ) {
      continue;
    }
    boxes.push(geom);
  }
  return boxes;
}

/**
 * Clip a straight motion guide to segments that only exist in the gaps between
 * boxes (the line boundaries). Returns undefined when no corridor remains.
 */
function boundaryOnlyMotionPath(
  d: string,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): string | undefined {
  const trimmed = d.trim();
  const hMatch =
    /^M\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+H\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/i.exec(
      trimmed,
    );
  if (
    hMatch?.[1] !== undefined &&
    hMatch[2] !== undefined &&
    hMatch[3] !== undefined
  ) {
    return horizontalBoundaryCorridors(
      Number(hMatch[1]),
      Number(hMatch[3]),
      Number(hMatch[2]),
      index,
      layoutOrigin,
    );
  }
  const vMatch =
    /^M\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+V\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/i.exec(
      trimmed,
    );
  if (
    vMatch?.[1] !== undefined &&
    vMatch[2] !== undefined &&
    vMatch[3] !== undefined
  ) {
    return verticalBoundaryCorridors(
      Number(vMatch[1]),
      Number(vMatch[2]),
      Number(vMatch[3]),
      index,
      layoutOrigin,
    );
  }
  const lMatch =
    /^M\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+L\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?:\s+|,)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/i.exec(
      trimmed,
    );
  if (
    lMatch?.[1] !== undefined &&
    lMatch[2] !== undefined &&
    lMatch[3] !== undefined &&
    lMatch[4] !== undefined
  ) {
    const x1 = Number(lMatch[1]);
    const y1 = Number(lMatch[2]);
    const x2 = Number(lMatch[3]);
    const y2 = Number(lMatch[4]);
    const dx = Math.abs(x2 - x1);
    const dy = Math.abs(y2 - y1);
    if (dy <= 1.5 && dx > dy) {
      return horizontalBoundaryCorridors(x1, x2, (y1 + y2) / 2, index, layoutOrigin);
    }
    if (dx <= 1.5 && dy > dx) {
      return verticalBoundaryCorridors(x1, y1, y2, index, layoutOrigin);
    }
    // Diagonal / elbow: keep only if the open segment does not pierce a box.
    if (!segmentPiercesBox(x1, y1, x2, y2, index, layoutOrigin)) {
      return trimmed;
    }
    return undefined;
  }
  // Complex authored curves: keep only when they stay outside box fills.
  if (!pathPiercesBoxes(trimmed, index, layoutOrigin)) {
    return trimmed;
  }
  return undefined;
}

function horizontalBoundaryCorridors(
  x1: number,
  x2: number,
  y: number,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): string | undefined {
  const xMin = Math.min(x1, x2);
  const xMax = Math.max(x1, x2);
  const worldY = y + layoutOrigin.y;
  const worldXMin = xMin + layoutOrigin.x;
  const worldXMax = xMax + layoutOrigin.x;
  const hit = boxGeometriesForMotion(index)
    .filter(
      (box) =>
        box.x < worldXMax &&
        box.x + box.width > worldXMin &&
        box.y < worldY + 2 &&
        box.y + box.height > worldY - 2,
    )
    .slice()
    .sort((a, b) => a.x - b.x);
  if (hit.length === 0) {
    return `M${formatPathNumber(x1)} ${formatPathNumber(y)} H${formatPathNumber(x2)}`;
  }
  // Keep authored corridor Y so travelers stay on the connector lane.
  const corridorY = y;
  const parts: string[] = [];
  const pushGap = (gapStartLocal: number, gapEndLocal: number) => {
    if (gapEndLocal - gapStartLocal < MOTION_BOUNDARY_GAP_MIN) {
      return;
    }
    const fromX = x1 <= x2 ? gapStartLocal : gapEndLocal;
    const toX = x1 <= x2 ? gapEndLocal : gapStartLocal;
    parts.push(
      `M${formatPathNumber(fromX)} ${formatPathNumber(corridorY)} H${formatPathNumber(toX)}`,
    );
  };
  // Leading span before the first box.
  pushGap(xMin, hit[0]!.x - layoutOrigin.x);
  for (let i = 0; i < hit.length - 1; i++) {
    const left = hit[i]!;
    const right = hit[i + 1]!;
    pushGap(left.x + left.width - layoutOrigin.x, right.x - layoutOrigin.x);
  }
  // Trailing span after the last box.
  const last = hit[hit.length - 1]!;
  pushGap(last.x + last.width - layoutOrigin.x, xMax);
  return parts.length > 0 ? parts.join(" ") : undefined;
}

function verticalBoundaryCorridors(
  x: number,
  y1: number,
  y2: number,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): string | undefined {
  const yMin = Math.min(y1, y2);
  const yMax = Math.max(y1, y2);
  const worldX = x + layoutOrigin.x;
  const worldYMin = yMin + layoutOrigin.y;
  const worldYMax = yMax + layoutOrigin.y;
  const hit = boxGeometriesForMotion(index)
    .filter(
      (box) =>
        box.y < worldYMax &&
        box.y + box.height > worldYMin &&
        box.x < worldX + 2 &&
        box.x + box.width > worldX - 2,
    )
    .slice()
    .sort((a, b) => a.y - b.y);
  if (hit.length === 0) {
    return `M${formatPathNumber(x)} ${formatPathNumber(y1)} V${formatPathNumber(y2)}`;
  }
  const corridorX = x;
  const parts: string[] = [];
  const pushGap = (gapStartLocal: number, gapEndLocal: number) => {
    if (gapEndLocal - gapStartLocal < MOTION_BOUNDARY_GAP_MIN) {
      return;
    }
    const fromY = y1 <= y2 ? gapStartLocal : gapEndLocal;
    const toY = y1 <= y2 ? gapEndLocal : gapStartLocal;
    parts.push(
      `M${formatPathNumber(corridorX)} ${formatPathNumber(fromY)} V${formatPathNumber(toY)}`,
    );
  };
  pushGap(yMin, hit[0]!.y - layoutOrigin.y);
  for (let i = 0; i < hit.length - 1; i++) {
    const top = hit[i]!;
    const bottom = hit[i + 1]!;
    pushGap(top.y + top.height - layoutOrigin.y, bottom.y - layoutOrigin.y);
  }
  const last = hit[hit.length - 1]!;
  pushGap(last.y + last.height - layoutOrigin.y, yMax);
  return parts.length > 0 ? parts.join(" ") : undefined;
}

function pointInsideBox(
  x: number,
  y: number,
  box: SceneGeometryLike,
  inset = 1,
): boolean {
  return (
    x > box.x + inset &&
    x < box.x + box.width - inset &&
    y > box.y + inset &&
    y < box.y + box.height - inset
  );
}

function segmentPiercesBox(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): boolean {
  const wx1 = x1 + layoutOrigin.x;
  const wy1 = y1 + layoutOrigin.y;
  const wx2 = x2 + layoutOrigin.x;
  const wy2 = y2 + layoutOrigin.y;
  for (const box of boxGeometriesForMotion(index)) {
    // Sample the open segment (skip endpoints which may sit on edges).
    for (let i = 1; i <= 4; i++) {
      const t = i / 5;
      const x = wx1 + (wx2 - wx1) * t;
      const y = wy1 + (wy2 - wy1) * t;
      if (pointInsideBox(x, y, box)) {
        return true;
      }
    }
  }
  return false;
}

function pathPiercesBoxes(
  d: string,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): boolean {
  if (typeof document === "undefined") {
    return false;
  }
  try {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    if (typeof path.getTotalLength !== "function") {
      return false;
    }
    const total = path.getTotalLength();
    if (!(total > 0)) {
      return false;
    }
    const boxes = boxGeometriesForMotion(index);
    const samples = Math.max(8, Math.min(48, Math.ceil(total / 12)));
    for (let i = 1; i < samples; i++) {
      const point = path.getPointAtLength((total * i) / samples);
      const x = point.x + layoutOrigin.x;
      const y = point.y + layoutOrigin.y;
      for (const box of boxes) {
        if (pointInsideBox(x, y, box)) {
          return true;
        }
      }
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Resolve an endpoint into the current drawing frame.
 * Explicit `x`/`y` are already in-frame; `nodeId` anchors are world-space and
 * rebased by subtracting `layoutOrigin` (ancestor group/container translates).
 */
function resolveEndpoint(
  endpoint: ScenePointLike | undefined,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): Readonly<{ x: number; y: number }> {
  if (endpoint === undefined) {
    return { x: 0, y: 0 };
  }
  const hasX = typeof endpoint.x === "number" && Number.isFinite(endpoint.x);
  const hasY = typeof endpoint.y === "number" && Number.isFinite(endpoint.y);
  // Require both coordinates for an absolute point; a lone axis would pin the
  // other to 0 and drag connectors into the wrong corner.
  if (hasX && hasY) {
    return {
      x: endpoint.x as number,
      y: endpoint.y as number,
    };
  }
  if (typeof endpoint.nodeId === "string" && endpoint.nodeId.length > 0) {
    const world =
      index.worldGeometryById.get(endpoint.nodeId) ??
      (index.nodesById.has(endpoint.nodeId)
        ? geometryOf(index.nodesById.get(endpoint.nodeId)!)
        : undefined);
    if (world !== undefined) {
      const point = nodeAnchorPoint(world, endpoint.anchor);
      return {
        x: point.x - layoutOrigin.x,
        y: point.y - layoutOrigin.y,
      };
    }
  }
  return { x: 0, y: 0 };
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

function pointCentroid(points: readonly ScenePoint[]): ScenePoint {
  if (points.length === 0) {
    return { x: 0, y: 0 };
  }
  const sum = points.reduce(
    (total, point) => ({ x: total.x + point.x, y: total.y + point.y }),
    { x: 0, y: 0 },
  );
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function facingAnchorToPoint(
  geometry: SceneGeometryLike,
  peer: ScenePoint,
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
function resolveFanEndpointWorld(
  endpoint: ScenePointLike,
  peer: ScenePoint,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): ScenePoint {
  const hasX = typeof endpoint.x === "number" && Number.isFinite(endpoint.x);
  const hasY = typeof endpoint.y === "number" && Number.isFinite(endpoint.y);
  if (hasX && hasY) {
    return {
      x: (endpoint.x as number) + layoutOrigin.x,
      y: (endpoint.y as number) + layoutOrigin.y,
    };
  }
  if (typeof endpoint.nodeId !== "string" || endpoint.nodeId.length === 0) {
    return { x: layoutOrigin.x, y: layoutOrigin.y };
  }
  const world =
    index.worldGeometryById.get(endpoint.nodeId) ??
    (index.nodesById.has(endpoint.nodeId)
      ? geometryOf(index.nodesById.get(endpoint.nodeId)!)
      : undefined);
  if (world === undefined) {
    return { x: layoutOrigin.x, y: layoutOrigin.y };
  }
  const anchor = isSoftMotionAnchor(endpoint.anchor)
    ? facingAnchorToPoint(world, peer)
    : endpoint.anchor;
  return nodeAnchorPoint(world, anchor);
}

function fanPath(points: readonly ScenePoint[]): string {
  const compact: ScenePoint[] = [];
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
      (d, point) =>
        `${d} L${formatPathNumber(point.x)} ${formatPathNumber(point.y)}`,
      `M${formatPathNumber(first.x)} ${formatPathNumber(first.y)}`,
    );
}

function fanBranchPoints(
  start: ScenePoint,
  junction: ScenePoint,
  axis: "x" | "y",
  incoming: boolean,
): readonly ScenePoint[] {
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
  start: ScenePoint,
  end: ScenePoint,
  axis: "x" | "y",
): readonly ScenePoint[] {
  return axis === "x"
    ? [start, { x: end.x, y: start.y }, end]
    : [start, { x: start.x, y: end.y }, end];
}

function automaticFanJunction(
  singleton: ScenePoint,
  many: readonly ScenePoint[],
  axis: "x" | "y",
): ScenePoint {
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

function pointsNear(a: ScenePoint, b: ScenePoint, eps = 0.001): boolean {
  return Math.abs(a.x - b.x) <= eps && Math.abs(a.y - b.y) <= eps;
}

/** Collapse a polyline into atomic horizontal / vertical spans. */
function atomicOrthogonalSpans(
  points: readonly ScenePoint[],
): readonly Readonly<{ start: ScenePoint; end: ScenePoint }>[] {
  const compact: ScenePoint[] = [];
  for (const point of points) {
    const previous = compact.at(-1);
    if (previous === undefined || !pointsNear(previous, point)) {
      compact.push(point);
    }
  }
  const spans: Array<Readonly<{ start: ScenePoint; end: ScenePoint }>> = [];
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

type FanAtomicSpan = Readonly<{
  axis: "h" | "v";
  fixed: number;
  from: number;
  to: number;
  role: "trunk" | "branch" | "merge-trunk";
  destination?: ScenePoint;
}>;

function toFanAtomicSpan(
  start: ScenePoint,
  end: ScenePoint,
  role: "trunk" | "branch" | "merge-trunk",
  destination: ScenePoint | undefined,
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
    const key = `${span.axis}:${formatPathNumber(span.fixed)}`;
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

function fanSegmentFromAtomic(
  id: string,
  span: FanAtomicSpan,
): FanSegment {
  const start =
    span.axis === "h"
      ? { x: span.from, y: span.fixed }
      : { x: span.fixed, y: span.from };
  const end =
    span.axis === "h"
      ? { x: span.to, y: span.fixed }
      : { x: span.fixed, y: span.to };
  const destination = span.destination;
  const directed =
    destination === undefined
      ? { start, end }
      : pointsNear(end, destination)
        ? { start, end }
        : pointsNear(start, destination)
          ? { start: end, end: start }
          : { start, end };
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
  trunkPoints: readonly ScenePoint[],
  trunkRole: "trunk" | "merge-trunk",
  branchPointSets: readonly (readonly ScenePoint[])[],
  destinations: readonly ScenePoint[],
): readonly FanSegment[] {
  const corridors: FanAtomicSpan[] = [];
  const terminals: FanAtomicSpan[] = [];
  const pushPoints = (
    points: readonly ScenePoint[],
    role: "trunk" | "branch" | "merge-trunk",
    destination: ScenePoint | undefined,
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
    const key = `${terminal.axis}:${formatPathNumber(terminal.fixed)}:${formatPathNumber(terminal.from)}:${formatPathNumber(terminal.to)}`;
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

function fanPathPoints(d: string): ScenePoint[] {
  const points: ScenePoint[] = [];
  const token =
    /[ML]\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)/gi;
  for (const match of d.matchAll(token)) {
    points.push({ x: Number(match[1]), y: Number(match[2]) });
  }
  return points;
}

/** Split a fan trajectory at the junction into trunk and branch ball paths. */
function splitFanTrajectoryAtJunction(
  d: string,
  junction: ScenePoint,
): Readonly<{ head: string; tail: string }> | undefined {
  const points = fanPathPoints(d);
  if (points.length < 2) {
    return undefined;
  }
  let junctionIndex = points.findIndex((point) => pointsNear(point, junction));
  if (junctionIndex < 0) {
      // Authored bend: pick the orthogonal corner closest to the junction.
    let best = 0;
    let bestDist = Number.POSITIVE_INFINITY;
    points.forEach((point, index) => {
      const dist =
        Math.abs(point.x - junction.x) + Math.abs(point.y - junction.y);
      if (dist < bestDist) {
        bestDist = dist;
        best = index;
      }
    });
    junctionIndex = best;
  }
  if (junctionIndex <= 0 || junctionIndex >= points.length - 1) {
    return undefined;
  }
  return {
    head: fanPath(points.slice(0, junctionIndex + 1)),
    tail: fanPath(points.slice(junctionIndex)),
  };
}

/**
 * Resolve fan topology in world space, then rebase it to the current layout.
 * Paint uses merged atomic spans; trajectories stay complete source→destination.
 */
export function resolveFanGeometry(
  node: SceneNodeLike,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin = ZERO_ORIGIN,
): ResolvedFanGeometry {
  const capability =
    capabilityOf(node) === "core.fan-in" ? "core.fan-in" : "core.fan-out";
  const fanOut = capability === "core.fan-out";
  const from = scenePoints(node.from);
  const to = scenePoints(node.to);
  const singletonEndpoint = (fanOut ? from[0] : to[0]) ?? {};
  const manyEndpoints = fanOut ? to : from;

  const roughSingleton = resolveFanEndpointWorld(
    singletonEndpoint,
    { x: layoutOrigin.x, y: layoutOrigin.y },
    index,
    layoutOrigin,
  );
  const roughMany = manyEndpoints.map((endpoint) =>
    resolveFanEndpointWorld(endpoint, roughSingleton, index, layoutOrigin),
  );
  const roughManyCentroid = pointCentroid(roughMany);
  const singleton = resolveFanEndpointWorld(
    singletonEndpoint,
    roughManyCentroid,
    index,
    layoutOrigin,
  );
  const many = manyEndpoints.map((endpoint) =>
    resolveFanEndpointWorld(endpoint, singleton, index, layoutOrigin),
  );
  const manyCentroid = pointCentroid(many);
  const axis =
    connectorAxisOf(node) ??
    (Math.abs(manyCentroid.x - singleton.x) >=
    Math.abs(manyCentroid.y - singleton.y)
      ? "x"
      : "y");
  const authoredJunction = singleScenePoint(node.junction);
  const junctionWorld =
    authoredJunction === undefined
      ? automaticFanJunction(singleton, many, axis)
      : resolveFanEndpointWorld(
          authoredJunction,
          fanOut ? manyCentroid : singleton,
          index,
          layoutOrigin,
        );
  const local = (point: ScenePoint): ScenePoint => ({
    x: point.x - layoutOrigin.x,
    y: point.y - layoutOrigin.y,
  });
  const singletonLocal = local(singleton);
  const manyLocal = many.map(local);
  const junction = local(junctionWorld);

  const trunkPoints = fanOut
    ? orthogonalFanPoints(singletonLocal, junction, axis)
    : orthogonalFanPoints(junction, singletonLocal, axis);
  const trunkRole = fanOut ? "trunk" : "merge-trunk";
  const branchPointSets = manyLocal.map((endpoint) =>
    fanBranchPoints(endpoint, junction, axis, !fanOut),
  );
  const destinations = fanOut ? manyLocal : [singletonLocal];
  const trajectories = manyLocal.map((_endpoint, branchIndex): FanTrajectory => {
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

function polylinePathData(
  points: readonly ScenePointLike[],
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): string | undefined {
  if (points.length === 0) {
    return undefined;
  }
  const resolved = points.map((point) =>
    resolveEndpoint(point, index, layoutOrigin),
  );
  const [first, ...rest] = resolved;
  if (first === undefined) {
    return undefined;
  }
  let d = `M${first.x} ${first.y}`;
  for (const point of rest) {
    d += ` L${point.x} ${point.y}`;
  }
  return d;
}

function timelineDurationMs(timeline: readonly SceneTimelineCueLike[]): number {
  return timeline.reduce(
    (maximum, cue) =>
      Math.max(maximum, finiteNumber(cue.at) + finiteNumber(cue.duration)),
    0,
  );
}

function isEnterLikeAction(action: string): boolean {
  return action === "enter" || action === "reveal";
}

function isDrawAction(action: string): boolean {
  return (
    action === "draw" || action === "trace" || action === "reveal-stroke"
  );
}

function isEmphasizeAction(action: string): boolean {
  return action === "emphasize" || action === "emphasis";
}

function isPulseAction(action: string): boolean {
  return action === "pulse";
}

function isFadeLikeAction(action: string): boolean {
  return action === "fade" || action === "exit";
}

function isStaggerLikeAction(action: string): boolean {
  return action === "stagger" || action === "enter-children";
}

/** Direct child ids of a group/panel node (for `enter-children` expansion). */
function directChildIds(node: SceneNodeLike | undefined): readonly string[] {
  if (node === undefined || !Array.isArray(node.children)) {
    return [];
  }
  return node.children
    .map((child) => child.id)
    .filter((id) => typeof id === "string" && id.length > 0);
}

function resolveStaggerTargets(
  cue: SceneTimelineCueLike,
  nodesById: ReadonlyMap<string, SceneNodeLike>,
): readonly string[] {
  if (Array.isArray(cue.targets) && cue.targets.length > 0) {
    return cue.targets.filter((id) => typeof id === "string" && id.length > 0);
  }
  if (cue.action === "enter-children" && cue.target.length > 0) {
    return directChildIds(nodesById.get(cue.target));
  }
  return [];
}

/**
 * Expand compact `stagger` / `enter-children` cues into per-target enter cues
 * with stepped `at`. Other cues pass through unchanged.
 */
function expandTimelineCues(
  timeline: readonly SceneTimelineCueLike[],
  nodesById: ReadonlyMap<string, SceneNodeLike>,
): readonly SceneTimelineCueLike[] {
  const expanded: SceneTimelineCueLike[] = [];
  for (const cue of timeline) {
    if (!isStaggerLikeAction(cue.action)) {
      expanded.push(cue);
      continue;
    }
    const targets = resolveStaggerTargets(cue, nodesById);
    if (targets.length === 0) {
      // Keep the cue so duration accounting is not silently lost.
      expanded.push({
        ...cue,
        action: "enter",
        target: cue.target.length > 0 ? cue.target : cue.id,
      });
      continue;
    }
    const step = Math.max(0, finiteNumber(cue.step, 80));
    targets.forEach((targetId, index) => {
      expanded.push({
        id: `${cue.id}__${index}`,
        at: finiteNumber(cue.at) + index * step,
        duration: finiteNumber(cue.duration),
        action: "enter",
        target: targetId,
        easing: cue.easing,
      });
    });
  }
  return expanded;
}

function isArrowLike(node: SceneNodeLike, capability: string): boolean {
  if (ARROW_CAPABILITIES.has(capability)) {
    return true;
  }
  return typeof node.kind === "string" && ARROW_KINDS.has(node.kind);
}

function isFanNode(node: SceneNodeLike, capability: string): boolean {
  return (
    capability === "core.fan-out" ||
    capability === "core.fan-in" ||
    node.kind === "fan"
  );
}

function isDotLike(node: SceneNodeLike, capability: string): boolean {
  if (DOT_CAPABILITIES.has(capability)) {
    return true;
  }
  if (typeof node.kind === "string" && DOT_KINDS.has(node.kind)) {
    return true;
  }
  // Never promote rect / panel chrome: authors use `r` as corner radius.
  if (
    capability === "core.rect" ||
    capability === "core.panel" ||
    capability === "core.header" ||
    capability === "core.circle" ||
    capability === "core.ellipse" ||
    node.kind === "rect" ||
    node.kind === "circle" ||
    node.kind === "ellipse"
  ) {
    return false;
  }
  // Legacy bare nodes with `style.r` and no capability → small motion/dot mark.
  if (capability.length > 0) {
    return false;
  }
  const radius = node.style?.r;
  return typeof radius === "number" && Number.isFinite(radius) && radius > 0;
}

function isCircleOrEllipse(node: SceneNodeLike, capability: string): boolean {
  if (capability === "core.circle" || capability === "core.ellipse") {
    return true;
  }
  return node.kind === "circle" || node.kind === "ellipse";
}

function isElbowConnector(node: SceneNodeLike, capability: string): boolean {
  if (capability === "core.elbow" || capability === "core.route") {
    return true;
  }
  if (node.style?.route === "elbow") {
    return true;
  }
  if (node.kind === "elbow") {
    return true;
  }
  return false;
}

function connectorAxisOf(
  node: SceneNodeLike,
): "x" | "y" | undefined {
  if (node.axis === "x" || node.axis === "y") {
    return node.axis;
  }
  const styled = node.style?.axis;
  if (styled === "x" || styled === "y") {
    return styled;
  }
  return undefined;
}

/**
 * Orthogonal elbow path: `M x1 y1 H/V mid H/V x2 y2`.
 * `via` supplies the bend coordinate; otherwise midpoint. `axis` prefers the
 * first segment direction (`x` → horizontal first).
 */
function elbowPathData(
  start: Readonly<{ x: number; y: number }>,
  end: Readonly<{ x: number; y: number }>,
  via: Readonly<{ x: number; y: number }> | undefined,
  axis: "x" | "y" | undefined,
): string {
  const dx = Math.abs(end.x - start.x);
  const dy = Math.abs(end.y - start.y);
  const preferX =
    axis === "y" ? false : axis === "x" ? true : dx >= dy;
  if (via !== undefined) {
    // Route through the via bend point (both axes), then finish to end.
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

/**
 * Legacy static companion beside a motion path (`s9-motion-sig-dot`).
 * MotionSignal on the path owns the traveling visual — drop these from the tree.
 */
function isMotionCompanionDot(node: SceneNodeLike, capability = ""): boolean {
  const cap = capability.length > 0 ? capability : capabilityOf(node);
  if (!isDotLike(node, cap)) {
    return false;
  }
  const role = node.style?.role;
  if (role === "motion-signal" || role === "motion-dot") {
    return true;
  }
  return /motion[-_]?sig/i.test(node.id) && /-dot$/i.test(node.id);
}

/** Drop obsolete motion companion dots (and strip them from nested children). */
function omitMotionCompanionDots(
  nodes: readonly SceneNodeLike[] | undefined,
): SceneNodeLike[] {
  if (!Array.isArray(nodes) || nodes.length === 0) {
    return [];
  }
  const out: SceneNodeLike[] = [];
  for (const node of nodes) {
    if (isMotionCompanionDot(node)) {
      continue;
    }
    const kids = node.children;
    if (!Array.isArray(kids) || kids.length === 0) {
      out.push(node);
      continue;
    }
    const nextKids = omitMotionCompanionDots(kids);
    out.push(
      nextKids.length === kids.length &&
        nextKids.every((child, i) => child === kids[i])
        ? node
        : { ...node, children: nextKids },
    );
  }
  return out;
}

/** Drop timeline cues whose only target was a stripped companion dot. */
function omitCompanionTimelineCues(
  cues: readonly SceneTimelineCueLike[],
  nodesById: ReadonlyMap<string, SceneNodeLike>,
): SceneTimelineCueLike[] {
  return cues.filter((cue) => {
    const target = cue.target;
    if (typeof target !== "string" || target.length === 0) {
      return true;
    }
    if (nodesById.has(target)) {
      return true;
    }
    // Orphan cue that named a companion (`…-motion-sig-dot`).
    return !(/motion[-_]?sig/i.test(target) && /-dot$/i.test(target));
  });
}

/** Traveling MentalModel-style motion dots (often authored as `motion-sig` paths). */
function isMotionSignalNode(node: SceneNodeLike, capability = ""): boolean {
  const cap = capability.length > 0 ? capability : capabilityOf(node);
  if (isDotLike(node, cap)) {
    return false;
  }
  if (MOTION_SIGNAL_CAPABILITIES.has(cap)) {
    return true;
  }
  if (/motion[-_]?sig/i.test(node.id)) {
    return true;
  }
  const label = (node.accessibility?.label ?? "").toLowerCase();
  if (label.includes("motion signal")) {
    return true;
  }
  const motion = node.style?.motion;
  const role = node.style?.role;
  return (
    motion === true ||
    motion === 1 ||
    motion === "signal" ||
    motion === "dot" ||
    role === "motion" ||
    role === "motion-signal"
  );
}

/** Rects tagged for a gentle float/pulse (style.pulse, motion.pulse, or pulse-* ids). */
function isPulseNode(node: SceneNodeLike, capability = ""): boolean {
  const cap = capability.length > 0 ? capability : capabilityOf(node);
  if (cap === "motion.pulse") {
    return true;
  }
  const pulse = node.style?.pulse;
  if (
    pulse === true ||
    pulse === 1 ||
    pulse === "true" ||
    (typeof pulse === "string" && pulse.length > 0)
  ) {
    return true;
  }
  const label = (node.accessibility?.label ?? "").toLowerCase();
  if (label.includes("motion pulse") || label === "pulse") {
    return true;
  }
  const id = node.id.toLowerCase();
  return (
    id.startsWith("pulse-") ||
    id.startsWith("pulse_") ||
    id.includes("-pulse") ||
    id.includes("_pulse")
  );
}

/**
 * True when pulse should animate outline opacity (legacy MentalModel overlay).
 * Filled content boxes keep opaque fills — only stroke scale / float pulse.
 */
function isOutlinePulseFill(fillPaint: string): boolean {
  const token = fillPaint.trim().toLowerCase();
  return token === "none" || token === "transparent" || token === "rgba(0, 0, 0, 0)";
}

function markerEndDisabled(style: SceneNodeLike["style"]): boolean {
  if (
    style?.arrowhead === false ||
    style?.arrowhead === 0 ||
    style?.arrowhead === "false"
  ) {
    return true;
  }
  return isMarkerEndNone(style?.markerEnd);
}

/** Resolve tip geometry when this node will show an arrowhead. */
function tipForArrowNode(
  node: SceneNodeLike,
  capability: string,
): ResolvedMarkerTip | null {
  if (!shouldShowArrowhead(node, capability)) {
    return null;
  }
  return resolveMarkerTip(node.style?.markerEnd, DEFAULT_MARKER_TIP);
}

/** True when the author opted into an explicit tip (not just kind defaults). */
function hasExplicitMarkerEnd(style: SceneNodeLike["style"]): boolean {
  return style?.markerEnd !== undefined && !markerEndDisabled(style);
}

/**
 * Directed edges get arrowheads; motion guides and undirected strokes do not.
 *
 * Package IR stamps `kind: "connector"` on `core.path` / `core.line` /
 * `core.bracket`. Capability must win for braces (undirected); authored
 * path/line edges with endpoints are directed like connectors.
 */
function shouldShowArrowhead(node: SceneNodeLike, capability: string): boolean {
  if (isMotionSignalNode(node, capability) || markerEndDisabled(node.style)) {
    return false;
  }
  // Visual dividers / rules are never directed (matches flow-verifier).
  if (/^(split|divider|rule|sep|guide)([-_]|$)/i.test(node.id)) {
    return false;
  }
  // Braces are undirected geometry (desugar also stamps markerEnd: "none").
  if (capability === "core.bracket" || node.kind === "bracket") {
    return hasExplicitMarkerEnd(node.style);
  }
  if (
    capability === "core.arrow" ||
    capability === "core.connector" ||
    capability === "core.elbow" ||
    capability === "core.route" ||
    capability === "core.fan-out" ||
    capability === "core.fan-in" ||
    capability === "core.path" ||
    capability === "core.line" ||
    node.kind === "arrow" ||
    node.kind === "connector" ||
    node.kind === "elbow" ||
    node.kind === "fan" ||
    node.kind === "path" ||
    node.kind === "line"
  ) {
    return true;
  }
  return false;
}

function isDashedStyle(style: SceneNodeLike["style"]): boolean {
  if (style === undefined) {
    return false;
  }
  if (style.dashed === true || style.dashed === 1 || style.dashed === "true") {
    return true;
  }
  const strokeStyle = style.strokeStyle ?? style.variant;
  if (strokeStyle === "dashed" || strokeStyle === "dotted") {
    return true;
  }
  const dash = style.strokeDasharray ?? style.dashArray;
  return (
    (typeof dash === "string" && dash.length > 0 && dash !== "none") ||
    (typeof dash === "number" && Number.isFinite(dash))
  );
}

function authoredStrokeDasharray(
  style: SceneNodeLike["style"],
): string | undefined {
  if (style === undefined) {
    return undefined;
  }
  const dash = style.strokeDasharray ?? style.dashArray;
  if (typeof dash === "string" && dash.length > 0 && dash !== "none") {
    return dash;
  }
  if (typeof dash === "number" && Number.isFinite(dash)) {
    return String(dash);
  }
  const strokeStyle = style.strokeStyle ?? style.variant;
  if (strokeStyle === "dashed") {
    return DASHED_STROKE;
  }
  if (strokeStyle === "dotted") {
    return DOTTED_STROKE;
  }
  if (style.dashed === true || style.dashed === 1 || style.dashed === "true") {
    return DASHED_STROKE;
  }
  return undefined;
}

function circleRadius(node: SceneNodeLike, geom: SceneGeometryLike): number {
  const styled = node.style?.r;
  if (typeof styled === "number" && Number.isFinite(styled) && styled > 0) {
    return styled;
  }
  const fromBox = Math.min(geom.width, geom.height) / 2;
  if (fromBox > 0) {
    return fromBox;
  }
  return DEFAULT_DOT_RADIUS;
}

function enterCueForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
): SceneTimelineCueLike | undefined {
  return timeline
    .filter(
      (candidate) =>
        candidate.target === nodeId && isEnterLikeAction(candidate.action),
    )
    .at(-1);
}

function fadeCueForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
): SceneTimelineCueLike | undefined {
  return timeline
    .filter(
      (candidate) =>
        candidate.target === nodeId && isFadeLikeAction(candidate.action),
    )
    .at(-1);
}

/** Map cue `at`/`duration` onto opacity and enter state for one node. */
function appearanceForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): TimelineAppearance {
  const enterCue = enterCueForNode(nodeId, timeline);
  const fadeCue = fadeCueForNode(nodeId, timeline);

  let enterOpacity = 1;
  let state: TimelineState = "unchanged";

  if (enterCue !== undefined) {
    const progress = cueProgress(enterCue, playbackTimeMs);
    if (progress <= 0) {
      enterOpacity = 0;
      state = "hidden";
    } else if (progress >= 1) {
      enterOpacity = 1;
      state = "revealed";
    } else {
      enterOpacity = progress;
      state = "entering";
    }
  }

  if (fadeCue !== undefined) {
    const fadeAt = Math.max(0, finiteNumber(fadeCue.at));
    if (playbackTimeMs >= fadeAt) {
      const fadeProgress = cueProgress(fadeCue, playbackTimeMs);
      const opacity = enterOpacity * (1 - fadeProgress);
      if (fadeCue.action === "exit" && fadeProgress >= 1) {
        return { state: "hidden", opacity: 0 };
      }
      if (fadeProgress >= 1) {
        return { state: "hidden", opacity: 0 };
      }
      return {
        state: opacity <= 0 ? "hidden" : state === "unchanged" ? "entering" : state,
        opacity,
      };
    }
  }

  if (enterCue === undefined) {
    return { state: "unchanged", opacity: 1 };
  }
  return { state, opacity: enterOpacity };
}

/** True once a fade/exit cue window has started for this node. */
function isFadingOut(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): boolean {
  const fadeCue = fadeCueForNode(nodeId, timeline);
  if (fadeCue === undefined) {
    return false;
  }
  return playbackTimeMs >= Math.max(0, finiteNumber(fadeCue.at));
}

/**
 * Stroke-reveal progress for a `draw` cue in [0, 1].
 * Undefined when the node has no draw cue.
 */
function drawProgressForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  includeTrace = true,
): number | undefined {
  const cue = timeline
    .filter(
      (candidate) =>
        candidate.target === nodeId &&
        isDrawAction(candidate.action) &&
        (includeTrace || candidate.action !== "trace"),
    )
    .at(-1);
  if (cue === undefined) {
    return undefined;
  }
  const atMs = finiteNumber(cue.at);
  if (playbackTimeMs <= atMs) {
    return 0;
  }
  return cueProgress(cue, playbackTimeMs);
}

function traceProgressForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): number | undefined {
  const cue = timeline
    .filter(
      (candidate) =>
        candidate.target === nodeId && candidate.action === "trace",
    )
    .at(-1);
  if (cue === undefined || playbackTimeMs < finiteNumber(cue.at)) {
    return undefined;
  }
  return cueProgress(cue, playbackTimeMs);
}

/**
 * Dash-gating progress for a fan connector's own stroke: 0 (hidden) before
 * its trace cue starts rather than `undefined` (which renders solid,
 * un-dashed). traceProgressForNode stays `undefined` pre-cue for ball
 * rendering, where that's correct — the connector body must not be.
 */
function fanStrokeProgress(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): number | undefined {
  const cue = timeline
    .filter(
      (candidate) =>
        candidate.target === nodeId && candidate.action === "trace",
    )
    .at(-1);
  if (cue === undefined) {
    return undefined;
  }
  if (playbackTimeMs < finiteNumber(cue.at)) {
    return 0;
  }
  return cueProgress(cue, playbackTimeMs);
}

/**
 * Emphasize/emphasis pulse for a node while its cue window is active.
 * Undefined when idle (no cue, before `at`, or after `at + duration`).
 */
function emphasisForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  accentColor: string,
): EmphasisAppearance | undefined {
  const cue = timeline
    .filter(
      (candidate) =>
        candidate.target === nodeId && isEmphasizeAction(candidate.action),
    )
    .at(-1);
  if (cue === undefined) {
    return undefined;
  }

  const atMs = Math.max(0, finiteNumber(cue.at));
  const durationMs = Math.max(0, finiteNumber(cue.duration));
  if (playbackTimeMs < atMs) {
    return undefined;
  }
  if (durationMs <= 0) {
    if (playbackTimeMs > atMs) {
      return undefined;
    }
  } else if (playbackTimeMs > atMs + durationMs) {
    return undefined;
  }

  const progress = cueProgress(cue, playbackTimeMs);
  const intensity = Math.sin(progress * Math.PI);
  if (intensity <= 0) {
    return undefined;
  }

  const glowPx = 2 + intensity * 10;
  return {
    intensity,
    strokeScale: 1 + intensity * 0.85,
    opacityScale: 0.55 + intensity * 0.45,
    filter: `drop-shadow(0 0 ${glowPx.toFixed(2)}px ${accentColor})`,
  };
}

/** Authored `pulse` cue envelope (same half-sine shape as emphasize, no glow). */
function pulseCueForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): EmphasisAppearance | undefined {
  const cue = timeline
    .filter(
      (candidate) =>
        candidate.target === nodeId && isPulseAction(candidate.action),
    )
    .at(-1);
  if (cue === undefined) {
    return undefined;
  }

  const atMs = Math.max(0, finiteNumber(cue.at));
  const durationMs = Math.max(0, finiteNumber(cue.duration));
  if (playbackTimeMs < atMs) {
    return undefined;
  }
  if (durationMs <= 0) {
    if (playbackTimeMs > atMs) {
      return undefined;
    }
  } else if (playbackTimeMs > atMs + durationMs) {
    return undefined;
  }

  const progress = cueProgress(cue, playbackTimeMs);
  const intensity = Math.sin(progress * Math.PI);
  if (intensity <= 0) {
    return undefined;
  }
  return {
    intensity,
    strokeScale: 1 + intensity * 0.35,
    opacityScale: 0.35 + intensity * 0.65,
    filter: "none",
  };
}

/**
 * Continuous outline pulse matching legacy MentalModel CSS keyframes
 * (`0 → 0.72 → 0` over 2.2s with an 0.8s start delay), driven by playback time
 * so `restartKey` resets the cycle.
 */
function continuousPulseForNode(
  node: SceneNodeLike,
  capability: string,
  appearance: TimelineAppearance,
  playbackTimeMs: number,
  playback: PlaybackContext,
): PulseAppearance | undefined {
  if (!isPulseNode(node, capability)) {
    return undefined;
  }
  if (appearance.state === "hidden") {
    return undefined;
  }
  // Match legacy MentalModel CSS: `.box-pulse { opacity: 0 }` with animation
  // only while playing. Pausing / reduced-motion must not leave a mid-pulse
  // outline parked on top of panels (reads as a ghost double border).
  if (playback.reducedMotion || !playback.playing) {
    return { intensity: 0, opacity: 0 };
  }
  if (playbackTimeMs < PULSE_DELAY_MS) {
    return { intensity: 0, opacity: 0 };
  }
  const cycle =
    ((playbackTimeMs - PULSE_DELAY_MS) % PULSE_CYCLE_MS) / PULSE_CYCLE_MS;
  let opacity = 0;
  if (cycle < 0.04) {
    opacity = (cycle / 0.04) * 0.72;
  } else if (cycle < 0.12) {
    opacity = 0.72;
  } else if (cycle < 0.21) {
    opacity = 0.72 * (1 - (cycle - 0.12) / 0.09);
  }
  return { intensity: opacity / 0.72, opacity };
}

/** Gentle vertical float for pulse-tagged shapes while playing. */
function pulseFloatStyle(
  playback: PlaybackContext,
  playbackTimeMs: number,
): CSSProperties | undefined {
  if (playback.reducedMotion || !playback.playing) {
    return undefined;
  }
  const wave = Math.sin((playbackTimeMs / 900) * Math.PI * 2);
  return {
    transform: `translateY(${(wave * 1.6).toFixed(2)}px)`,
    transformBox: "fill-box",
    transformOrigin: "center",
  };
}

/**
 * Map Flow `@theme.*` role strings (and bare role paths) onto host Theme paints.
 * Literals (`#hex`, `none`, css colors) pass through unchanged.
 */
function resolveThemePaint(
  value: unknown,
  theme: Theme,
  fallback: string,
): string {
  if (typeof value !== "string" || value.length === 0) {
    return fallback;
  }
  if (value === "none" || value === "transparent") {
    return value;
  }

  const role = value.startsWith("@theme.")
    ? value.slice("@theme.".length)
    : value.startsWith("theme.")
      ? value.slice("theme.".length)
      : value;
  const isBareThemeRole =
    role.startsWith("surface.") ||
    role.startsWith("bg.") ||
    role.startsWith("ink.") ||
    role.startsWith("text.") ||
    role.startsWith("stroke.") ||
    role.startsWith("structure.") ||
    role.startsWith("accent.");
  if (!value.startsWith("@theme.") && !value.startsWith("theme.") && !isBareThemeRole) {
    return value;
  }

  switch (role) {
    case "surface.elevated":
      // Scene cards need a solid layer above the graphite stage. A translucent
      // fill reads as black once diagrams are scaled down.
      return theme.bg.panel;
    case "bg.elevated":
      return theme.bg.elevated;
    case "surface.primary":
      return theme.fill.quaternary;
    case "bg.primary":
      return theme.bg.chrome;
    case "surface.secondary":
      // Quieter fill for headers / chips (distinct from elevated panels).
      return theme.fill.tertiary;
    case "surface.chrome":
    case "bg.chrome":
      return theme.bg.chrome;
    case "surface.editor":
    case "bg.editor":
      return theme.bg.editor;
    case "ink.primary":
    case "text.primary":
      return theme.text.primary;
    case "ink.secondary":
    case "text.secondary":
      return theme.text.secondary;
    case "ink.tertiary":
    case "text.tertiary":
      return theme.text.tertiary;
    case "ink.quaternary":
    case "text.quaternary":
      return theme.text.quaternary;
    case "ink.link":
    case "text.link":
      return theme.text.link;
    case "ink.onAccent":
    case "text.onAccent":
      return theme.text.onAccent;
    case "stroke.primary":
    case "structure.primary":
      return theme.stroke.primary;
    case "stroke.secondary":
    case "structure.secondary":
    case "structure.divider":
      return theme.stroke.secondary;
    case "stroke.tertiary":
    case "structure.tertiary":
      return theme.stroke.tertiary;
    case "accent.primary":
    case "accent.control":
    case "accent.execute":
      return theme.accent.primary;
    case "accent.secondary":
      return theme.category.blue;
    case "accent.tertiary":
      return theme.category.purple;
    case "accent.danger":
      return theme.category.red;
    case "accent.warning":
    case "accent.attention":
      return theme.category.yellow;
    case "accent.caution":
      return theme.category.orange;
    case "accent.cyan":
      return theme.category.cyan;
    case "accent.orange":
      return theme.category.orange;
    case "accent.gray":
      return theme.category.gray;
    case "accent.green":
      return theme.category.green;
    default:
      return fallback;
  }
}

function paintFromStyle(
  style: SceneNodeLike["style"],
  key: string,
  theme: Theme,
  fallback: string,
): string {
  return resolveThemePaint(style?.[key], theme, fallback);
}

/** True when a style value is an `@theme.accent.*` (or bare `accent.*`) role. */
function isAccentThemeRole(value: unknown): boolean {
  if (typeof value !== "string" || value.length === 0) {
    return false;
  }
  const role = value.startsWith("@theme.")
    ? value.slice("@theme.".length)
    : value.startsWith("theme.")
      ? value.slice("theme.".length)
      : value;
  return role.startsWith("accent.");
}

/**
 * GTC skin for diagram panels: accent-tagged `core.rect` boxes get a real,
 * saturated accent fill (with the accent doubling as the glowing border)
 * instead of the old chalk convention of an outline-only bg fill.
 */
function chalkRectPaints(
  style: SceneNodeLike["style"],
  theme: Theme,
  themeBg: string,
  themeStroke: string,
): { fill: string; stroke: string } {
  const fillRole = style?.fill;
  const strokeRole = style?.stroke;
  if (isAccentThemeRole(fillRole)) {
    const accent = resolveThemePaint(fillRole, theme, theme.accent.primary);
    const stroke = isAccentThemeRole(strokeRole)
      ? resolveThemePaint(strokeRole, theme, accent)
      : accent;
    return {
      fill: `color-mix(in srgb, ${accent} 38%, ${themeBg})`,
      stroke: accent,
    };
  }
  // GTC skin: any box outlined in an accent color gets a real, saturated,
  // stage-lit fill (authored `fill: none` is overridden here) instead of
  // the old chalk convention of leaving accent-bordered boxes unfilled.
  if (isAccentThemeRole(strokeRole)) {
    const accent = resolveThemePaint(strokeRole, theme, theme.accent.primary);
    return {
      fill: `color-mix(in srgb, ${accent} 32%, ${themeBg})`,
      stroke: accent,
    };
  }
  return {
    fill: paintFromStyle(style, "fill", theme, themeBg),
    stroke: paintFromStyle(style, "stroke", theme, themeStroke),
  };
}

/** CSS from node.style with `@theme.*` paints resolved; omits fill/stroke attrs. */
function styleToCss(
  style: SceneNodeLike["style"],
  theme: Theme,
): CSSProperties {
  if (style === undefined) {
    return {};
  }
  const css: CSSProperties = {};
  for (const [key, value] of Object.entries(style)) {
    if (
      key === "fill" ||
      key === "stroke" ||
      key === "markerEnd" ||
      key === "role" ||
      key === "motion" ||
      key === "pulse" ||
      key === "dashed" ||
      key === "strokeDasharray" ||
      key === "dashArray" ||
      key === "strokeStyle" ||
      key === "variant" ||
      key === "r" ||
      key === "rx" ||
      key === "ry" ||
      key === "radius" ||
      key === "direction" ||
      key === "cols" ||
      key === "gap" ||
      key === "route" ||
      key === "axis" ||
      key === "coordinateSpace" ||
      // Caps are owned by FlowArrow (butt under markers / draw reveal).
      key === "strokeLinecap" ||
      key === "stroke-linecap"
    ) {
      continue;
    }
    if (typeof value === "string") {
      (css as Record<string, string | number>)[key] = resolveThemePaint(
        value,
        theme,
        value,
      );
    } else if (typeof value === "number") {
      (css as Record<string, string | number>)[key] = value;
    }
  }
  return css;
}

/**
 * Resolve SVG path data for line / path / arrow / connector / elbow nodes.
 * Precedence: authored `d` → `path` → `points` polyline → elbow/`from`/`to`.
 */
function arrowPathData(
  node: SceneNodeLike,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): string | undefined {
  if (typeof node.d === "string" && node.d.length > 0) {
    return node.d;
  }
  if (typeof node.path === "string" && node.path.length > 0) {
    return node.path;
  }
  if (Array.isArray(node.points) && node.points.length > 0) {
    return polylinePathData(node.points, index, layoutOrigin);
  }
  const from = singleScenePoint(node.from);
  const to = singleScenePoint(node.to);
  if (from !== undefined || to !== undefined) {
    const start = resolveEndpoint(from, index, layoutOrigin);
    const end = resolveEndpoint(to, index, layoutOrigin);
    if (isElbowConnector(node, capabilityOf(node))) {
      const via =
        node.via !== undefined
          ? resolveEndpoint(node.via, index, layoutOrigin)
          : undefined;
      return elbowPathData(start, end, via, connectorAxisOf(node));
    }
    return `M${start.x} ${start.y} L${end.x} ${end.y}`;
  }
  return undefined;
}

function formatPathNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  const rounded = Math.round(value * 1000) / 1000;
  return String(rounded);
}

/**
 * Rewrite the final absolute H / V / L / C endpoint when the rest of the path
 * can stay intact (avoids polyline-approximating whole curves).
 * Relative endings are refused so absolute coords are not written under `h/v/l/c`.
 */
function rewriteLastEndpoint(
  d: string,
  x: number,
  y: number,
): string | undefined {
  const trimmed = d.trim();
  // Absolute only — no `/i` (relative endings fall through to polyline approx).
  let match =
    /^(.*)H\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$/.exec(trimmed);
  if (match?.[1] !== undefined) {
    return `${match[1]}H${formatPathNumber(x)}`;
  }
  match =
    /^(.*)V\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$/.exec(trimmed);
  if (match?.[1] !== undefined) {
    return `${match[1]}V${formatPathNumber(y)}`;
  }
  match =
    /^(.*)L\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:\s+|,)\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$/.exec(
      trimmed,
    );
  if (match?.[1] !== undefined) {
    return `${match[1]}L${formatPathNumber(x)} ${formatPathNumber(y)}`;
  }
  // Absolute cubic: keep control points, pull only the terminal point back.
  match =
    /^(.*C(?:\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:\s+|,)\s*){4})[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:\s+|,)\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?\s*$/.exec(
      trimmed,
    );
  if (match?.[1] !== undefined) {
    return `${match[1]}${formatPathNumber(x)} ${formatPathNumber(y)}`;
  }
  return undefined;
}

function shortenPathEndWithDom(d: string, inset: number): string | undefined {
  try {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    if (typeof path.getTotalLength !== "function") {
      return undefined;
    }
    const total = path.getTotalLength();
    if (!(total > inset + 0.5)) {
      return d;
    }
    const cutAt = total - inset;
    const end = path.getPointAtLength(cutAt);
    const rewritten = rewriteLastEndpoint(d, end.x, end.y);
    if (rewritten !== undefined) {
      return rewritten;
    }
    // Fallback for uncommon commands: polyline up to the cut.
    const steps = Math.max(12, Math.min(64, Math.ceil(cutAt / 3)));
    const parts: string[] = [];
    for (let i = 0; i <= steps; i++) {
      const point = path.getPointAtLength((cutAt * i) / steps);
      const x = formatPathNumber(point.x);
      const y = formatPathNumber(point.y);
      parts.push(i === 0 ? `M${x} ${y}` : `L${x} ${y}`);
    }
    return parts.join("");
  } catch {
    return undefined;
  }
}

/**
 * Pure fallback when DOM path measurement is unavailable (or failed).
 * Walks M/L/H/V (abs + rel) so tip inset uses the true pen position.
 */
function lastPointFromPrefix(
  prefix: string,
): Readonly<{ x: number; y: number }> | undefined {
  const trimmed = prefix.trim();
  if (trimmed.length === 0) {
    return undefined;
  }
  const cmdRe =
    /([MmLlHhVv])\s*([^MmLlHhVvCcSsQqTtAaZz]*)/g;
  let x = 0;
  let y = 0;
  let found = false;
  for (const match of trimmed.matchAll(cmdRe)) {
    const cmd = match[1] ?? "";
    const nums = [
      ...(match[2] ?? "").matchAll(
        /[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g,
      ),
    ].map((token) => Number(token[0]));
    const relative = cmd === cmd.toLowerCase();
    const op = cmd.toUpperCase();
    if (op === "M" || op === "L") {
      for (let i = 0; i + 1 < nums.length; i += 2) {
        const nx = nums[i]!;
        const ny = nums[i + 1]!;
        if (relative && found) {
          x += nx;
          y += ny;
        } else {
          x = nx;
          y = ny;
        }
        found = true;
        // Extra M pairs are treated as implicit LineTos.
      }
    } else if (op === "H") {
      for (const nx of nums) {
        x = relative && found ? x + nx : nx;
        found = true;
      }
    } else if (op === "V") {
      for (const ny of nums) {
        y = relative && found ? y + ny : ny;
        found = true;
      }
    }
  }
  return found ? { x, y } : undefined;
}

function shortenPathEndParsed(d: string, inset: number): string {
  const trimmed = d.trim();
  // Relative endings: shorten the delta, keep the relative command letter.
  const hRel =
    /^(.*)h\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/.exec(trimmed);
  if (hRel?.[1] !== undefined && hRel[2] !== undefined) {
    const dx = Number(hRel[2]);
    if (Math.abs(dx) > inset) {
      return `${hRel[1]}h${formatPathNumber(dx - Math.sign(dx) * inset)}`;
    }
    return d;
  }
  const vRel =
    /^(.*)v\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/.exec(trimmed);
  if (vRel?.[1] !== undefined && vRel[2] !== undefined) {
    const dy = Number(vRel[2]);
    if (Math.abs(dy) > inset) {
      return `${vRel[1]}v${formatPathNumber(dy - Math.sign(dy) * inset)}`;
    }
    return d;
  }
  const lRel =
    /^(.*)l\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?:\s+|,)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/.exec(
      trimmed,
    );
  if (
    lRel?.[1] !== undefined &&
    lRel[2] !== undefined &&
    lRel[3] !== undefined
  ) {
    const dx = Number(lRel[2]);
    const dy = Number(lRel[3]);
    const length = Math.hypot(dx, dy);
    if (length > inset) {
      const scale = (length - inset) / length;
      return `${lRel[1]}l${formatPathNumber(dx * scale)} ${formatPathNumber(dy * scale)}`;
    }
    return d;
  }

  const hMatch =
    /^(.*)H\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/.exec(trimmed);
  if (hMatch?.[1] !== undefined && hMatch[2] !== undefined) {
    const prev = lastPointFromPrefix(hMatch[1]);
    const xEnd = Number(hMatch[2]);
    if (prev !== undefined) {
      const dx = xEnd - prev.x;
      if (Math.abs(dx) > inset) {
        return `${hMatch[1]}H${formatPathNumber(xEnd - Math.sign(dx) * inset)}`;
      }
    }
  }
  const vMatch =
    /^(.*)V\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/.exec(trimmed);
  if (vMatch?.[1] !== undefined && vMatch[2] !== undefined) {
    const prev = lastPointFromPrefix(vMatch[1]);
    const yEnd = Number(vMatch[2]);
    if (prev !== undefined) {
      const dy = yEnd - prev.y;
      if (Math.abs(dy) > inset) {
        return `${vMatch[1]}V${formatPathNumber(yEnd - Math.sign(dy) * inset)}`;
      }
    }
  }
  const lMatch =
    /^(.*)L\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?:\s+|,)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/.exec(
      trimmed,
    );
  if (
    lMatch?.[1] !== undefined &&
    lMatch[2] !== undefined &&
    lMatch[3] !== undefined
  ) {
    const prev = lastPointFromPrefix(lMatch[1]);
    if (prev !== undefined) {
      const xEnd = Number(lMatch[2]);
      const yEnd = Number(lMatch[3]);
      const dx = xEnd - prev.x;
      const dy = yEnd - prev.y;
      const length = Math.hypot(dx, dy);
      if (length > inset) {
        const scale = (length - inset) / length;
        return `${lMatch[1]}L${formatPathNumber(prev.x + dx * scale)} ${formatPathNumber(prev.y + dy * scale)}`;
      }
    }
  }
  // Absolute cubic: pull only the terminal point back along the chord.
  const cMatch =
    /^(.*C(?:\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:\s+|,)\s*){4})([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(?:\s+|,)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/.exec(
      trimmed,
    );
  if (
    cMatch?.[1] !== undefined &&
    cMatch[2] !== undefined &&
    cMatch[3] !== undefined
  ) {
    const prev = lastPointFromPrefix(cMatch[1]);
    if (prev !== undefined) {
      const xEnd = Number(cMatch[2]);
      const yEnd = Number(cMatch[3]);
      const dx = xEnd - prev.x;
      const dy = yEnd - prev.y;
      const length = Math.hypot(dx, dy);
      if (length > inset) {
        const scale = (length - inset) / length;
        return `${cMatch[1]}${formatPathNumber(prev.x + dx * scale)} ${formatPathNumber(prev.y + dy * scale)}`;
      }
    }
  }
  return d;
}

/**
 * Pull the stroke end back by the arrowhead length so a `refX=0` tip lands on
 * the authored endpoint instead of poking into the destination box.
 */
function shortenPathForArrowhead(
  d: string,
  strokeWidth: number,
  tipInsetUnits: number,
): string {
  const inset = tipInsetUnits * strokeWidth;
  if (!(inset > 0) || d.length === 0) {
    return d;
  }
  if (typeof document !== "undefined") {
    const shortened = shortenPathEndWithDom(d, inset);
    if (shortened !== undefined) {
      return shortened;
    }
  }
  return shortenPathEndParsed(d, inset);
}

/** Collect unique tips used by arrow-like nodes in the scene tree. */
function collectSceneTips(
  roots: readonly SceneNodeLike[],
): readonly ResolvedMarkerTip[] {
  const byKey = new Map<string, ResolvedMarkerTip>();
  const visit = (node: SceneNodeLike) => {
    const tip = tipForArrowNode(node, capabilityOf(node));
    if (tip !== null) {
      byKey.set(tip.key, tip);
    }
    if (Array.isArray(node.children)) {
      for (const child of node.children) {
        visit(child);
      }
    }
  };
  for (const root of roots) {
    visit(root);
  }
  if (byKey.size === 0) {
    byKey.set(DEFAULT_MARKER_TIP.key, DEFAULT_MARKER_TIP);
  }
  return [...byKey.values()];
}

function strokeWidthFromStyle(
  style: SceneNodeLike["style"],
  fallback = DEFAULT_ARROW_STROKE_WIDTH,
): number {
  const width = style?.strokeWidth;
  return typeof width === "number" && Number.isFinite(width) ? width : fallback;
}

function cornerRadiusFromStyle(
  style: SceneNodeLike["style"],
  fallback = 14,
): number {
  const radius = style?.radius ?? style?.rx ?? style?.borderRadius;
  return typeof radius === "number" && Number.isFinite(radius) ? radius : fallback;
}

/**
 * Recursively render nested `children` into sibling `<g>` wrappers.
 * When `layoutOffset` is set, children are parent-local and wrapped in
 * `translate(offset)` so group/container origins shift the subtree.
 * `childGeoms` supplies stack/grid local placements when present.
 * Arrow-like siblings paint after non-arrows so tip heads (which extend
 * past the stroke end) are not buried under destination box fills.
 */
function renderChildren(
  children: readonly SceneNodeLike[] | undefined,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  theme: Theme,
  markerPrefix: string,
  index: SceneNodeIndex,
  parentLayoutOrigin: LayoutOrigin,
  layoutOffset: LayoutOrigin | undefined,
  playback: PlaybackContext,
  childGeoms: readonly SceneGeometryLike[] | undefined,
): ReactNode {
  if (!Array.isArray(children) || children.length === 0) {
    return null;
  }
  const childOrigin: LayoutOrigin =
    layoutOffset === undefined
      ? parentLayoutOrigin
      : {
          x: parentLayoutOrigin.x + layoutOffset.x,
          y: parentLayoutOrigin.y + layoutOffset.y,
        };
  const ordered = orderArrowSiblingsLast(children);
  const nested = ordered.map((child) => {
    const authoredIndex = children.indexOf(child);
    const geometryOverride =
      authoredIndex >= 0 ? childGeoms?.[authoredIndex] : undefined;
    return renderNode(
      child,
      timeline,
      playbackTimeMs,
      theme,
      markerPrefix,
      index,
      childOrigin,
      playback,
      geometryOverride,
    );
  });
  if (
    layoutOffset === undefined ||
    (layoutOffset.x === 0 && layoutOffset.y === 0)
  ) {
    return nested;
  }
  return (
    <g
      data-flow-layout-offset={`${layoutOffset.x},${layoutOffset.y}`}
      transform={`translate(${layoutOffset.x} ${layoutOffset.y})`}
    >
      {nested}
    </g>
  );
}

/** Keep document order within each partition; arrows paint on top. */
function orderArrowSiblingsLast(
  nodes: readonly SceneNodeLike[],
): readonly SceneNodeLike[] {
  const background: SceneNodeLike[] = [];
  const arrows: SceneNodeLike[] = [];
  for (const node of nodes) {
    if (isArrowLike(node, capabilityOf(node))) {
      arrows.push(node);
    } else {
      background.push(node);
    }
  }
  if (arrows.length === 0) {
    return nodes;
  }
  return [...background, ...arrows];
}

function renderNode(
  node: SceneNodeLike,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  theme: Theme,
  markerPrefix: string,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin = ZERO_ORIGIN,
  playback: PlaybackContext,
  geometryOverride?: SceneGeometryLike,
): ReactNode {
  const capability = capabilityOf(node);
  let authoredGeom = geometryOverride ?? geometryOf(node);
  const relativePosition = node.relativePosition;
  if (geometryOverride === undefined && relativePosition !== undefined) {
    const targetWorldGeom = index.worldGeometryById.get(relativePosition.nodeId);
    if (targetWorldGeom !== undefined) {
      const anchorPoint = nodeAnchorPoint(targetWorldGeom, relativePosition.anchor);
      const worldX = anchorPoint.x + finiteNumber(relativePosition.dx);
      const worldY = anchorPoint.y + finiteNumber(relativePosition.dy);
      authoredGeom = {
        ...authoredGeom,
        x: worldX - layoutOrigin.x,
        y: worldY - layoutOrigin.y,
      };
    }
  }
  const kids = node.children;
  const { parentGeom: geom, childGeoms } = resolveContainerLayout(
    node,
    authoredGeom,
    kids,
  );
  const appearance = appearanceForNode(node.id, timeline, playbackTimeMs);
  const fanNode = isFanNode(node, capability);
  const drawProgress = drawProgressForNode(
    node.id,
    timeline,
    playbackTimeMs,
    !fanNode,
  );
  const traceProgress = fanNode
    ? traceProgressForNode(node.id, timeline, playbackTimeMs)
    : undefined;
  const themeAccent = theme.accent.primary;
  const emphasis = emphasisForNode(
    node.id,
    timeline,
    playbackTimeMs,
    themeAccent,
  );
  const pulseCue = pulseCueForNode(node.id, timeline, playbackTimeMs);
  const continuousPulse = continuousPulseForNode(
    node,
    capability,
    appearance,
    playbackTimeMs,
    playback,
  );
  const activeEmphasis = emphasis ?? pulseCue;
  const label = node.accessibility?.label ?? node.id;
  const description = node.accessibility?.description;
  const descriptionId =
    typeof description === "string" && description.length > 0
      ? `flow-node-${node.id}-desc`
      : undefined;
  const localChildren = childrenUseLocalLayout(node, geom, kids);
  const nested = renderChildren(
    kids,
    timeline,
    playbackTimeMs,
    theme,
    markerPrefix,
    index,
    layoutOrigin,
    localChildren ? { x: geom.x, y: geom.y } : undefined,
    playback,
    childGeoms,
  );

  const themeBg = theme.bg.elevated;
  const themeStroke = theme.stroke.secondary;
  const themeText = theme.text.primary;
  const groupLike = isGroupLike(node, capability);
  const strokeScale = activeEmphasis?.strokeScale ?? 1;
  const pulseTagged = isPulseNode(node, capability);
  const pulseFloat = pulseTagged
    ? pulseFloatStyle(playback, playbackTimeMs)
    : undefined;

  // Capability before kind: package IR often stamps `kind: "rect"` on
  // `core.dot` nodes. Dots must win over chalk rects.
  let body: ReactNode = null;
  if (isDotLike(node, capability) && !isArrowLike(node, capability)) {
    const radius = circleRadius(node, geom);
    const cx =
      geom.width > 0 || geom.height > 0 ? geom.x + geom.width / 2 : geom.x;
    const cy =
      geom.width > 0 || geom.height > 0 ? geom.y + geom.height / 2 : geom.y;
    body = (
      <circle
        cx={cx}
        cy={cy}
        r={radius}
        fill={paintFromStyle(node.style, "fill", theme, themeAccent)}
        stroke={paintFromStyle(node.style, "stroke", theme, "none")}
        strokeWidth={strokeWidthFromStyle(node.style, 0) * strokeScale}
        focusable={false}
        aria-hidden="true"
        data-flow-dot="true"
        style={styleToCss(node.style, theme)}
      />
    );
  } else if (isCircleOrEllipse(node, capability)) {
    const cx =
      geom.width > 0 || geom.height > 0 ? geom.x + geom.width / 2 : geom.x;
    const cy =
      geom.width > 0 || geom.height > 0 ? geom.y + geom.height / 2 : geom.y;
    const styledRx =
      typeof node.style?.rx === "number" && Number.isFinite(node.style.rx)
        ? node.style.rx
        : typeof node.style?.r === "number" && Number.isFinite(node.style.r)
          ? node.style.r
          : undefined;
    const styledRy =
      typeof node.style?.ry === "number" && Number.isFinite(node.style.ry)
        ? node.style.ry
        : styledRx;
    const rx =
      styledRx !== undefined && styledRx > 0
        ? styledRx
        : geom.width > 0
          ? geom.width / 2
          : circleRadius(node, geom);
    const ry =
      styledRy !== undefined && styledRy > 0
        ? styledRy
        : geom.height > 0
          ? geom.height / 2
          : rx;
    const { fill: fillPaint, stroke: strokePaint } = {
      fill: paintFromStyle(node.style, "fill", theme, themeAccent),
      stroke: paintFromStyle(node.style, "stroke", theme, "none"),
    };
    if (
      capability === "core.ellipse" ||
      node.kind === "ellipse" ||
      Math.abs(rx - ry) > 0.5
    ) {
      body = (
        <ellipse
          cx={cx}
          cy={cy}
          rx={rx}
          ry={ry}
          fill={fillPaint}
          stroke={strokePaint}
          strokeWidth={strokeWidthFromStyle(node.style, 1.3) * strokeScale}
          focusable={false}
          aria-hidden="true"
          style={styleToCss(node.style, theme)}
          data-flow-ellipse="true"
        />
      );
    } else {
      body = (
        <circle
          cx={cx}
          cy={cy}
          r={rx}
          fill={fillPaint}
          stroke={strokePaint}
          strokeWidth={strokeWidthFromStyle(node.style, 1.3) * strokeScale}
          focusable={false}
          aria-hidden="true"
          data-flow-circle="true"
          style={styleToCss(node.style, theme)}
        />
      );
    }
  } else if (capability === "core.rect" || node.kind === "rect") {
    const { fill: fillPaint, stroke: strokePaint } = chalkRectPaints(
      node.style,
      theme,
      themeBg,
      themeStroke,
    );
    const remappedAccentFill =
      isAccentThemeRole(node.style?.fill) || isAccentThemeRole(node.style?.stroke);
    // Outline-only pulses fade opacity; filled content boxes stay opaque and
    // pulse via stroke scale / float (MentalModel overlay parity).
    const pulseOpacity =
      continuousPulse !== undefined && isOutlinePulseFill(fillPaint)
        ? continuousPulse.opacity
        : undefined;
    const rx = cornerRadiusFromStyle(node.style);
    const ry =
      typeof node.style?.ry === "number" && Number.isFinite(node.style.ry)
        ? node.style.ry
        : rx;
    body = (
      <rect
        x={geom.x}
        y={geom.y}
        width={geom.width}
        height={geom.height}
        rx={rx}
        ry={ry}
        fill={fillPaint}
        stroke={strokePaint}
        strokeWidth={
          strokeWidthFromStyle(node.style, remappedAccentFill ? 1.8 : 1.3) *
          strokeScale *
          (pulseTagged && continuousPulse !== undefined
            ? 1 + continuousPulse.intensity * 0.25
            : 1)
        }
        focusable={false}
        aria-hidden="true"
        style={{
          filter: remappedAccentFill
            ? `drop-shadow(0 8px 14px rgba(0, 0, 0, 0.4)) drop-shadow(0 0 8px color-mix(in srgb, ${strokePaint} 45%, transparent))`
            : "drop-shadow(0 6px 10px rgba(0, 0, 0, 0.3))",
          ...styleToCss(node.style, theme),
          ...pulseFloat,
          ...(pulseOpacity !== undefined ? { opacity: pulseOpacity } : {}),
        }}
        data-flow-pulse={pulseTagged ? "true" : undefined}
        data-pulse-intensity={
          continuousPulse === undefined
            ? undefined
            : String(continuousPulse.intensity)
        }
      />
    );
  } else if (CHROME_GROUP_CAPABILITIES.has(capability)) {
    // Panel / header groups paint chrome; title/detail children are local.
    // Enter opacity rides the chrome only (MentalModel: rect fades, text snaps).
    const { fill: fillPaint, stroke: strokePaint } = chalkRectPaints(
      node.style,
      theme,
      themeBg,
      themeStroke,
    );
    const chromeEnterOpacity =
      appearance.state === "entering" &&
      !isFadingOut(node.id, timeline, playbackTimeMs)
        ? appearance.opacity
        : undefined;
    body = (
      <rect
        x={geom.x}
        y={geom.y}
        width={geom.width}
        height={geom.height}
        rx={cornerRadiusFromStyle(node.style, 10)}
        fill={fillPaint}
        stroke={strokePaint}
        strokeWidth={
          strokeWidthFromStyle(
            node.style,
            capability === "core.panel" ? 1.6 : 1.3,
          ) * strokeScale
        }
        focusable={false}
        aria-hidden="true"
        style={{
          ...(capability === "core.panel"
            ? {
                filter:
                  "drop-shadow(0 5px 7px rgba(0, 0, 0, 0.32))",
              }
            : {}),
          ...styleToCss(node.style, theme),
          ...(chromeEnterOpacity !== undefined
            ? { opacity: chromeEnterOpacity }
            : {}),
        }}
        data-flow-panel-chrome="true"
      />
    );
  } else if (capability === "core.text" || node.kind === "text") {
    const fontSize =
      typeof node.style?.fontSize === "number" ? node.style.fontSize : 14;
    const rawAnchor =
      typeof node.style?.textAnchor === "string"
        ? node.style.textAnchor
        : undefined;
    const textAnchor =
      rawAnchor === "start" ||
      rawAnchor === "middle" ||
      rawAnchor === "end" ||
      rawAnchor === "inherit"
        ? rawAnchor
        : undefined;
    // Geometry is a layout box: start/middle/end anchor to left/center/right.
    // Using `geom.x` for `end` draws long captions leftward over sibling titles.
    const textX =
      textAnchor === "middle"
        ? geom.x + geom.width / 2
        : textAnchor === "end"
          ? geom.x + Math.max(geom.width, 0)
          : geom.x;
    // Middle-anchored labels in a tall box center optically (SceneBox parity).
    const centerVertically =
      textAnchor === "middle" && geom.height > fontSize * 1.25;
    const textY = centerVertically ? geom.y + geom.height / 2 : geom.y;
    body = (
      <text
        x={textX}
        y={textY}
        dominantBaseline={centerVertically ? "middle" : "hanging"}
        textAnchor={textAnchor}
        fill={paintFromStyle(node.style, "fill", theme, themeText)}
        fontSize={fontSize}
        focusable={false}
        aria-hidden="true"
        style={styleToCss(node.style, theme)}
      >
        {node.text ?? ""}
      </text>
    );
  } else if (fanNode) {
    const geometry = resolveFanGeometry(node, index, layoutOrigin);
    const stroke = paintFromStyle(node.style, "stroke", theme, themeAccent);
    const tip = tipForArrowNode(node, capability);
    const markerId =
      tip !== null ? markerDomId(markerPrefix, tip) : markerPrefix;
    // Fan connectors draw exclusively via `trace` cues (drawProgress
    // excludes trace for fan nodes to avoid double-driving the ball
    // animation below); fall back to fanStrokeProgress so the stroke
    // itself stays dash-hidden until its own trace cue fires instead of
    // rendering solid from t=0, ahead of the target nodes fading in.
    const strokeProgress =
      drawProgress ?? fanStrokeProgress(node.id, timeline, playbackTimeMs);
    const drawing = strokeProgress !== undefined;
    const strokeWidth = strokeWidthFromStyle(node.style) * strokeScale;
    const authoredDash = authoredStrokeDasharray(node.style);
    const dashed = isDashedStyle(node.style);
    const resolvedSegments = geometry.segments.map((segment) => {
      const semanticTip = segment.showMarker ? tip : null;
      return {
        ...segment,
        d:
          semanticTip === null
            ? segment.d
            : shortenPathForArrowhead(
                segment.d,
                strokeWidth,
                semanticTip.insetUnits,
              ),
        showMarker:
          segment.showMarker &&
          semanticTip !== null &&
          (!drawing || strokeProgress >= 1 || playback.reducedMotion),
      };
    });
    const firstHalf =
      traceProgress === undefined ? undefined : clamp01(traceProgress * 2);
    const secondHalf =
      traceProgress === undefined
        ? undefined
        : traceProgress === 0.5
          ? 0.001
          : clamp01((traceProgress - 0.5) * 2);
    const trajectorySplits = geometry.trajectories.map((trajectory) => ({
      trajectory,
      split: splitFanTrajectoryAtJunction(trajectory.d, geometry.junction),
    }));
    const trunkBallPath =
      geometry.capability === "core.fan-out"
        ? trajectorySplits[0]?.split?.head
        : trajectorySplits[0]?.split?.tail;
    const branchBallPaths =
      geometry.capability === "core.fan-out"
        ? trajectorySplits.map(({ trajectory, split }) => ({
            id: trajectory.id,
            d: split?.tail,
          }))
        : trajectorySplits.map(({ trajectory, split }) => ({
            id: trajectory.id,
            d: split?.head,
          }));
    const traceSignals =
      traceProgress === undefined || playback.reducedMotion
        ? null
        : geometry.capability === "core.fan-out"
          ? (
              <>
                {traceProgress < 0.5
                  ? trunkBallPath !== undefined
                    ? (
                        <MotionSignal
                          key={`${playback.restartKey}-${node.id}-trunk-trace`}
                          path={trunkBallPath}
                          color={stroke}
                          progress={firstHalf}
                          active
                          reducedMotion={playback.reducedMotion}
                          r={DEFAULT_DOT_RADIUS}
                          data-flow-fan-ball="trunk"
                        />
                      )
                    : null
                  : branchBallPaths.map((branch) =>
                      branch.d === undefined ? null : (
                        <MotionSignal
                          key={`${playback.restartKey}-${branch.id}-trace`}
                          path={branch.d}
                          color={stroke}
                          progress={secondHalf}
                          active
                          reducedMotion={playback.reducedMotion}
                          r={DEFAULT_DOT_RADIUS}
                          data-flow-fan-ball="branch"
                          data-flow-fan-branch={branch.id}
                        />
                      ),
                    )}
              </>
            )
          : (
              <>
                {traceProgress < 0.5
                  ? branchBallPaths.map((branch) =>
                      branch.d === undefined ? null : (
                        <MotionSignal
                          key={`${playback.restartKey}-${branch.id}-trace`}
                          path={branch.d}
                          color={stroke}
                          progress={firstHalf}
                          active
                          reducedMotion={playback.reducedMotion}
                          r={DEFAULT_DOT_RADIUS}
                          data-flow-fan-ball="branch"
                          data-flow-fan-branch={branch.id}
                        />
                      ),
                    )
                  : trunkBallPath !== undefined
                    ? (
                        <MotionSignal
                          key={`${playback.restartKey}-${node.id}-trunk-trace`}
                          path={trunkBallPath}
                          color={stroke}
                          progress={secondHalf}
                          active
                          reducedMotion={playback.reducedMotion}
                          r={DEFAULT_DOT_RADIUS}
                          data-flow-fan-ball="trunk"
                        />
                      )
                    : null}
              </>
            );
    body = (
      <>
        {resolvedSegments.map((segment) => (
          <FlowArrow
            key={segment.id}
            d={segment.d}
            markerId={markerId}
            showMarker={segment.showMarker}
            color={stroke}
            dashed={!drawing && dashed}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            pathLength={drawing ? 1 : undefined}
            strokeDasharray={
              drawing
                ? 1
                : authoredDash !== undefined
                  ? authoredDash
                  : undefined
            }
            strokeDashoffset={drawing ? 1 - strokeProgress : undefined}
            focusable={false}
            aria-hidden="true"
            style={styleToCss(node.style, theme)}
            data-flow-segment-id={segment.id}
            data-flow-fan-role={segment.role}
            data-flow-arrowhead={segment.showMarker ? "true" : "false"}
            data-flow-tip={segment.showMarker ? tip?.key : undefined}
          />
        ))}
        {traceSignals}
      </>
    );
  } else if (isMotionSignalNode(node, capability)) {
    const d = motionSignalPathData(node, index, layoutOrigin);
    if (d !== undefined) {
      const stroke = paintFromStyle(node.style, "stroke", theme, themeAccent);
      const authoredOpacity =
        typeof node.style?.opacity === "number" &&
        Number.isFinite(node.style.opacity)
          ? Math.min(1, Math.max(0, node.style.opacity))
          : typeof node.style?.opacity === "string" &&
              node.style.opacity.length > 0 &&
              Number.isFinite(Number(node.style.opacity))
            ? Math.min(1, Math.max(0, Number(node.style.opacity)))
            : 1;
      // Travel with a calm SMIL loop. Short `draw` cues (~0.9s) used to drive
      // progress 0→1 once and park the ball at the destination — that felt like
      // a zip through boxes. Keep draw/enter for visibility only.
      const smilActive =
        playback.playing &&
        !playback.reducedMotion &&
        appearance.state !== "hidden";
      // One traveler per inter-box corridor (joined M… paths would teleport).
      const segments = d.match(/M[^M]+/gi) ?? [d];
      body = (
        <g opacity={authoredOpacity < 1 ? authoredOpacity : undefined}>
          {segments.map((segment, segmentIndex) => {
            const path = segment.trim();
            if (path.length === 0) {
              return null;
            }
            const delayS = MOTION_DOT_DELAY_S + segmentIndex * 0.45;
            return (
              <g key={`motion-seg-${node.id}-${segmentIndex}`}>
                <path
                  d={path}
                  fill="none"
                  stroke="none"
                  aria-hidden="true"
                  focusable={false}
                />
                <MotionSignal
                  key={`motion-${playback.restartKey}-${node.id}-${segmentIndex}`}
                  path={path}
                  color={stroke}
                  delay={smilSeconds(delayS, playback.playbackRate)}
                  duration={smilSeconds(
                    MOTION_DOT_DURATION_S,
                    playback.playbackRate,
                  )}
                  reducedMotion={playback.reducedMotion}
                  active={smilActive}
                  r={DEFAULT_DOT_RADIUS}
                  data-flow-motion-signal={node.id}
                  data-flow-motion-segment={String(segmentIndex)}
                />
              </g>
            );
          })}
        </g>
      );
    }
  } else if (isArrowLike(node, capability)) {
    const dRaw = arrowPathData(node, index, layoutOrigin);
    if (dRaw !== undefined) {
      const stroke = paintFromStyle(node.style, "stroke", theme, themeAccent);
      const tip = tipForArrowNode(node, capability);
      const wantsMarker = tip !== null;
      const drawing = drawProgress !== undefined;
      // SVG marker-end sits at the path tip immediately; only attach it once
      // the stroke has fully revealed so tips never lead the line.
      const showMarker =
        wantsMarker &&
        (!drawing || drawProgress >= 1 || playback.reducedMotion);
      const strokeWidth = strokeWidthFromStyle(node.style) * strokeScale;
      // Keep tip length reserved on the path whenever a head will show, so
      // draw-reveal and the final frame share the same endpoint (no jump).
      const d =
        tip !== null
          ? shortenPathForArrowhead(dRaw, strokeWidth, tip.insetUnits)
          : dRaw;
      const authoredDash = authoredStrokeDasharray(node.style);
      const dashed = isDashedStyle(node.style);
      const markerId =
        tip !== null ? markerDomId(markerPrefix, tip) : markerPrefix;
      body = (
        <>
          <FlowArrow
            d={d}
            markerId={markerId}
            showMarker={showMarker}
            color={stroke}
            dashed={!drawing && dashed}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            pathLength={drawing ? 1 : undefined}
            strokeDasharray={
              drawing
                ? 1
                : authoredDash !== undefined
                  ? authoredDash
                  : undefined
            }
            strokeDashoffset={drawing ? 1 - drawProgress : undefined}
            focusable={false}
            aria-hidden="true"
            style={styleToCss(node.style, theme)}
            data-flow-arrowhead={showMarker ? "true" : "false"}
            data-flow-tip={tip?.key}
            data-flow-elbow={
              isElbowConnector(node, capability) ? "true" : undefined
            }
          />
          {/* Tip only near the end of long draws — short cues used to zip a
              progress-driven ball along the whole stroke in <300ms. */}
          {drawing &&
          drawProgress >= 0.88 &&
          drawProgress < 1 &&
          !playback.reducedMotion ? (
            <MotionSignal
              path={d}
              color={stroke}
              progress={drawProgress}
              reducedMotion={playback.reducedMotion}
              active
              r={DEFAULT_DOT_RADIUS}
            />
          ) : null}
        </>
      );
    }
  }

  const baseOpacity =
    appearance.state === "unchanged" ? 1 : appearance.opacity;
  const fadingOut = isFadingOut(node.id, timeline, playbackTimeMs);
  // MentalModel only faded box chrome — labels popped in fully opaque. Keep
  // enter fades off text / panel wrappers; chrome rect carries the ramp.
  const snapEnterOpaque =
    !fadingOut &&
    activeEmphasis === undefined &&
    (CHROME_GROUP_CAPABILITIES.has(capability) ||
      capability === "core.text" ||
      node.kind === "text");
  const groupOpacity =
    appearance.state === "hidden"
      ? 0
      : activeEmphasis !== undefined
        ? baseOpacity * activeEmphasis.opacityScale
        : fadingOut
          ? appearance.opacity
          : snapEnterOpaque || appearance.state === "unchanged"
            ? undefined
            : appearance.opacity;
  const groupStyle: CSSProperties | undefined =
    groupOpacity === undefined &&
    (emphasis === undefined || emphasis.filter === "none")
      ? undefined
      : {
          ...(groupOpacity !== undefined ? { opacity: groupOpacity } : {}),
          ...(emphasis !== undefined && emphasis.filter !== "none"
            ? { filter: emphasis.filter }
            : {}),
        };

  const flowKind = groupLike
    ? "group"
    : node.kind !== undefined && node.kind.length > 0
      ? node.kind
      : capability.length > 0
        ? capability
        : undefined;

  return (
    <g
      key={node.id}
      data-flow-node-id={node.id}
      data-flow-kind={flowKind}
      data-flow-capability={capability.length > 0 ? capability : undefined}
      data-flow-local-layout={localChildren ? "true" : undefined}
      data-flow-motion-signal={
        isMotionSignalNode(node, capability) ? "true" : undefined
      }
      data-flow-pulse={pulseTagged ? "true" : undefined}
      data-timeline-state={appearance.state}
      data-draw-progress={
        drawProgress === undefined ? undefined : String(drawProgress)
      }
      data-emphasis-intensity={
        emphasis === undefined ? undefined : String(emphasis.intensity)
      }
      data-pulse-cue-intensity={
        pulseCue === undefined ? undefined : String(pulseCue.intensity)
      }
      aria-label={label}
      aria-describedby={descriptionId}
      role="img"
      focusable={false}
      style={groupStyle}
    >
      {descriptionId !== undefined ? (
        <desc id={descriptionId}>{description}</desc>
      ) : null}
      {body}
      {nested}
    </g>
  );
}

/**
 * Renders Flow Scene IR into an ExplainerShell diagram slot.
 * Plays authored timeline cues when `playing`, freezes when paused,
 * restarts on `restartKey`, and collapses to the final frame under reduced motion.
 *
 * Supports `core.rect` / `core.text` / `core.circle|ellipse` / `core.dot` /
 * `core.line|path|arrow|connector|elbow|fan-out|fan-in`, `core.panel` /
 * `core.header`,
 * `layout.stack` / `layout.grid`, nested local children, enter/draw/emphasize/pulse
 * /fade/exit/stagger cues (with easing), camera viewBox, theme paints, motion
 * signals, dashed strokes, and arrowheads.
 */
export function SceneRenderer({
  scene,
  playing,
  restartKey,
  reducedMotion = false,
  playbackRate = 1,
}: SceneRendererProps): ReactNode {
  const theme = useHostTheme();
  const reactId = useId().replaceAll(":", "");
  const markerPrefix = `scene-arrow-${reactId}`;
  const roots = omitMotionCompanionDots(scene.roots ?? []);
  const index = indexSceneNodes(roots);
  const authoredTimeline = Array.isArray(scene.timeline) ? scene.timeline : [];
  const timeline = expandTimelineCues(
    omitCompanionTimelineCues(authoredTimeline, index.nodesById),
    index.nodesById,
  );
  const durationMs = timelineDurationMs(timeline);
  const [playbackTimeMs, setPlaybackTimeMs] = useState(0);
  const playbackTimeMsRef = useRef(0);
  const rate = playbackRate > 0 ? playbackRate : 1;

  const commitTime = (nextMs: number) => {
    const clamped = Math.min(durationMs, Math.max(0, nextMs));
    playbackTimeMsRef.current = clamped;
    setPlaybackTimeMs(clamped);
  };

  useEffect(() => {
    commitTime(0);
  }, [restartKey, scene, durationMs]);

  useEffect(() => {
    if (reducedMotion) {
      commitTime(durationMs);
      return;
    }
    if (!playing) {
      return;
    }

    const wallOriginMs = performance.now();
    const playOriginMs = playbackTimeMsRef.current;
    let intervalId = 0;

    const syncFromWallClock = () => {
      const elapsed = Math.min(
        durationMs,
        playOriginMs + Math.max(0, performance.now() - wallOriginMs) * rate,
      );
      commitTime(elapsed);
      if (elapsed >= durationMs && intervalId !== 0) {
        window.clearInterval(intervalId);
        intervalId = 0;
      }
    };

    // 1ms cadence makes `vi.advanceTimersByTime(n)` land on exact cue times.
    intervalId = window.setInterval(syncFromWallClock, 1);
    syncFromWallClock();
    return () => {
      if (intervalId !== 0) {
        window.clearInterval(intervalId);
      }
    };
  }, [playing, reducedMotion, durationMs, restartKey, scene, rate]);

  const effectiveTimeMs = reducedMotion ? durationMs : playbackTimeMs;
  const ariaLabel =
    scene.accessibility?.label ?? scene.title ?? "Flow scene diagram";
  const summaryDescId =
    typeof scene.summary === "string" && scene.summary.length > 0
      ? `scene-summary-${reactId}`
      : undefined;
  const sceneTips = collectSceneTips(roots);
  const { width: viewportWidth, height: viewportHeight } = resolveViewportSize(
    scene.viewport,
  );
  const camera = authoredCameraAt(scene.camera, effectiveTimeMs);
  const viewBox = sceneViewBox(viewportWidth, viewportHeight, camera);
  const playback: PlaybackContext = {
    playing: playing && !reducedMotion,
    reducedMotion,
    restartKey,
    playbackRate: rate,
  };

  return (
    <svg
      className="scene-renderer"
      viewBox={viewBox}
      role="img"
      aria-label={ariaLabel}
      aria-describedby={summaryDescId}
      focusable={false}
      style={{ display: "block", width: "100%" }}
      data-scene-playing={playing ? "true" : "false"}
      data-scene-reduced-motion={reducedMotion ? "true" : "false"}
      data-scene-restart-key={String(restartKey)}
    >
      <defs>
        {sceneTips.map((tip) => {
          const geom = markerGeometry(tip);
          return (
            <marker
              key={tip.key}
              id={markerDomId(markerPrefix, tip)}
              markerWidth={geom.markerWidth}
              markerHeight={geom.markerHeight}
              // Attach at the tip base so the stroke stops before the head.
              refX={geom.refX}
              refY={geom.refY}
              orient="auto"
              markerUnits="strokeWidth"
            >
              {geom.children}
            </marker>
          );
        })}
      </defs>
      {summaryDescId !== undefined ? (
        <desc id={summaryDescId}>{scene.summary}</desc>
      ) : null}
      {orderArrowSiblingsLast(roots).map((node) =>
        renderNode(
          node,
          timeline,
          effectiveTimeMs,
          theme,
          markerPrefix,
          index,
          ZERO_ORIGIN,
          playback,
        ),
      )}
    </svg>
  );
}
