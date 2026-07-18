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
import { FlowArrow } from "./FlowArrow";
import { MotionSignal } from "./MotionSignal";
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
 * Explicit `x`/`y` win; otherwise `nodeId` resolves to the target node center.
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

/** Minimal render node supporting rect / text / path / line-like shapes. */
export type SceneNodeLike = Readonly<{
  id: string;
  kind?: string;
  capabilityId?: string;
  capability?: string;
  geometry?: SceneGeometryLike;
  layout?: SceneGeometryLike;
  style?: Readonly<Record<string, string | number>>;
  /** Glyph content for `core.text` / `kind: "text"` nodes. */
  text?: string;
  accessibility?: SceneNodeAccessibilityLike;
  children?: readonly SceneNodeLike[];
  /** SVG path data for path / arrow nodes (FlowArrow `d`). */
  d?: string;
  path?: string;
  /** Polyline waypoints for path / connector-like nodes. */
  points?: readonly ScenePointLike[];
  /** Endpoint coordinates or node refs for core.line / core.connector. */
  from?: ScenePointLike;
  to?: ScenePointLike;
}>;

/** Minimal timeline cue (enter/reveal/draw/emphasize/pulse). */
export type SceneTimelineCueLike = Readonly<{
  id: string;
  at: number;
  duration: number;
  action: string;
  target: string;
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
}>;

const VIEWPORT_WIDTH = 700;
const VIEWPORT_HEIGHT = 400;
const DEFAULT_ARROW_STROKE_WIDTH = 2.2;
const PULSE_CYCLE_MS = 2200;
const PULSE_DELAY_MS = 800;
const MOTION_DOT_DURATION = "2.2s";
const MOTION_DOT_DELAY = "0.8s";

type CameraTransform = Readonly<{ x: number; y: number; zoom: number }>;

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

const ARROW_CAPABILITIES = new Set([
  "core.line",
  "core.path",
  "core.arrow",
  "core.connector",
]);

const ARROW_KINDS = new Set(["line", "path", "arrow", "connector"]);

const DOT_CAPABILITIES = new Set(["core.dot", "core.circle"]);

const DOT_KINDS = new Set(["dot", "circle"]);

const MOTION_SIGNAL_CAPABILITIES = new Set([
  "motion.signal",
  "motion.dot",
  "core.motion",
]);

const DEFAULT_DOT_RADIUS = 5;
const DASHED_STROKE = "8 4";
const DOTTED_STROKE = "2 3";

type TimelineState = "hidden" | "entering" | "revealed" | "unchanged";

type TimelineAppearance = Readonly<{
  state: TimelineState;
  opacity: number;
}>;

/** Transient emphasize/emphasis/pulse envelope applied during an active cue. */
type EmphasisAppearance = Readonly<{
  /** 0–1 sine envelope peaking mid-cue. */
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
}>;

function finiteNumber(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
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

type LayoutOrigin = Readonly<{ x: number; y: number }>;

const ZERO_ORIGIN: LayoutOrigin = { x: 0, y: 0 };

/** Scene node index plus world-space geometry (after group/container offsets). */
type SceneNodeIndex = Readonly<{
  nodesById: ReadonlyMap<string, SceneNodeLike>;
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>;
}>;

/** Pure group container: nests children in a `<g>` with no leaf body of its own. */
function isGroupLike(node: SceneNodeLike, capability: string): boolean {
  if (capability === "core.group") {
    return true;
  }
  return node.kind === "group";
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
 */
function childrenUseLocalLayout(
  node: SceneNodeLike,
  parentGeom: SceneGeometryLike,
  children: readonly SceneNodeLike[] | undefined,
): boolean {
  if (!Array.isArray(children) || children.length === 0) {
    return false;
  }
  if (childrenFitParentLocalBox(parentGeom, children)) {
    return true;
  }
  if (capabilityOf(node) !== "core.group") {
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

/** Flatten scene roots (and nested children) into id → node / world-geometry maps. */
function indexSceneNodes(roots: readonly SceneNodeLike[]): SceneNodeIndex {
  const nodesById = new Map<string, SceneNodeLike>();
  const worldGeometryById = new Map<string, SceneGeometryLike>();

  const visit = (
    node: SceneNodeLike,
    originX: number,
    originY: number,
    coordsAreLocal: boolean,
  ): void => {
    const geom = geometryOf(node);
    const worldGeom: SceneGeometryLike = coordsAreLocal
      ? {
          x: originX + geom.x,
          y: originY + geom.y,
          width: geom.width,
          height: geom.height,
        }
      : geom;
    nodesById.set(node.id, node);
    worldGeometryById.set(node.id, worldGeom);

    const kids = node.children;
    if (!Array.isArray(kids) || kids.length === 0) {
      return;
    }
    const local = childrenUseLocalLayout(node, geom, kids);
    for (const child of kids) {
      if (local) {
        visit(child, worldGeom.x, worldGeom.y, true);
      } else {
        visit(child, 0, 0, false);
      }
    }
  };

  for (const root of roots) {
    visit(root, 0, 0, false);
  }
  return { nodesById, worldGeometryById };
}

/** Center point of world-space geometry (matches Flow runtime connectors). */
function nodeCenter(
  worldGeom: SceneGeometryLike,
): Readonly<{ x: number; y: number }> {
  return {
    x: worldGeom.x + worldGeom.width / 2,
    y: worldGeom.y + worldGeom.height / 2,
  };
}

/**
 * Resolve an endpoint into the current drawing frame.
 * Explicit `x`/`y` are already in-frame; `nodeId` centers are world-space and
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
  if (hasX || hasY) {
    return {
      x: hasX ? (endpoint.x as number) : 0,
      y: hasY ? (endpoint.y as number) : 0,
    };
  }
  if (typeof endpoint.nodeId === "string" && endpoint.nodeId.length > 0) {
    const world =
      index.worldGeometryById.get(endpoint.nodeId) ??
      (index.nodesById.has(endpoint.nodeId)
        ? geometryOf(index.nodesById.get(endpoint.nodeId)!)
        : undefined);
    if (world !== undefined) {
      const center = nodeCenter(world);
      return {
        x: center.x - layoutOrigin.x,
        y: center.y - layoutOrigin.y,
      };
    }
  }
  return { x: 0, y: 0 };
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
  return action === "draw";
}

function isEmphasizeAction(action: string): boolean {
  return action === "emphasize" || action === "emphasis";
}

function isPulseAction(action: string): boolean {
  return action === "pulse";
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

/** True when this path is a traveling motion-dot guide (not a directed edge). */
function isMotionSignalNode(node: SceneNodeLike): boolean {
  const label = (node.accessibility?.label ?? "").toLowerCase();
  if (label.includes("motion signal")) {
    return true;
  }
  const id = node.id.toLowerCase();
  if (id.includes("motion-sig") || id.includes("motion_sig")) {
    return true;
  }
  const role = node.style?.role;
  return role === "motion" || role === "motion-signal";
}

/** True when this rect is a legacy-style pulsing outline box. */
function isPulseNode(node: SceneNodeLike): boolean {
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

function markerEndDisabled(style: SceneNodeLike["style"]): boolean {
  const markerEnd = style?.markerEnd;
  return markerEnd === "none" || markerEnd === "false" || markerEnd === 0;
}

/** Directed edges get arrowheads; motion guides and undirected lines do not. */
function shouldShowArrowhead(node: SceneNodeLike, capability: string): boolean {
  if (isMotionSignalNode(node) || markerEndDisabled(node.style)) {
    return false;
  }
  if (
    capability === "core.arrow" ||
    capability === "core.connector" ||
    node.kind === "arrow" ||
    node.kind === "connector"
  ) {
    return true;
  }
  if (capability === "core.path" || node.kind === "path") {
    return true;
  }
  if (capability === "core.line" || node.kind === "line") {
    return node.style?.markerEnd !== undefined
      ? !markerEndDisabled(node.style)
      : true;
  }
  return false;
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

/** Map cue `at`/`duration` onto opacity and enter state for one node. */
function appearanceForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): TimelineAppearance {
  const cue = enterCueForNode(nodeId, timeline);
  if (cue === undefined) {
    return { state: "unchanged", opacity: 1 };
  }

  const atMs = Math.max(0, finiteNumber(cue.at));
  const cueDurationMs = Math.max(0, finiteNumber(cue.duration));

  let progress = 0;
  if (playbackTimeMs < atMs) {
    progress = 0;
  } else if (cueDurationMs === 0) {
    progress = 1;
  } else {
    progress = clamp01((playbackTimeMs - atMs) / cueDurationMs);
  }

  if (progress <= 0) {
    return { state: "hidden", opacity: 0 };
  }
  if (progress >= 1) {
    return { state: "revealed", opacity: 1 };
  }
  return { state: "entering", opacity: progress };
}

/**
 * Stroke-reveal progress for a `draw` cue in [0, 1].
 * Undefined when the node has no draw cue.
 */
function drawProgressForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): number | undefined {
  const cue = timeline
    .filter(
      (candidate) =>
        candidate.target === nodeId && isDrawAction(candidate.action),
    )
    .at(-1);
  if (cue === undefined) {
    return undefined;
  }
  const atMs = finiteNumber(cue.at);
  const durationMs = finiteNumber(cue.duration);
  if (playbackTimeMs <= atMs) {
    return 0;
  }
  if (durationMs <= 0) {
    return 1;
  }
  return clamp01((playbackTimeMs - atMs) / durationMs);
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

  const progress =
    durationMs <= 0
      ? 1
      : clamp01((playbackTimeMs - atMs) / durationMs);
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

/**
 * Authored `pulse` cue envelope (same half-sine shape as emphasize, no glow).
 */
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

  const progress =
    durationMs <= 0
      ? 1
      : clamp01((playbackTimeMs - atMs) / durationMs);
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
  appearance: TimelineAppearance,
  playbackTimeMs: number,
  playback: PlaybackContext,
): PulseAppearance | undefined {
  if (!isPulseNode(node) || playback.reducedMotion || !playback.playing) {
    return undefined;
  }
  if (appearance.state === "hidden") {
    return undefined;
  }
  if (playbackTimeMs < PULSE_DELAY_MS) {
    return { intensity: 0, opacity: 0 };
  }
  const cycle = ((playbackTimeMs - PULSE_DELAY_MS) % PULSE_CYCLE_MS) / PULSE_CYCLE_MS;
  // Match rust-arch-box-pulse: peak early, fade by ~21% of the cycle.
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
  if (value === "none" || value === "transparent" || !value.startsWith("@")) {
    return value;
  }

  const role = value.startsWith("@theme.")
    ? value.slice("@theme.".length)
    : value.startsWith("theme.")
      ? value.slice("theme.".length)
      : value;

  switch (role) {
    case "surface.primary":
    case "surface.elevated":
    case "surface.secondary":
    case "bg.elevated":
    case "bg.primary":
      return theme.bg.elevated;
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
      key === "role"
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

function isArrowLike(node: SceneNodeLike, capability: string): boolean {
  if (ARROW_CAPABILITIES.has(capability)) {
    return true;
  }
  return typeof node.kind === "string" && ARROW_KINDS.has(node.kind);
}

function isDotLike(node: SceneNodeLike, capability: string): boolean {
  if (DOT_CAPABILITIES.has(capability)) {
    return true;
  }
  if (typeof node.kind === "string" && DOT_KINDS.has(node.kind)) {
    return true;
  }
  const radius = node.style?.r;
  return typeof radius === "number" && Number.isFinite(radius) && radius > 0;
}

/** Traveling MentalModel-style motion dots (often authored as `motion-sig` paths). */
function isMotionSignalNode(node: SceneNodeLike, capability: string): boolean {
  if (MOTION_SIGNAL_CAPABILITIES.has(capability)) {
    return true;
  }
  if (/motion[-_]?sig/i.test(node.id)) {
    return true;
  }
  const label = node.accessibility?.label ?? "";
  if (/motion\s*signal/i.test(label)) {
    return true;
  }
  const motion = node.style?.motion;
  return (
    motion === true ||
    motion === 1 ||
    motion === "signal" ||
    motion === "dot" ||
    node.style?.role === "motion"
  );
}

/** Rects tagged for a gentle float/pulse (style.pulse, motion.pulse, or pulse-* ids). */
function isPulseTagged(node: SceneNodeLike, capability: string): boolean {
  if (capability === "motion.pulse") {
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
  if (/^pulse[-_]/i.test(node.id)) {
    return true;
  }
  return /motion\s*pulse/i.test(node.accessibility?.label ?? "");
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

/**
 * Soft opacity envelope for pulse-tagged rects (MentalModel box-pulse parity).
 * Returns undefined when the node should keep authored opacity only.
 */
function pulseOpacityScale(
  playbackTimeMs: number,
  reducedMotion: boolean,
): number | undefined {
  if (reducedMotion) {
    return undefined;
  }
  const cycleMs = 2200;
  const phase = (Math.max(0, playbackTimeMs) % cycleMs) / cycleMs;
  // 0 → 0.04 fade in, hold to 0.12, fade out by 0.21, idle until loop.
  if (phase < 0.04) {
    return (phase / 0.04) * 0.72;
  }
  if (phase < 0.12) {
    return 0.72;
  }
  if (phase < 0.21) {
    return 0.72 * (1 - (phase - 0.12) / 0.09);
  }
  return 0;
}

/** Gentle vertical float + stroke breathe for pulse-tagged shapes. */
function pulseFloatStyle(
  playbackTimeMs: number,
  reducedMotion: boolean,
): CSSProperties | undefined {
  if (reducedMotion) {
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
 * Recursively render nested `children` into sibling `<g>` wrappers.
 * When `layoutOffset` is set, children are parent-local and wrapped in
 * `translate(offset)` so group/container origins shift the subtree.
 */
function renderChildren(
  children: readonly SceneNodeLike[] | undefined,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  theme: Theme,
  arrowMarkerId: string,
  index: SceneNodeIndex,
  parentLayoutOrigin: LayoutOrigin,
  layoutOffset: LayoutOrigin | undefined,
  playback: PlaybackContext,
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
  const nested = children.map((child) =>
    renderNode(
      child,
      timeline,
      playbackTimeMs,
      theme,
      arrowMarkerId,
      index,
      childOrigin,
      playback,
    ),
  );
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

/**
 * Resolve SVG path data for line / path / arrow / connector nodes.
 * Precedence: authored `d` → `path` → `points` polyline → `from`/`to`.
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
  if (node.from !== undefined || node.to !== undefined) {
    const start = resolveEndpoint(node.from, index, layoutOrigin);
    const end = resolveEndpoint(node.to, index, layoutOrigin);
    return `M${start.x} ${start.y} L${end.x} ${end.y}`;
  }
  return undefined;
}

function strokeWidthFromStyle(
  style: SceneNodeLike["style"],
  fallback = DEFAULT_ARROW_STROKE_WIDTH,
): number {
  const width = style?.strokeWidth;
  return typeof width === "number" && Number.isFinite(width) ? width : fallback;
}

function cornerRadiusFromStyle(style: SceneNodeLike["style"], fallback = 10): number {
  const radius = style?.radius ?? style?.rx ?? style?.borderRadius;
  return typeof radius === "number" && Number.isFinite(radius) ? radius : fallback;
}

function renderNode(
  node: SceneNodeLike,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  theme: Theme,
  arrowMarkerId: string,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin = ZERO_ORIGIN,
  playback: PlaybackContext,
): ReactNode {
  const capability = capabilityOf(node);
  const geom = geometryOf(node);
  const appearance = appearanceForNode(node.id, timeline, playbackTimeMs);
  const drawProgress = drawProgressForNode(node.id, timeline, playbackTimeMs);
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
    appearance,
    playbackTimeMs,
    playback,
  );
  const activeEmphasis = emphasis ?? pulseCue;
  // Prefer authored accessibility.label; fall back to id only when absent.
  const label = node.accessibility?.label ?? node.id;
  const description = node.accessibility?.description;
  const descriptionId =
    typeof description === "string" && description.length > 0
      ? `flow-node-${node.id}-desc`
      : undefined;
  const localChildren = childrenUseLocalLayout(node, geom, node.children);
  const nested = renderChildren(
    node.children,
    timeline,
    playbackTimeMs,
    theme,
    arrowMarkerId,
    index,
    layoutOrigin,
    localChildren ? { x: geom.x, y: geom.y } : undefined,
    playback,
  );

  const themeBg = theme.bg.elevated;
  const themeStroke = theme.stroke.secondary;
  const themeText = theme.text.primary;
  const groupLike = isGroupLike(node, capability);
  const strokeScale = activeEmphasis?.strokeScale ?? 1;

  let body: ReactNode = null;
  if (capability === "core.rect" || node.kind === "rect") {
    const strokePaint = paintFromStyle(node.style, "stroke", theme, themeStroke);
    const fillPaint = paintFromStyle(node.style, "fill", theme, themeBg);
    const pulseOpacity =
      continuousPulse !== undefined
        ? continuousPulse.opacity
        : undefined;
    body = (
      <rect
        x={geom.x}
        y={geom.y}
        width={geom.width}
        height={geom.height}
        rx={cornerRadiusFromStyle(node.style)}
        fill={fillPaint}
        stroke={strokePaint}
        strokeWidth={strokeWidthFromStyle(node.style, 1.3) * strokeScale}
        focusable={false}
        aria-hidden="true"
        style={{
          ...styleToCss(node.style, theme),
          ...(pulseOpacity !== undefined ? { opacity: pulseOpacity } : {}),
        }}
        data-pulse-intensity={
          continuousPulse === undefined
            ? undefined
            : String(continuousPulse.intensity)
        }
      />
    );
  } else if (capability === "core.text" || node.kind === "text") {
    const fontSize =
      typeof node.style?.fontSize === "number" ? node.style.fontSize : 14;
    const textAnchor =
      typeof node.style?.textAnchor === "string"
        ? node.style.textAnchor
        : undefined;
    const textX =
      textAnchor === "middle" ? geom.x + geom.width / 2 : geom.x;
    body = (
      <text
        x={textX}
        y={geom.y}
        dominantBaseline="hanging"
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
  } else if (isArrowLike(node, capability)) {
    const d = arrowPathData(node, index, layoutOrigin);
    if (d !== undefined) {
      const stroke = paintFromStyle(node.style, "stroke", theme, themeAccent);
      const motionSignal = isMotionSignalNode(node);
      const showMarker = shouldShowArrowhead(node, capability);
      const nodeMarkerId = `${arrowMarkerId}-${node.id}`;
      const motionActive =
        motionSignal &&
        playback.playing &&
        !playback.reducedMotion &&
        appearance.state !== "hidden" &&
        (drawProgress === undefined || drawProgress > 0);

      body = (
        <>
          <FlowArrow
            d={d}
            markerId={nodeMarkerId}
            showMarker={showMarker}
            color={stroke}
            strokeWidth={strokeWidthFromStyle(node.style) * strokeScale}
            pathLength={drawProgress === undefined ? undefined : 1}
            strokeDasharray={drawProgress === undefined ? undefined : 1}
            strokeDashoffset={
              drawProgress === undefined ? undefined : 1 - drawProgress
            }
            focusable={false}
            aria-hidden="true"
            style={styleToCss(node.style, theme)}
            data-flow-arrowhead={showMarker ? "true" : "false"}
          />
          {motionSignal ? (
            <MotionSignal
              key={`motion-${playback.restartKey}-${node.id}`}
              path={d}
              color={stroke}
              delay={MOTION_DOT_DELAY}
              duration={MOTION_DOT_DURATION}
              reducedMotion={playback.reducedMotion}
              active={motionActive}
              data-flow-motion-signal={node.id}
            />
          ) : null}
        </>
      );
    }
  }

  const baseOpacity =
    appearance.state === "unchanged" ? 1 : appearance.opacity;
  const groupOpacity =
    activeEmphasis !== undefined
      ? baseOpacity * activeEmphasis.opacityScale
      : appearance.state === "unchanged"
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
      data-flow-local-layout={localChildren ? "true" : undefined}
      data-flow-motion-signal={isMotionSignalNode(node) ? "true" : undefined}
      data-flow-pulse={isPulseNode(node) ? "true" : undefined}
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
 * Supports `core.rect` / `core.text` / `core.line|path|arrow|connector`, nested
 * children with layout offsets, enter/draw/emphasize/pulse cues, camera viewBox,
 * theme paints, motion dots on motion-signal paths, and arrowheads on directed edges.
 */
export function SceneRenderer({
  scene,
  playing,
  restartKey,
  reducedMotion = false,
}: SceneRendererProps): ReactNode {
  const theme = useHostTheme();
  const reactId = useId().replaceAll(":", "");
  const arrowMarkerId = `scene-arrow-${reactId}`;
  const timeline = Array.isArray(scene.timeline) ? scene.timeline : [];
  const durationMs = timelineDurationMs(timeline);
  const [playbackTimeMs, setPlaybackTimeMs] = useState(0);
  const playbackTimeMsRef = useRef(0);

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
        playOriginMs + Math.max(0, performance.now() - wallOriginMs),
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
  }, [playing, reducedMotion, durationMs, restartKey, scene]);

  const effectiveTimeMs = reducedMotion ? durationMs : playbackTimeMs;
  const ariaLabel =
    scene.accessibility?.label ?? scene.title ?? "Flow scene diagram";
  const summaryDescId =
    typeof scene.summary === "string" && scene.summary.length > 0
      ? `scene-summary-${reactId}`
      : undefined;
  const roots = scene.roots ?? [];
  const index = indexSceneNodes(roots);
  const { width: viewportWidth, height: viewportHeight } = resolveViewportSize(
    scene.viewport,
  );
  const camera = authoredCameraAt(scene.camera, effectiveTimeMs);
  const viewBox = sceneViewBox(viewportWidth, viewportHeight, camera);
  const playback: PlaybackContext = {
    playing: playing && !reducedMotion,
    reducedMotion,
    restartKey,
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
      {summaryDescId !== undefined ? (
        <desc id={summaryDescId}>{scene.summary}</desc>
      ) : null}
      {roots.map((node) =>
        renderNode(
          node,
          timeline,
          effectiveTimeMs,
          theme,
          arrowMarkerId,
          index,
          ZERO_ORIGIN,
          playback,
        ),
      )}
    </svg>
  );
}
