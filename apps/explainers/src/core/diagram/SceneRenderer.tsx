/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  useEffect,
  useId,
  useMemo,
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
import {
  elbowPathData,
  isCurveRoute,
  isElbowRoute,
  normalizeCurveRouteOptions,
  routeCurve,
  type CurveRouteOptions,
  type RouteObstacle,
} from "./connector-routing.js";
import { FlowArrow } from "./FlowArrow";
import { MotionSignal } from "./MotionSignal";
import { isFanNode, isMotionSignalNode } from "./node-classification.js";
import { hasNativeSemanticChrome } from "./capabilities/chrome.js";
import { resolveScene } from "./resolution/resolve-scene.js";
import { isRoutingObstacle } from "./resolution/resolve-connectors.js";
import type {
  ResolvedConnector,
  ResolvedFanGeometry,
  ResolvedGeneratedPart,
} from "./resolution/types.js";
import type {
  SceneCameraKeyframeLike,
  SceneGeometryLike,
  SceneIrLike,
  SceneNodeLike,
  ScenePointLike,
  SceneTimelineCueLike,
  SceneViewportLike,
} from "./scene-types.js";
import { scaledSceneFontSize, wrapTextToWidth } from "./text-metrics.js";

export type {
  SceneCameraKeyframeLike,
  SceneGeometryLike,
  SceneIrLike,
  SceneNodeAccessibilityLike,
  SceneNodeLike,
  ScenePointLike,
  SceneRelativePositionLike,
  SceneSourceRangeLike,
  SceneStyleValue,
  SceneTimelineCueLike,
  SceneViewportLike,
} from "./scene-types.js";

// Re-exported for direct unit testing of the shared fan segment builder.
export { fanSegmentFromAtomic } from "./resolution/resolve-fans.js";

export type SceneRendererProps = Readonly<{
  scene: SceneIrLike;
  playing: boolean;
  restartKey: number;
  reducedMotion?: boolean;
  /** Wall-clock multiplier for timeline advance (1 = realtime). */
  playbackRate?: number;
}>;

/** 16:9 default canvas; scales exactly 2x to a 3840x2160 4K export. */
const VIEWPORT_WIDTH = 1920;
const VIEWPORT_HEIGHT = 1080;
const DEFAULT_ARROW_STROKE_WIDTH: number = tokens.diagram.strokeWidth;
const SVG_NS = "http://www.w3.org/2000/svg";
const DEFAULT_DOT_RADIUS = 13.5;
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
  ancestorIdsById: ReadonlyMap<string, readonly string[]>;
  generatedPartsById: ReadonlyMap<string, ResolvedGeneratedPart>;
  connectorsById: ReadonlyMap<string, ResolvedConnector>;
  fanGeometryById: ReadonlyMap<string, ResolvedFanGeometry>;
}>;

/** @internal exported for direct unit testing of fan/arrowhead geometry helpers. */
export type ScenePoint = Readonly<{ x: number; y: number }>;

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
  const sorted = camera
    .map((keyframe, index) => ({ keyframe, index }))
    .sort(
      (left, right) =>
        finiteNumber(left.keyframe.at) - finiteNumber(right.keyframe.at) ||
        left.index - right.index,
    )
    .map(({ keyframe }) => keyframe);
  // Duplicate `at` values (authoring mistake or a generated override) would
  // otherwise leave `findIndex` picking whichever duplicate sorted first;
  // collapse adjacent equal-`at` keyframes and keep the last so the later
  // authored keyframe consistently wins.
  const keyframes: SceneCameraKeyframeLike[] = [];
  for (const keyframe of sorted) {
    const at = finiteNumber(keyframe.at);
    const previous = keyframes[keyframes.length - 1];
    if (previous !== undefined && finiteNumber(previous.at) === at) {
      keyframes[keyframes.length - 1] = keyframe;
    } else {
      keyframes.push(keyframe);
    }
  }
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
  if (!Number.isFinite(visibleWidth) || !Number.isFinite(visibleHeight)) {
    // An extreme (near-zero) zoom can overflow finite-but-huge viewport math
    // to Infinity; fall back to the static, unzoomed viewport rather than
    // emit a non-finite viewBox that browsers reject outright.
    return `0 0 ${width} ${height}`;
  }
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
        return fanGeometryFor(candidate, index, layoutOrigin)?.trajectories[
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
  const resolved = index.connectorsById.get(node.id);
  if (resolved?.d !== undefined && resolved.d.length > 0) {
    return boundaryOnlyMotionPath(resolved.d, index, layoutOrigin);
  }

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
    raw = motionConnectorPathData(
      node,
      start,
      end,
      from,
      to,
      index,
      layoutOrigin,
    );
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
const MOTION_BOUNDARY_GAP_MIN = 21.6;

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

function pointsNear(a: ScenePoint, b: ScenePoint, eps = 0.001): boolean {
  return Math.abs(a.x - b.x) <= eps && Math.abs(a.y - b.y) <= eps;
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

/** Closest point to `point` on segment `a`-`b`, clamped to the segment. */
function projectPointOntoSegment(
  point: ScenePoint,
  a: ScenePoint,
  b: ScenePoint,
): Readonly<{ point: ScenePoint; distance: number }> {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSquared = dx * dx + dy * dy;
  const t =
    lengthSquared === 0
      ? 0
      : clamp01(((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSquared);
  const projected = { x: a.x + dx * t, y: a.y + dy * t };
  return {
    point: projected,
    distance: Math.hypot(point.x - projected.x, point.y - projected.y),
  };
}

/**
 * Split a fan trajectory at the junction into trunk and branch ball paths.
 * @internal exported for direct unit testing of off-trajectory junctions.
 */
export function splitFanTrajectoryAtJunction(
  d: string,
  junction: ScenePoint,
): Readonly<{ head: string; tail: string }> | undefined {
  const points = fanPathPoints(d);
  if (points.length < 2) {
    return undefined;
  }
  const vertexIndex = points.findIndex((point) => pointsNear(point, junction));
  if (vertexIndex > 0 && vertexIndex < points.length - 1) {
    return {
      head: fanPath(points.slice(0, vertexIndex + 1)),
      tail: fanPath(points.slice(vertexIndex)),
    };
  }
  // No interior vertex sits on the junction — an authored junction can land
  // off-trajectory (a bend the automatic routing did not produce), and the
  // nearest vertex can be a trajectory endpoint. Snapping to that endpoint
  // used to bail out with `undefined`, dropping the ball entirely; instead
  // project onto the nearest edge and splice the projected point into the
  // polyline so every trajectory still yields a head/tail split.
  let bestSegmentIndex = 0;
  let bestProjection = projectPointOntoSegment(junction, points[0]!, points[1]!);
  for (let index = 1; index < points.length - 1; index++) {
    const projection = projectPointOntoSegment(
      junction,
      points[index]!,
      points[index + 1]!,
    );
    if (projection.distance < bestProjection.distance) {
      bestProjection = projection;
      bestSegmentIndex = index;
    }
  }
  const spliced = [
    ...points.slice(0, bestSegmentIndex + 1),
    bestProjection.point,
    ...points.slice(bestSegmentIndex + 1),
  ];
  const junctionIndex = bestSegmentIndex + 1;
  if (junctionIndex <= 0 || junctionIndex >= spliced.length - 1) {
    return undefined;
  }
  return {
    head: fanPath(spliced.slice(0, junctionIndex + 1)),
    tail: fanPath(spliced.slice(junctionIndex)),
  };
}

/** Rebase canonical world-space fan geometry into the current layout's local space. */
function localizeFanGeometry(
  geometry: ResolvedFanGeometry,
  layoutOrigin: LayoutOrigin,
): ResolvedFanGeometry {
  if (layoutOrigin.x === 0 && layoutOrigin.y === 0) {
    return geometry;
  }
  const shift = (d: string): string =>
    fanPath(
      fanPathPoints(d).map((point) => ({
        x: point.x - layoutOrigin.x,
        y: point.y - layoutOrigin.y,
      })),
    );
  return {
    ...geometry,
    junction: {
      x: geometry.junction.x - layoutOrigin.x,
      y: geometry.junction.y - layoutOrigin.y,
    },
    segments: geometry.segments.map((segment) => ({
      ...segment,
      d: shift(segment.d),
    })),
    trajectories: geometry.trajectories.map((trajectory) => ({
      ...trajectory,
      d: shift(trajectory.d),
    })),
  };
}

/** Look up a node's canonical fan geometry from the resolved index, rebased to layout space. */
function fanGeometryFor(
  node: SceneNodeLike,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): ResolvedFanGeometry | undefined {
  const geometry = index.fanGeometryById.get(node.id);
  return geometry === undefined
    ? undefined
    : localizeFanGeometry(geometry, layoutOrigin);
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

/**
 * Scene duration spans the latest cue end and the latest authored camera
 * keyframe `at` — a camera move scheduled after the last cue must still get
 * to play instead of being clamped away by a duration derived from cues alone.
 */
function timelineDurationMs(
  timeline: readonly SceneTimelineCueLike[],
  camera?: readonly SceneCameraKeyframeLike[],
): number {
  const cueEnd = timeline.reduce(
    (maximum, cue) =>
      Math.max(maximum, finiteNumber(cue.at) + finiteNumber(cue.duration)),
    0,
  );
  const cameraEnd = Array.isArray(camera)
    ? camera.reduce(
        (maximum, keyframe) => Math.max(maximum, finiteNumber(keyframe.at)),
        0,
      )
    : 0;
  return Math.max(cueEnd, cameraEnd);
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
 * Nodes eligible to act as routing obstacles. Connectors, fans, motion signals,
 * and the arrow family are never obstacles; positive-area filtering happens in
 * {@link curveObstacles} via {@link isRoutingObstacle}.
 */
function isRouteObstacleNode(node: SceneNodeLike): boolean {
  const capability = capabilityOf(node);
  return (
    !ARROW_CAPABILITIES.has(capability) &&
    !isMotionSignalNode(node, capability) &&
    node.kind !== "connector" &&
    node.kind !== "fan"
  );
}

/**
 * World-space obstacle rectangles for one curved edge. The source, target, and
 * their ancestor containers are excluded so a curve is never blocked by the very
 * boxes it connects; zero-area and non-finite geometry is dropped. The result is
 * id-sorted for deterministic routing.
 */
function curveObstacles(
  from: ScenePointLike | undefined,
  to: ScenePointLike | undefined,
  index: SceneNodeIndex,
  options: CurveRouteOptions,
): readonly RouteObstacle[] {
  if (!options.avoidObstacles) {
    return [];
  }
  const excludedIds = new Set<string>();
  for (const endpoint of [from, to]) {
    const id = endpoint?.nodeId;
    if (typeof id === "string" && id.length > 0) {
      excludedIds.add(id);
      for (const ancestor of index.ancestorIdsById.get(id) ?? []) {
        excludedIds.add(ancestor);
      }
    }
  }
  const obstacles: RouteObstacle[] = [];
  const generatedPartIds =
    index.generatedPartsById.size > 0
      ? new Set(index.generatedPartsById.keys())
      : undefined;
  for (const [id, geometry] of index.worldGeometryById) {
    if (excludedIds.has(id)) {
      continue;
    }
    const candidate = index.nodesById.get(id);
    if (
      candidate === undefined ||
      !isRouteObstacleNode(candidate) ||
      !isRoutingObstacle(candidate, geometry, generatedPartIds)
    ) {
      continue;
    }
    if (
      !Number.isFinite(geometry.x) ||
      !Number.isFinite(geometry.y) ||
      !Number.isFinite(geometry.width) ||
      !Number.isFinite(geometry.height) ||
      !(geometry.width > 0) ||
      !(geometry.height > 0)
    ) {
      continue;
    }
    obstacles.push({ id, bounds: geometry });
  }
  obstacles.sort((left, right) => left.id.localeCompare(right.id));
  return obstacles;
}

/** Rebase world-space route obstacles into a connector's local drawing frame. */
function frameRouteObstacles(
  from: ScenePointLike | undefined,
  to: ScenePointLike | undefined,
  index: SceneNodeIndex,
  options: CurveRouteOptions,
  layoutOrigin: LayoutOrigin,
): readonly RouteObstacle[] {
  return curveObstacles(from, to, index, options).map((obstacle) => ({
    id: obstacle.id,
    bounds: {
      x: obstacle.bounds.x - layoutOrigin.x,
      y: obstacle.bounds.y - layoutOrigin.y,
      width: obstacle.bounds.width,
      height: obstacle.bounds.height,
    },
  }));
}

/**
 * Read the canonical curved route, with a compatibility fallback for temporary
 * scene indexes that do not contain this node.
 */
function resolvedCurveRoute(
  node: SceneNodeLike,
  from: ScenePointLike | undefined,
  to: ScenePointLike | undefined,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): string {
  const connector = index.connectorsById.get(node.id);
  if (connector !== undefined) {
    return connector.d;
  }
  const frameStart = resolveEndpoint(from, index, layoutOrigin);
  const frameEnd = resolveEndpoint(to, index, layoutOrigin);
  const start = { x: frameStart.x + layoutOrigin.x, y: frameStart.y + layoutOrigin.y };
  const end = { x: frameEnd.x + layoutOrigin.x, y: frameEnd.y + layoutOrigin.y };
  const options = normalizeCurveRouteOptions(node.style);
  const fromId = typeof from?.nodeId === "string" && from.nodeId.length > 0 ? from.nodeId : undefined;
  const toId = typeof to?.nodeId === "string" && to.nodeId.length > 0 ? to.nodeId : undefined;
  const route = routeCurve({
    edgeId: node.id,
    start,
    end,
    fromAnchor: typeof from?.anchor === "string" ? from.anchor : undefined,
    toAnchor: typeof to?.anchor === "string" ? to.anchor : undefined,
    sourceId: fromId,
    targetId: toId,
    sourceBounds: fromId !== undefined ? index.worldGeometryById.get(fromId) : undefined,
    targetBounds: toId !== undefined ? index.worldGeometryById.get(toId) : undefined,
    obstacles: curveObstacles(from, to, index, options),
    siblings: [],
    options,
  });
  return route.d;
}

/** Resolve connector geometry for motion overlays on node-linked endpoints. */
function motionConnectorPathData(
  node: SceneNodeLike,
  start: Readonly<{ x: number; y: number }>,
  end: Readonly<{ x: number; y: number }>,
  from: ScenePointLike | undefined,
  to: ScenePointLike | undefined,
  index: SceneNodeIndex,
  layoutOrigin: LayoutOrigin,
): string {
  if (isCurveRoute(node)) {
    return resolvedCurveRoute(node, from, to, index, layoutOrigin);
  }
  if (isElbowRoute(node) || node.via !== undefined || connectorAxisOf(node) !== undefined) {
    const options = normalizeCurveRouteOptions(node.style);
    return elbowPathData(
      start,
      end,
      node.via === undefined
        ? undefined
        : resolveEndpoint(node.via, index, layoutOrigin),
      connectorAxisOf(node),
      typeof from?.anchor === "string" ? from.anchor : undefined,
      typeof to?.anchor === "string" ? to.anchor : undefined,
      frameRouteObstacles(from, to, index, options, layoutOrigin),
      options.clearance,
    );
  }
  return `M${formatPathNumber(start.x)} ${formatPathNumber(start.y)} L${formatPathNumber(end.x)} ${formatPathNumber(end.y)}`;
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
  // Braces are undirected geometry (also stamp markerEnd: "none").
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

/**
 * Picks the cue whose window most recently began at or before
 * `playbackTimeMs`, falling back to the earliest-authored cue when none
 * has started yet. Authoring the same target with more than one cue of a
 * kind (e.g. two `trace` cues, meant as "draw early, then confirm later")
 * is a supported idiom — picking blindly by declaration order (`.at(-1)`)
 * instead of by which window is actually live discards the earlier cue's
 * entire animation window, causing a pop-then-restart glitch.
 */
function mostRecentlyStartedCue(
  cues: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): SceneTimelineCueLike | undefined {
  let started: SceneTimelineCueLike | undefined;
  let earliest: SceneTimelineCueLike | undefined;
  for (const cue of cues) {
    const atMs = finiteNumber(cue.at);
    if (earliest === undefined || atMs < finiteNumber(earliest.at)) {
      earliest = cue;
    }
    if (
      atMs <= playbackTimeMs &&
      (started === undefined || atMs >= finiteNumber(started.at))
    ) {
      started = cue;
    }
  }
  return started ?? earliest;
}

/**
 * Picks the cue whose `[at, at + duration]` window currently contains
 * `playbackTimeMs` (last match wins on overlap), unlike `.at(-1)` which
 * blindly takes the last-authored cue even if its window isn't active —
 * silently masking an earlier, currently-active cue of the same kind.
 */
/**
 * Forward tolerance for a zero-duration cue's active window. Real playback
 * advances in animation-frame steps, so `playbackTimeMs === atMs` almost
 * never holds exactly — an authored `duration: 0` emphasize/pulse cue would
 * otherwise never have a frame in which it is considered active.
 */
const ZERO_DURATION_CUE_WINDOW_MS = 48;

function activeWindowCue(
  cues: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): SceneTimelineCueLike | undefined {
  let match: SceneTimelineCueLike | undefined;
  for (const cue of cues) {
    const atMs = Math.max(0, finiteNumber(cue.at));
    const durationMs = Math.max(0, finiteNumber(cue.duration));
    const inWindow =
      durationMs <= 0
        ? playbackTimeMs >= atMs &&
          playbackTimeMs <= atMs + ZERO_DURATION_CUE_WINDOW_MS
        : playbackTimeMs >= atMs && playbackTimeMs <= atMs + durationMs;
    if (inWindow) {
      match = cue;
    }
  }
  return match;
}

function enterCueForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): SceneTimelineCueLike | undefined {
  const cues = timeline.filter(
    (candidate) =>
      candidate.target === nodeId && isEnterLikeAction(candidate.action),
  );
  return mostRecentlyStartedCue(cues, playbackTimeMs);
}

function fadeCueForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): SceneTimelineCueLike | undefined {
  const cues = timeline.filter(
    (candidate) =>
      candidate.target === nodeId && isFadeLikeAction(candidate.action),
  );
  return mostRecentlyStartedCue(cues, playbackTimeMs);
}

/** Map cue `at`/`duration` onto opacity and enter state for one node. */
function appearanceForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): TimelineAppearance {
  const enterCue = enterCueForNode(nodeId, timeline, playbackTimeMs);
  const fadeCue = fadeCueForNode(nodeId, timeline, playbackTimeMs);

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

  // A fade authored before the currently active enter cue is stale — a
  // later enter/reveal supersedes it and must not keep the node hidden or
  // faded forever.
  const fadeSuperseded =
    fadeCue !== undefined &&
    enterCue !== undefined &&
    Math.max(0, finiteNumber(fadeCue.at)) <
      Math.max(0, finiteNumber(enterCue.at));

  if (fadeCue !== undefined && !fadeSuperseded) {
    const fadeAt = Math.max(0, finiteNumber(fadeCue.at));
    if (playbackTimeMs >= fadeAt) {
      const fadeProgress = cueProgress(fadeCue, playbackTimeMs);
      if (fadeProgress >= 1) {
        // `exit` is terminal — the node stays hidden. A completed `fade`
        // (not `exit`) is not terminal; fall through to the enter-derived
        // state so a later `reveal` cue can bring the node back.
        if (fadeCue.action === "exit") {
          return { state: "hidden", opacity: 0 };
        }
      } else {
        const opacity = enterOpacity * (1 - fadeProgress);
        return {
          state:
            opacity <= 0 ? "hidden" : state === "unchanged" ? "entering" : state,
          opacity,
        };
      }
    }
  }

  if (enterCue === undefined) {
    return { state: "unchanged", opacity: 1 };
  }
  return { state, opacity: enterOpacity };
}

/**
 * Cap connector visibility to its node-backed endpoints and their ancestors.
 * Coordinate-only endpoints have no timeline dependency.
 */
function connectorEndpointOpacity(
  node: SceneNodeLike,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  index: SceneNodeIndex,
): number | undefined {
  const dependencyIds = new Set<string>();
  let brokenReference = false;
  const endpoints = [
    ...scenePoints(node.from),
    ...scenePoints(node.to),
    ...scenePoints(node.junction),
  ];
  for (const endpoint of endpoints) {
    const endpointId = endpoint.nodeId;
    if (typeof endpointId !== "string" || endpointId.length === 0) {
      continue;
    }
    if (!index.nodesById.has(endpointId)) {
      // A non-empty nodeId that does not resolve is an authoring error, not
      // "no dependency" — fail closed instead of rendering an orphaned
      // connector at full opacity.
      brokenReference = true;
      continue;
    }
    dependencyIds.add(endpointId);
    for (const ancestorId of index.ancestorIdsById.get(endpointId) ?? []) {
      dependencyIds.add(ancestorId);
    }
  }
  if (brokenReference) {
    return 0;
  }
  if (dependencyIds.size === 0) {
    return undefined;
  }
  let opacity = 1;
  for (const dependencyId of dependencyIds) {
    opacity = Math.min(
      opacity,
      appearanceForNode(dependencyId, timeline, playbackTimeMs).opacity,
    );
  }
  return opacity;
}

/** True only while a fade/exit cue's window is actively progressing. */
function isFadingOut(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): boolean {
  const fadeCue = fadeCueForNode(nodeId, timeline, playbackTimeMs);
  if (fadeCue === undefined) {
    return false;
  }
  const fadeAt = Math.max(0, finiteNumber(fadeCue.at));
  const enterCue = enterCueForNode(nodeId, timeline, playbackTimeMs);
  if (
    enterCue !== undefined &&
    fadeAt < Math.max(0, finiteNumber(enterCue.at))
  ) {
    // Superseded by a newer enter cue — the stale fade is not in effect.
    return false;
  }
  if (playbackTimeMs < fadeAt) {
    return false;
  }
  const fadeProgress = cueProgress(fadeCue, playbackTimeMs);
  return fadeProgress > 0 && fadeProgress < 1;
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
  const cues = timeline.filter(
    (candidate) =>
      candidate.target === nodeId &&
      isDrawAction(candidate.action) &&
      (includeTrace || candidate.action !== "trace"),
  );
  const cue = mostRecentlyStartedCue(cues, playbackTimeMs);
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
  const cues = timeline.filter(
    (candidate) => candidate.target === nodeId && candidate.action === "trace",
  );
  const cue = mostRecentlyStartedCue(cues, playbackTimeMs);
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
  const cues = timeline.filter(
    (candidate) => candidate.target === nodeId && candidate.action === "trace",
  );
  const cue = mostRecentlyStartedCue(cues, playbackTimeMs);
  if (cue === undefined) {
    return undefined;
  }
  if (playbackTimeMs < finiteNumber(cue.at)) {
    return 0;
  }
  return cueProgress(cue, playbackTimeMs);
}

/**
 * Dev-only, once-per-node warning: `emphasize` and `pulse` cues both
 * active for the same node at once silently drop `pulse` (emphasize
 * wins via `emphasis ?? pulseCue`). Not a rendering bug by itself, but
 * an authoring ambiguity worth flagging rather than masking.
 */
const warnedOverlappingEmphasisNodeIds = new Set<string>();
function warnOverlappingEmphasisOnce(nodeId: string): void {
  if (warnedOverlappingEmphasisNodeIds.has(nodeId)) {
    return;
  }
  warnedOverlappingEmphasisNodeIds.add(nodeId);
  console.warn(
    `[flow] node "${nodeId}" has overlapping emphasize and pulse cues; ` +
      "emphasize wins and pulse is ignored for the overlap. Stagger the " +
      "cue windows if both effects are intended.",
  );
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
  const cues = timeline.filter(
    (candidate) =>
      candidate.target === nodeId && isEmphasizeAction(candidate.action),
  );
  const cue = activeWindowCue(cues, playbackTimeMs);
  if (cue === undefined) {
    return undefined;
  }

  const atMs = Math.max(0, finiteNumber(cue.at));
  const durationMs = Math.max(0, finiteNumber(cue.duration));
  if (playbackTimeMs < atMs) {
    return undefined;
  }
  if (durationMs <= 0) {
    if (playbackTimeMs > atMs + ZERO_DURATION_CUE_WINDOW_MS) {
      return undefined;
    }
  } else if (playbackTimeMs > atMs + durationMs) {
    return undefined;
  }

  // A zero-duration cue has no window to ease across — treat it as an
  // instantaneous one-frame peak (the midpoint of the half-sine envelope)
  // instead of `sin(progress * PI)` at progress 1, which rounds to ~0.
  const intensity =
    durationMs <= 0 ? 1 : Math.sin(cueProgress(cue, playbackTimeMs) * Math.PI);
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
  const cues = timeline.filter(
    (candidate) => candidate.target === nodeId && isPulseAction(candidate.action),
  );
  const cue = activeWindowCue(cues, playbackTimeMs);
  if (cue === undefined) {
    return undefined;
  }

  const atMs = Math.max(0, finiteNumber(cue.at));
  const durationMs = Math.max(0, finiteNumber(cue.duration));
  if (playbackTimeMs < atMs) {
    return undefined;
  }
  if (durationMs <= 0) {
    if (playbackTimeMs > atMs + ZERO_DURATION_CUE_WINDOW_MS) {
      return undefined;
    }
  } else if (playbackTimeMs > atMs + durationMs) {
    return undefined;
  }

  // See emphasisForNode: a zero-duration cue peaks for one frame instead of
  // evaluating the half-sine envelope at progress 1 (~0 intensity).
  const intensity =
    durationMs <= 0 ? 1 : Math.sin(cueProgress(cue, playbackTimeMs) * Math.PI);
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
      stroke,
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
      key === "clearance" ||
      key === "curvature" ||
      key === "avoidObstacles" ||
      key === "preferredSide" ||
      key === "bundle" ||
      key === "parallelGap" ||
      key === "laneGap" ||
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
  const resolved = index.connectorsById.get(node.id);
  if (resolved !== undefined) {
    return resolved.d;
  }
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
    if (isCurveRoute(node)) {
      return resolvedCurveRoute(node, from, to, index, layoutOrigin);
    }
    const start = resolveEndpoint(from, index, layoutOrigin);
    const end = resolveEndpoint(to, index, layoutOrigin);
    if (isElbowRoute(node)) {
      const options = normalizeCurveRouteOptions(node.style);
      const via =
        node.via !== undefined
          ? resolveEndpoint(node.via, index, layoutOrigin)
          : undefined;
      return elbowPathData(
        start,
        end,
        via,
        connectorAxisOf(node),
        typeof from?.anchor === "string" ? from.anchor : undefined,
        typeof to?.anchor === "string" ? to.anchor : undefined,
        frameRouteObstacles(from, to, index, options, layoutOrigin),
        options.clearance,
      );
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

/** Index of the last SVG path command letter in `d` (path always starts with `M`). */
function lastCommandStartIndex(d: string): number {
  const commandLetters = /[MmLlHhVvCcSsQqTtAaZz]/g;
  let lastIndex = -1;
  let match: RegExpExecArray | null;
  while ((match = commandLetters.exec(d)) !== null) {
    lastIndex = match.index;
  }
  return lastIndex;
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
    const trimmed = d.trim();
    const lastCommandIndex = lastCommandStartIndex(trimmed);
    const lastLetter =
      lastCommandIndex >= 0 ? trimmed[lastCommandIndex] : undefined;
    // A cubic's chord rewrite keeps the original control points and only
    // moves the terminal anchor, which is a different curve rather than a
    // truncated one — always prefer the arc-length-accurate polyline below.
    const isCubicEnding = lastLetter === "C" || lastLetter === "c";
    let lastSegmentStart = 0;
    if (lastCommandIndex > 0) {
      const prefixPath = document.createElementNS(SVG_NS, "path");
      prefixPath.setAttribute("d", trimmed.slice(0, lastCommandIndex));
      lastSegmentStart = prefixPath.getTotalLength();
    }
    // `rewriteLastEndpoint` only moves the last command's own endpoint, so
    // it is only correct when the cut point still falls within that last
    // segment. A cut deep enough to land on an earlier segment must drop
    // whole trailing commands instead — handled by the polyline fallback.
    const cutWithinLastSegment =
      !isCubicEnding && cutAt >= lastSegmentStart - 0.01;
    if (cutWithinLastSegment) {
      const rewritten = rewriteLastEndpoint(d, end.x, end.y);
      if (rewritten !== undefined) {
        return rewritten;
      }
    }
    // Multi-segment cuts that land before the final command, cubic endings,
    // and any other unsupported ending rebuild the path as an arc-length
    // polyline up to the cut instead of rewriting just the last endpoint.
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
 * @internal exported for direct unit testing of multi-segment/cubic cuts.
 */
export function shortenPathForArrowhead(
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
  fallback = 0,
): number {
  const radius = style?.radius ?? style?.rx ?? style?.borderRadius;
  return typeof radius === "number" && Number.isFinite(radius) ? radius : fallback;
}

/** Recursively paint canonical child geometry in stable sibling order. */
function renderChildren(
  children: readonly SceneNodeLike[] | undefined,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  theme: Theme,
  markerPrefix: string,
  index: SceneNodeIndex,
  playback: PlaybackContext,
): ReactNode {
  if (!Array.isArray(children) || children.length === 0) {
    return null;
  }
  const ordered = orderArrowSiblingsLast(children);
  return ordered.map((child) =>
    renderNode(
      child,
      timeline,
      playbackTimeMs,
      theme,
      markerPrefix,
      index,
      ZERO_ORIGIN,
      playback,
    ),
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
): ReactNode {
  const capability = capabilityOf(node);
  const worldGeometry = index.worldGeometryById.get(node.id) ?? geometryOf(node);
  const geom = {
    x: worldGeometry.x - layoutOrigin.x,
    y: worldGeometry.y - layoutOrigin.y,
    width: worldGeometry.width,
    height: worldGeometry.height,
  };
  const kids = node.children;
  const appearance = appearanceForNode(node.id, timeline, playbackTimeMs);
  const fadingOut = isFadingOut(node.id, timeline, playbackTimeMs);
  const nodeHidden = appearance.state === "hidden" || appearance.opacity <= 0;
  const fanNode = isFanNode(node, capability);
  const endpointOpacity =
    fanNode ||
    isArrowLike(node, capability) ||
    isMotionSignalNode(node, capability)
      ? connectorEndpointOpacity(node, timeline, playbackTimeMs, index)
      : undefined;
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
  // Emphasize/pulse cues ease across a window keyed to `playbackTimeMs`; while
  // paused that time is frozen, but pausing mid-envelope must not leave the
  // glow/scale parked at whatever intensity the cue happened to be at (same
  // rationale as continuousPulseForNode's pause gate below). Reduced motion
  // already snaps `playbackTimeMs` to the final frame, so it needs no
  // separate gate here.
  const cueFrozenByPause = !playback.playing && !playback.reducedMotion;
  const emphasis = cueFrozenByPause
    ? undefined
    : emphasisForNode(node.id, timeline, playbackTimeMs, themeAccent);
  const pulseCue = cueFrozenByPause
    ? undefined
    : pulseCueForNode(node.id, timeline, playbackTimeMs);
  // Continuous pulse and emphasize/pulse cues animate a visible node; a
  // hidden node (pre-enter or fully faded) has nothing to glow/scale and
  // must not keep animating underneath its own invisibility.
  const continuousPulse =
    nodeHidden || fadingOut
      ? undefined
      : continuousPulseForNode(node, capability, appearance, playbackTimeMs, playback);
  if (import.meta.env.DEV && emphasis !== undefined && pulseCue !== undefined) {
    warnOverlappingEmphasisOnce(node.id);
  }
  const activeEmphasis = nodeHidden ? undefined : emphasis ?? pulseCue;
  const label = node.accessibility?.label ?? node.id;
  const description = node.accessibility?.description;
  const descriptionId =
    typeof description === "string" && description.length > 0
      ? `flow-node-${node.id}-desc`
      : undefined;
  const localChildren = false;
  const nested = renderChildren(
    kids,
    timeline,
    playbackTimeMs,
    theme,
    markerPrefix,
    index,
    playback,
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
  } else if (
    capability === "core.image" &&
    typeof node.props?.src === "string" &&
    node.props.src.length > 0
  ) {
    const fit = node.props.fit;
    const preserveAspectRatio =
      fit === "cover" ? "xMidYMid slice" : fit === "fill" ? "none" : "xMidYMid meet";
    body = (
      <image
        href={node.props.src}
        x={geom.x}
        y={geom.y}
        width={geom.width}
        height={geom.height}
        preserveAspectRatio={preserveAspectRatio}
        focusable={false}
        aria-hidden="true"
        style={styleToCss(node.style, theme)}
        data-flow-image="true"
      />
    );
  } else if (hasNativeSemanticChrome(node)) {
    const semanticParts = [...index.generatedPartsById.values()].filter(
      (part) => part.ownerId === node.id,
    );
    const rootBox = semanticParts.find(
      (part) => part.kind === "box" && part.role === "chrome",
    );
    const boxes = semanticParts.filter(
      (part) => part.kind === "box" && part.role !== "chrome",
    );
    const texts = semanticParts.filter((part) => part.kind === "text");
    const { fill: fillPaint, stroke: strokePaint } = chalkRectPaints(
      node.style,
      theme,
      themeBg,
      themeStroke,
    );
    body = (
      <>
        {rootBox === undefined ? null : (
          <rect
            id={rootBox.id}
            x={rootBox.geometry.x}
            y={rootBox.geometry.y}
            width={rootBox.geometry.width}
            height={rootBox.geometry.height}
            rx={rootBox.radius}
            fill={fillPaint}
            stroke={strokePaint}
            strokeWidth={strokeWidthFromStyle(node.style, 1.3) * strokeScale}
            focusable={false}
            aria-hidden="true"
            style={styleToCss(node.style, theme)}
            data-flow-semantic-chrome={capability}
          />
        )}
        {boxes.map((box) => (
          <rect
            key={`generated-box-${box.id}`}
            id={box.id}
            x={box.geometry.x}
            y={box.geometry.y}
            width={box.geometry.width}
            height={box.geometry.height}
            rx={box.radius}
            fill={fillPaint}
            stroke={strokePaint}
            strokeWidth={strokeWidthFromStyle(node.style, 1.3) * strokeScale}
            focusable={false}
            aria-hidden="true"
            data-flow-semantic-chrome={capability}
          />
        ))}
        {texts.map((part) => {
          const anchor = part.anchor ?? "start";
          const centered = anchor === "middle";
          const inkFallback =
            part.tone === "secondary"
              ? theme.text.secondary
              : theme.text.primary;
          return (
            <text
              key={`generated-text-${part.id}`}
              id={part.id}
              x={
                centered
                  ? part.geometry.x + part.geometry.width / 2
                  : anchor === "end"
                    ? part.geometry.x + part.geometry.width
                    : part.geometry.x
              }
              y={
                centered
                  ? part.geometry.y + part.geometry.height / 2
                  : part.geometry.y
              }
              dominantBaseline={centered ? "middle" : "hanging"}
              textAnchor={anchor}
              fill={
                part.inkRole !== undefined
                  ? resolveThemePaint(part.inkRole, theme, inkFallback)
                  : inkFallback
              }
              fontSize={scaledSceneFontSize(part.fontSize)}
              fontWeight={part.fontWeight}
              fontFamily={part.fontFamily}
              fontStyle={part.fontStyle}
              style={
                part.whiteSpace !== undefined
                  ? { whiteSpace: part.whiteSpace }
                  : undefined
              }
              focusable={false}
              aria-hidden="true"
              data-flow-semantic-text={capability}
            >
              {part.text ?? ""}
            </text>
          );
        })}
      </>
    );
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
            ? `drop-shadow(0 8px 14px rgba(15, 12, 8, 0.16)) drop-shadow(0 0 8px color-mix(in srgb, ${strokePaint} 45%, transparent))`
            : "drop-shadow(0 6px 10px rgba(15, 12, 8, 0.12))",
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
      appearance.state === "entering" && !fadingOut
        ? appearance.opacity
        : undefined;
    body = (
      <rect
        x={geom.x}
        y={geom.y}
        width={geom.width}
        height={geom.height}
        rx={cornerRadiusFromStyle(node.style, 0)}
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
                  "drop-shadow(0 5px 7px rgba(15, 12, 8, 0.13))",
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
    const fontSize = scaledSceneFontSize(node.style?.fontSize);
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
    const content = node.text ?? "";
    // Auto-wrap to the box width by default. Manual `\n` (or `whiteSpace: "pre"`)
    // is author-authoritative and never re-wrapped; `whiteSpace: "nowrap"` opts
    // a node out entirely (short kickers/stats that must stay one line even if
    // they overflow). `estimateTextWidth` only distinguishes normal vs bold, so
    // any non-"normal" weight is measured as bold.
    const whiteSpaceStyle = node.style?.whiteSpace;
    const hasManualBreaks =
      content.includes("\n") || whiteSpaceStyle === "pre";
    const fontWeight =
      node.style?.fontWeight === "bold" || node.style?.fontWeight === 700
        ? "bold"
        : "normal";
    const textLines = hasManualBreaks
      ? content.split("\n")
      : whiteSpaceStyle === "nowrap"
        ? undefined
        : geom.width > 0
          ? wrapTextToWidth(content, geom.width, fontSize, fontWeight)
          : undefined;
    const lineHeight =
      typeof node.style?.lineHeight === "number" &&
      Number.isFinite(node.style.lineHeight)
        ? node.style.lineHeight
        : fontSize * 1.3;
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
        style={{ ...styleToCss(node.style, theme), fontSize }}
      >
        {textLines === undefined
          ? content
          : textLines.map((line, index) => (
              <tspan
                key={`${node.id}-line-${index}`}
                x={textX}
                dy={index === 0 ? 0 : lineHeight}
              >
                {line}
              </tspan>
            ))}
      </text>
    );
  } else if (fanNode) {
    const geometry = fanGeometryFor(node, index, layoutOrigin);
    if (geometry === undefined) {
      body = null;
    } else {
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
    const strokeWidth = strokeWidthFromStyle(node.style) * strokeScale;
    const authoredDash = authoredStrokeDasharray(node.style);
    const dashed = isDashedStyle(node.style);
    const firstHalf =
      traceProgress === undefined ? undefined : clamp01(traceProgress * 2);
    const secondHalf =
      traceProgress === undefined
        ? undefined
        : traceProgress === 0.5
          ? 0.001
          : clamp01((traceProgress - 0.5) * 2);
    // A trace-driven fan reveals its own stroke in the same two phases as
    // the MotionSignal ball below: fan-out travels trunk-then-branch
    // (single -> many), fan-in travels branch-then-merge-trunk (many ->
    // single). An explicit `draw`/`reveal-stroke` cue (drawProgress
    // defined) has no ball phase and drives every segment uniformly.
    const trunkSideRole: "trunk" | "merge-trunk" =
      geometry.capability === "core.fan-out" ? "trunk" : "merge-trunk";
    const trunkFirst = geometry.capability === "core.fan-out";
    const segmentProgress = (
      role: "trunk" | "branch" | "merge-trunk",
    ): number | undefined => {
      if (drawProgress !== undefined) {
        return drawProgress;
      }
      if (traceProgress === undefined) {
        return strokeProgress;
      }
      const isTrunkSide = role === trunkSideRole;
      return (
        (trunkFirst === isTrunkSide ? firstHalf : secondHalf) ?? strokeProgress
      );
    };
    const resolvedSegments = geometry.segments.map((segment) => {
      const semanticTip = segment.showMarker ? tip : null;
      const progress = segmentProgress(segment.role);
      const segmentDrawing = progress !== undefined && progress < 1;
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
          (!segmentDrawing || playback.reducedMotion),
        progress,
        drawing: segmentDrawing,
      };
    });
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
            dashed={!segment.drawing && dashed}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            pathLength={segment.drawing ? 1 : undefined}
            strokeDasharray={
              segment.drawing
                ? 1
                : authoredDash !== undefined
                  ? authoredDash
                  : undefined
            }
            strokeDashoffset={
              segment.drawing ? 1 - (segment.progress ?? 0) : undefined
            }
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
    }
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
      //
      // `active` is deliberately not gated on `playback.playing`: SMIL's
      // indefinite `<animate>`/`<animateMotion>` loop is un-keyed to
      // `playbackTimeMs` (it is a decorative, self-timed loop, not a
      // timeline-driven progress). Toggling `active` off on pause would
      // unmount the circle, and remounting on resume restarts `begin={delay}`
      // from zero — desyncing the dot from where it was before pausing.
      // Leaving it mounted keeps the loop's own wall-clock continuity across
      // pause/resume; only a genuinely inactive state (reduced motion,
      // hidden node, or an endpoint not yet on stage) should unmount it.
      // Pausing must still freeze the dot in place, which `paused` below
      // does via the SVG document's own pauseAnimations without unmounting.
      const smilActive =
        !playback.reducedMotion &&
        appearance.state !== "hidden" &&
        // Hold the traveler until every node-backed endpoint is on stage, so
        // the dot never zips across a corridor before its boxes appear.
        (endpointOpacity === undefined || endpointOpacity > 0);
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
                  paused={!playback.playing}
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
    const resolvedConnector = index.connectorsById.get(node.id);
    // Glyph icons / sparklines emit `core.path` with authored `path` and are
    // excluded from connector resolution — fall back to arrowPathData so they
    // still paint, then translate local glyph coords by the node origin.
    const dRaw =
      resolvedConnector?.d ?? arrowPathData(node, index, layoutOrigin);
    if (dRaw !== undefined) {
      const stroke = paintFromStyle(node.style, "stroke", theme, themeAccent);
      // Routed connectors use resolver policy; authored core.path glyphs only
      // show a tip when markerEnd is explicitly enabled (icons stamp "none").
      const tip =
        resolvedConnector !== undefined
          ? resolvedConnector.showArrowhead
            ? resolveMarkerTip(node.style?.markerEnd, DEFAULT_MARKER_TIP)
            : null
          : hasExplicitMarkerEnd(node.style)
            ? resolveMarkerTip(node.style?.markerEnd, DEFAULT_MARKER_TIP)
            : null;
      const wantsMarker = tip !== null;
      const drawing = drawProgress !== undefined && drawProgress < 1;
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
      const translateLocalGlyph =
        resolvedConnector === undefined &&
        capability === "core.path" &&
        (geom.x !== 0 || geom.y !== 0);
      const arrow = (
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
            data-flow-resolved-path={dRaw}
            data-flow-tip={tip?.key}
            data-flow-elbow={isElbowRoute(node) ? "true" : undefined}
            data-flow-curve={isCurveRoute(node) ? "true" : undefined}
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
      body = translateLocalGlyph ? (
        <g transform={`translate(${geom.x} ${geom.y})`}>{arrow}</g>
      ) : (
        arrow
      );
    }
  }

  const baseOpacity =
    appearance.state === "unchanged" ? 1 : appearance.opacity;
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
  const endpointBoundOpacity =
    endpointOpacity !== undefined && endpointOpacity < 1
      ? (groupOpacity ?? 1) * endpointOpacity
      : groupOpacity;
  const groupStyle: CSSProperties | undefined =
    endpointBoundOpacity === undefined &&
    (emphasis === undefined || emphasis.filter === "none")
      ? undefined
      : {
          ...(endpointBoundOpacity !== undefined
            ? { opacity: endpointBoundOpacity }
            : {}),
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
  const clipsChildren =
    node.style?.overflow === "hidden" || node.style?.clip === true;
  const clipPathId = `${markerPrefix}-clip-${node.id.replaceAll(
    /[^a-zA-Z0-9_-]/g,
    "-",
  )}`;

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
      {clipsChildren ? (
        <>
          <defs>
            <clipPath id={clipPathId}>
              <rect
                x={geom.x}
                y={geom.y}
                width={geom.width}
                height={geom.height}
              />
            </clipPath>
          </defs>
          <g clipPath={`url(#${clipPathId})`}>{nested}</g>
        </>
      ) : (
        nested
      )}
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
  const { roots, index, timeline, durationMs, sceneTips } = useMemo(() => {
    const nextRoots = omitMotionCompanionDots(scene.roots ?? []);
    const resolved = resolveScene({ ...scene, roots: nextRoots });
    const nextIndex: SceneNodeIndex = {
      nodesById: resolved.nodesById,
      worldGeometryById: resolved.worldGeometryById,
      ancestorIdsById: resolved.ancestorIdsById,
      generatedPartsById: resolved.generatedPartsById,
      connectorsById: resolved.connectorsById,
      fanGeometryById: resolved.fanGeometryById,
    };
    const authoredTimeline = Array.isArray(scene.timeline) ? scene.timeline : [];
    const nextTimeline = expandTimelineCues(
      omitCompanionTimelineCues(authoredTimeline, nextIndex.nodesById),
      nextIndex.nodesById,
    );

    return {
      roots: nextRoots,
      index: nextIndex,
      timeline: nextTimeline,
      durationMs: timelineDurationMs(nextTimeline, scene.camera),
      sceneTips: collectSceneTips(nextRoots),
    };
  }, [scene.id, scene.roots, scene.timeline, scene.camera]);
  const [playbackTimeMs, setPlaybackTimeMs] = useState(0);
  const playbackTimeMsRef = useRef(0);
  const rate = playbackRate > 0 ? playbackRate : 1;
  const scenePlaybackIdentity = scene.id ?? scene.roots;

  const commitTime = (nextMs: number) => {
    const clamped = Math.min(durationMs, Math.max(0, nextMs));
    playbackTimeMsRef.current = clamped;
    setPlaybackTimeMs(clamped);
  };

  useEffect(() => {
    commitTime(0);
  }, [restartKey, scenePlaybackIdentity, durationMs]);

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
    let animationFrameId = 0;

    const syncFromWallClock = (wallTimeMs: number) => {
      const elapsed = Math.min(
        durationMs,
        playOriginMs + Math.max(0, wallTimeMs - wallOriginMs) * rate,
      );
      commitTime(elapsed);
      if (elapsed < durationMs) {
        animationFrameId = window.requestAnimationFrame(syncFromWallClock);
      }
    };

    animationFrameId = window.requestAnimationFrame(syncFromWallClock);
    return () => {
      if (animationFrameId !== 0) {
        window.cancelAnimationFrame(animationFrameId);
      }
    };
  }, [
    playing,
    reducedMotion,
    durationMs,
    restartKey,
    scenePlaybackIdentity,
    rate,
  ]);

  const effectiveTimeMs = reducedMotion ? durationMs : playbackTimeMs;
  const ariaLabel =
    scene.accessibility?.label ?? scene.title ?? "Flow scene diagram";
  const summaryDescId =
    typeof scene.summary === "string" && scene.summary.length > 0
      ? `scene-summary-${reactId}`
      : undefined;
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
