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

/** Minimal timeline cue (enter/reveal/draw). */
export type SceneTimelineCueLike = Readonly<{
  id: string;
  at: number;
  duration: number;
  action: string;
  target: string;
}>;

/** Minimal Scene IR shape consumed by ExplainerShell diagrams. */
export type SceneIrLike = Readonly<{
  id?: string;
  title?: string;
  summary?: string;
  roots: readonly SceneNodeLike[];
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

const ARROW_CAPABILITIES = new Set([
  "core.line",
  "core.path",
  "core.arrow",
  "core.connector",
]);

const ARROW_KINDS = new Set(["line", "path", "arrow", "connector"]);

type TimelineState = "hidden" | "entering" | "revealed" | "unchanged";

type TimelineAppearance = Readonly<{
  state: TimelineState;
  opacity: number;
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

/** Flatten scene roots (and nested children) into an id → node index. */
function indexSceneNodes(
  roots: readonly SceneNodeLike[],
): ReadonlyMap<string, SceneNodeLike> {
  const index = new Map<string, SceneNodeLike>();
  const visit = (node: SceneNodeLike): void => {
    index.set(node.id, node);
    if (Array.isArray(node.children)) {
      for (const child of node.children) {
        visit(child);
      }
    }
  };
  for (const root of roots) {
    visit(root);
  }
  return index;
}

/** Center point of a node's geometry (matches Flow runtime connectors). */
function nodeCenter(node: SceneNodeLike): Readonly<{ x: number; y: number }> {
  const geom = geometryOf(node);
  return {
    x: geom.x + geom.width / 2,
    y: geom.y + geom.height / 2,
  };
}

/**
 * Resolve an endpoint: prefer explicit coordinates, else nodeId center,
 * else origin.
 */
function resolveEndpoint(
  endpoint: ScenePointLike | undefined,
  nodesById: ReadonlyMap<string, SceneNodeLike>,
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
    const target = nodesById.get(endpoint.nodeId);
    if (target !== undefined) {
      return nodeCenter(target);
    }
  }
  return { x: 0, y: 0 };
}

function polylinePathData(
  points: readonly ScenePointLike[],
  nodesById: ReadonlyMap<string, SceneNodeLike>,
): string | undefined {
  if (points.length === 0) {
    return undefined;
  }
  const resolved = points.map((point) => resolveEndpoint(point, nodesById));
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

function clamp01(value: number): number {
  if (value <= 0) {
    return 0;
  }
  if (value >= 1) {
    return 1;
  }
  return value;
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
    if (key === "fill" || key === "stroke") {
      // Applied as SVG presentation attributes via paintFromStyle.
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

/** Pure group container: nests children in a `<g>` with no leaf body of its own. */
function isGroupLike(node: SceneNodeLike, capability: string): boolean {
  if (capability === "core.group") {
    return true;
  }
  return node.kind === "group";
}

/**
 * Recursively render nested `children` into sibling `<g>` wrappers.
 * Absolute child geometry is preserved (no parent translate).
 */
function renderChildren(
  children: readonly SceneNodeLike[] | undefined,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  theme: Theme,
  arrowMarkerId: string,
  nodesById: ReadonlyMap<string, SceneNodeLike>,
): ReactNode {
  if (!Array.isArray(children) || children.length === 0) {
    return null;
  }
  return children.map((child) =>
    renderNode(
      child,
      timeline,
      playbackTimeMs,
      theme,
      arrowMarkerId,
      nodesById,
    ),
  );
}

/**
 * Resolve SVG path data for line / path / arrow / connector nodes.
 * Precedence: authored `d` → `path` → `points` polyline → `from`/`to`.
 */
function arrowPathData(
  node: SceneNodeLike,
  nodesById: ReadonlyMap<string, SceneNodeLike>,
): string | undefined {
  if (typeof node.d === "string" && node.d.length > 0) {
    return node.d;
  }
  if (typeof node.path === "string" && node.path.length > 0) {
    return node.path;
  }
  if (Array.isArray(node.points) && node.points.length > 0) {
    return polylinePathData(node.points, nodesById);
  }
  if (node.from !== undefined || node.to !== undefined) {
    const start = resolveEndpoint(node.from, nodesById);
    const end = resolveEndpoint(node.to, nodesById);
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

function renderNode(
  node: SceneNodeLike,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  theme: Theme,
  arrowMarkerId: string,
  nodesById: ReadonlyMap<string, SceneNodeLike>,
): ReactNode {
  const capability = capabilityOf(node);
  const geom = geometryOf(node);
  const appearance = appearanceForNode(node.id, timeline, playbackTimeMs);
  const drawProgress = drawProgressForNode(node.id, timeline, playbackTimeMs);
  // Prefer authored accessibility.label; fall back to id only when absent.
  const label = node.accessibility?.label ?? node.id;
  const description = node.accessibility?.description;
  const descriptionId =
    typeof description === "string" && description.length > 0
      ? `flow-node-${node.id}-desc`
      : undefined;
  const nested = renderChildren(
    node.children,
    timeline,
    playbackTimeMs,
    theme,
    arrowMarkerId,
    nodesById,
  );

  const themeBg = theme.bg.elevated;
  const themeStroke = theme.stroke.secondary;
  const themeText = theme.text.primary;
  const themeAccent = theme.accent.primary;
  const groupLike = isGroupLike(node, capability);

  let body: ReactNode = null;
  // `core.group` has no leaf body; other capabilities still draw even when
  // `kind` was normalized to "group" because the node has nested children.
  if (capability === "core.rect" || node.kind === "rect") {
    body = (
      <rect
        x={geom.x}
        y={geom.y}
        width={geom.width}
        height={geom.height}
        rx={10}
        fill={paintFromStyle(node.style, "fill", theme, themeBg)}
        stroke={paintFromStyle(node.style, "stroke", theme, themeStroke)}
        strokeWidth={strokeWidthFromStyle(node.style, 1.3)}
        focusable={false}
        aria-hidden="true"
        style={styleToCss(node.style, theme)}
      />
    );
  } else if (capability === "core.text" || node.kind === "text") {
    const fontSize =
      typeof node.style?.fontSize === "number" ? node.style.fontSize : 14;
    body = (
      <text
        x={geom.x}
        y={geom.y}
        dominantBaseline="hanging"
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
    const d = arrowPathData(node, nodesById);
    if (d !== undefined) {
      body = (
        <FlowArrow
          d={d}
          markerId={arrowMarkerId}
          color={paintFromStyle(node.style, "stroke", theme, themeAccent)}
          strokeWidth={strokeWidthFromStyle(node.style)}
          pathLength={drawProgress === undefined ? undefined : 1}
          strokeDasharray={drawProgress === undefined ? undefined : 1}
          strokeDashoffset={
            drawProgress === undefined ? undefined : 1 - drawProgress
          }
          focusable={false}
          aria-hidden="true"
          style={styleToCss(node.style, theme)}
        />
      );
    }
  }

  const groupStyle: CSSProperties | undefined =
    appearance.state === "unchanged"
      ? undefined
      : { opacity: appearance.opacity };

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
      data-timeline-state={appearance.state}
      data-draw-progress={
        drawProgress === undefined ? undefined : String(drawProgress)
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
  // Preserve scene.accessibility.label when authored; else title / default.
  const ariaLabel =
    scene.accessibility?.label ?? scene.title ?? "Flow scene diagram";
  const summaryDescId =
    typeof scene.summary === "string" && scene.summary.length > 0
      ? `scene-summary-${reactId}`
      : undefined;
  const arrowColor = theme.accent.primary;
  const roots = scene.roots ?? [];
  const nodesById = indexSceneNodes(roots);

  return (
    <svg
      className="scene-renderer"
      viewBox={`0 0 ${VIEWPORT_WIDTH} ${VIEWPORT_HEIGHT}`}
      role="img"
      aria-label={ariaLabel}
      aria-describedby={summaryDescId}
      focusable={false}
      style={{ display: "block", width: "100%" }}
    >
      <defs>
        <marker
          id={arrowMarkerId}
          markerWidth={8}
          markerHeight={8}
          refX={6}
          refY={3}
          orient="auto"
        >
          <path d="M0,0 L6,3 L0,6 Z" fill={arrowColor} focusable={false} />
        </marker>
      </defs>
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
          nodesById,
        ),
      )}
    </svg>
  );
}
