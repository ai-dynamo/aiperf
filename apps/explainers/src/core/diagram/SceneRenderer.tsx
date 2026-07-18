/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  useEffect,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { useHostTheme } from "../ui";

/** Minimal geometry for a scene node. */
export type SceneGeometryLike = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
}>;

/** Minimal accessibility metadata for a scene node. */
export type SceneNodeAccessibilityLike = Readonly<{
  label?: string;
  description?: string;
}>;

/** Minimal render node supporting core.rect / rect-like shapes. */
export type SceneNodeLike = Readonly<{
  id: string;
  kind?: string;
  capabilityId?: string;
  capability?: string;
  geometry?: SceneGeometryLike;
  layout?: SceneGeometryLike;
  style?: Readonly<Record<string, string | number>>;
  accessibility?: SceneNodeAccessibilityLike;
  children?: readonly SceneNodeLike[];
}>;

/** Minimal timeline cue (enter/reveal/trace). */
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

type TimelineState = "hidden" | "revealed" | "unchanged";

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

function timelineStateForNode(
  nodeId: string,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
): TimelineState {
  const cue = timeline.filter(
    (candidate) =>
      candidate.target === nodeId && isEnterLikeAction(candidate.action),
  ).at(-1);
  if (cue === undefined) {
    return "unchanged";
  }
  const endMs = finiteNumber(cue.at) + finiteNumber(cue.duration);
  return playbackTimeMs >= endMs ? "revealed" : "hidden";
}

function styleToCss(style: SceneNodeLike["style"]): CSSProperties {
  if (style === undefined) {
    return {};
  }
  const css: CSSProperties = {};
  for (const [key, value] of Object.entries(style)) {
    if (typeof value === "string" || typeof value === "number") {
      (css as Record<string, string | number>)[key] = value;
    }
  }
  return css;
}

function renderNode(
  node: SceneNodeLike,
  timeline: readonly SceneTimelineCueLike[],
  playbackTimeMs: number,
  themeFill: string,
  themeStroke: string,
): ReactNode {
  const capability = capabilityOf(node);
  const geom = geometryOf(node);
  const timelineState = timelineStateForNode(node.id, timeline, playbackTimeMs);
  const label = node.accessibility?.label ?? node.id;
  const children = Array.isArray(node.children)
    ? node.children.map((child) =>
        renderNode(child, timeline, playbackTimeMs, themeFill, themeStroke),
      )
    : null;

  let body: ReactNode = null;
  if (capability === "core.rect" || node.kind === "rect") {
    body = (
      <rect
        x={geom.x}
        y={geom.y}
        width={geom.width}
        height={geom.height}
        rx={10}
        fill={themeFill}
        stroke={themeStroke}
        strokeWidth={1.3}
        style={styleToCss(node.style)}
      />
    );
  }

  return (
    <g
      key={node.id}
      data-flow-node-id={node.id}
      data-timeline-state={timelineState}
      aria-label={label}
      role="img"
      style={timelineState === "hidden" ? { opacity: 0 } : undefined}
    >
      {body}
      {children}
    </g>
  );
}

/**
 * Renders Flow Scene IR into an ExplainerShell diagram slot.
 * Plays authored timeline cues when `playing`, restarts on `restartKey`,
 * and collapses to the final frame under reduced motion.
 */
export function SceneRenderer({
  scene,
  playing,
  restartKey,
  reducedMotion = false,
}: SceneRendererProps): ReactNode {
  const theme = useHostTheme();
  const timeline = Array.isArray(scene.timeline) ? scene.timeline : [];
  const durationMs = timelineDurationMs(timeline);
  const [playbackTimeMs, setPlaybackTimeMs] = useState(0);

  useEffect(() => {
    setPlaybackTimeMs(0);
  }, [restartKey, scene]);

  useEffect(() => {
    if (reducedMotion) {
      setPlaybackTimeMs(durationMs);
      return;
    }
    if (!playing) {
      return;
    }

    let frameId = 0;
    const startedAt = performance.now();

    const tick = (now: number) => {
      const elapsed = Math.min(durationMs, Math.max(0, now - startedAt));
      setPlaybackTimeMs(elapsed);
      if (elapsed < durationMs) {
        frameId = requestAnimationFrame(tick);
      }
    };

    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, [playing, reducedMotion, durationMs, restartKey, scene]);

  const effectiveTimeMs = reducedMotion ? durationMs : playbackTimeMs;
  const ariaLabel =
    scene.accessibility?.label ?? scene.title ?? "Flow scene diagram";

  return (
    <svg
      className="scene-renderer"
      viewBox={`0 0 ${VIEWPORT_WIDTH} ${VIEWPORT_HEIGHT}`}
      role="img"
      aria-label={ariaLabel}
      style={{ display: "block", width: "100%" }}
    >
      {scene.summary ? <desc>{scene.summary}</desc> : null}
      {(scene.roots ?? []).map((node) =>
        renderNode(
          node,
          timeline,
          effectiveTimeMs,
          theme.bg.elevated,
          theme.stroke.secondary,
        ),
      )}
    </svg>
  );
}
