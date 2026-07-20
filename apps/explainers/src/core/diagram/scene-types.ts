/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Renderer-independent structural contracts for authored Scene IR.

/** Minimal geometry for a scene node. */
export type SceneGeometryLike = Readonly<{
  x: number;
  y: number;
  width: number;
  height: number;
}>;

/** Point or connector endpoint, optionally anchored to another scene node. */
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

/** Structural source location retained from authored Flow input. */
export type SceneSourceRangeLike = Readonly<{
  source: string;
  start: Readonly<{ offset: number; line: number; column: number }>;
  end: Readonly<{ offset: number; line: number; column: number }>;
}>;

/** Style values may be scalars or nested scalar records. */
export type SceneStyleValue =
  | string
  | number
  | boolean
  | Readonly<Record<string, string | number | boolean>>;

/** Placement relative to an already-resolved node in document order. */
export type SceneRelativePositionLike = Readonly<{
  nodeId: string;
  anchor?: string;
  dx?: number;
  dy?: number;
}>;

/** Structural scene node independent of React and renderer state. */
export type SceneNodeLike = Readonly<{
  id: string;
  kind?: string;
  capabilityId?: string;
  capability?: string;
  geometry?: SceneGeometryLike;
  layout?: SceneGeometryLike;
  relativePosition?: SceneRelativePositionLike;
  style?: Readonly<Record<string, SceneStyleValue>>;
  props?: Readonly<Record<string, unknown>>;
  text?: string;
  accessibility?: SceneNodeAccessibilityLike;
  sourceMap?: SceneSourceRangeLike;
  children?: readonly SceneNodeLike[];
  d?: string;
  path?: string;
  points?: readonly ScenePointLike[];
  from?: ScenePointLike | readonly ScenePointLike[];
  to?: ScenePointLike | readonly ScenePointLike[];
  via?: ScenePointLike;
  axis?: string;
  junction?: ScenePointLike;
  edgeRef?: string;
}>;

/** Minimal timeline cue authored for a scene. */
export type SceneTimelineCueLike = Readonly<{
  id: string;
  at: number;
  duration: number;
  action: string;
  target: string;
  targets?: readonly string[];
  step?: number;
  easing?: string;
}>;

/** Optional logical SVG bounds. */
export type SceneViewportLike = Readonly<{
  width: number;
  height: number;
}>;

/** Authored camera keyframe. */
export type SceneCameraKeyframeLike = Readonly<{
  id?: string;
  at: number;
  x: number;
  y: number;
  zoom: number;
}>;

/** Minimal Scene IR consumed by canonical resolution and rendering. */
export type SceneIrLike = Readonly<{
  id?: string;
  title?: string;
  summary?: string;
  viewport?: SceneViewportLike;
  roots: readonly SceneNodeLike[];
  camera?: readonly SceneCameraKeyframeLike[];
  timeline: readonly SceneTimelineCueLike[];
  accessibility?: Readonly<{ label?: string }>;
}>;
