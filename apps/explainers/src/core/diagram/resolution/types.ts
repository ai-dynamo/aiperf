/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Canonical resolved-scene contracts shared by geometry consumers.

import type {
  SceneGeometryLike,
  SceneIrLike,
  SceneNodeLike,
  SceneSourceRangeLike,
  SceneViewportLike,
} from "../scene-types.js";

/** One source-mapped finding produced during canonical scene resolution. */
export type SceneResolutionDiagnostic = Readonly<{
  code: string;
  severity: "error" | "warning" | "info";
  message: string;
  range: SceneSourceRangeLike;
  nodeIds: readonly string[];
  repair?: string;
}>;

/** One visual part generated and owned by a semantic scene node. */
export type ResolvedGeneratedPart = Readonly<{
  id: string;
  ownerId: string;
  kind: "box" | "text";
  role:
    | "chrome"
    | "title"
    | "detail"
    | "subtitle"
    | "caption"
    | "label"
    | "step";
  geometry: SceneGeometryLike;
  radius?: number;
  text?: string;
  fontSize?: number;
  fontWeight?: string;
  fontFamily?: string;
  fontStyle?: string;
  whiteSpace?: string;
  anchor?: "start" | "middle" | "end";
  tone?: "primary" | "secondary";
  inkRole?: string;
}>;

/** Finite world-space point in a resolved scene. */
export type ResolvedPoint = Readonly<{ x: number; y: number }>;

/** One canonical connector path and its resolution metadata. */
export type ResolvedConnector = Readonly<{
  id: string;
  source: ResolvedPoint;
  target: ResolvedPoint;
  sourceId?: string;
  targetId?: string;
  d: string;
  directed: boolean;
  showArrowhead: boolean;
  usedFallback: boolean;
  penetratedObstacleIds: readonly string[];
}>;

/** One resolved fan painted stroke segment (trunk, branch, or merge-trunk). */
export type FanSegment = Readonly<{
  id: string;
  d: string;
  directed: true;
  showMarker: boolean;
  role: "trunk" | "branch" | "merge-trunk";
}>;

/** One resolved fan complete source-to-destination trajectory. */
export type FanTrajectory = Readonly<{
  id: string;
  d: string;
  role: "trunk" | "branch" | "merge-trunk";
}>;

/** One canonical fan-out/fan-in topology resolution and its painted geometry. */
export type ResolvedFanGeometry = Readonly<{
  id: string;
  capability: "core.fan-out" | "core.fan-in";
  junction: ResolvedPoint;
  segments: readonly FanSegment[];
  trajectories: readonly FanTrajectory[];
}>;

/** Pure canonical scene output consumed by rendering and later verification. */
export type ResolvedScene = Readonly<{
  scene: SceneIrLike;
  nodesById: ReadonlyMap<string, SceneNodeLike>;
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>;
  ancestorIdsById: ReadonlyMap<string, readonly string[]>;
  generatedPartsById: ReadonlyMap<string, ResolvedGeneratedPart>;
  connectorsById: ReadonlyMap<string, ResolvedConnector>;
  fanGeometryById: ReadonlyMap<string, ResolvedFanGeometry>;
  diagnostics: readonly SceneResolutionDiagnostic[];
}>;

/** JSON-safe canonical scene representation consumed by external verifiers. */
export type ResolvedSceneSnapshot = Readonly<{
  sceneId?: string;
  viewport: SceneViewportLike;
  nodes: readonly Readonly<{
    id: string;
    capability: string;
    bounds: SceneGeometryLike;
    ancestorIds: readonly string[];
  }>[];
  generatedParts: readonly ResolvedGeneratedPart[];
  connectors: readonly ResolvedConnector[];
  fans: readonly ResolvedFanGeometry[];
  diagnostics: readonly SceneResolutionDiagnostic[];
}>;
