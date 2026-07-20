/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared contracts for native semantic Scene IR capability layout.

import type {
  SceneGeometryLike,
  SceneNodeLike,
} from "../scene-types.js";

/** Stable layout finding promoted to a source-mapped scene diagnostic upstream. */
export type CapabilityLayoutDiagnostic = Readonly<{
  code: "SCENE_MANAGED_CONTENT_OVERFLOW" | "SCENE_MANAGED_CHILD_OVERLAP";
  severity: "error";
  message: string;
  nodeIds: readonly string[];
}>;

/** One deterministic layout result shared by indexing and rendering. */
export type CapabilityLayout = Readonly<{
  bounds: SceneGeometryLike;
  contentBounds: SceneGeometryLike;
  childGeometries: readonly SceneGeometryLike[];
  generatedPorts?: Readonly<Record<string, SceneGeometryLike>>;
  diagnostics?: readonly CapabilityLayoutDiagnostic[];
}>;

/** Pure layout implementation for one semantic Scene capability. */
export type NativeSceneCapability = Readonly<{
  capabilityId: string;
  resolveLayout(
    node: SceneNodeLike,
    children: readonly SceneNodeLike[],
  ): CapabilityLayout;
}>;

