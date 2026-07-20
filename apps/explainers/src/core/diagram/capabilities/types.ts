/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared contracts for native semantic Scene IR capability layout.

import type {
  SceneGeometryLike,
  SceneNodeLike,
} from "../SceneRenderer.js";

/** One deterministic layout result shared by indexing and rendering. */
export type CapabilityLayout = Readonly<{
  bounds: SceneGeometryLike;
  childGeometries: readonly SceneGeometryLike[];
}>;

/** Pure layout implementation for one semantic Scene capability. */
export type NativeSceneCapability = Readonly<{
  capabilityId: string;
  resolveLayout(
    node: SceneNodeLike,
    children: readonly SceneNodeLike[],
  ): CapabilityLayout;
}>;

