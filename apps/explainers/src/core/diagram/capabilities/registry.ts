/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit native Scene capability registry.

import type { SceneNodeLike } from "../scene-types.js";
import {
  LAYOUT_CAPABILITIES,
  resolveIdentityLayout,
} from "./layout.js";
import type { CapabilityLayout, NativeSceneCapability } from "./types.js";

/** Build an immutable registry and reject ambiguous capability ownership. */
export function createCapabilityRegistry(
  definitions: readonly NativeSceneCapability[],
): ReadonlyMap<string, NativeSceneCapability> {
  const registry = new Map<string, NativeSceneCapability>();
  for (const definition of definitions) {
    if (registry.has(definition.capabilityId)) {
      throw new Error(
        `duplicate native Scene capability "${definition.capabilityId}"`,
      );
    }
    registry.set(definition.capabilityId, definition);
  }
  return registry;
}

const NATIVE_SCENE_CAPABILITIES = createCapabilityRegistry(
  LAYOUT_CAPABILITIES,
);

function capabilityOf(node: SceneNodeLike): string {
  if (typeof node.capabilityId === "string" && node.capabilityId.length > 0) {
    return node.capabilityId;
  }
  if (typeof node.capability === "string" && node.capability.length > 0) {
    return node.capability;
  }
  return node.kind ?? "core.group";
}

/**
 * Resolve one layout through the native registry.
 *
 * Identity fallback is the migration adapter for already-first-class
 * primitives; the compiler cutover removes reliance on this fallback.
 */
export function resolveCapabilityLayout(
  node: SceneNodeLike,
  children: readonly SceneNodeLike[],
): CapabilityLayout {
  const definition = NATIVE_SCENE_CAPABILITIES.get(capabilityOf(node));
  return (definition?.resolveLayout ?? resolveIdentityLayout)(node, children);
}

