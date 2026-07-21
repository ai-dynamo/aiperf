/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JSON-safe serialization of canonical resolved scenes for verifier consumers.

import type { SceneNodeLike } from "../scene-types.js";
import { capabilityOf as nodeCapabilityOf } from "../node-classification.js";
import type {
  ResolvedGeneratedPart,
  ResolvedScene,
  ResolvedSceneSnapshot,
} from "./types.js";

const DEFAULT_VIEWPORT = Object.freeze({ width: 1920, height: 1080 });

function capabilityOf(node: SceneNodeLike): string {
  return nodeCapabilityOf(node);
}

/**
 * Convert a resolved scene into plain arrays and records safe for JSON output.
 *
 * Map-backed values retain semantic document order; generated parts are grouped
 * by owner document order and all ties use stable IDs.
 */
export function resolvedSceneSnapshot(
  resolved: ResolvedScene,
): ResolvedSceneSnapshot {
  const documentOrder = new Map(
    [...resolved.nodesById.keys()].map((id, index) => [id, index]),
  );
  const orderOf = (id: string): number =>
    documentOrder.get(id) ?? Number.MAX_SAFE_INTEGER;
  const compareIds = (left: string, right: string): number =>
    orderOf(left) - orderOf(right) || left.localeCompare(right);
  const compareGeneratedParts = (
    left: ResolvedGeneratedPart,
    right: ResolvedGeneratedPart,
  ): number =>
    orderOf(left.ownerId) - orderOf(right.ownerId) ||
    left.id.localeCompare(right.id);

  const nodes = [...resolved.nodesById.entries()]
    .sort(([left], [right]) => compareIds(left, right))
    .flatMap(([id, node]) => {
      const bounds = resolved.worldGeometryById.get(id);
      if (bounds === undefined) return [];
      return [
        {
          id,
          capability: capabilityOf(node),
          bounds,
          ancestorIds: resolved.ancestorIdsById.get(id) ?? [],
        },
      ];
    });

  return {
    ...(resolved.scene.id === undefined ? {} : { sceneId: resolved.scene.id }),
    viewport: resolved.scene.viewport ?? DEFAULT_VIEWPORT,
    nodes,
    generatedParts: [...resolved.generatedPartsById.values()].sort(
      compareGeneratedParts,
    ),
    connectors: [...resolved.connectorsById.values()].sort((left, right) =>
      compareIds(left.id, right.id),
    ),
    fans: [...resolved.fanGeometryById.values()].sort((left, right) =>
      compareIds(left.id, right.id),
    ),
    diagnostics: resolved.diagnostics,
  };
}
