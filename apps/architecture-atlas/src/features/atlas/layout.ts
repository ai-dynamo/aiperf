// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import ELK from "elkjs/lib/elk.bundled.js";

import type {
  ArchitectureComponent,
  ArchitectureEdge,
  LifecycleStage,
  Ownership,
} from "../../domain/architecture";

export type LayoutPerspective = "ownership" | "lifecycle";

export interface LayoutPosition {
  id: string;
  x: number;
  y: number;
}

const elk = new ELK();
const layoutCache = new Map<string, Promise<LayoutPosition[]>>();
const ownerOrder: Ownership[] = ["python", "rust", "external", "legacy"];

function componentOrder(
  component: ArchitectureComponent,
  perspective: LayoutPerspective,
  stages: readonly LifecycleStage[],
): number {
  if (perspective === "ownership") {
    return ownerOrder.indexOf(component.owner);
  }
  const stage = stages.find(({ componentIds }) =>
    componentIds.includes(component.id),
  );
  return stage?.order ?? stages.length;
}

export function layoutAtlas(
  components: readonly ArchitectureComponent[],
  edges: readonly ArchitectureEdge[],
  perspective: LayoutPerspective,
  stages: readonly LifecycleStage[],
): Promise<LayoutPosition[]> {
  const ordered = [...components].sort(
    (left, right) =>
      componentOrder(left, perspective, stages) -
        componentOrder(right, perspective, stages) ||
      left.id.localeCompare(right.id),
  );
  const key = [
    perspective,
    ordered.map(({ id }) => id).join(","),
    [...edges]
      .map(({ id }) => id)
      .sort()
      .join(","),
  ].join("|");
  const cached = layoutCache.get(key);
  if (cached) {
    return cached;
  }
  const pending = elk
    .layout({
      id: "atlas",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": perspective === "ownership" ? "RIGHT" : "DOWN",
        "elk.edgeRouting": "ORTHOGONAL",
        "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
        "elk.spacing.nodeNode": "42",
        "elk.layered.spacing.nodeNodeBetweenLayers": "88",
      },
      children: ordered.map((component) => ({
        id: component.id,
        width: 248,
        height: 112,
      })),
      edges: [...edges]
        .sort((left, right) => left.id.localeCompare(right.id))
        .map((edge) => ({
          id: edge.id,
          sources: [edge.from],
          targets: [edge.to],
        })),
    })
    .then(
      ({ children = [] }) =>
        children.map((node) => ({
          id: node.id,
          x: node.x ?? 0,
          y: node.y ?? 0,
        })),
    );
  layoutCache.set(key, pending);
  return pending;
}
