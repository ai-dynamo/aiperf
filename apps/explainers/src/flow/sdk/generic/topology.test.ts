/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { createSdkRegistry } from "../registry.js";
import type { SceneFragment, SdkExpansionContext } from "../types.js";
import type { RenderNodeIr } from "../../schema/ir.js";

const SOURCE_MAP = {
  source: "topology.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function context(instanceId: string): SdkExpansionContext {
  return {
    instanceId,
    sourceMap: SOURCE_MAP,
    themeTokens: new Map(),
  };
}

function rectNode(
  id: string,
  geometry: { x: number; y: number; width: number; height: number },
): RenderNodeIr {
  return {
    kind: "rect",
    id,
    capabilityId: "core.rect",
    geometry,
    style: {},
    accessibility: { label: id },
    fallback: id,
    sourceMap: SOURCE_MAP,
  };
}

function stageFragment(
  primaryId: string,
  extras: readonly RenderNodeIr[] = [],
  primaryGeometry = { x: 0, y: 0, width: 80, height: 40 },
): SceneFragment {
  const primary = rectNode(primaryId, primaryGeometry);
  return {
    roots: [primary, ...extras],
    ports: {
      self: { nodeId: primaryId },
      output: { nodeId: primaryId, anchor: "e" },
      input: { nodeId: primaryId, anchor: "w" },
    },
    actions: { enter: [primaryId, ...extras.map((node) => node.id)] },
  };
}

describe("sdk.pipeline topology factory", () => {
  it("includes every multi-root stage fragment root in the pipeline group children", () => {
    const definition = createSdkRegistry().lookup("sdk.pipeline")!;
    const badge = rectNode("stage-a-badge", { x: 12, y: 28, width: 24, height: 16 });
    const stageA = stageFragment("stage-a", [badge], { x: 10, y: 4, width: 80, height: 40 });
    const stageB = stageFragment("stage-b");

    const result = definition.factory(
      { id: "pipe", gap: 20 },
      { nodes: [stageA, stageB] },
      context("pipe"),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    const group = result.value.roots[0];
    expect(group?.kind).toBe("group");
    if (group?.kind !== "group") {
      return;
    }

    const childIds = group.children.map((child) => child.id);
    expect(childIds).toContain("stage-a");
    expect(childIds).toContain("stage-a-badge");
    expect(childIds).toContain("stage-b");

    const placedPrimary = group.children.find((child) => child.id === "stage-a");
    const placedBadge = group.children.find((child) => child.id === "stage-a-badge");
    expect(placedPrimary?.geometry).toMatchObject({ x: 0, y: 0 });
    // Badge keeps its offset relative to the primary (dx=12-10, dy=28-4).
    expect(placedBadge?.geometry).toMatchObject({ x: 2, y: 24 });

    expect(result.value.actions.enter).toContain("stage-a");
    expect(result.value.actions.enter).toContain("stage-a-badge");
    expect(result.value.actions.enter).toContain("stage-b");
  });
});
