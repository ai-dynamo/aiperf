/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Focused contracts for canonical fan-out/fan-in topology resolution.

import { describe, expect, it } from "vitest";

import type { SceneGeometryLike, SceneNodeLike } from "../scene-types.js";
import { resolveFans } from "./resolve-fans.js";

function panel(id: string, geometry: SceneGeometryLike): SceneNodeLike {
  return {
    id,
    kind: "rect",
    capabilityId: "core.panel",
    geometry,
  };
}

function nodesAndGeometry(
  nodes: readonly SceneNodeLike[],
): Readonly<{
  nodesById: ReadonlyMap<string, SceneNodeLike>;
  worldGeometryById: ReadonlyMap<string, SceneGeometryLike>;
}> {
  const nodesById = new Map(nodes.map((node) => [node.id, node]));
  const worldGeometryById = new Map(
    nodes
      .filter((node) => node.geometry !== undefined)
      .map((node) => [node.id, node.geometry as SceneGeometryLike]),
  );
  return { nodesById, worldGeometryById };
}

describe("resolveFans", () => {
  it("resolves a fan-out node into a junction plus one trajectory per branch", () => {
    const fan: SceneNodeLike = {
      id: "fan",
      kind: "fan",
      capabilityId: "core.fan-out",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      axis: "x",
      from: { x: 0, y: 100 },
      to: [
        { x: 300, y: 50 },
        { x: 300, y: 150 },
      ],
    };
    const { nodesById, worldGeometryById } = nodesAndGeometry([fan]);

    const fanGeometryById = resolveFans({ nodesById, worldGeometryById });

    const geometry = fanGeometryById.get("fan");
    expect(geometry).toBeDefined();
    expect(geometry?.id).toBe("fan");
    expect(geometry?.capability).toBe("core.fan-out");
    expect(geometry?.trajectories).toHaveLength(2);
    expect(geometry?.trajectories.map(({ id }) => id)).toEqual([
      "fan-trajectory-0",
      "fan-trajectory-1",
    ]);
    // Trunk travels along x from the singleton toward the branch centroid.
    expect(geometry?.junction).toMatchObject({ y: 100 });
    expect(geometry?.segments.length).toBeGreaterThan(0);
    expect(
      geometry?.segments.some((segment) => segment.role === "trunk"),
    ).toBe(true);
    expect(
      geometry?.segments.some((segment) => segment.role === "branch"),
    ).toBe(true);
  });

  it("resolves a fan-in node into a merge-trunk trajectory per source branch", () => {
    const fan: SceneNodeLike = {
      id: "merge",
      kind: "fan",
      capabilityId: "core.fan-in",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      axis: "x",
      from: [
        { x: 0, y: 50 },
        { x: 0, y: 150 },
      ],
      to: { x: 300, y: 100 },
    };
    const { nodesById, worldGeometryById } = nodesAndGeometry([fan]);

    const fanGeometryById = resolveFans({ nodesById, worldGeometryById });

    const geometry = fanGeometryById.get("merge");
    expect(geometry?.capability).toBe("core.fan-in");
    expect(geometry?.trajectories).toHaveLength(2);
    for (const trajectory of geometry?.trajectories ?? []) {
      expect(trajectory.role).toBe("merge-trunk");
    }
  });

  it("resolves node-referenced endpoints from world geometry, not just literal points", () => {
    const source = panel("source", { x: 0, y: 80, width: 40, height: 40 });
    const targetA = panel("target-a", { x: 300, y: 0, width: 40, height: 40 });
    const targetB = panel("target-b", { x: 300, y: 160, width: 40, height: 40 });
    const fan: SceneNodeLike = {
      id: "fan",
      kind: "fan",
      capabilityId: "core.fan-out",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      from: { nodeId: "source", anchor: "e" },
      to: [
        { nodeId: "target-a", anchor: "w" },
        { nodeId: "target-b", anchor: "w" },
      ],
    };
    const { nodesById, worldGeometryById } = nodesAndGeometry([
      source,
      targetA,
      targetB,
      fan,
    ]);

    const fanGeometryById = resolveFans({ nodesById, worldGeometryById });

    const geometry = fanGeometryById.get("fan");
    expect(geometry).toBeDefined();
    // The trunk starts at "source"'s east-facing anchor (x=40, y=100).
    expect(geometry?.trajectories[0]?.d.startsWith("M40 100")).toBe(true);
  });

  it("returns undefined for a fan with fewer than two many-side endpoints", () => {
    const fan: SceneNodeLike = {
      id: "fan",
      kind: "fan",
      capabilityId: "core.fan-out",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      from: { x: 0, y: 100 },
      to: [],
    };
    const { nodesById, worldGeometryById } = nodesAndGeometry([fan]);

    const fanGeometryById = resolveFans({ nodesById, worldGeometryById });

    expect(fanGeometryById.has("fan")).toBe(false);
  });

  it("ignores non-fan nodes entirely", () => {
    const node = panel("panel", { x: 0, y: 0, width: 40, height: 40 });
    const { nodesById, worldGeometryById } = nodesAndGeometry([node]);

    const fanGeometryById = resolveFans({ nodesById, worldGeometryById });

    expect(fanGeometryById.size).toBe(0);
  });

  it("respects an authored junction point instead of the automatic midpoint", () => {
    const fan: SceneNodeLike = {
      id: "fan",
      kind: "fan",
      capabilityId: "core.fan-out",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      axis: "x",
      from: { x: 0, y: 100 },
      to: [
        { x: 300, y: 50 },
        { x: 300, y: 150 },
      ],
      junction: { x: 250, y: 100 },
    };
    const { nodesById, worldGeometryById } = nodesAndGeometry([fan]);

    const fanGeometryById = resolveFans({ nodesById, worldGeometryById });

    expect(fanGeometryById.get("fan")?.junction).toEqual({ x: 250, y: 100 });
  });
});
