/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JSON snapshot contracts for canonical resolved scenes.

import { describe, expect, it } from "vitest";

import type { SceneIrLike } from "../scene-types.js";
import { resolveScene } from "./resolve-scene.js";
import { resolvedSceneSnapshot } from "./serialize.js";
import type { ResolvedConnector, ResolvedScene } from "./types.js";

describe("resolvedSceneSnapshot", () => {
  it("serializes resolved maps into a JSON-safe document-order snapshot", () => {
    const scene: SceneIrLike = {
      id: "snapshot-scene",
      viewport: { width: 700, height: 400 },
      roots: [
        {
          id: "task-1",
          kind: "rect",
          capabilityId: "core.panel",
          geometry: { x: 40, y: 50, width: 80, height: 30 },
          children: [],
        },
        {
          id: "credit",
          kind: "connector",
          capabilityId: "core.connector",
          from: { nodeId: "task-1", anchor: "e" },
          to: { x: 240, y: 65 },
        },
      ],
      timeline: [],
    };
    const base = resolveScene(scene);
    const credit: ResolvedConnector = {
      id: "credit",
      source: { x: 120, y: 65 },
      target: { x: 240, y: 65 },
      sourceId: "task-1",
      d: "M120 65 L240 65",
      directed: true,
      showArrowhead: true,
      usedFallback: false,
      penetratedObstacleIds: [],
    };
    const resolved: ResolvedScene = {
      ...base,
      connectorsById: new Map([["credit", credit]]),
    };

    const snapshot = JSON.parse(
      JSON.stringify(resolvedSceneSnapshot(resolved)),
    ) as ReturnType<typeof resolvedSceneSnapshot>;

    expect(snapshot.sceneId).toBe("snapshot-scene");
    expect(snapshot.nodes.find(({ id }) => id === "task-1")?.bounds).toEqual(
      resolved.worldGeometryById.get("task-1"),
    );
    expect(snapshot.connectors.find(({ id }) => id === "credit")?.d).toBe(
      resolved.connectorsById.get("credit")?.d,
    );
    expect(snapshot.diagnostics).toEqual(resolved.diagnostics);
    expect(snapshot.nodes.map(({ id }) => id)).toEqual(["task-1", "credit"]);
  });

  it("serializes canonical fan geometry into the snapshot's `fans` array", () => {
    const scene: SceneIrLike = {
      id: "fan-snapshot-scene",
      roots: [
        {
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
        },
      ],
      timeline: [],
    };
    const resolved = resolveScene(scene);

    const snapshot = JSON.parse(
      JSON.stringify(resolvedSceneSnapshot(resolved)),
    ) as ReturnType<typeof resolvedSceneSnapshot>;

    expect(snapshot.fans).toHaveLength(1);
    expect(snapshot.fans[0]).toMatchObject({
      id: "fan",
      capability: "core.fan-out",
    });
    expect(snapshot.fans[0]?.trajectories).toEqual(
      resolved.fanGeometryById.get("fan")?.trajectories,
    );
  });
});
