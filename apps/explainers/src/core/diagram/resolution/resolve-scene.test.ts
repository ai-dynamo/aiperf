/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Focused contracts for deterministic canonical Scene IR resolution.

import { describe, expect, it } from "vitest";

import type {
  SceneGeometryLike,
  SceneIrLike,
  SceneNodeLike,
} from "../scene-types.js";
import { resolveScene } from "./resolve-scene.js";

function panel(
  id: string,
  width: number,
  height: number,
  position: Readonly<{ x: number; y: number }> = { x: 0, y: 0 },
): SceneNodeLike {
  return {
    id,
    kind: "rect",
    capabilityId: "core.panel",
    geometry: { ...position, width, height },
    children: [],
  };
}

describe("resolveScene", () => {
  it("resolves managed children into stable world geometry without mutating authored input", () => {
    const scene: SceneIrLike = {
      viewport: { width: 700, height: 400 },
      roots: [
        {
          id: "stack",
          kind: "group",
          capabilityId: "layout.stack",
          geometry: { x: 40, y: 60, width: 0, height: 0 },
          style: { direction: "column", gap: 8 },
          children: [
            panel("one", 100, 30),
            panel("two", 100, 30),
          ],
        },
        panel("absolute", 80, 24, { x: 500, y: 300 }),
      ],
      timeline: [],
    };
    const authoredStackGeometry = scene.roots[0]?.geometry;

    const resolved = resolveScene(scene);

    expect(resolved.worldGeometryById.get("one")).toEqual({
      x: 40,
      y: 60,
      width: 100,
      height: 30,
    });
    expect(resolved.worldGeometryById.get("two")).toEqual({
      x: 40,
      y: 98,
      width: 100,
      height: 30,
    });
    expect(resolved.worldGeometryById.get("absolute")?.x).toBe(500);
    expect(resolved.ancestorIdsById.get("two")).toEqual(["stack"]);
    expect(scene.roots[0]?.geometry).toBe(authoredStackGeometry);
    expect(scene.roots[0]?.geometry).toEqual({
      x: 40,
      y: 60,
      width: 0,
      height: 0,
    });
    expect(resolveScene(scene)).toEqual(resolved);
    expect(resolved.connectorsById.size).toBe(0);
  });

  it("resolves relative positions only from previously visited world geometry", () => {
    const geometry: SceneGeometryLike = {
      x: 10,
      y: 20,
      width: 40,
      height: 30,
    };
    const scene: SceneIrLike = {
      roots: [
        panel("anchor", geometry.width, geometry.height, geometry),
        {
          ...panel("after", 20, 10),
          relativePosition: {
            nodeId: "anchor",
            anchor: "se",
            dx: 5,
            dy: 7,
          },
        },
        {
          ...panel("before", 20, 10, { x: 300, y: 200 }),
          relativePosition: {
            nodeId: "later",
            anchor: "center",
          },
        },
        panel("later", 20, 10, { x: 400, y: 300 }),
      ],
      timeline: [],
    };

    const resolved = resolveScene(scene);

    expect(resolved.worldGeometryById.get("after")).toMatchObject({
      x: 55,
      y: 57,
    });
    expect(resolved.worldGeometryById.get("before")).toMatchObject({
      x: 300,
      y: 200,
    });
  });

  it("promotes managed layout diagnostics into source-mapped scene findings", () => {
    const overflowRange = {
      source: "test.flow",
      start: { offset: 0, line: 1, column: 1 },
      end: { offset: 10, line: 1, column: 11 },
    };
    const overlapRange = {
      source: "overlap.flow",
      start: { offset: 20, line: 2, column: 1 },
      end: { offset: 40, line: 2, column: 21 },
    };
    const scene: SceneIrLike = {
      viewport: { width: 200, height: 200 },
      roots: [
        {
          id: "stack",
          kind: "group",
          capabilityId: "layout.stack",
          geometry: { x: 10, y: 20, width: 60, height: 60 },
          style: {
            direction: "column",
            padding: 10,
            fixedWidth: true,
          },
          sourceMap: overflowRange,
          children: [panel("wide", 80, 20)],
        },
        {
          id: "overlap-stack",
          kind: "group",
          capabilityId: "layout.stack",
          geometry: { x: 100, y: 20, width: 120, height: 80 },
          style: { direction: "column", gap: 8, padding: 8 },
          sourceMap: overlapRange,
          children: [
            panel("flow", 80, 30),
            {
              ...panel("absolute", 80, 30, { x: 8, y: 8 }),
              style: { position: "absolute" },
            },
          ],
        },
      ],
      timeline: [],
    };

    const resolved = resolveScene(scene);

    expect(resolved.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_MANAGED_CONTENT_OVERFLOW",
        severity: "error",
        range: overflowRange,
        nodeIds: expect.arrayContaining(["stack", "wide"]),
      }),
    );
    expect(resolved.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_MANAGED_CHILD_OVERLAP",
        severity: "error",
        range: overlapRange,
        nodeIds: expect.arrayContaining([
          "overlap-stack",
          "flow",
          "absolute",
        ]),
      }),
    );
  });
});
