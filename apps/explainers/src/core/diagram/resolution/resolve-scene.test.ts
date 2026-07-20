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

  it("applies relativePosition on top of a managed-layout geometry override", () => {
    const scene: SceneIrLike = {
      viewport: { width: 700, height: 400 },
      roots: [
        panel("anchor", 40, 30, { x: 10, y: 20 }),
        {
          id: "stack",
          kind: "group",
          capabilityId: "layout.stack",
          geometry: { x: 200, y: 200, width: 0, height: 0 },
          style: { direction: "column", gap: 8 },
          children: [
            {
              ...panel("shifted", 100, 30),
              relativePosition: {
                nodeId: "anchor",
                anchor: "se",
                dx: 5,
                dy: 7,
              },
            },
          ],
        },
      ],
      timeline: [],
    };

    const resolved = resolveScene(scene);

    expect(resolved.worldGeometryById.get("shifted")).toEqual({
      x: 55,
      y: 57,
      width: 100,
      height: 30,
    });
  });

  it("diagnoses a relativePosition target with no resolved world geometry", () => {
    const scene: SceneIrLike = {
      roots: [
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

    expect(resolved.worldGeometryById.get("before")).toMatchObject({
      x: 300,
      y: 200,
    });
    expect(resolved.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_RELATIVE_POSITION_TARGET_MISSING",
        severity: "error",
        nodeIds: ["before", "later"],
      }),
    );
  });

  it("diagnoses duplicate node ids while still resolving the second occurrence", () => {
    const scene: SceneIrLike = {
      roots: [
        panel("dup", 40, 20, { x: 10, y: 20 }),
        panel("dup", 40, 20, { x: 200, y: 20 }),
      ],
      timeline: [],
    };

    const resolved = resolveScene(scene);

    expect(resolved.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_DUPLICATE_NODE_ID",
        severity: "error",
        nodeIds: ["dup"],
      }),
    );
    expect(resolved.nodesById.get("dup")).toBe(scene.roots[1]);
    expect(resolved.worldGeometryById.get("dup")).toEqual({
      x: 200,
      y: 20,
      width: 40,
      height: 20,
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

  it("includes canonical connector paths and diagnostics in the resolved scene", () => {
    const scene: SceneIrLike = {
      roots: [
        panel("source", 80, 40, { x: 10, y: 20 }),
        panel("target", 80, 40, { x: 210, y: 20 }),
        {
          id: "edge",
          kind: "connector",
          capabilityId: "core.connector",
          geometry: { x: 0, y: 0, width: 0, height: 0 },
          style: {},
          from: { nodeId: "source", anchor: "e" },
          to: { nodeId: "target", anchor: "w" },
        },
      ],
      timeline: [],
    };

    const resolved = resolveScene(scene);

    expect(resolved.connectorsById.get("edge")).toMatchObject({
      d: "M90 40 L210 40",
      directed: true,
      showArrowhead: true,
    });
    expect(resolved.diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_DIRECTED_ARROWHEAD_DEFAULTED",
        severity: "info",
      }),
    );
  });

  it("indexes native semantic chrome under capability-specific generated IDs", () => {
    const scene: SceneIrLike = {
      viewport: { width: 700, height: 400 },
      roots: [
        {
          id: "header",
          kind: "group",
          capabilityId: "core.header",
          geometry: { x: 20, y: 30, width: 300, height: 66 },
          props: { title: "Profile", caption: "source" },
          children: [],
        },
        {
          id: "chip",
          kind: "group",
          capabilityId: "core.chip",
          geometry: { x: 20, y: 120, width: 84, height: 26 },
          props: { label: "Ready" },
          children: [],
        },
        {
          id: "note",
          kind: "group",
          capabilityId: "core.note",
          geometry: { x: 20, y: 170, width: 180, height: 48 },
          props: { text: "The worker only executes" },
          children: [],
        },
      ],
      timeline: [],
    };

    const resolved = resolveScene(scene);

    expect([...resolved.generatedPartsById.keys()]).toEqual([
      "header__chrome",
      "header__title",
      "header__caption",
      "chip__chrome",
      "chip__label",
      "note__chrome",
      "note__caption",
    ]);
    expect(resolved.generatedPartsById.get("note__caption")).toMatchObject({
      ownerId: "note",
      role: "caption",
    });
    expect(resolved.worldGeometryById.get("note__caption")).toEqual(
      resolved.generatedPartsById.get("note__caption")?.geometry,
    );
  });

  it("rejects an authored node that claims a semantic chrome generated ID", () => {
    const scene: SceneIrLike = {
      roots: [
        {
          id: "panel",
          kind: "group",
          capabilityId: "core.panel",
          geometry: { x: 20, y: 30, width: 180, height: 70 },
          props: { title: "Profile" },
          children: [
            {
              id: "panel__title",
              kind: "text",
              capabilityId: "core.text",
              geometry: { x: 8, y: 8, width: 160, height: 22 },
              text: "Compatibility title",
            },
          ],
        },
      ],
      timeline: [],
    };

    const resolved = resolveScene(scene);

    expect(
      resolved.diagnostics.filter(
        ({ code }) => code === "SCENE_DUPLICATE_PAINT_OWNER",
      ),
    ).toEqual([
      expect.objectContaining({
        severity: "error",
        nodeIds: ["panel__title", "panel"],
      }),
    ]);
  });

  it("does not generate paint parts for layout-managed stepper children", () => {
    const scene: SceneIrLike = {
      roots: [
        {
          id: "s7-steps",
          kind: "group",
          capabilityId: "core.stepper",
          geometry: { x: 20, y: 30, width: 0, height: 0 },
          props: { steps: ["Plan", "Run"] },
          children: [
            {
              id: "s7-steps-step-0",
              kind: "group",
              capabilityId: "core.chip",
              geometry: { x: 0, y: 0, width: 0, height: 0 },
              props: { label: "1. Plan" },
            },
            {
              id: "s7-steps-step-1",
              kind: "group",
              capabilityId: "core.chip",
              geometry: { x: 0, y: 0, width: 0, height: 0 },
              props: { label: "2. Run" },
            },
          ],
        },
      ],
      timeline: [],
    };

    const resolved = resolveScene(scene);

    expect(
      resolved.diagnostics.filter(
        ({ code }) => code === "SCENE_DUPLICATE_PAINT_OWNER",
      ),
    ).toEqual([]);
    expect(
      [...resolved.generatedPartsById.values()].filter(
        ({ ownerId }) => ownerId === "s7-steps",
      ),
    ).toEqual([]);
    expect(resolved.worldGeometryById.get("s7-steps-step-0")).toEqual(
      expect.objectContaining({ width: expect.any(Number) }),
    );
  });

  it("reports repeated generated IDs independently of authored ownership", () => {
    const duplicate = {
      id: "panel",
      kind: "group",
      capabilityId: "core.panel",
      geometry: { x: 20, y: 30, width: 180, height: 70 },
      props: { title: "Profile" },
      children: [],
    } satisfies SceneNodeLike;
    const scene: SceneIrLike = {
      roots: [
        duplicate,
        {
          ...duplicate,
          geometry: { ...duplicate.geometry, x: 240 },
        },
      ],
      timeline: [],
    };

    const resolved = resolveScene(scene);

    expect(
      resolved.diagnostics.filter(
        ({ code }) => code === "SCENE_DUPLICATE_GENERATED_ID",
      ),
    ).toEqual([
      expect.objectContaining({
        severity: "error",
        nodeIds: ["panel", "panel"],
      }),
      expect.objectContaining({
        severity: "error",
        nodeIds: ["panel", "panel"],
      }),
    ]);
  });
});
