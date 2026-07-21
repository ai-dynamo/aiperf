/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Final validation contracts over canonical resolved bounds.

import { describe, expect, it } from "vitest";

import type { SceneIrLike, SceneNodeLike } from "../scene-types.js";
import { resolveScene } from "./resolve-scene.js";

function sdkOrigin(
  componentId: string,
  instanceId: string,
  generatedRole: string,
): NonNullable<SceneNodeLike["sdkOrigin"]> {
  return {
    componentId,
    instanceId,
    generatedRole,
  };
}

function overlapWarnings(scene: SceneIrLike) {
  return resolveScene(scene).diagnostics.filter(
    ({ code }) => code === "SCENE_ABSOLUTE_SIBLING_OVERLAP",
  );
}

describe("resolved scene final validation", () => {
  it("ignores decorative bands and brackets but reports overlapping content siblings", () => {
    const scene: SceneIrLike = {
      viewport: { width: 200, height: 100 },
      roots: [
        {
          id: "band",
          kind: "rect",
          capabilityId: "core.band",
          geometry: { x: 0, y: 10, width: 180, height: 30 },
          children: [],
        },
        {
          id: "band-chip",
          kind: "group",
          capabilityId: "core.chip",
          geometry: { x: 20, y: 20, width: 80, height: 20 },
          children: [],
        },
        {
          id: "bracket",
          kind: "path",
          capabilityId: "core.bracket",
          geometry: { x: 100, y: 10, width: 60, height: 30 },
          children: [],
        },
        {
          id: "bracket-panel",
          kind: "rect",
          capabilityId: "core.panel",
          geometry: { x: 120, y: 20, width: 50, height: 20 },
          children: [],
        },
        {
          id: "first-chip",
          kind: "group",
          capabilityId: "core.chip",
          geometry: { x: 20, y: 60, width: 80, height: 20 },
          children: [],
        },
        {
          id: "second-chip",
          kind: "group",
          capabilityId: "core.chip",
          geometry: { x: 60, y: 70, width: 80, height: 20 },
          children: [],
        },
      ],
      timeline: [],
    };

    const overlaps = resolveScene(scene).diagnostics.filter(
      ({ code }) => code === "SCENE_ABSOLUTE_SIBLING_OVERLAP",
    );

    expect(overlaps).not.toContainEqual(
      expect.objectContaining({ nodeIds: ["band", "band-chip"] }),
    );
    expect(overlaps).not.toContainEqual(
      expect.objectContaining({ nodeIds: ["bracket", "bracket-panel"] }),
    );
    expect(overlaps).toContainEqual(
      expect.objectContaining({ nodeIds: ["first-chip", "second-chip"] }),
    );
  });

  it("ignores sdk indicator track and value parts sharing one instance", () => {
    const scene: SceneIrLike = {
      viewport: { width: 700, height: 400 },
      roots: [
        {
          id: "progress-hero",
          kind: "group",
          capabilityId: "core.group",
          geometry: { x: 44, y: 135, width: 350, height: 28 },
          sdkOrigin: sdkOrigin("sdk.progress", "progress-hero", "root"),
          children: [
            {
              id: "progress-hero__track",
              kind: "rect",
              capabilityId: "core.rect",
              geometry: { x: 0, y: 0, width: 350, height: 28 },
              sdkOrigin: sdkOrigin("sdk.progress", "progress-hero", "track"),
              children: [],
            },
            {
              id: "progress-hero__value",
              kind: "rect",
              capabilityId: "core.rect",
              geometry: { x: 0, y: 0, width: 210, height: 28 },
              sdkOrigin: sdkOrigin("sdk.progress", "progress-hero", "value"),
              children: [],
            },
          ],
        },
      ],
      timeline: [],
    };

    expect(overlapWarnings(scene)).toEqual([]);
  });

  it("still warns for track and value ids without sdk provenance", () => {
    const scene: SceneIrLike = {
      viewport: { width: 700, height: 400 },
      roots: [
        {
          id: "progress-hero",
          kind: "group",
          capabilityId: "core.group",
          geometry: { x: 44, y: 135, width: 350, height: 28 },
          children: [
            {
              id: "progress-hero__track",
              kind: "rect",
              capabilityId: "core.rect",
              geometry: { x: 0, y: 0, width: 350, height: 28 },
              children: [],
            },
            {
              id: "progress-hero__value",
              kind: "rect",
              capabilityId: "core.rect",
              geometry: { x: 0, y: 0, width: 210, height: 28 },
              children: [],
            },
          ],
        },
      ],
      timeline: [],
    };

    expect(overlapWarnings(scene)).toContainEqual(
      expect.objectContaining({
        nodeIds: ["progress-hero__track", "progress-hero__value"],
      }),
    );
  });

  it("ignores sdk chips intentionally anchored on sibling panels", () => {
    const scene: SceneIrLike = {
      viewport: { width: 700, height: 400 },
      roots: [
        {
          id: "algo-built",
          kind: "group",
          capabilityId: "core.panel",
          geometry: { x: 35, y: 132, width: 190, height: 72 },
          sdkOrigin: sdkOrigin("sdk.panel", "algo-built", "root"),
          children: [],
        },
        {
          id: "st-built",
          kind: "group",
          capabilityId: "core.chip",
          geometry: { x: 155, y: 120, width: 70, height: 22 },
          sdkOrigin: sdkOrigin("sdk.chip", "st-built", "root"),
          children: [],
        },
      ],
      timeline: [],
    };

    expect(overlapWarnings(scene)).toEqual([]);
  });

  it("still warns when sdk chips overlap unrelated panels", () => {
    const scene: SceneIrLike = {
      viewport: { width: 700, height: 400 },
      roots: [
        {
          id: "left-panel",
          kind: "group",
          capabilityId: "core.panel",
          geometry: { x: 35, y: 132, width: 190, height: 100 },
          sdkOrigin: sdkOrigin("sdk.panel", "left-panel", "root"),
          children: [],
        },
        {
          id: "right-chip",
          kind: "group",
          capabilityId: "core.chip",
          geometry: { x: 120, y: 150, width: 70, height: 22 },
          sdkOrigin: sdkOrigin("sdk.chip", "right-chip", "root"),
          children: [],
        },
      ],
      timeline: [],
    };

    expect(overlapWarnings(scene)).toContainEqual(
      expect.objectContaining({
        nodeIds: ["left-panel", "right-chip"],
      }),
    );
  });

  it("reports overlapping absolute siblings and viewport escape", () => {
    const scene: SceneIrLike = {
      viewport: { width: 100, height: 80 },
      roots: [
        {
          id: "first",
          kind: "rect",
          capabilityId: "core.panel",
          geometry: { x: 10, y: 10, width: 50, height: 30 },
          children: [],
        },
        {
          id: "second",
          kind: "rect",
          capabilityId: "core.panel",
          geometry: { x: 40, y: 20, width: 70, height: 30 },
          children: [],
        },
      ],
      timeline: [],
    };

    const diagnostics = resolveScene(scene).diagnostics;

    expect(diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_ABSOLUTE_SIBLING_OVERLAP",
        severity: "warning",
        nodeIds: ["first", "second"],
      }),
    );
    expect(diagnostics).toContainEqual(
      expect.objectContaining({
        code: "SCENE_VIEWPORT_ESCAPE",
        severity: "warning",
        nodeIds: ["second"],
      }),
    );
  });

  it("reports an escaping semantic chrome owner without duplicate generated-part escapes", () => {
    const scene: SceneIrLike = {
      viewport: { width: 100, height: 80 },
      roots: [
        {
          id: "s2-note",
          kind: "group",
          capabilityId: "core.note",
          geometry: { x: 80, y: 20, width: 60, height: 36 },
          props: { text: "Escapes with its generated chrome." },
          children: [],
        },
      ],
      timeline: [],
    };

    const escapes = resolveScene(scene).diagnostics.filter(
      ({ code }) => code === "SCENE_VIEWPORT_ESCAPE",
    );

    expect(escapes).toEqual([
      expect.objectContaining({ nodeIds: ["s2-note"] }),
    ]);
  });
});
