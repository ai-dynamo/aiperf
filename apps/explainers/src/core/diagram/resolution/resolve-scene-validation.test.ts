/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Final validation contracts over canonical resolved bounds.

import { describe, expect, it } from "vitest";

import type { SceneIrLike } from "../scene-types.js";
import { resolveScene } from "./resolve-scene.js";

describe("resolved scene final validation", () => {
  it("ignores decorative bands but reports overlapping content siblings", () => {
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
    expect(overlaps).toContainEqual(
      expect.objectContaining({ nodeIds: ["first-chip", "second-chip"] }),
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
});
