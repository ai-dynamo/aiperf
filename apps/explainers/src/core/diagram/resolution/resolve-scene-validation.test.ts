/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Final validation contracts over canonical resolved bounds.

import { describe, expect, it } from "vitest";

import type { SceneIrLike } from "../scene-types.js";
import { resolveScene } from "./resolve-scene.js";

describe("resolved scene final validation", () => {
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
