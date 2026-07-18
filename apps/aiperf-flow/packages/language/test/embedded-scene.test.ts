// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";

import {
  detectEmbeddedSceneForm,
  parsePackageSceneBody,
} from "../src/embedded-scene.js";
import { parseNativeEmbeddedScene } from "../src/parser.js";

describe("embedded scene dialects", () => {
  it("detects package form from roots/timeline/camera leaders", () => {
    expect(detectEmbeddedSceneForm("roots: []")).toBe("package");
    expect(detectEmbeddedSceneForm("timeline: []")).toBe("package");
    expect(detectEmbeddedSceneForm("camera: []")).toBe("package");
    expect(detectEmbeddedSceneForm("rect box { x 0 y 0 width 1 height 1 }")).toBe(
      "native",
    );
  });

  it("parses package roots/timeline with @theme refs", () => {
    const scene = parsePackageSceneBody(`
      roots: [
        {
          id: "box"
          capability: "core.rect"
          layout: { x: 10, y: 20, width: 100, height: 40 }
          style: { fill: @theme.surface.primary }
        }
      ]
      timeline: [
        { id: "enter-box", at: 0, duration: 400, target: "box", action: "enter" }
      ]
    `);

    expect(scene.kind).toBe("package-scene");
    expect(scene.roots).toHaveLength(1);
    expect(scene.roots[0]).toMatchObject({
      id: "box",
      capability: "core.rect",
      style: { fill: "@theme.surface.primary" },
    });
    expect(scene.timeline).toEqual([
      {
        id: "enter-box",
        at: 0,
        duration: 400,
        target: "box",
        action: "enter",
      },
    ]);
  });

  it("parses native rect/timeline via shared cinematic path", () => {
    const result = parseNativeEmbeddedScene(`
      summary "One box"
      rect box {
        x 10
        y 20
        width 100
        height 40
        fill "#244a35"
        label "Box"
        role "img"
        description "A box"
        fallback "Box"
      }
      timeline primary {
        at 0ms duration 400ms reveal box
      }
      narrate "The diagram appears."
      reading-order [box]
      fallback "Box appears."
    `);

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.kind).toBe("scene");
    expect(result.value.renderDeclarations[0]).toMatchObject({
      kind: "rect",
      id: "box",
    });
    expect(result.value.timelines[0]?.cues[0]).toMatchObject({
      target: "box",
      action: "reveal",
    });
  });
});
