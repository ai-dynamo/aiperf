/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for camera / interactions / responsive / theme-role /
//! token resolution surviving package/native-package scene lowering
//! (`lowerPackageScene` and `nativeSceneToPackageScene`).

import { describe, expect, it } from "vitest";

import type {
  CameraAst,
  InteractionAst,
  RectAst,
  ResponsiveAst,
  SceneAst,
  ScenePrimitiveAst,
} from "../language/ast.js";
import { lowerExplainerScene } from "./lower-explainer-scene.js";

const SOURCE_MAP = {
  source: "lower-explainer-scene.camera.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

/** Minimal rect used as a camera / interaction / responsive target. */
function rectNode(
  id: string,
  fill: RectAst["fill"] = { kind: "literal", value: "#f00", sourceMap: SOURCE_MAP },
  extras: Partial<Pick<RectAst, "x" | "y" | "width" | "height" | "stroke">> = {},
): RectAst {
  return {
    kind: "rect",
    id,
    x: extras.x ?? 10,
    y: extras.y ?? 20,
    width: extras.width ?? 40,
    height: extras.height ?? 30,
    fill,
    ...(extras.stroke !== undefined ? { stroke: extras.stroke } : {}),
    label: id,
    role: "",
    description: "",
    fallback: { kind: "fallback", text: id, sourceMap: SOURCE_MAP },
    sourceMap: SOURCE_MAP,
  };
}

/** Fade cue forces `nativeSceneNeedsPackageLower` onto the package path. */
function fadeCue(target: string): SceneAst["timelines"][number] {
  return {
    kind: "timeline",
    id: "main",
    sourceMap: SOURCE_MAP,
    cues: [
      {
        kind: "timeline-cue",
        sourceMap: SOURCE_MAP,
        timing: { mode: "at", ms: 0 },
        action: "fade",
        target,
        duration: 200,
      },
    ],
  };
}

describe("camera keyframes through explainer scene lowering", () => {
  it("passes through authored package-scene camera keyframes", () => {
    const result = lowerExplainerScene({
      kind: "package-scene",
      id: "scene",
      title: "Scene",
      roots: [{ id: "node", capability: "core.rect", layout: { x: 0, y: 0, width: 10, height: 10 } }],
      timeline: [{ id: "cue-0", at: 0, duration: 100, target: "node", action: "enter" }],
      camera: [
        { id: "cam-0", at: 0, x: 5, y: 5, zoom: 1 },
        { id: "cam-1", at: 500, x: 40, y: 20, zoom: 1.8 },
        // Non-finite keyframes must be dropped, not forwarded as garbage IR.
        { id: "cam-bad", at: Number.NaN, x: 0, y: 0, zoom: 1 },
      ],
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.scene.camera).toEqual([
      { id: "cam-0", at: 0, x: 5, y: 5, zoom: 1, sourceMap: expect.anything() },
      { id: "cam-1", at: 500, x: 40, y: 20, zoom: 1.8, sourceMap: expect.anything() },
    ]);
  });

  it("preserves native camera keyframes when a scene needs the package lowerer", () => {
    const camera: CameraAst = {
      kind: "camera",
      id: "cam",
      keyframes: [
        {
          kind: "camera-keyframe",
          time: 0,
          targets: { kind: "reference-list", references: ["node"], sourceMap: SOURCE_MAP },
          zoom: 1,
          sourceMap: SOURCE_MAP,
        },
        {
          kind: "camera-keyframe",
          time: 500,
          targets: { kind: "reference-list", references: ["node"], sourceMap: SOURCE_MAP },
          zoom: 2,
          sourceMap: SOURCE_MAP,
        },
      ],
      sourceMap: SOURCE_MAP,
    };

    // `fade` forces `nativeSceneNeedsPackageLower` to route this scene through
    // `nativeSceneToPackageScene` + `lowerPackageScene` instead of `lower.ts`.
    const scene: SceneAst = {
      kind: "scene",
      id: "scene",
      title: "Scene",
      sourceMap: SOURCE_MAP,
      renderDeclarations: [rectNode("node")],
      cameras: [camera],
      timelines: [fadeCue("node")],
      interactions: [],
      responsiveVariants: [],
    };

    const result = lowerExplainerScene(scene);
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.scene.camera).toEqual([
      { id: "cam-0", at: 0, x: 30, y: 35, zoom: 1, sourceMap: expect.anything() },
      { id: "cam-1", at: 500, x: 30, y: 35, zoom: 2, sourceMap: expect.anything() },
    ]);
  });

  it("preserves native interactions when a scene needs the package lowerer", () => {
    const interaction: InteractionAst = {
      kind: "interaction",
      id: "inspect-node",
      event: {
        kind: "interaction-event",
        name: "select",
        target: "node",
        sourceMap: SOURCE_MAP,
      },
      action: {
        kind: "interaction-action",
        name: "inspect",
        target: "node",
        sourceMap: SOURCE_MAP,
      },
      sourceMap: SOURCE_MAP,
    };

    const scene: SceneAst = {
      kind: "scene",
      id: "scene",
      title: "Scene",
      sourceMap: SOURCE_MAP,
      renderDeclarations: [rectNode("node")],
      cameras: [],
      timelines: [fadeCue("node")],
      interactions: [interaction],
      responsiveVariants: [],
    };

    const result = lowerExplainerScene(scene);
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.scene.interactions).toEqual([
      {
        id: "inspect-node",
        event: "select",
        target: "node",
        action: "inspect",
        sourceMap: expect.anything(),
      },
    ]);
  });

  it("preserves native responsive variants when a scene needs the package lowerer", () => {
    const responsive: ResponsiveAst = {
      kind: "responsive",
      id: "narrow",
      condition: {
        kind: "responsive-condition",
        property: "width",
        operator: "<",
        value: 600,
        sourceMap: SOURCE_MAP,
      },
      overrides: [
        {
          kind: "responsive-override",
          target: "node",
          property: "x",
          value: 4,
          sourceMap: SOURCE_MAP,
        },
      ],
      sourceMap: SOURCE_MAP,
    };

    const scene: SceneAst = {
      kind: "scene",
      id: "scene",
      title: "Scene",
      sourceMap: SOURCE_MAP,
      renderDeclarations: [rectNode("node")],
      cameras: [],
      timelines: [fadeCue("node")],
      interactions: [],
      responsiveVariants: [responsive],
    };

    const result = lowerExplainerScene(scene);
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.scene.responsive).toHaveLength(1);
    const variant = result.value.scene.responsive[0]!;
    expect(variant).toMatchObject({
      id: "narrow",
      condition: "width < 600",
    });
    const patched = variant.roots.find((root) => root.id === "node");
    expect(patched?.geometry.x).toBe(4);
    expect(patched?.geometry.y).toBe(20);
  });

  it("passes through authored package-scene interactions and responsive variants", () => {
    const result = lowerExplainerScene({
      kind: "package-scene",
      id: "scene",
      title: "Scene",
      roots: [
        {
          id: "node",
          capability: "core.rect",
          layout: { x: 10, y: 20, width: 40, height: 30 },
        },
      ],
      timeline: [
        { id: "cue-0", at: 0, duration: 100, target: "node", action: "enter" },
      ],
      camera: [],
      interactions: [
        {
          id: "inspect-node",
          event: "select",
          target: "node",
          action: "inspect",
        },
      ],
      responsive: [
        {
          id: "narrow",
          condition: "width < 600",
          overrides: [{ target: "node", property: "width", value: 80 }],
        },
      ],
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.scene.interactions).toEqual([
      {
        id: "inspect-node",
        event: "select",
        target: "node",
        action: "inspect",
        sourceMap: expect.anything(),
      },
    ]);
    expect(result.value.scene.responsive).toHaveLength(1);
    expect(result.value.scene.responsive[0]).toMatchObject({
      id: "narrow",
      condition: "width < 600",
    });
    expect(
      result.value.scene.responsive[0]!.roots.find((root) => root.id === "node")
        ?.geometry.width,
    ).toBe(80);
  });
});

describe("theme-role and token style values through package lowering", () => {
  it("preserves theme-role fill/stroke when a native scene needs the package lowerer", () => {
    const scene: SceneAst = {
      kind: "scene",
      id: "scene",
      title: "Scene",
      sourceMap: SOURCE_MAP,
      renderDeclarations: [
        rectNode("node", {
          kind: "theme-role-reference",
          role: "surface.panel",
          sourceMap: SOURCE_MAP,
        }, {
          stroke: {
            kind: "theme-role-reference",
            role: "line.structural",
            sourceMap: SOURCE_MAP,
          },
        }),
      ],
      cameras: [],
      timelines: [fadeCue("node")],
      interactions: [],
      responsiveVariants: [],
    };

    const result = lowerExplainerScene(scene);
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const node = result.value.scene.roots.find((root) => root.id === "node");
    expect(node?.style.fill).toEqual({
      kind: "theme-role",
      role: "surface.panel",
    });
    expect(node?.style.stroke).toEqual({
      kind: "theme-role",
      role: "line.structural",
    });
  });

  it("resolves @token fill/stroke via options.tokens on the package path", () => {
    const scene: SceneAst = {
      kind: "scene",
      id: "scene",
      title: "Scene",
      sourceMap: SOURCE_MAP,
      renderDeclarations: [
        rectNode(
          "node",
          { kind: "token-reference", token: "brandBlue", sourceMap: SOURCE_MAP },
          {
            stroke: {
              kind: "token-reference",
              token: "brandLine",
              sourceMap: SOURCE_MAP,
            },
          },
        ),
      ],
      cameras: [],
      timelines: [fadeCue("node")],
      interactions: [],
      responsiveVariants: [],
    };

    const result = lowerExplainerScene(scene, {
      tokens: new Map([
        ["brandBlue", "#00aaff"],
        ["brandLine", "#112233"],
      ]),
    });
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const node = result.value.scene.roots.find((root) => root.id === "node");
    expect(node?.style.fill).toBe("#00aaff");
    expect(node?.style.stroke).toBe("#112233");
  });

  it("resolves @token props on scene-primitives via options.tokens", () => {
    // Scene-primitive forces the package path; timeline targets a rect so link
    // can resolve the cue (scene-primitive ids are not link reference targets).
    const primitive: ScenePrimitiveAst = {
      kind: "scene-primitive",
      id: "chip",
      capability: "core.chip",
      props: [
        {
          kind: "prop-assignment",
          name: "fill",
          value: {
            kind: "token-reference",
            token: "chipFill",
            sourceMap: SOURCE_MAP,
          },
          sourceMap: SOURCE_MAP,
        },
      ],
      sourceMap: SOURCE_MAP,
      fallback: { kind: "fallback", text: "chip", sourceMap: SOURCE_MAP },
    };

    const scene: SceneAst = {
      kind: "scene",
      id: "scene",
      title: "Scene",
      sourceMap: SOURCE_MAP,
      renderDeclarations: [rectNode("node"), primitive],
      cameras: [],
      timelines: [fadeCue("node")],
      interactions: [],
      responsiveVariants: [],
    };

    const result = lowerExplainerScene(scene, {
      tokens: new Map([["chipFill", "#ffaa00"]]),
    });
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const chip = result.value.scene.roots.find((root) => root.id === "chip");
    expect(chip?.style.fill).toBe("#ffaa00");
  });
});
