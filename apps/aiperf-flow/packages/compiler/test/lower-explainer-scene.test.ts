/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SceneAst } from "@aiperf/flow-language";
import type { SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { lowerExplainerScene } from "../src/lower-explainer-scene.js";

function range(): SourceRange {
  return {
    source: "<test>",
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 0, line: 1, column: 1 },
  };
}

function validScene(overrides: Partial<SceneAst> = {}): SceneAst {
  return {
    kind: "scene",
    id: "diagram",
    title: "Diagram",
    summary: {
      kind: "summary",
      text: "A minimal explainer diagram.",
      sourceMap: range(),
    },
    renderDeclarations: [
      {
        kind: "rect",
        id: "box",
        x: 10,
        y: 20,
        width: 100,
        height: 40,
        fill: { kind: "literal", value: "#244a35", sourceMap: range() },
        label: "Box",
        role: "img",
        description: "A box",
        fallback: { kind: "fallback", text: "Box", sourceMap: range() },
        sourceMap: range(),
      },
    ],
    cameras: [],
    timelines: [
      {
        kind: "timeline",
        id: "primary",
        cues: [
          {
            kind: "timeline-cue",
            time: 0,
            duration: 400,
            target: "box",
            action: "reveal",
            sourceMap: range(),
          },
        ],
        sourceMap: range(),
      },
    ],
    interactions: [],
    responsiveVariants: [],
    narration: {
      kind: "narration",
      text: "The diagram appears.",
      sourceMap: range(),
    },
    readingOrder: {
      kind: "reading-order",
      references: ["box"],
      sourceMap: range(),
    },
    fallback: { kind: "fallback", text: "Box appears.", sourceMap: range() },
    sourceMap: range(),
    ...overrides,
  };
}

describe("lowerExplainerScene", () => {
  test("lowers an embedded @scene AST to { kind: 'scene', scene }", () => {
    const result = lowerExplainerScene(validScene());

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.kind).toBe("scene");
    expect(result.value.scene).toMatchObject({
      id: "diagram",
      title: "Diagram",
      summary: "A minimal explainer diagram.",
      narration: "The diagram appears.",
      fallback: "Box appears.",
      accessibility: {
        label: "Diagram",
        readingOrder: ["box"],
      },
    });
    expect(result.value.scene.roots).toEqual([
      expect.objectContaining({
        kind: "rect",
        id: "box",
        geometry: { x: 10, y: 20, width: 100, height: 40 },
        style: { fill: "#244a35" },
      }),
    ]);
    expect(result.value.scene.timeline).toEqual([
      expect.objectContaining({
        id: "primary-0",
        at: 0,
        duration: 400,
        target: "box",
        action: "reveal",
      }),
    ]);
  });

  test("resolves token references through the shared lowerer", () => {
    const scene = validScene({
      renderDeclarations: [
        {
          kind: "rect",
          id: "box",
          x: 0,
          y: 0,
          width: 50,
          height: 50,
          fill: {
            kind: "token-reference",
            token: "accent",
            sourceMap: range(),
          },
          label: "Box",
          role: "img",
          description: "",
          fallback: { kind: "fallback", text: "Box", sourceMap: range() },
          sourceMap: range(),
        },
      ],
    });

    const result = lowerExplainerScene(scene, {
      tokens: new Map([["accent", "#7aa2f7"]]),
    });

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.scene.roots[0]?.style.fill).toBe("#7aa2f7");
  });

  test("fails when the input is not a scene AST", () => {
    const result = lowerExplainerScene({ kind: "slide", title: "Nope" });

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics[0]?.code).toBe("EXPLAINER_SCENE_INVALID");
  });

  test("lowers a decks-flow package-scene to SceneRender", () => {
    const result = lowerExplainerScene(
      {
        kind: "package-scene",
        roots: [
          {
            id: "box",
            capability: "core.rect",
            layout: { x: 10, y: 20, width: 100, height: 40 },
            style: { fill: "@theme.surface.primary" },
          },
        ],
        timeline: [
          {
            id: "enter-box",
            at: 0,
            duration: 400,
            target: "box",
            action: "enter",
          },
        ],
        camera: [],
      },
      { defaults: { id: "diagram", title: "Diagram", summary: "A box" } },
    );

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.kind).toBe("scene");
    expect(result.value.scene.roots[0]).toMatchObject({
      kind: "rect",
      id: "box",
      geometry: { x: 10, y: 20, width: 100, height: 40 },
      style: { fill: "@theme.surface.primary" },
    });
    expect(result.value.scene.timeline[0]).toMatchObject({
      id: "enter-box",
      action: "enter",
      target: "box",
    });
  });

  test("fails when lowered SceneIr is schema-invalid", () => {
    const result = lowerExplainerScene(
      validScene({
        summary: undefined,
        fallback: undefined,
      }),
    );

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }
    expect(result.diagnostics.length).toBeGreaterThan(0);
    expect(result.diagnostics.every((d) => d.code === "EXPLAINER_SCENE_INVALID")).toBe(
      true,
    );
  });
});
