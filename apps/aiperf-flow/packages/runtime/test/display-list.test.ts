// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  buildDisplayList,
  type DisplayListInput,
} from "../src/display-list.js";
import type {
  EvaluatedScene,
  SemanticProjection,
} from "../src/evaluate/types.js";

const bounds = { x: 0, y: 0, width: 100, height: 40 };

function command(
  id: string,
  order: number,
): DisplayListInput["commands"][number] {
  return {
    kind: "path",
    id,
    order,
    path: "M 0 0 L 100 40",
    paintBounds: bounds,
    damageBounds: bounds,
  };
}

describe("display list", () => {
  test("represents every backend-neutral command without host objects", () => {
    const displayList = buildDisplayList({
      commands: [
        {
          kind: "group",
          id: "group",
          order: 0,
          paintBounds: bounds,
          damageBounds: bounds,
          children: [
            command("path", 0),
            {
              kind: "text",
              id: "text",
              order: 1,
              text: "Runtime",
              origin: { x: 8, y: 24 },
              font: { family: "Inter", sizePx: 16 },
              paintBounds: bounds,
              damageBounds: bounds,
            },
            {
              kind: "image",
              id: "image",
              order: 2,
              assetId: "runtime-icon",
              destination: bounds,
              paintBounds: bounds,
              damageBounds: bounds,
            },
          ],
        },
        {
          kind: "clip",
          id: "clip",
          order: 1,
          path: "M 0 0 H 100 V 40 H 0 Z",
          paintBounds: bounds,
          damageBounds: bounds,
          children: [command("clipped-path", 0)],
        },
        {
          kind: "layer",
          id: "layer",
          order: 2,
          opacity: 0.8,
          blendMode: "screen",
          paintBounds: bounds,
          damageBounds: bounds,
          children: [command("layer-path", 0)],
        },
      ],
      hitRegions: [
        {
          id: "runtime-hit",
          semanticId: "runtime",
          order: 0,
          bounds,
        },
      ],
      paintBounds: bounds,
      damageBounds: bounds,
    });

    const semantic: SemanticProjection = {
      sceneId: "request-path",
      entities: [{ id: "runtime", label: "Runtime", role: "group" }],
      relations: [],
      readingOrder: ["runtime"],
    };
    const scene: EvaluatedScene = {
      sceneId: "request-path",
      atMs: 2500,
      displayList,
      semantic,
    };

    expect(displayList.commands.map(({ kind }) => kind)).toEqual([
      "group",
      "clip",
      "layer",
    ]);
    expect(scene.semantic.readingOrder).toEqual(["runtime"]);
    expect(Object.isFrozen(displayList)).toBe(true);
    expect(Object.isFrozen(displayList.commands)).toBe(true);
    expect(Object.isFrozen(displayList.commands[0])).toBe(true);
  });

  test.each([
    ["display-list paint bounds", { ...bounds, width: Number.NaN }],
    ["display-list damage bounds", { ...bounds, height: Number.POSITIVE_INFINITY }],
    ["command paint bounds", { ...bounds, x: Number.NEGATIVE_INFINITY }],
    ["hit-region bounds", { ...bounds, y: Number.NaN }],
  ])("rejects non-finite %s", (location, invalidBounds) => {
    const baseInput: DisplayListInput = {
      commands: [command("path", 0)],
      hitRegions: [
        {
          id: "runtime-hit",
          semanticId: "runtime",
          order: 0,
          bounds,
        },
      ],
      paintBounds: bounds,
      damageBounds: bounds,
    };

    const input: DisplayListInput =
      location === "display-list paint bounds"
        ? { ...baseInput, paintBounds: invalidBounds }
        : location === "display-list damage bounds"
          ? { ...baseInput, damageBounds: invalidBounds }
          : location === "command paint bounds"
            ? {
                ...baseInput,
                commands: [
                  { ...command("path", 0), paintBounds: invalidBounds },
                ],
              }
            : {
                ...baseInput,
                hitRegions: [
                  { ...baseInput.hitRegions[0]!, bounds: invalidBounds },
                ],
              };

    expect(() => buildDisplayList(input)).toThrow(/finite bounds/);
  });

  test("orders commands and hit regions by order then id deterministically", () => {
    const inputs: readonly DisplayListInput[] = [
      {
        commands: [
          command("zebra", 1),
          {
            kind: "group",
            id: "group",
            order: 0,
            paintBounds: bounds,
            damageBounds: bounds,
            children: [command("beta", 0), command("alpha", 0)],
          },
          command("alpha", 1),
        ],
        hitRegions: [
          { id: "zebra-hit", semanticId: "zebra", order: 0, bounds },
          { id: "alpha-hit", semanticId: "alpha", order: 0, bounds },
        ],
        paintBounds: bounds,
        damageBounds: bounds,
      },
      {
        commands: [
          command("alpha", 1),
          {
            kind: "group",
            id: "group",
            order: 0,
            paintBounds: bounds,
            damageBounds: bounds,
            children: [command("alpha", 0), command("beta", 0)],
          },
          command("zebra", 1),
        ],
        hitRegions: [
          { id: "alpha-hit", semanticId: "alpha", order: 0, bounds },
          { id: "zebra-hit", semanticId: "zebra", order: 0, bounds },
        ],
        paintBounds: bounds,
        damageBounds: bounds,
      },
    ];

    const serialized = inputs.map((input) =>
      JSON.stringify(buildDisplayList(input)),
    );

    expect(serialized[0]).toBe(serialized[1]);
    const result = buildDisplayList(inputs[0]!);
    expect(result.commands.map(({ id }) => id)).toEqual([
      "group",
      "alpha",
      "zebra",
    ]);
    const group = result.commands[0];
    expect(group?.kind).toBe("group");
    if (group?.kind === "group") {
      expect(group.children.map(({ id }) => id)).toEqual(["alpha", "beta"]);
    }
    expect(result.hitRegions.map(({ id }) => id)).toEqual([
      "alpha-hit",
      "zebra-hit",
    ]);
  });
});
