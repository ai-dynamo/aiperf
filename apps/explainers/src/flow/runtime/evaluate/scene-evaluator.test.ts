/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { ConnectorNodeIr, RectNodeIr, SceneIr } from "../../schema/ir.js";
import { evaluateScene } from "./scene-evaluator.js";

const SOURCE_MAP = {
  source: "scene-evaluator.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function rect(
  id: string,
  geometry: Readonly<{ x: number; y: number; width: number; height: number }>,
): RectNodeIr {
  return {
    id,
    kind: "rect",
    geometry,
    style: {},
    accessibility: { label: id },
    fallback: id,
    sourceMap: SOURCE_MAP,
  };
}

function connector(
  partial: Omit<ConnectorNodeIr, "kind" | "style" | "accessibility" | "fallback" | "sourceMap"> &
    Partial<Pick<ConnectorNodeIr, "style" | "accessibility" | "fallback">>,
): ConnectorNodeIr {
  return {
    kind: "connector",
    style: {},
    accessibility: { label: partial.id },
    fallback: partial.id,
    sourceMap: SOURCE_MAP,
    ...partial,
  };
}

function scene(roots: SceneIr["roots"], readingOrder: readonly string[]): SceneIr {
  return {
    id: "scene",
    title: "Scene",
    summary: "test",
    roots,
    camera: [],
    timeline: [],
    narration: "",
    interactions: [],
    responsive: [],
    accessibility: { label: "Scene", readingOrder: [...readingOrder] },
    fallback: "Scene",
    sourceMap: SOURCE_MAP,
  };
}

describe("evaluateScene connector endpoints", () => {
  it("evaluates free-coordinate connectors without throwing", () => {
    const edge = connector({
      id: "edge",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      from: { x: 10, y: 20 },
      to: { x: 110, y: 80 },
    });

    const evaluated = evaluateScene(scene([edge], ["edge"]));

    expect(evaluated.displayList.commands).toHaveLength(1);
    const command = evaluated.displayList.commands[0];
    expect(command?.kind).toBe("path");
    if (command?.kind === "path") {
      expect(command.path).toBe("M 10 20 L 110 80");
      expect(command.paintBounds).toEqual({
        x: 10,
        y: 20,
        width: 100,
        height: 60,
      });
    }
    expect(evaluated.semantic.relations).toEqual([
      {
        id: "edge",
        fromId: "point:10,20",
        toId: "point:110,80",
        label: "edge",
      },
    ]);
  });

  it("still resolves node-anchored endpoints via node centers", () => {
    const a = rect("a", { x: 0, y: 0, width: 40, height: 20 });
    const b = rect("b", { x: 100, y: 40, width: 40, height: 20 });
    const edge = connector({
      id: "edge",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      from: { nodeId: "a" },
      to: { nodeId: "b" },
    });

    const evaluated = evaluateScene(scene([a, b, edge], ["a", "b", "edge"]));

    const command = evaluated.displayList.commands.find(({ id }) => id === "edge");
    expect(command?.kind).toBe("path");
    if (command?.kind === "path") {
      // centers: a=(20,10), b=(120,50)
      expect(command.path).toBe("M 20 10 L 120 50");
    }
    expect(evaluated.semantic.relations).toContainEqual({
      id: "edge",
      fromId: "a",
      toId: "b",
      label: "edge",
    });
  });

  it("supports mixed node-anchored and free-coordinate endpoints", () => {
    const a = rect("a", { x: 0, y: 0, width: 40, height: 20 });
    const edge = connector({
      id: "edge",
      geometry: { x: 0, y: 0, width: 0, height: 0 },
      from: { nodeId: "a" },
      to: { x: 200, y: 100 },
    });

    const evaluated = evaluateScene(scene([a, edge], ["a", "edge"]));

    const command = evaluated.displayList.commands.find(({ id }) => id === "edge");
    expect(command?.kind).toBe("path");
    if (command?.kind === "path") {
      expect(command.path).toBe("M 20 10 L 200 100");
    }
    expect(evaluated.semantic.relations).toContainEqual({
      id: "edge",
      fromId: "a",
      toId: "point:200,100",
      label: "edge",
    });
  });
});
