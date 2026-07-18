// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SceneIr } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import type { DrawCommand } from "../../src/display-list.js";
import { evaluateScene } from "../../src/evaluate/scene-evaluator.js";

const sourceMap = {
  source: "scene.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

function scene(): SceneIr {
  return {
    id: "foundation",
    title: "Foundation",
    summary: "Backend-neutral evaluation",
    roots: [
      {
        kind: "group",
        id: "root",
        geometry: { x: 0, y: 0, width: 640, height: 360 },
        style: {},
        accessibility: { label: "Foundation group" },
        fallback: "Group unavailable",
        sourceMap,
        children: [
          {
            kind: "rect",
            id: "panel",
            geometry: { x: 20, y: 30, width: 200, height: 100 },
            style: { fill: "#123456" },
            accessibility: { label: "Request panel" },
            fallback: "Panel unavailable",
            sourceMap,
          },
          {
            kind: "text",
            id: "label",
            geometry: { x: 40, y: 50, width: 100, height: 24 },
            style: { fill: "#ffffff" },
            accessibility: {
              label: "Request label",
              description: "Names the request",
            },
            fallback: "Label unavailable",
            sourceMap,
            text: "Request",
          },
        ],
      },
      {
        kind: "connector",
        id: "route",
        geometry: { x: 0, y: 0, width: 0, height: 0 },
        style: { stroke: "#76b900" },
        accessibility: { label: "Request route" },
        fallback: "Route unavailable",
        sourceMap,
        from: { nodeId: "panel" },
        to: { nodeId: "label" },
      },
    ],
    camera: [],
    timeline: [],
    narration: "",
    interactions: [],
    responsive: [],
    accessibility: {
      label: "Foundation scene",
      readingOrder: ["panel", "label", "route"],
    },
    fallback: "Scene unavailable",
    sourceMap,
  };
}

describe("evaluateScene", () => {
  test("preserves stable ids in deterministic depth-first draw order", () => {
    const first = evaluateScene(scene());
    const second = evaluateScene(scene());

    expect(first).toEqual(second);
    const flatten = (commands: readonly DrawCommand[]): readonly DrawCommand[] =>
      commands.flatMap((command) =>
        command.kind === "group" ||
        command.kind === "clip" ||
        command.kind === "layer"
          ? [command, ...flatten(command.children)]
          : [command],
      );
    const commands = flatten(first.displayList.commands);
    expect(commands.map(({ id }) => id)).toEqual([
      "root",
      "panel",
      "label",
      "route",
    ]);
    expect(first.displayList.commands.map(({ order }) => order)).toEqual([0, 1]);
    expect(
      first.displayList.commands[0]?.kind === "group"
        ? first.displayList.commands[0].children.map(({ order }) => order)
        : [],
    ).toEqual([0, 1]);
  });

  test("projects authored accessibility reading order", () => {
    const evaluated = evaluateScene(scene());

    expect(evaluated.semantic).toEqual({
      sceneId: "foundation",
      readingOrder: ["panel", "label", "route"],
      entities: [
        { id: "panel", label: "Request panel" },
        {
          id: "label",
          label: "Request label",
          description: "Names the request",
        },
      ],
      relations: [
        {
          id: "route",
          fromId: "panel",
          toId: "label",
          label: "Request route",
        },
      ],
    });
  });

  test("rejects duplicate node ids and non-finite geometry", () => {
    const duplicate = scene();
    const duplicatePanel = duplicate.roots[0]!;
    const invalidGeometry = scene();
    const invalidPanel = invalidGeometry.roots[0]!;

    expect(() =>
      evaluateScene({
        ...duplicate,
        roots: [...duplicate.roots, duplicatePanel],
      }),
    ).toThrow('Duplicate scene node id "root".');
    expect(() =>
      evaluateScene({
        ...invalidGeometry,
        roots: [
          {
            ...invalidPanel,
            geometry: { ...invalidPanel.geometry, x: Number.NaN },
          },
          ...invalidGeometry.roots.slice(1),
        ],
      }),
    ).toThrow('Node "root" geometry must contain finite numbers.');
  });
});
