// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ComponentNodeIr, SceneIr } from "@aiperf/flow-schema";
import { describe, expect, test, vi } from "vitest";

import type { DrawCommand } from "../../src/display-list.js";
import {
  CapabilityEvaluatorRegistry,
  type CapabilityContribution,
  type CapabilityEvaluator,
  UnknownCapabilityEvaluatorError,
} from "../../src/evaluate/registry.js";
import { evaluateScene } from "../../src/evaluate/scene-evaluator.js";

const sourceMap = {
  source: "component.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

function glyphRun(
  id: string,
  capabilityId = "core.glyph-run",
): ComponentNodeIr {
  return {
    kind: "component",
    id,
    capabilityId,
    geometry: { x: 10, y: 20, width: 120, height: 24 },
    style: {},
    accessibility: { label: "Glyph run" },
    fallback: "Glyph run unavailable",
    sourceMap,
    props: { text: "café" },
    children: [],
  };
}

function sceneWithComponents(
  roots: SceneIr["roots"],
  readingOrder: readonly string[],
): SceneIr {
  return {
    id: "components",
    title: "Components",
    summary: "Component dispatch",
    roots,
    camera: [],
    timeline: [],
    narration: "",
    interactions: [],
    responsive: [],
    accessibility: {
      label: "Component scene",
      readingOrder: [...readingOrder],
    },
    fallback: "Scene unavailable",
    sourceMap,
  };
}

function stubEvaluator(capabilityId: string): CapabilityEvaluator {
  return {
    capabilityId,
    evaluate(node, context): CapabilityContribution {
      const bounds = node.geometry;
      return {
        display: {
          commands: [
            {
              kind: "text",
              id: `${node.id}:text`,
              order: 0,
              paintBounds: bounds,
              damageBounds: bounds,
              text: String(node.props.text ?? ""),
              origin: { x: bounds.x, y: bounds.y + bounds.height },
              font: { family: "Inter", sizePx: 16 },
            },
          ],
          hitRegions: [
            {
              id: `hit:${node.id}:text`,
              semanticId: `${node.id}:glyph`,
              order: 0,
              bounds,
            },
          ],
        },
        semantic: {
          entities: [
            {
              id: `${node.id}:glyph`,
              label: String(node.props.text ?? node.accessibility.label),
              role: "text",
            },
          ],
          relations: [],
          readingOrder: [`${node.id}:glyph`],
        },
      };
    },
  };
}

describe("evaluateScene component dispatch", () => {
  test("preserves foundation evaluation without an injected registry", () => {
    const foundation: SceneIr = {
      id: "foundation",
      title: "Foundation",
      summary: "No components",
      roots: [
        {
          kind: "rect",
          id: "panel",
          geometry: { x: 0, y: 0, width: 40, height: 20 },
          style: { fill: "#123456" },
          accessibility: { label: "Panel" },
          fallback: "Panel unavailable",
          sourceMap,
        },
      ],
      camera: [],
      timeline: [],
      narration: "",
      interactions: [],
      responsive: [],
      accessibility: { label: "Foundation", readingOrder: ["panel"] },
      fallback: "Scene unavailable",
      sourceMap,
    };

    const evaluated = evaluateScene(foundation, 0);
    expect(evaluated.displayList.commands).toHaveLength(1);
    expect(evaluated.semantic.entities).toEqual([
      { id: "panel", label: "Panel" },
    ]);
  });

  test("fails closed on components when no evaluators are injected", () => {
    expect(() =>
      evaluateScene(sceneWithComponents([glyphRun("prompt")], ["prompt"])),
    ).toThrow('Foundation evaluator cannot evaluate component "prompt".');
  });

  test("dispatches components through injected evaluators at integer virtual time", () => {
    const dateNow = vi.spyOn(Date, "now").mockImplementation(() => {
      throw new Error("wall time must not be read");
    });
    const seenTimes: number[] = [];
    const registry = new CapabilityEvaluatorRegistry([
      {
        capabilityId: "core.glyph-run",
        evaluate(node, context): CapabilityContribution {
          seenTimes.push(context.atMs);
          return stubEvaluator("core.glyph-run").evaluate(node, context);
        },
      },
    ]).freeze();

    const evaluated = evaluateScene(
      sceneWithComponents([glyphRun("prompt")], ["prompt"]),
      17,
      { evaluators: registry },
    );

    expect(seenTimes).toEqual([17]);
    expect(dateNow).not.toHaveBeenCalled();
    const root = evaluated.displayList.commands[0];
    expect(root).toMatchObject({
      kind: "group",
      id: "prompt",
    });
    expect(
      root?.kind === "group" ? root.children.map(({ id }) => id) : [],
    ).toEqual(["prompt:text"]);
    expect(evaluated.semantic.entities.map(({ id }) => id)).toEqual([
      "prompt",
      "prompt:glyph",
    ]);
    expect(evaluated.semantic.readingOrder).toEqual(["prompt", "prompt:glyph"]);
    expect(evaluated.displayList.hitRegions.map(({ id }) => id)).toEqual([
      "hit:prompt",
      "hit:prompt:text",
    ]);
  });

  test("merges multiple component contributions in deterministic source order", () => {
    const registry = new CapabilityEvaluatorRegistry([
      stubEvaluator("core.glyph-run"),
    ]).freeze();
    const evaluated = evaluateScene(
      sceneWithComponents(
        [glyphRun("zeta"), glyphRun("alpha")],
        ["zeta", "alpha"],
      ),
      0,
      { evaluators: registry },
    );

    expect(evaluated.displayList.commands.map(({ id }) => id)).toEqual([
      "zeta",
      "alpha",
    ]);
    expect(evaluated.semantic.entities.map(({ id }) => id)).toEqual([
      "zeta",
      "alpha",
      "zeta:glyph",
      "alpha:glyph",
    ]);
    expect(evaluated.semantic.readingOrder).toEqual([
      "zeta",
      "alpha",
      "zeta:glyph",
      "alpha:glyph",
    ]);
    expect(evaluated.displayList.hitRegions.map(({ semanticId }) => semanticId)).toEqual([
      "zeta",
      "alpha",
      "zeta:glyph",
      "alpha:glyph",
    ]);
  });

  test("reports unknown injected capability ids", () => {
    const registry = new CapabilityEvaluatorRegistry([
      stubEvaluator("core.glyph-run"),
    ]).freeze();

    expect(() =>
      evaluateScene(
        sceneWithComponents(
          [glyphRun("prompt", "viz.missing")],
          ["prompt"],
        ),
        0,
        { evaluators: registry },
      ),
    ).toThrow(UnknownCapabilityEvaluatorError);
  });

  test("rejects duplicate contributed semantic ids against the foundation projection", () => {
    const registry = new CapabilityEvaluatorRegistry([
      {
        capabilityId: "core.glyph-run",
        evaluate(node): CapabilityContribution {
          return {
            display: { commands: [], hitRegions: [] },
            semantic: {
              entities: [{ id: node.id, label: "Duplicate" }],
              relations: [],
              readingOrder: [],
            },
          };
        },
      },
    ]).freeze();

    expect(() =>
      evaluateScene(
        sceneWithComponents([glyphRun("prompt")], ["prompt"]),
        0,
        { evaluators: registry },
      ),
    ).toThrow('Duplicate semantic entity id "prompt".');
  });

  test("keeps nested component dispatch stable across repeated evaluation", () => {
    const registry = new CapabilityEvaluatorRegistry([
      stubEvaluator("core.glyph-run"),
    ]).freeze();
    const scene = sceneWithComponents(
      [
        {
          kind: "group",
          id: "root",
          geometry: { x: 0, y: 0, width: 200, height: 100 },
          style: {},
          accessibility: { label: "Root" },
          fallback: "Root unavailable",
          sourceMap,
          children: [glyphRun("prompt")],
        },
      ],
      ["prompt"],
    );

    const first = evaluateScene(scene, 3, { evaluators: registry });
    const second = evaluateScene(scene, 3, { evaluators: registry });
    expect(first).toEqual(second);

    const flatten = (commands: readonly DrawCommand[]): readonly string[] =>
      commands.flatMap((command) =>
        command.kind === "group" ||
        command.kind === "clip" ||
        command.kind === "layer"
          ? [command.id, ...flatten(command.children)]
          : [command.id],
      );
    expect(flatten(first.displayList.commands)).toEqual([
      "root",
      "prompt",
      "prompt:text",
    ]);
  });
});
