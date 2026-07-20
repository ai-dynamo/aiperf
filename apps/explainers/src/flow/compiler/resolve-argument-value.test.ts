/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type {
  ArgumentValueAst,
  ComponentInvocationAst,
  DocumentAst,
  PropAssignmentAst,
  SceneAst,
} from "../language/ast.js";
import type { LinkedDocument } from "./link.js";
import { lower } from "./lower.js";
import { lowerExplainerScene } from "./lower-explainer-scene.js";

const SOURCE_MAP = {
  source: "resolve-argument-value.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function literal(value: string | number | boolean): ArgumentValueAst {
  return { kind: "literal", value, sourceMap: SOURCE_MAP };
}

function prop(name: string, value: ArgumentValueAst): PropAssignmentAst {
  return { kind: "prop-assignment", name, value, sourceMap: SOURCE_MAP };
}

function arrayLiteral(
  items: readonly ArgumentValueAst[],
): ArgumentValueAst {
  return { kind: "array-literal", items, sourceMap: SOURCE_MAP };
}

function ref(target: string): ArgumentValueAst {
  const dot = target.indexOf(".");
  return {
    kind: "ref",
    target,
    instance: dot === -1 ? target : target.slice(0, dot),
    port: dot === -1 ? "" : target.slice(dot + 1),
    sourceMap: SOURCE_MAP,
  };
}

function objectLiteral(
  properties: Readonly<Record<string, ArgumentValueAst>>,
): ArgumentValueAst {
  return {
    kind: "object-literal",
    properties: Object.entries(properties).map(([name, value]) => ({
      kind: "object-property" as const,
      name,
      value,
      sourceMap: SOURCE_MAP,
    })),
    sourceMap: SOURCE_MAP,
  };
}

describe("resolveArgumentValue array-literal and ref", () => {
  it("preserves array literals and refs when lowering component props", () => {
    const invocation: ComponentInvocationAst = {
      kind: "component-invocation",
      name: "Widget",
      sourceMap: SOURCE_MAP,
      props: [
        prop("id", literal("widget")),
        prop(
          "columns",
          arrayLiteral([
            literal("layout"),
            objectLiteral({ width: literal(120) }),
            arrayLiteral([literal("nested"), ref("controller.output")]),
          ]),
        ),
        prop("endpoint", ref("cells.worker.0")),
      ],
    };

    const scene: SceneAst = {
      kind: "scene",
      id: "scene",
      title: "Scene",
      sourceMap: SOURCE_MAP,
      renderDeclarations: [invocation],
      cameras: [],
      timelines: [],
      interactions: [],
      responsiveVariants: [],
    };

    const document: DocumentAst = {
      kind: "document",
      id: "doc",
      title: "Doc",
      sourceMap: SOURCE_MAP,
      language: { kind: "language", version: 1, sourceMap: SOURCE_MAP },
      requirements: [],
      tokens: [],
      themes: [],
      symbols: [],
      scenes: [scene],
    };

    const linked: LinkedDocument = {
      document,
      tokens: new Map(),
      scenes: new Map([["scene", { nodes: new Map() }]]),
      imports: new Map(),
      qualifiedNames: new Map(),
      themes: [],
    };

    const ir = lower(linked);
    const root = ir.scenes[0]?.roots[0];
    expect(root?.kind).toBe("component");
    if (root?.kind !== "component") {
      return;
    }

    expect(root.props.columns).toEqual([
      "layout",
      { width: 120 },
      ["nested", { ref: "controller.output" }],
    ]);
    expect(root.props.endpoint).toEqual({ ref: "cells.worker.0" });
  });

  it("preserves array literals and refs when lowering explainer scenes", () => {
    // `fade` forces the package-scene path in lowerExplainerScene, which uses
    // that module's resolveArgumentValue (not lower.ts).
    const invocation: ComponentInvocationAst = {
      kind: "component-invocation",
      name: "core.stepper",
      sourceMap: SOURCE_MAP,
      props: [
        prop("id", literal("steps")),
        prop(
          "steps",
          arrayLiteral([
            literal("layout"),
            literal("slots"),
            literal("timeline"),
          ]),
        ),
        prop("source", ref("controller.output")),
      ],
    };

    const scene: SceneAst = {
      kind: "scene",
      id: "scene",
      title: "Scene",
      sourceMap: SOURCE_MAP,
      renderDeclarations: [invocation],
      cameras: [],
      timelines: [
        {
          kind: "timeline",
          id: "main",
          sourceMap: SOURCE_MAP,
          cues: [
            {
              kind: "timeline-cue",
              sourceMap: SOURCE_MAP,
              timing: { mode: "at", ms: 0 },
              action: "fade",
              target: "steps",
              duration: 200,
            },
          ],
        },
      ],
      interactions: [],
      responsiveVariants: [],
    };

    const result = lowerExplainerScene(scene);
    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    const root = result.value.scene.roots[0];
    expect(root).toMatchObject({
      id: "steps",
      capabilityId: "core.stepper",
      props: {
        steps: ["layout", "slots", "timeline"],
        source: { ref: "controller.output" },
      },
    });
  });
});
