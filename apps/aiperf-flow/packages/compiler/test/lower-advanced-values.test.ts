/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  ArgumentValueAst,
  ComponentInvocationAst,
  DocumentAst,
} from "@aiperf/flow-language";
import type { JsonValue, SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { lower } from "../src/lower.js";
import type { LinkedDocument } from "../src/link.js";

function range(): SourceRange {
  return {
    source: "<test>",
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 0, line: 1, column: 1 },
  };
}

function linkedWithInvocation(
  invocation: ComponentInvocationAst,
  requiredCapability?: string,
): LinkedDocument {
  const document: DocumentAst = {
    kind: "document",
    id: "advanced-values",
    title: "Advanced values",
    language: { kind: "language", version: 1, sourceMap: range() },
    imports: [],
    requirements:
      requiredCapability === undefined
        ? []
        : [
            {
              kind: "requirement",
              capability: requiredCapability,
              versionRange: "^1.0.0",
              sourceMap: range(),
            },
          ],
    tokens: [],
    symbols: [],
    scenes: [
      {
        kind: "scene",
        id: "main",
        title: "Main",
        renderDeclarations: [invocation],
        cameras: [],
        timelines: [],
        interactions: [],
        responsiveVariants: [],
        sourceMap: range(),
      },
    ],
    sourceMap: range(),
  };

  return {
    document,
    tokens: new Map(),
    scenes: new Map([["main", { nodes: new Map() }]]),
  };
}

function invocation(
  name: string,
  props: Readonly<Record<string, ArgumentValueAst>>,
): ComponentInvocationAst {
  return {
    kind: "component-invocation",
    name,
    props: Object.entries(props).map(([propName, value]) => ({
      kind: "prop-assignment",
      name: propName,
      value,
      sourceMap: range(),
    })),
    sourceMap: range(),
  };
}

function literal(value: string | number | boolean): ArgumentValueAst {
  return { kind: "literal", value, sourceMap: range() };
}

function componentProps(
  name: string,
  props: Readonly<Record<string, ArgumentValueAst>>,
): Readonly<Record<string, JsonValue>> {
  const root = lower(linkedWithInvocation(invocation(name, props))).scenes[0]?.roots[0];
  if (root?.kind !== "component") {
    throw new Error("Expected a lowered component.");
  }
  return root.props;
}

describe("advanced component value lowering", () => {
  test("lowers canonical component ids and JSON-safe argument values", () => {
    const linked = linkedWithInvocation(
      invocation("GlyphRun", {
        id: literal("run"),
        label: literal("Prompt"),
        direction: {
          kind: "identifier-reference",
          name: "ltr",
          sourceMap: range(),
        },
        options: {
          kind: "object-literal",
          properties: [
            {
              kind: "object-property",
              name: "wrap",
              value: literal(true),
              sourceMap: range(),
            },
            {
              kind: "object-property",
              name: "overflow",
              value: {
                kind: "identifier-reference",
                name: "clip",
                sourceMap: range(),
              },
              sourceMap: range(),
            },
          ],
          sourceMap: range(),
        },
      }),
      "core.glyph-run",
    );

    const root = lower(linked).scenes[0]?.roots[0];
    expect(root).toMatchObject({
      kind: "component",
      capabilityId: "core.glyph-run",
      props: {
        id: "run",
        label: "Prompt",
        direction: "ltr",
        options: { wrap: true, overflow: "clip" },
      },
    });
  });

  test.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    "rejects non-finite component values: %s",
    (value) => {
      expect(() =>
        componentProps("GlyphRun", { id: literal("run"), invalid: literal(value) }),
      ).toThrow(/finite JSON number/);
    },
  );
});
