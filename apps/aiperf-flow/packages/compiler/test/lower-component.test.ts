/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ComponentInvocationAst, DocumentAst } from "@aiperf/flow-language";
import {
  FOUNDATION_CAPABILITIES,
  safeParseFlowIr,
  type SourceRange,
} from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { compileSource } from "../src/index.js";
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
): LinkedDocument {
  const document: DocumentAst = {
    kind: "document",
    id: "demo",
    title: "Demo",
    language: { kind: "language", version: 1, sourceMap: range() },
    requirements: [],
    tokens: [],
    symbols: [],
    scenes: [
      {
        kind: "scene",
        id: "solo",
        title: "Solo",
        summary: { kind: "summary", text: "Component lowering.", sourceMap: range() },
        renderDeclarations: [invocation],
        cameras: [],
        timelines: [],
        interactions: [],
        responsiveVariants: [],
        narration: {
          kind: "narration",
          text: "A scene used only to exercise component capability lowering.",
          sourceMap: range(),
        },
        readingOrder: {
          kind: "reading-order",
          references: [],
          sourceMap: range(),
        },
        fallback: { kind: "fallback", text: "Component.", sourceMap: range() },
        sourceMap: range(),
      },
    ],
    sourceMap: range(),
  };

  return {
    document,
    tokens: new Map(),
    scenes: new Map([["solo", { nodes: new Map() }]]),
  };
}

function invocation(
  name: string,
  props: ReadonlyArray<readonly [string, string | number | boolean]>,
): ComponentInvocationAst {
  return {
    kind: "component-invocation",
    name,
    props: props.map(([propName, value]) => ({
      kind: "prop-assignment",
      name: propName,
      value: { kind: "literal", value, sourceMap: range() },
      sourceMap: range(),
    })),
    sourceMap: range(),
  };
}

describe("lowerComponentInvocation capabilityId resolution", () => {
  test("prefers an explicit capabilityId prop over the invocation name", () => {
    const ir = lower(
      linkedWithInvocation(
        invocation("SpanMap", [
          ["capabilityId", "core.span-map"],
          ["id", "tok"],
          ["label", "Tokens"],
        ]),
      ),
    );

    const parsed = safeParseFlowIr(ir);
    expect(parsed.ok, JSON.stringify(parsed.diagnostics)).toBe(true);

    const component = ir.scenes[0]?.roots[0];
    expect(component).toMatchObject({
      kind: "component",
      id: "tok",
      capabilityId: "core.span-map",
      accessibility: { label: "Tokens" },
    });
  });

  test("uses a dotted invocation name as the capability id", () => {
    const ir = lower(
      linkedWithInvocation(
        invocation("core.span-map", [
          ["id", "tok"],
          ["label", "Tokens"],
        ]),
      ),
    );

    const parsed = safeParseFlowIr(ir);
    expect(parsed.ok, JSON.stringify(parsed.diagnostics)).toBe(true);

    expect(ir.scenes[0]?.roots[0]).toMatchObject({
      kind: "component",
      capabilityId: "core.span-map",
    });
  });

  test("keeps a PascalCase symbol name when no prop or dotted id is present", () => {
    const ir = lower(
      linkedWithInvocation(
        invocation("SpanMap", [
          ["id", "tok"],
          ["label", "Tokens"],
        ]),
      ),
    );

    const parsed = safeParseFlowIr(ir);
    expect(parsed.ok, JSON.stringify(parsed.diagnostics)).toBe(true);

    expect(ir.scenes[0]?.roots[0]).toMatchObject({
      kind: "component",
      capabilityId: "SpanMap",
    });
  });
});

describe("compileSource component invocation lowering", () => {
  test("compiles a component node whose capabilityId comes from props", () => {
    const source = `flow "Demo" as demo {
  language 1
  require core.rect "^1.0.0"

  scene "Solo" as solo {
    summary "A scene with one component and one rect for reading order."

    SpanMap(capabilityId = "core.span-map", id = "tok", label = "Tokens")

    rect box {
      x 0
      y 0
      width 10
      height 10
      fill "#000000"
      label "Box"
      description "Anchor rect"
      fallback "Box"
    }

    narrate "This scene exercises component capability id prop lowering."
    reading-order box
    fallback "Tokens beside a box."
  }
}
`;

    const result = compileSource({
      source,
      sourceName: "component-lower.flow",
      capabilities: FOUNDATION_CAPABILITIES,
      strict: false,
    });

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    const component = result.value.scenes[0]?.roots.find(
      (node) => node.kind === "component",
    );
    expect(component).toMatchObject({
      kind: "component",
      id: "tok",
      capabilityId: "core.span-map",
      props: {
        capabilityId: "core.span-map",
        id: "tok",
        label: "Tokens",
      },
      accessibility: { label: "Tokens" },
    });
  });
});
