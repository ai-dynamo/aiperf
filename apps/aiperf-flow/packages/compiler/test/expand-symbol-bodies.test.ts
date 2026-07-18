/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type ComponentInvocationAst,
  type DocumentAst,
  type SymbolDefinitionAst,
} from "@aiperf/flow-language";
import type { SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { expandSymbolInvocations } from "../src/expand-symbols.js";
import type { SymbolTable } from "../src/symbols.js";

function range(source: string): SourceRange {
  return {
    source,
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 1, line: 1, column: 2 },
  };
}

function document(...renderDeclarations: readonly ComponentInvocationAst[]): DocumentAst {
  return {
    kind: "document",
    title: "Test",
    id: "test",
    language: { kind: "language", version: 1, sourceMap: range("language") },
    requirements: [],
    tokens: [],
    symbols: [],
    scenes: [
      {
        kind: "scene",
        title: "Scene",
        id: "scene",
        renderDeclarations,
        cameras: [],
        timelines: [],
        interactions: [],
        responsiveVariants: [],
        sourceMap: range("scene"),
      },
    ],
    sourceMap: range("document"),
  };
}

function invocation(
  name: string,
  props: ComponentInvocationAst["props"] = [],
  source = `call:${name}`,
): ComponentInvocationAst {
  return { kind: "component-invocation", name, props, sourceMap: range(source) };
}

function literalProp(
  name: string,
  value: string | number | boolean,
  source = `argument:${name}`,
): ComponentInvocationAst["props"][number] {
  return {
    kind: "prop-assignment",
    name,
    value: { kind: "literal", value, sourceMap: range(source) },
    sourceMap: range(`assignment:${name}`),
  };
}

function parameterProp(
  name: string,
  parameter: string,
  source = `reference:${parameter}`,
): ComponentInvocationAst["props"][number] {
  return {
    kind: "prop-assignment",
    name,
    value: {
      kind: "identifier-reference",
      name: parameter,
      sourceMap: range(source),
    },
    sourceMap: range(`assignment:${name}`),
  } as ComponentInvocationAst["props"][number];
}

function symbol(
  name: string,
  params: readonly [name: string, type: string][],
  body: SymbolDefinitionAst["body"],
): SymbolDefinitionAst {
  return {
    kind: "symbol-definition",
    name,
    params: params.map(([paramName, type]) => ({
      kind: "param",
      name: paramName,
      type: { kind: "type-ref", name: type, sourceMap: range(`type:${type}`) },
      sourceMap: range(`parameter:${paramName}`),
    })),
    body,
    sourceMap: range(`symbol:${name}`),
  };
}

function table(...symbols: readonly SymbolDefinitionAst[]): SymbolTable {
  return new Map(symbols.map((entry) => [entry.name, entry]));
}

describe("flat symbol body expansion", () => {
  test("preserves empty-body erasure without mutating authored inputs", () => {
    const call = invocation("Empty");
    const definition = symbol("Empty", [], []);
    const input = document(call);
    const beforeInput = structuredClone(input);
    const beforeDefinition = structuredClone(definition);

    const first = expandSymbolInvocations(input, table(definition));
    const second = expandSymbolInvocations(input, table(definition));

    expect(first).toEqual(second);
    expect(first.ok).toBe(true);
    if (first.ok) {
      expect(first.value.scenes[0]!.renderDeclarations).toEqual([]);
    }
    expect(input).toEqual(beforeInput);
    expect(definition).toEqual(beforeDefinition);
  });

  test("expands in authored order and binds named parameters with provenance", () => {
    const firstBody = invocation(
      "SemanticEntity",
      [parameterProp("label", "label"), parameterProp("id", "id")],
      "body:first",
    );
    const secondBody = invocation(
      "Gauge",
      [parameterProp("value", "count")],
      "body:second",
    );
    const definition = symbol(
      "Summary",
      [
        ["id", "EntityId"],
        ["label", "string"],
        ["count", "number"],
      ],
      [firstBody, secondBody],
    );
    const call = invocation("Summary", [
      literalProp("count", 7, "call:count"),
      literalProp("id", "queue", "call:id"),
      literalProp("label", "Ready", "call:label"),
    ]);

    const result = expandSymbolInvocations(document(call), table(definition));

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const expanded = result.value.scenes[0]!.renderDeclarations;
    expect(expanded.map((entry) => entry.kind === "component-invocation" && entry.name))
      .toEqual(["SemanticEntity", "Gauge"]);
    expect(expanded[0]).toMatchObject({
      sourceMap: range("body:first"),
      props: [
        { name: "label", value: { value: "Ready", sourceMap: range("call:label") } },
        { name: "id", value: { value: "queue", sourceMap: range("call:id") } },
      ],
    });
    expect(expanded[1]).toMatchObject({
      sourceMap: range("body:second"),
      props: [{ name: "value", value: { value: 7, sourceMap: range("call:count") } }],
    });
  });

  test("recursively expands symbols while allowing independent reuse", () => {
    const leaf = symbol(
      "Leaf",
      [["label", "string"]],
      [invocation("Text", [parameterProp("text", "label")], "body:leaf")],
    );
    const wrapper = symbol(
      "Wrapper",
      [["label", "string"]],
      [invocation("Leaf", [parameterProp("label", "label")], "body:wrapper")],
    );
    const first = invocation("Wrapper", [literalProp("label", "First")]);
    const second = invocation("Wrapper", [literalProp("label", "Second")]);

    const result = expandSymbolInvocations(
      document(first, second),
      table(leaf, wrapper),
    );

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    const expanded = result.value.scenes[0]!.renderDeclarations;
    expect(expanded).toHaveLength(2);
    expect(expanded[0]).toMatchObject({
      name: "Text",
      props: [{ value: { value: "First" } }],
    });
    expect(expanded[1]).toMatchObject({
      name: "Text",
      props: [{ value: { value: "Second" } }],
    });
    expect(expanded[0]).not.toBe(expanded[1]);
    expect((expanded[0] as ComponentInvocationAst).props).not.toBe(
      (expanded[1] as ComponentInvocationAst).props,
    );
  });

  test.each([
    {
      name: "direct",
      symbols: () => {
        const recursive = symbol("A", [], [invocation("A")]);
        return table(recursive);
      },
    },
    {
      name: "indirect",
      symbols: () => {
        const a = symbol("A", [], [invocation("B")]);
        const b = symbol("B", [], [invocation("A")]);
        return table(a, b);
      },
    },
  ])("reports SYMBOL_EXPANSION_CYCLE for $name recursion", ({ symbols }) => {
    const result = expandSymbolInvocations(document(invocation("A")), symbols());

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "SYMBOL_EXPANSION_CYCLE",
          range: range("call:A"),
        }),
      ]),
    );
  });

  test("rejects unknown parameter references at their authored source", () => {
    const definition = symbol(
      "Broken",
      [],
      [invocation("Text", [parameterProp("text", "missing", "bad-reference")])],
    );

    const result = expandSymbolInvocations(
      document(invocation("Broken")),
      table(definition),
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "SYMBOL_UNKNOWN_PARAMETER",
          range: range("bad-reference"),
        }),
      ]),
    );
  });

  test.each(["slot", "for-loop"])(
    "fails closed for unsupported %s body constructs",
    (kind) => {
      const definition = symbol("Unsupported", [], [
        {
          kind,
          sourceMap: range(`body:${kind}`),
        } as SymbolDefinitionAst["body"][number],
      ]);

      const result = expandSymbolInvocations(
        document(invocation("Unsupported")),
        table(definition),
      );

      expect(result.ok).toBe(false);
      expect(result.diagnostics).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            code: "SYMBOL_EXPANSION_UNSUPPORTED",
            range: range(`body:${kind}`),
          }),
        ]),
      );
    },
  );

  test.each([
    {
      name: "unknown props",
      call: invocation("Typed", [
        literalProp("label", "ok"),
        literalProp("extra", true),
      ]),
      code: "STRICT_UNKNOWN_PROP",
    },
    {
      name: "missing props",
      call: invocation("Typed"),
      code: "PROP_MISSING_REQUIRED",
    },
    {
      name: "mismatched props",
      call: invocation("Typed", [literalProp("label", 5)]),
      code: "PROP_TYPE_MISMATCH",
    },
  ])("rejects $name through strict binding validation", ({ call, code }) => {
    const definition = symbol("Typed", [["label", "string"]], []);

    const result = expandSymbolInvocations(document(call), table(definition));

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([expect.objectContaining({ code })]),
    );
  });
});
