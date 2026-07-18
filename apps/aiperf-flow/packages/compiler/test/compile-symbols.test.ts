/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { FOUNDATION_CAPABILITIES } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { compileSource } from "../src/index.js";

function source(
  declarations: string,
  renderDeclarations = "",
): string {
  return `flow "Symbol pipeline" as symbol-pipeline {
  language 1
  require core.rect "^1.0.0"

${declarations}

  scene "Symbol scene" as symbol-scene {
    summary "A scene that exercises symbol collection and expansion."

${renderDeclarations}
    rect anchor {
      x 0
      y 0
      width 10
      height 10
      fill "#000000"
      label "Anchor"
      role "img"
      description "Anchor node"
      fallback "Anchor"
    }

    narrate "This scene exercises the complete compiler symbol pipeline."
    reading-order anchor
    fallback "Symbol pipeline scene."
  }
}
`;
}

function compile(sourceText: string) {
  return compileSource({
    source: sourceText,
    sourceName: "compile-symbols.flow",
    capabilities: FOUNDATION_CAPABILITIES,
    strict: false,
  });
}

describe("compileSource symbol collection and expansion", () => {
  test("removes an invocation whose collected symbol has an empty body", () => {
    const result = compile(
      source(
        `  symbol EmptyMarker() {
  }`,
        "    EmptyMarker()\n",
      ),
    );

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.scenes[0]?.roots.map(({ id }) => id)).toEqual(["anchor"]);
    expect(result.value.scenes[0]?.roots).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ capabilityId: "EmptyMarker" }),
      ]),
    );
  });

  test("returns duplicate symbol diagnostics before later compiler stages", () => {
    const result = compile(
      source(`  symbol Repeated() {
  }

  symbol Repeated() {
  }`),
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        code: "SYMBOL_DUPLICATE_EXPORT",
        severity: "error",
        message: 'Duplicate symbol export "Repeated".',
        repair: 'Rename this symbol or remove the earlier "Repeated" declaration.',
        range: expect.objectContaining({
          source: "compile-symbols.flow",
          start: expect.objectContaining({ line: 8 }),
        }),
      }),
    ]);
  });

  test("expands a flat symbol body through the compile pipeline", () => {
    const result = compile(
      source(
        `  symbol Wrapper() {
    SemanticEntity(id = "entity", label = "Entity")
  }`,
        "    Wrapper()\n",
      ),
    );

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.scenes[0]?.roots.map((root) =>
      root.kind === "component" ? root.capabilityId : root.id,
    )).toEqual([
      "SemanticEntity",
      "anchor",
    ]);
    expect(result.value.scenes[0]?.roots[0]).toMatchObject({
      kind: "component",
      capabilityId: "SemanticEntity",
      props: {
        id: "entity",
        label: "Entity",
      },
    });
  });

  test("fails closed at parsing for parameter-bound symbol bodies", () => {
    const result = compile(
      source(
        `  symbol LabeledEntity(id: string, label: string) {
    SemanticEntity(id = id, label = label)
  }`,
        '    LabeledEntity(id = "entity", label = "Entity")\n',
      ),
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "PARSE_UNEXPECTED_TOKEN",
          severity: "error",
          range: expect.objectContaining({ source: "compile-symbols.flow" }),
        }),
      ]),
    );
    expect(result.diagnostics).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "SYMBOL_EXPANSION_UNSUPPORTED" }),
      ]),
    );
  });
});
