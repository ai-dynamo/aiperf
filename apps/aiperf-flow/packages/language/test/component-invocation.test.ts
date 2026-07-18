/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, test } from "vitest";

import { formatDocument, parseDocument } from "../src/index.js";

const SOURCE = `flow "Demo" as demo {
  language 1

  symbol SemanticEntity(id: EntityId, label: string) {
  }

  scene "S" as s {
    summary "x"

    SemanticEntity(id = "e0", label = "CLI")

    rect cli {
      x 0
      y 0
      width 10
      height 10
      fill "#000"
      label "CLI"
      role "img"
      description "CLI"
      fallback "CLI"
    }
  }
}
`;

describe("component invocations", () => {
  test("parses a component invocation as a scene render declaration alongside rect", () => {
    const result = parseDocument(SOURCE, "demo.flow");

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    const scene = result.value.scenes[0];
    expect(scene?.renderDeclarations.map(({ kind }) => kind)).toEqual([
      "component-invocation",
      "rect",
    ]);

    const invocation = scene?.renderDeclarations[0];
    expect(invocation).toMatchObject({
      kind: "component-invocation",
      name: "SemanticEntity",
      props: [
        {
          kind: "prop-assignment",
          name: "id",
          value: { kind: "literal", value: "e0" },
        },
        {
          kind: "prop-assignment",
          name: "label",
          value: { kind: "literal", value: "CLI" },
        },
      ],
    });
  });

  test("round-trips a component invocation through the formatter", () => {
    const parsed = parseDocument(SOURCE, "demo.flow");
    expect(parsed.ok, JSON.stringify(parsed.diagnostics)).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const formatted = formatDocument(parsed.value);
    expect(formatted).toContain('SemanticEntity(id = "e0", label = "CLI")');

    const reparsed = parseDocument(formatted, "formatted.flow");
    expect(reparsed.ok, JSON.stringify(reparsed.diagnostics)).toBe(true);
    if (!reparsed.ok) {
      return;
    }
    expect(formatDocument(reparsed.value)).toBe(formatted);
  });

  test("supports numeric and token-reference prop values", () => {
    const source = `flow "Demo" as demo {
  language 1
  token accent = "#7aa2f7"

  scene "S" as s {
    summary "x"

    Gauge(value = 42, tint = token(accent))
  }
}
`;
    const result = parseDocument(source, "demo.flow");
    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    const invocation = result.value.scenes[0]?.renderDeclarations[0];
    expect(invocation).toMatchObject({
      kind: "component-invocation",
      name: "Gauge",
      props: [
        { name: "value", value: { kind: "literal", value: 42 } },
        { name: "tint", value: { kind: "token-reference", token: "accent" } },
      ],
    });
  });
});
