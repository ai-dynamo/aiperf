/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, test } from "vitest";

import { formatDocument, parseDocument } from "../src/index.js";

describe("symbol definitions", () => {
  test("parses a symbol definition with typed params and an empty body", () => {
    const source = `flow "Demo" as demo {
  language 1

  symbol SemanticEntity(id: EntityId, label: string) {
  }

  scene "S" as s {
    summary "x"
  }
}
`;
    const result = parseDocument(source, "demo.flow");

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.symbols).toHaveLength(1);
    const symbol = result.value.symbols[0];
    expect(symbol).toMatchObject({
      kind: "symbol-definition",
      name: "SemanticEntity",
      params: [
        { kind: "param", name: "id", type: { kind: "type-ref", name: "EntityId" } },
        { kind: "param", name: "label", type: { kind: "type-ref", name: "string" } },
      ],
      body: [],
    });
  });

  test("parses a symbol definition whose body contains nested component calls", () => {
    const source = `flow "Demo" as demo {
  language 1

  symbol Wrapper(id: EntityId) {
    SemanticEntity(id = "e0", label = "CLI")
  }

  scene "S" as s {
    summary "x"
  }
}
`;
    const result = parseDocument(source, "demo.flow");

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    const symbol = result.value.symbols[0];
    expect(symbol?.body).toHaveLength(1);
    expect(symbol?.body[0]).toMatchObject({
      kind: "component-invocation",
      name: "SemanticEntity",
    });
  });

  test("round-trips a symbol definition through the formatter", () => {
    const source = `flow "Demo" as demo {
  language 1

  symbol SemanticEntity(id: EntityId, label: string) {
  }

  scene "S" as s {
    summary "x"
  }
}
`;
    const parsed = parseDocument(source, "demo.flow");
    expect(parsed.ok, JSON.stringify(parsed.diagnostics)).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const formatted = formatDocument(parsed.value);
    expect(formatted).toContain("symbol SemanticEntity(id: EntityId, label: string) {\n  }");

    const reparsed = parseDocument(formatted, "formatted.flow");
    expect(reparsed.ok, JSON.stringify(reparsed.diagnostics)).toBe(true);
    if (!reparsed.ok) {
      return;
    }
    expect(formatDocument(reparsed.value)).toBe(formatted);
  });
});
