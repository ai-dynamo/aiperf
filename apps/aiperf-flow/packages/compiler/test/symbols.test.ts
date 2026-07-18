/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { parseDocument } from "@aiperf/flow-language";
import type { SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  collectSymbols,
  type SymbolDeclarationAst,
} from "../src/symbols.js";
import { FOUNDATION_SOURCE } from "./fixture.js";

function range(): SourceRange {
  return {
    source: "<test>",
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 0, line: 1, column: 1 },
  };
}

function makeSymbol(name: string): SymbolDeclarationAst {
  return {
    kind: "symbol-definition",
    name,
    params: [],
    body: [],
    sourceMap: range(),
  };
}

function parsedDocument() {
  const parsed = parseDocument(FOUNDATION_SOURCE, "request-flow.flow");
  if (!parsed.ok) {
    throw new Error(
      `Expected the foundation source to parse: ${JSON.stringify(parsed.diagnostics)}`,
    );
  }
  return parsed.value;
}

describe("collectSymbols", () => {
  test("returns an empty table for a document that declares no symbols", () => {
    const result = collectSymbols(parsedDocument());

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.size).toBe(0);
    }
  });

  test("collects declared symbols into a table keyed by export name", () => {
    const document = {
      ...parsedDocument(),
      symbols: [makeSymbol("TokenSpanMorph"), makeSymbol("PromptSegmentComposer")],
    };

    const result = collectSymbols(document);

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(result.value.size).toBe(2);
    expect(result.value.get("TokenSpanMorph")?.name).toBe("TokenSpanMorph");
    expect(result.value.has("PromptSegmentComposer")).toBe(true);
  });

  test("diagnoses duplicate symbol exports with SYMBOL_DUPLICATE_EXPORT", () => {
    const document = {
      ...parsedDocument(),
      symbols: [makeSymbol("Queue"), makeSymbol("Queue")],
    };

    const result = collectSymbols(document);

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "SYMBOL_DUPLICATE_EXPORT",
          severity: "error",
        }),
      ]),
    );
  });

  test("collects symbols from parsed source that declares symbol definitions", () => {
    const source = `flow "Demo" as demo {
  language 1

  symbol SemanticEntity(id: EntityId, label: string) {
  }

  scene "S" as s {
    summary "A scene long enough for validation."
    reading-order cli
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
    narrate "This scene includes a symbol declaration for export-table tests."
    fallback "Scene."
  }
}
`;
    const parsed = parseDocument(source, "demo.flow");
    expect(parsed.ok, JSON.stringify(parsed.diagnostics)).toBe(true);
    if (!parsed.ok) {
      return;
    }

    const result = collectSymbols(parsed.value);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.get("SemanticEntity")?.params.map(({ name }) => name)).toEqual([
        "id",
        "label",
      ]);
    }
  });
});
