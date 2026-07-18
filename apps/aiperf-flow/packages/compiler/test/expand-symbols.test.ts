/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { parseDocument, type DocumentAst } from "@aiperf/flow-language";
import type { SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { expandSymbolInvocations } from "../src/expand-symbols.js";
import {
  collectSymbols,
  type SymbolDeclarationAst,
  type SymbolTable,
} from "../src/symbols.js";
import { FOUNDATION_SOURCE } from "./fixture.js";

function range(): SourceRange {
  return {
    source: "<test>",
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 0, line: 1, column: 1 },
  };
}

function table(...declarations: readonly SymbolDeclarationAst[]): SymbolTable {
  return new Map(
    declarations.map((declaration) => [declaration.name, declaration]),
  );
}

function document(): DocumentAst {
  const result = parseDocument(FOUNDATION_SOURCE, "<test>");
  if (!result.ok) {
    throw new Error(`Expected fixture to parse: ${JSON.stringify(result.diagnostics)}`);
  }
  return result.value;
}

describe("expandSymbolInvocations", () => {
  test("returns the document unchanged when the symbol table is empty", () => {
    const parsed = document();

    const result = expandSymbolInvocations(parsed, table());

    expect(result).toEqual({ ok: true, value: parsed, diagnostics: [] });
  });

  test("returns the document unchanged when declared symbols have empty bodies", () => {
    const parsed = document();
    const symbols = table({
      kind: "symbol-definition",
      name: "Queue",
      params: [],
      body: [],
      sourceMap: range(),
    });

    const result = expandSymbolInvocations(parsed, symbols);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toEqual(parsed);
    }
  });

  test("expands a matching empty symbol invocation to no render declarations", () => {
    const parsed = document();
    const invocation = {
      kind: "component-invocation" as const,
      name: "Queue",
      props: [],
      sourceMap: range(),
    };
    const withSymbolInvocation = {
      ...parsed,
      scenes: [
        {
          ...parsed.scenes[0]!,
          renderDeclarations: [
            invocation,
            ...parsed.scenes[0]!.renderDeclarations,
          ],
        },
      ],
    };
    const symbols = table({
      kind: "symbol-definition",
      name: "Queue",
      params: [],
      body: [],
      sourceMap: range(),
    });

    const result = expandSymbolInvocations(withSymbolInvocation, symbols);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.scenes[0]?.renderDeclarations).not.toContain(invocation);
      expect(result.value.scenes[0]?.renderDeclarations).toEqual(
        parsed.scenes[0]?.renderDeclarations,
      );
    }
  });

  test("expands a flat non-empty symbol body at the invocation site", () => {
    const parsed = document();
    const invocation = {
      kind: "component-invocation" as const,
      name: "Queue",
      props: [],
      sourceMap: range(),
    };
    const withSymbolInvocation = {
      ...parsed,
      scenes: [
        {
          ...parsed.scenes[0]!,
          renderDeclarations: [
            invocation,
            ...parsed.scenes[0]!.renderDeclarations,
          ],
        },
      ],
    };
    const symbols = table({
      kind: "symbol-definition",
      name: "Queue",
      params: [],
      body: [
        {
          kind: "component-invocation",
          name: "Gauge",
          props: [],
          sourceMap: range(),
        },
      ],
      sourceMap: range(),
    });

    const result = expandSymbolInvocations(withSymbolInvocation, symbols);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.scenes[0]?.renderDeclarations).not.toContain(invocation);
      expect(result.value.scenes[0]?.renderDeclarations[0]).toMatchObject({
        kind: "component-invocation",
        name: "Gauge",
      });
    }
  });

  test("preserves duplicate detection when collecting symbols before expansion", () => {
    const parsed = document();
    const duplicate = {
      ...parsed,
      symbols: [
        {
          kind: "symbol-definition" as const,
          name: "Queue",
          params: [],
          body: [],
          sourceMap: range(),
        },
        {
          kind: "symbol-definition" as const,
          name: "Queue",
          params: [],
          body: [],
          sourceMap: range(),
        },
      ],
    };

    const result = collectSymbols(duplicate);

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "SYMBOL_DUPLICATE_EXPORT" }),
      ]),
    );
  });
});
