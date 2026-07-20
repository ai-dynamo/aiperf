/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type {
  ComponentInvocationAst,
  DocumentAst,
  ImportDeclarationAst,
  SceneAst,
  SymbolDefinitionAst,
  UseThemeAst,
} from "../language/ast.js";
import { link, type ResolvedModule } from "./link.js";

const SOURCE_MAP = {
  source: "link.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function emptyScene(id = "scene"): SceneAst {
  return {
    kind: "scene",
    id,
    title: "Scene",
    sourceMap: SOURCE_MAP,
    renderDeclarations: [],
    cameras: [],
    timelines: [],
    interactions: [],
    responsiveVariants: [],
  };
}

function baseDocument(
  overrides: Partial<DocumentAst> = {},
): DocumentAst {
  return {
    kind: "document",
    id: "doc",
    title: "Doc",
    sourceMap: SOURCE_MAP,
    language: { kind: "language", version: 1, sourceMap: SOURCE_MAP },
    requirements: [],
    tokens: [],
    themes: [],
    symbols: [],
    scenes: [emptyScene()],
    ...overrides,
  };
}

function importDecl(path: string, alias: string): ImportDeclarationAst {
  return {
    kind: "import",
    path,
    alias,
    sourceMap: SOURCE_MAP,
  };
}

function useTheme(themeId: string): UseThemeAst {
  return { kind: "use-theme", themeId, sourceMap: SOURCE_MAP };
}

function invocation(
  name: string,
  namespace?: string,
): ComponentInvocationAst {
  return {
    kind: "component-invocation",
    name,
    ...(namespace === undefined ? {} : { namespace }),
    props: [],
    sourceMap: SOURCE_MAP,
  };
}

function symbolDef(
  name: string,
  body: ComponentInvocationAst[] = [],
): SymbolDefinitionAst {
  return {
    kind: "symbol-definition",
    name,
    params: [],
    body,
    sourceMap: SOURCE_MAP,
  };
}

describe("link theme flattening (B1)", () => {
  it("does not double-count themes when the same module is imported under two aliases", () => {
    const sharedModule: ResolvedModule = {
      canonicalUri: "file:///shared.flow",
      exports: new Set(["Widget"]),
      themes: [
        {
          kind: "theme-declaration",
          id: "shared-theme",
          extends: "base",
          assignments: [],
          sourceMap: SOURCE_MAP,
        },
      ],
      useTheme: useTheme("shared-theme"),
    };

    const document = baseDocument({
      imports: [
        importDecl("./shared.flow", "alpha"),
        importDecl("./shared.flow", "beta"),
      ],
    });

    const result = link(document, {
      resolveModule: () => sharedModule,
    });

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }
    expect(
      result.diagnostics.some((d) => d.code === "THEME_DUPLICATE_DEFAULT"),
    ).toBe(false);
    expect(result.value.themes).toHaveLength(1);
    expect(result.value.themes[0]?.id).toBe("shared-theme");
    expect(result.value.useTheme?.themeId).toBe("shared-theme");
  });

  it("still reports THEME_DUPLICATE_DEFAULT for distinct modules each declaring a default", () => {
    const first: ResolvedModule = {
      canonicalUri: "file:///first.flow",
      exports: new Set(),
      useTheme: useTheme("theme-a"),
    };
    const second: ResolvedModule = {
      canonicalUri: "file:///second.flow",
      exports: new Set(),
      useTheme: useTheme("theme-b"),
    };

    const document = baseDocument({
      imports: [
        importDecl("./first.flow", "a"),
        importDecl("./second.flow", "b"),
      ],
    });

    const result = link(document, {
      resolveModule: ({ path }) =>
        path.includes("first") ? first : second,
    });

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some((d) => d.code === "THEME_DUPLICATE_DEFAULT"),
    ).toBe(true);
  });
});

describe("link local component/symbol resolution (B2)", () => {
  it("emits LINK_UNKNOWN_NAME for a mistyped local component invocation", () => {
    const document = baseDocument({
      symbols: [symbolDef("WidgetFunc")],
      scenes: [
        {
          ...emptyScene(),
          renderDeclarations: [invocation("Widgfunc")],
        },
      ],
    });

    const result = link(document);

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some(
        (d) =>
          d.code === "LINK_UNKNOWN_NAME" &&
          d.message.includes("Widgfunc"),
      ),
    ).toBe(true);
  });

  it("accepts an unqualified invocation that matches a local symbol", () => {
    const document = baseDocument({
      symbols: [symbolDef("WidgetFunc")],
      scenes: [
        {
          ...emptyScene(),
          renderDeclarations: [invocation("WidgetFunc")],
        },
      ],
    });

    const result = link(document);

    expect(result.ok).toBe(true);
  });

  it("emits LINK_UNKNOWN_NAME for a mistyped local call inside a symbol body", () => {
    const document = baseDocument({
      symbols: [
        symbolDef("Outer", [invocation("MissingInner")]),
        symbolDef("Inner"),
      ],
    });

    const result = link(document);

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some(
        (d) =>
          d.code === "LINK_UNKNOWN_NAME" &&
          d.message.includes("MissingInner"),
      ),
    ).toBe(true);
  });
});
