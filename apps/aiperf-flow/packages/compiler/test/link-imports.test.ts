/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { DocumentAst } from "@aiperf/flow-language";
import type { SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test, vi } from "vitest";

import {
  link,
  type ModuleImportAst,
  type ModuleResolver,
} from "../src/link.js";

function sourceMap(offset: number): SourceRange {
  return {
    source: "entry.flow",
    start: { offset, line: 1, column: offset + 1 },
    end: { offset: offset + 1, line: 1, column: offset + 2 },
  };
}

function importDeclaration(
  path: string,
  alias: string,
  offset: number,
): ModuleImportAst {
  return {
    kind: "import",
    path,
    alias,
    sourceMap: sourceMap(offset),
  };
}

function document(
  imports: readonly ModuleImportAst[] = [],
  componentNames: readonly string[] = [],
): DocumentAst {
  return {
    kind: "document",
    title: "Entry",
    id: "entry",
    language: {
      kind: "language",
      version: 1,
      sourceMap: sourceMap(0),
    },
    imports,
    requirements: [],
    tokens: [],
    symbols: [],
    scenes: [
      {
        kind: "scene",
        title: "Scene",
        id: "scene",
        renderDeclarations: componentNames.map((qualifiedName, index) => {
          const [namespace, name] = qualifiedName.split(".");
          return {
            kind: "component-invocation" as const,
            ...(name === undefined ? { name: namespace ?? "" } : { namespace, name }),
            props: [],
            sourceMap: sourceMap(100 + index),
          };
        }),
        cameras: [],
        timelines: [],
        interactions: [],
        responsiveVariants: [],
        sourceMap: sourceMap(50),
      },
    ],
    sourceMap: sourceMap(0),
  };
}

describe("link module imports", () => {
  test("preserves single-document linking without invoking a resolver", () => {
    const resolveModule = vi.fn<ModuleResolver>();

    const result = link(document(), { resolveModule });

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(resolveModule).not.toHaveBeenCalled();
    if (result.ok) {
      expect(result.value.imports.size).toBe(0);
      expect(result.value.qualifiedNames.size).toBe(0);
    }
  });

  test("resolves imports in authored order through the injected hook", () => {
    const resolveModule = vi.fn<ModuleResolver>(({ path }) => ({
      canonicalUri: `module:${path}`,
      exports: new Set(["Panel"]),
    }));

    const result = link(
      document(
        [
          importDeclaration("./alpha.flow", "alpha", 1),
          importDeclaration("@aiperf/flow-stdlib/beta", "beta", 2),
        ],
        ["alpha.Panel", "beta.Panel"],
      ),
      { resolveModule },
    );

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    expect(resolveModule.mock.calls.map(([request]) => request.path)).toEqual([
      "./alpha.flow",
      "@aiperf/flow-stdlib/beta",
    ]);
    if (result.ok) {
      expect([...result.value.imports.keys()]).toEqual(["alpha", "beta"]);
      expect(
        [...result.value.qualifiedNames.values()].map(
          ({ canonicalUri, exportName }) => [canonicalUri, exportName],
        ),
      ).toEqual([
        ["module:./alpha.flow", "Panel"],
        ["module:@aiperf/flow-stdlib/beta", "Panel"],
      ]);
    }
  });

  test("rejects a duplicate alias at the duplicate declaration", () => {
    const result = link(
      document([
        importDeclaration("./alpha.flow", "shared", 1),
        importDeclaration("./beta.flow", "shared", 8),
      ]),
      {
        resolveModule: ({ path }) => ({
          canonicalUri: `module:${path}`,
          exports: new Set(),
        }),
      },
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "LINK_DUPLICATE_BINDING",
          range: sourceMap(8),
        }),
      ]),
    );
  });

  test("rejects an alias that collides with a local module binding", () => {
    const entry = document([importDeclaration("./theme.flow", "accent", 9)]);
    const result = link(
      {
        ...entry,
        tokens: [
          {
            kind: "token",
            id: "accent",
            value: {
              kind: "literal",
              value: "#ffffff",
              sourceMap: sourceMap(3),
            },
            sourceMap: sourceMap(3),
          },
        ],
      },
      {
        resolveModule: () => ({
          canonicalUri: "module:theme",
          exports: new Set(),
        }),
      },
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "LINK_DUPLICATE_BINDING",
          range: sourceMap(9),
        }),
      ]),
    );
  });

  test.each([
    {
      name: "an import without a resolver",
      document: document([importDeclaration("./missing.flow", "missing", 4)]),
      options: {},
      code: "MODULE_NOT_FOUND",
      range: sourceMap(4),
    },
    {
      name: "an invalid import specifier",
      document: document([importDeclaration("/absolute.flow", "absolute", 5)]),
      options: {
        resolveModule: (() => {
          throw new Error("must not be called");
        }) satisfies ModuleResolver,
      },
      code: "MODULE_INVALID_SPECIFIER",
      range: sourceMap(5),
    },
    {
      name: "an unknown namespace",
      document: document([], ["missing.Panel"]),
      options: {},
      code: "LINK_UNKNOWN_NAME",
      range: sourceMap(100),
    },
    {
      name: "an unknown namespace member",
      document: document(
        [importDeclaration("./widgets.flow", "widgets", 6)],
        ["widgets.Missing"],
      ),
      options: {
        resolveModule: (() => ({
          canonicalUri: "module:widgets",
          exports: new Set(["Panel"]),
        })) satisfies ModuleResolver,
      },
      code: "LINK_UNKNOWN_NAMESPACE_MEMBER",
      range: sourceMap(100),
    },
  ])("fails closed for $name with a source-mapped diagnostic", (fixture) => {
    const result = link(fixture.document, fixture.options);

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: fixture.code,
          severity: "error",
          range: fixture.range,
        }),
      ]),
    );
  });
});
