/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  DocumentAst,
  LiteralAst,
  ThemeAssignmentAst,
  ThemeDeclarationAst,
  ThemeFontLiteralAst,
  ThemeValueKindAst,
  UseThemeAst,
} from "@aiperf/flow-language";
import type { SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { link, type ModuleResolver, type ResolvedModule } from "../src/link.js";
import { BUNDLED_THEME_IDS, validateThemes } from "../src/themes.js";

function range(offset = 0): SourceRange {
  return {
    source: "theme.flow",
    start: { offset, line: 1, column: offset + 1 },
    end: { offset: offset + 1, line: 1, column: offset + 2 },
  };
}

function literal(value: string | number | boolean, offset = 0): LiteralAst {
  return { kind: "literal", value, sourceMap: range(offset) };
}

function fontLiteral(
  families: readonly string[],
  offset = 0,
): ThemeFontLiteralAst {
  return { kind: "theme-font-literal", families, sourceMap: range(offset) };
}

function assignment(
  valueKind: ThemeValueKindAst,
  role: string,
  value: LiteralAst | ThemeFontLiteralAst,
  offset = 0,
): ThemeAssignmentAst {
  return { kind: "theme-assignment", valueKind, role, value, sourceMap: range(offset) };
}

function theme(
  id: string,
  extendsId: string,
  assignments: readonly ThemeAssignmentAst[] = [],
  offset = 0,
): ThemeDeclarationAst {
  return {
    kind: "theme-declaration",
    id,
    extends: extendsId,
    assignments,
    sourceMap: range(offset),
  };
}

function useTheme(themeId: string, offset = 0): UseThemeAst {
  return { kind: "use-theme", themeId, sourceMap: range(offset) };
}

describe("validateThemes", () => {
  test("exposes systems_chalk as the bundled default", () => {
    expect(BUNDLED_THEME_IDS).toEqual(["systems_chalk"]);
  });

  test("accepts lab_chalk extends systems_chalk with typed overrides", () => {
    const result = validateThemes({
      themes: [
        theme("lab_chalk", "systems_chalk", [
          assignment("color", "accent.control", literal("#78dce8")),
          assignment("color", "accent.execution", literal("#ffd866")),
          assignment("number", "stroke.standard", literal(2)),
          assignment("duration", "motion.draw", literal(420)),
          assignment("font", "font.body", fontLiteral(["Inter", "sans-serif"])),
          assignment("enum", "stroke.cap", literal("round")),
        ]),
      ],
      useTheme: useTheme("lab_chalk"),
    });

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (result.ok) {
      expect(result.value.defaultTheme).toBe("lab_chalk");
      expect(result.value.themes).toHaveLength(1);
    }
  });

  test("accepts a document without themes or a default unchanged", () => {
    const result = validateThemes({ themes: [] });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.defaultTheme).toBeUndefined();
      expect(result.value.themes).toEqual([]);
    }
  });

  test("resolves a bundled default even without authored themes", () => {
    const result = validateThemes({
      themes: [],
      useTheme: useTheme("systems_chalk"),
    });

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.defaultTheme).toBe("systems_chalk");
    }
  });

  test("reports THEME_DUPLICATE_ID for a repeated authored id", () => {
    const result = validateThemes({
      themes: [
        theme("lab_chalk", "systems_chalk", [], 10),
        theme("lab_chalk", "systems_chalk", [], 40),
      ],
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "THEME_DUPLICATE_ID",
          severity: "error",
          message: expect.stringContaining('Duplicate theme id "lab_chalk"'),
          range: range(40),
        }),
      ]),
    );
  });

  test("reports THEME_RESERVED_ID when an author redefines a bundled id", () => {
    const result = validateThemes({
      themes: [theme("systems_chalk", "systems_chalk", [], 5)],
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "THEME_RESERVED_ID",
          severity: "error",
          message: expect.stringContaining('"systems_chalk"'),
          range: range(5),
        }),
      ]),
    );
  });

  test("reports THEME_UNKNOWN_BASE for an unresolved extends", () => {
    const result = validateThemes({
      themes: [theme("lab_chalk", "ghost_theme", [], 7)],
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "THEME_UNKNOWN_BASE",
          severity: "error",
          message: expect.stringContaining('"ghost_theme"'),
          range: range(7),
        }),
      ]),
    );
  });

  test("reports THEME_INHERITANCE_CYCLE among authored themes", () => {
    const result = validateThemes({
      themes: [
        theme("alpha", "beta", [], 1),
        theme("beta", "alpha", [], 2),
      ],
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "THEME_INHERITANCE_CYCLE",
          severity: "error",
          message: expect.stringContaining("alpha"),
        }),
      ]),
    );
  });

  test("reports THEME_UNKNOWN_ROLE for a role outside THEME_ROLES", () => {
    const result = validateThemes({
      themes: [
        theme("lab_chalk", "systems_chalk", [
          assignment("color", "surface.bogus", literal("#000000"), 12),
        ]),
      ],
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "THEME_UNKNOWN_ROLE",
          severity: "error",
          message: expect.stringContaining('"surface.bogus"'),
          range: range(12),
        }),
      ]),
    );
  });

  test("reports THEME_ROLE_KIND_MISMATCH when the declared kind is wrong", () => {
    const result = validateThemes({
      themes: [
        theme("lab_chalk", "systems_chalk", [
          assignment("number", "accent.control", literal(2), 15),
        ]),
      ],
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "THEME_ROLE_KIND_MISMATCH",
          severity: "error",
          message: expect.stringContaining('"accent.control"'),
          range: range(15),
        }),
      ]),
    );
    const mismatch = result.diagnostics.find(
      (entry) => entry.code === "THEME_ROLE_KIND_MISMATCH",
    );
    expect(mismatch?.message).toContain("color");
    expect(mismatch?.message).toContain("number");
  });

  test.each([
    {
      name: "bad hex",
      assignment: assignment("color", "accent.control", literal("red"), 20),
      received: "red",
    },
    {
      name: "bad enum",
      assignment: assignment("enum", "stroke.cap", literal("diagonal"), 21),
      received: "diagonal",
    },
    {
      name: "empty font stack",
      assignment: assignment("font", "font.body", fontLiteral([]), 22),
      received: undefined,
    },
    {
      name: "non-finite number",
      assignment: assignment(
        "number",
        "stroke.standard",
        literal(Number.POSITIVE_INFINITY),
        23,
      ),
      received: undefined,
    },
  ])("reports THEME_INVALID_VALUE for $name", ({ assignment: badAssignment, received }) => {
    const result = validateThemes({
      themes: [theme("lab_chalk", "systems_chalk", [badAssignment])],
    });

    expect(result.ok).toBe(false);
    const invalid = result.diagnostics.find(
      (entry) => entry.code === "THEME_INVALID_VALUE",
    );
    expect(invalid?.severity).toBe("error");
    expect(invalid?.message).toContain(badAssignment.role);
    if (received !== undefined) {
      expect(invalid?.message).toContain(received);
    }
  });

  test("reports THEME_UNKNOWN_DEFAULT for a missing use theme id", () => {
    const result = validateThemes({
      themes: [theme("lab_chalk", "systems_chalk")],
      useTheme: useTheme("phantom", 33),
    });

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "THEME_UNKNOWN_DEFAULT",
          severity: "error",
          message: expect.stringContaining('"phantom"'),
          range: range(33),
        }),
      ]),
    );
  });
});

function themedDocument(overrides: Partial<DocumentAst> = {}): DocumentAst {
  return {
    kind: "document",
    title: "Entry",
    id: "entry",
    language: { kind: "language", version: 1, sourceMap: range(0) },
    imports: [],
    requirements: [],
    tokens: [],
    themes: [],
    symbols: [],
    scenes: [],
    sourceMap: range(0),
    ...overrides,
  };
}

describe("link theme collection", () => {
  test("collects authored and imported themes into the validation set", () => {
    const importedTheme = theme("imported_chalk", "systems_chalk", [], 200);
    const resolveModule: ModuleResolver = (): ResolvedModule => ({
      canonicalUri: "module:themes",
      exports: new Set(),
      themes: [importedTheme],
    });

    const result = link(
      themedDocument({
        imports: [
          { kind: "import", path: "./themes.flow", alias: "themes", sourceMap: range(1) },
        ],
        themes: [theme("lab_chalk", "systems_chalk", [], 100)],
        useTheme: useTheme("lab_chalk", 150),
      }),
      { resolveModule },
    );

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (result.ok) {
      expect(result.value.themes.map((entry) => entry.id)).toEqual([
        "lab_chalk",
        "imported_chalk",
      ]);
      expect(result.value.useTheme?.themeId).toBe("lab_chalk");
    }
  });

  test("reports THEME_DUPLICATE_DEFAULT when primary and import both select", () => {
    const resolveModule: ModuleResolver = (): ResolvedModule => ({
      canonicalUri: "module:themes",
      exports: new Set(),
      useTheme: useTheme("systems_chalk", 250),
    });

    const result = link(
      themedDocument({
        imports: [
          { kind: "import", path: "./themes.flow", alias: "themes", sourceMap: range(1) },
        ],
        useTheme: useTheme("systems_chalk", 150),
      }),
      { resolveModule },
    );

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "THEME_DUPLICATE_DEFAULT",
          severity: "error",
        }),
      ]),
    );
  });
});
