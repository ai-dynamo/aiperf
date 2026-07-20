/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type {
  ThemeAssignmentAst,
  ThemeDeclarationAst,
} from "../language/ast.js";
import { validateThemes } from "./themes.js";

const SOURCE_MAP = {
  source: "themes.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function numberAssignment(
  role: string,
  value: number,
): ThemeAssignmentAst {
  return {
    kind: "theme-assignment",
    valueKind: "number",
    role,
    value: { kind: "literal", value, sourceMap: SOURCE_MAP },
    sourceMap: SOURCE_MAP,
  };
}

function durationAssignment(
  role: string,
  value: number,
): ThemeAssignmentAst {
  return {
    kind: "theme-assignment",
    valueKind: "duration",
    role,
    value: { kind: "literal", value, sourceMap: SOURCE_MAP },
    sourceMap: SOURCE_MAP,
  };
}

function theme(
  assignments: readonly ThemeAssignmentAst[],
): ThemeDeclarationAst {
  return {
    kind: "theme-declaration",
    id: "custom",
    extends: "systems_chalk",
    assignments,
    sourceMap: SOURCE_MAP,
  };
}

describe("validateThemes numeric role bounds", () => {
  it("rejects negative stroke numbers with THEME_INVALID_VALUE", () => {
    const result = validateThemes({
      themes: [theme([numberAssignment("stroke.standard", -1)])],
    });

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some((d) => d.code === "THEME_INVALID_VALUE"),
    ).toBe(true);
  });

  it("rejects weight outside 100–900 with THEME_INVALID_VALUE", () => {
    const result = validateThemes({
      themes: [theme([numberAssignment("weight.regular", 50)])],
    });

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some(
        (d) =>
          d.code === "THEME_INVALID_VALUE" &&
          d.message.includes("100") &&
          d.message.includes("900"),
      ),
    ).toBe(true);
  });

  it("rejects non-integer weight with THEME_INVALID_VALUE", () => {
    const result = validateThemes({
      themes: [theme([numberAssignment("weight.emphasis", 400.5)])],
    });

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some((d) => d.code === "THEME_INVALID_VALUE"),
    ).toBe(true);
  });

  it("rejects non-positive size with THEME_INVALID_VALUE", () => {
    const result = validateThemes({
      themes: [theme([numberAssignment("size.body", 0)])],
    });

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some((d) => d.code === "THEME_INVALID_VALUE"),
    ).toBe(true);
  });

  it("rejects negative duration with THEME_INVALID_VALUE", () => {
    const result = validateThemes({
      themes: [theme([durationAssignment("motion.draw", -10)])],
    });

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some((d) => d.code === "THEME_INVALID_VALUE"),
    ).toBe(true);
  });

  it("rejects non-integer duration with THEME_INVALID_VALUE", () => {
    const result = validateThemes({
      themes: [theme([durationAssignment("motion.enter", 12.5)])],
    });

    expect(result.ok).toBe(false);
    expect(
      result.diagnostics.some((d) => d.code === "THEME_INVALID_VALUE"),
    ).toBe(true);
  });

  it("accepts in-range weight, positive size, and nonnegative duration", () => {
    const result = validateThemes({
      themes: [
        theme([
          numberAssignment("weight.regular", 400),
          numberAssignment("size.body", 14),
          numberAssignment("stroke.standard", 1),
          durationAssignment("motion.draw", 0),
          durationAssignment("motion.enter", 240),
        ]),
      ],
    });

    expect(result.ok).toBe(true);
  });
});
