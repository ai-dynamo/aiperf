// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  REQUIRED_CONTRAST_PAIRS,
  THEME_ROLES,
  flowThemeIrSchema,
  parseThemeRole,
  themeValueIrSchema,
  themeRoleReferenceIrSchema,
} from "../src/theme.js";
import { parseFlowIr, safeParseFlowIr, upgradeFlowIrV1ToV2 } from "../src/ir.js";

const sourceMap = {
  source: "theme.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

describe("theme schema", () => {
  test("accepts the closed role vocabulary and rejects unknown roles", () => {
    expect(THEME_ROLES).toContain("accent.execution");
    expect(parseThemeRole("motion.easing")).toBe("motion.easing");
    expect(() => parseThemeRole("glow.ambient")).toThrow(/unknown theme role/i);
  });

  test("parses typed theme values and rejects bad discriminants", () => {
    expect(
      themeValueIrSchema.parse({ kind: "color", value: "#71D8D0" }),
    ).toEqual({ kind: "color", value: "#71D8D0" });
    expect(
      themeValueIrSchema.parse({ kind: "duration", valueMs: 420 }),
    ).toEqual({ kind: "duration", valueMs: 420 });
    expect(
      themeValueIrSchema.safeParse({ kind: "color", valueMs: 1 }).success,
    ).toBe(false);
    expect(
      themeValueIrSchema.safeParse({ kind: "number", value: Number.NaN })
        .success,
    ).toBe(false);
  });

  test("parses theme-role style references and rejects unknown fields", () => {
    expect(
      themeRoleReferenceIrSchema.parse({
        kind: "theme-role",
        role: "surface.raised",
      }),
    ).toEqual({ kind: "theme-role", role: "surface.raised" });
    expect(
      themeRoleReferenceIrSchema.safeParse({
        kind: "theme-role",
        role: "surface.raised",
        extra: true,
      }).success,
    ).toBe(false);
  });

  test("lists required WCAG AA contrast pairs", () => {
    expect(REQUIRED_CONTRAST_PAIRS.length).toBeGreaterThanOrEqual(8);
    expect(REQUIRED_CONTRAST_PAIRS).toContainEqual({
      foreground: "ink.primary",
      background: "surface.canvas",
      minRatio: 4.5,
    });
  });

  test("enforces each role's value kind and constraints", () => {
    const theme = {
      id: "strict",
      extends: "systems_chalk",
      sourceMap,
    };

    expect(
      flowThemeIrSchema.safeParse({
        ...theme,
        values: { "font.body": { kind: "color", value: "#71D8D0" } },
      }).success,
    ).toBe(false);
    expect(
      flowThemeIrSchema.safeParse({
        ...theme,
        values: { "weight.label": { kind: "number", value: 950 } },
      }).success,
    ).toBe(false);
    expect(
      flowThemeIrSchema.safeParse({
        ...theme,
        values: { "size.body": { kind: "number", value: 0 } },
      }).success,
    ).toBe(false);
    expect(
      flowThemeIrSchema.safeParse({
        ...theme,
        values: { "stroke.cap": { kind: "enum", value: "miter" } },
      }).success,
    ).toBe(false);
    expect(
      flowThemeIrSchema.safeParse({
        ...theme,
        values: { "motion.easing": { kind: "enum", value: "ease_in_out" } },
      }).success,
    ).toBe(true);
  });
});

describe("Flow IR v2 themes", () => {
  test("parses themes, defaultTheme, and theme-role style values", () => {
    const flow = parseFlowIr({
      irVersion: 2,
      id: "themed",
      title: "Themed",
      capabilities: [],
      tokens: {},
      themes: [
        {
          id: "lab_chalk",
          extends: "systems_chalk",
          values: {
            "accent.control": { kind: "color", value: "#78dce8" },
            "stroke.standard": { kind: "number", value: 2 },
            "motion.draw": { kind: "duration", valueMs: 420 },
          },
          sourceMap,
        },
      ],
      defaultTheme: "lab_chalk",
      scenes: [
        {
          id: "main",
          title: "Main",
          summary: "s",
          roots: [
            {
              kind: "rect",
              id: "r",
              geometry: { x: 0, y: 0, width: 10, height: 10 },
              style: {
                fill: { kind: "theme-role", role: "surface.raised" },
              },
              accessibility: { label: "r" },
              fallback: "r",
              sourceMap,
            },
          ],
          camera: [],
          timeline: [],
          narration: "n",
          interactions: [],
          responsive: [],
          accessibility: { label: "main", readingOrder: ["r"] },
          fallback: "f",
          sourceMap,
        },
      ],
      sourceMap,
    });
    expect(flow.irVersion).toBe(2);
    expect(flow.defaultTheme).toBe("lab_chalk");
    expect(flow.themes[0]?.values["accent.control"]).toEqual({
      kind: "color",
      value: "#78dce8",
    });
    expect(flow.scenes[0]?.roots[0]?.style.fill).toEqual({
      kind: "theme-role",
      role: "surface.raised",
    });
  });

  test("rejects irVersion 1 without upgrade and upgrades v1 payloads", () => {
    const v1 = {
      irVersion: 1,
      id: "legacy",
      title: "Legacy",
      capabilities: [],
      tokens: { accent: "#7aa2f7" },
      scenes: [],
      sourceMap,
    };
    expect(safeParseFlowIr(v1).ok).toBe(false);
    const upgraded = parseFlowIr(upgradeFlowIrV1ToV2(v1));
    expect(upgraded.irVersion).toBe(2);
    expect(upgraded.themes).toEqual([]);
    expect(upgraded.defaultTheme).toBeUndefined();
    expect(upgraded.tokens.accent).toBe("#7aa2f7");
  });
});
