/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import type { FlowThemeIr } from "../../schema/theme.js";
import { resolveThemeValue, type ThemeContext } from "./with-theme.js";

const SOURCE_MAP = {
  source: "with-theme.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function theme(
  partial: Omit<FlowThemeIr, "sourceMap"> & Partial<Pick<FlowThemeIr, "sourceMap">>,
): FlowThemeIr {
  return {
    sourceMap: SOURCE_MAP,
    ...partial,
  };
}

describe("resolveThemeValue", () => {
  it("resolves a role from the active theme", () => {
    const active = theme({
      id: "active",
      extends: "base",
      values: { "ink.primary": { kind: "color", value: "#112233" } },
    });
    const context: ThemeContext = { activeTheme: active, allThemes: [active] };

    expect(resolveThemeValue("ink.primary", context)).toEqual({
      kind: "color",
      value: "#112233",
    });
  });

  it("walks a linear extends chain for missing roles", () => {
    const base = theme({
      id: "base",
      extends: "none",
      values: { "ink.muted": { kind: "color", value: "#445566" } },
    });
    const active = theme({
      id: "active",
      extends: "base",
      values: {},
    });
    const context: ThemeContext = {
      activeTheme: active,
      allThemes: [active, base],
    };

    expect(resolveThemeValue("ink.muted", context)).toEqual({
      kind: "color",
      value: "#445566",
    });
  });

  it("fails closed with a diagnostic on mutual extends cycles", () => {
    const a = theme({
      id: "a",
      extends: "b",
      values: {},
    });
    const b = theme({
      id: "b",
      extends: "a",
      values: {},
    });
    const context: ThemeContext = { activeTheme: a, allThemes: [a, b] };

    expect(() => resolveThemeValue("ink.primary", context)).toThrow(
      /theme extends cycle/i,
    );
  });

  it("fails closed on a longer extends cycle", () => {
    const a = theme({ id: "a", extends: "b", values: {} });
    const b = theme({ id: "b", extends: "c", values: {} });
    const c = theme({ id: "c", extends: "a", values: {} });
    const context: ThemeContext = {
      activeTheme: a,
      allThemes: [a, b, c],
    };

    expect(() => resolveThemeValue("surface.canvas", context)).toThrow(
      /theme extends cycle/i,
    );
  });

  it("fails closed when a theme extends itself", () => {
    const loop = theme({
      id: "loop",
      extends: "loop",
      values: {},
    });
    const context: ThemeContext = { activeTheme: loop, allThemes: [loop] };

    expect(() => resolveThemeValue("ink.primary", context)).toThrow(
      /theme extends cycle/i,
    );
  });
});
