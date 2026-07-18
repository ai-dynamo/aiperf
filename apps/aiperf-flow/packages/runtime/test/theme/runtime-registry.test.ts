// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type FlowThemeIr,
  type SourceRange,
  type ThemeRole,
  type ThemeValueIr,
} from "@aiperf/flow-schema";
import { beforeEach, describe, expect, test } from "vitest";

import {
  freezeThemeRegistry,
  getActiveTheme,
  getActiveThemeId,
  getRegisteredThemeIds,
  hasTheme,
  registerTheme,
  registerThemes,
  resetThemeRegistry,
  resolveTheme,
  setActiveThemeId,
  UnknownThemeIdError,
  DuplicateThemeIdError,
  ReservedThemeIdError,
} from "../../src/theme/index.js";

const sourceMap: SourceRange = {
  source: "runtime-registry.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

function theme(
  id: string,
  extendsId: string,
  values: Readonly<Partial<Record<ThemeRole, ThemeValueIr>>> = {},
): FlowThemeIr {
  return { id, extends: extendsId, values, sourceMap };
}

describe("Runtime Theme Registry", () => {
  beforeEach(() => {
    resetThemeRegistry();
  });

  describe("registerTheme()", () => {
    test("registers a single theme", () => {
      registerTheme(theme("custom", "systems_chalk"));

      expect(hasTheme("custom")).toBe(true);
      expect(getRegisteredThemeIds()).toContain("custom");
    });

    test("registers multiple themes sequentially", () => {
      registerTheme(theme("first", "systems_chalk"));
      registerTheme(theme("second", "first"));

      expect(hasTheme("first")).toBe(true);
      expect(hasTheme("second")).toBe(true);
    });

    test("throws on duplicate theme ID", () => {
      registerTheme(theme("custom", "systems_chalk"));

      expect(() => registerTheme(theme("custom", "systems_chalk"))).toThrow(
        /Duplicate theme ID/,
      );
    });

    test("throws on reserved theme IDs", () => {
      expect(() => registerTheme(theme("systems_chalk", "systems_chalk"))).toThrow(
        /reserved/i,
      );
    });

    test("throws after registry is frozen", () => {
      registerTheme(theme("custom", "systems_chalk"));
      freezeThemeRegistry();

      expect(() => registerTheme(theme("late", "systems_chalk"))).toThrow(
        Error,
        "Cannot register theme after registry is frozen",
      );
    });
  });

  describe("registerThemes()", () => {
    test("registers multiple themes atomically", () => {
      registerThemes([
        theme("first", "systems_chalk"),
        theme("second", "first"),
      ]);

      expect(hasTheme("first")).toBe(true);
      expect(hasTheme("second")).toBe(true);
    });

    test("does not register if batch has duplicates", () => {
      expect(() =>
        registerThemes([
          theme("first", "systems_chalk"),
          theme("first", "systems_chalk"),
        ]),
      ).toThrow(/Duplicate theme ID/);

      expect(hasTheme("first")).toBe(false);
    });

    test("throws after registry is frozen", () => {
      registerTheme(theme("custom", "systems_chalk"));
      freezeThemeRegistry();

      expect(() =>
        registerThemes([theme("late", "systems_chalk")]),
      ).toThrow(Error, "Cannot register themes after registry is frozen");
    });
  });

  describe("freezeThemeRegistry()", () => {
    test("freezes the registry after registration", () => {
      registerTheme(theme("custom", "systems_chalk"));
      const frozen = freezeThemeRegistry();

      expect(frozen).toBeDefined();
      expect(frozen.has("custom")).toBe(true);
    });

    test("returns the same frozen instance on subsequent calls", () => {
      registerTheme(theme("custom", "systems_chalk"));
      const first = freezeThemeRegistry();
      const second = freezeThemeRegistry();

      expect(first).toBe(second);
    });
  });

  describe("resolveTheme()", () => {
    test("resolves a registered theme with inheritance", () => {
      registerTheme(
        theme("custom", "systems_chalk", {
          "accent.control": { kind: "color", value: "#8BE8E0" },
        }),
      );

      const resolved = resolveTheme("custom");

      expect(resolved.id).toBe("custom");
      expect(resolved.values["accent.control"]).toEqual({
        kind: "color",
        value: "#8BE8E0",
      });
      // Should inherit from systems_chalk
      expect(resolved.values["surface.canvas"]).toEqual({
        kind: "color",
        value: "#232526",
      });
    });

    test("caches resolved themes", () => {
      registerTheme(theme("custom", "systems_chalk"));

      const first = resolveTheme("custom");
      const second = resolveTheme("custom");

      expect(first).toBe(second);
    });

    test("automatically freezes registry on first resolution", () => {
      registerTheme(theme("custom", "systems_chalk"));

      const frozen = resolveTheme("custom");

      expect(frozen).toBeDefined();
      expect(() => registerTheme(theme("late", "systems_chalk"))).toThrow(
        Error,
        "Cannot register theme after registry is frozen",
      );
    });

    test("throws on unknown theme ID", () => {
      expect(() => resolveTheme("missing")).toThrow(/Unknown theme ID/);
    });

    test("deep-freezes resolved themes", () => {
      registerTheme(theme("custom", "systems_chalk"));

      const resolved = resolveTheme("custom");

      expect(Object.isFrozen(resolved)).toBe(true);
      expect(Object.isFrozen(resolved.values)).toBe(true);
    });
  });

  describe("setActiveThemeId() and getActiveThemeId()", () => {
    test("sets and retrieves active theme ID", () => {
      registerTheme(theme("custom", "systems_chalk"));

      setActiveThemeId("custom");

      expect(getActiveThemeId()).toBe("custom");
    });

    test("returns null when no active theme is set", () => {
      expect(getActiveThemeId()).toBeNull();
    });

    test("clears active theme when set to null", () => {
      registerTheme(theme("custom", "systems_chalk"));
      setActiveThemeId("custom");

      setActiveThemeId(null);

      expect(getActiveThemeId()).toBeNull();
    });
  });

  describe("getActiveTheme()", () => {
    test("returns resolved active theme", () => {
      registerTheme(
        theme("custom", "systems_chalk", {
          "accent.control": { kind: "color", value: "#8BE8E0" },
        }),
      );
      setActiveThemeId("custom");

      const active = getActiveTheme();

      expect(active).not.toBeNull();
      expect(active?.id).toBe("custom");
      expect(active?.values["accent.control"]).toEqual({
        kind: "color",
        value: "#8BE8E0",
      });
    });

    test("returns null when no active theme is set", () => {
      expect(getActiveTheme()).toBeNull();
    });

    test("throws if active theme ID is unknown", () => {
      setActiveThemeId("missing");

      expect(() => getActiveTheme()).toThrow(/Unknown theme ID/);
    });

    test("caches active theme resolutions", () => {
      registerTheme(theme("custom", "systems_chalk"));
      setActiveThemeId("custom");

      const first = getActiveTheme();
      const second = getActiveTheme();

      expect(first).toBe(second);
    });
  });

  describe("getRegisteredThemeIds()", () => {
    test("returns all registered theme IDs sorted", () => {
      registerThemes([
        theme("zebra", "systems_chalk"),
        theme("amber", "systems_chalk"),
      ]);

      const ids = getRegisteredThemeIds();

      expect(ids).toEqual(["amber", "systems_chalk", "zebra"]);
      expect(Object.isFrozen(ids)).toBe(true);
    });

    test("includes bundled themes", () => {
      const ids = getRegisteredThemeIds();

      expect(ids).toContain("systems_chalk");
    });

    test("automatically freezes registry", () => {
      registerTheme(theme("custom", "systems_chalk"));
      getRegisteredThemeIds();

      expect(() => registerTheme(theme("late", "systems_chalk"))).toThrow(
        Error,
        "Cannot register theme after registry is frozen",
      );
    });
  });

  describe("hasTheme()", () => {
    test("returns true for registered themes", () => {
      registerTheme(theme("custom", "systems_chalk"));

      expect(hasTheme("custom")).toBe(true);
    });

    test("returns false for unregistered themes", () => {
      expect(hasTheme("missing")).toBe(false);
    });

    test("returns true for bundled themes", () => {
      expect(hasTheme("systems_chalk")).toBe(true);
    });

    test("automatically freezes registry", () => {
      registerTheme(theme("custom", "systems_chalk"));
      hasTheme("custom");

      expect(() => registerTheme(theme("late", "systems_chalk"))).toThrow(
        Error,
        "Cannot register theme after registry is frozen",
      );
    });
  });

  describe("resetThemeRegistry()", () => {
    test("resets registry to initial state", () => {
      registerTheme(theme("custom", "systems_chalk"));
      setActiveThemeId("custom");
      freezeThemeRegistry();

      resetThemeRegistry();

      // Check state without freezing (avoid calling hasTheme which freezes)
      expect(getActiveThemeId()).toBeNull();
      // After reset, should be able to register new themes
      registerTheme(theme("new", "systems_chalk"));
      // Verify new theme was registered
      expect(getRegisteredThemeIds()).toContain("new");
    });

    test("preserves bundled themes after reset", () => {
      registerTheme(theme("custom", "systems_chalk"));
      resetThemeRegistry();

      expect(hasTheme("systems_chalk")).toBe(true);
      expect(hasTheme("custom")).toBe(false);
    });
  });

  describe("Integration scenarios", () => {
    test("complete workflow: register, freeze, resolve, and set active", () => {
      registerThemes([
        theme("light-variant", "systems_chalk"),
        theme("dark-variant", "systems_chalk"),
      ]);

      setActiveThemeId("light-variant");
      const active = getActiveTheme();

      expect(active?.id).toBe("light-variant");
      // Verify it inherits from systems_chalk
      expect(active?.values["surface.canvas"]).toEqual({
        kind: "color",
        value: "#232526",
      });

      setActiveThemeId("dark-variant");
      const darkActive = getActiveTheme();

      expect(darkActive?.id).toBe("dark-variant");
      // Verify it also inherits from systems_chalk
      expect(darkActive?.values["surface.canvas"]).toEqual({
        kind: "color",
        value: "#232526",
      });
    });

    test("multiple active theme switches use cached resolutions", () => {
      registerThemes([
        theme("variant-a", "systems_chalk"),
        theme("variant-b", "systems_chalk"),
      ]);

      const variantA1 = (
        setActiveThemeId("variant-a"), resolveTheme("variant-a")
      );
      const variantB1 = (
        setActiveThemeId("variant-b"), resolveTheme("variant-b")
      );
      const variantA2 = (
        setActiveThemeId("variant-a"), resolveTheme("variant-a")
      );

      // Cache ensures same instance for variant-a
      expect(variantA1).toBe(variantA2);
      // But variant-b is a different instance since it's a different theme
      expect(variantB1.id).toBe("variant-b");
      expect(variantA1.id).toBe("variant-a");
    });
  });

  describe("Existing theme registry tests compatibility", () => {
    test("passes bundled Systems Chalk presence test", () => {
      expect(hasTheme("systems_chalk")).toBe(true);
      const resolved = resolveTheme("systems_chalk");
      expect(resolved.id).toBe("systems_chalk");
      expect(resolved.values["surface.canvas"]).toEqual({
        kind: "color",
        value: "#232526",
      });
    });

    test("passes inherited theme resolution test", () => {
      registerTheme(
        theme("custom", "systems_chalk", {
          "accent.control": { kind: "color", value: "#8BE8E0" },
        }),
      );

      const resolved = resolveTheme("custom");

      expect(resolved.values["accent.control"]).toEqual({
        kind: "color",
        value: "#8BE8E0",
      });
      expect(resolved.values["surface.canvas"]).toEqual({
        kind: "color",
        value: "#232526",
      });
    });

    test("passes cache identity test", () => {
      registerTheme(theme("custom", "systems_chalk"));

      const first = resolveTheme("custom");
      const second = resolveTheme("custom");

      expect(first).toBe(second);
    });

    test("passes frozen theme test", () => {
      registerTheme(theme("custom", "systems_chalk"));
      const resolved = resolveTheme("custom");

      expect(Object.isFrozen(resolved)).toBe(true);
      expect(Object.isFrozen(resolved.values)).toBe(true);
    });
  });
});
