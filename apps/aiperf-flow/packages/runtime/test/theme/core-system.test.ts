/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, test } from "vitest";

import {
  CoreTheme,
  CoreThemeInheritanceCycleError,
  CoreThemeNotFoundError,
  CoreThemeRegistry,
  CoreThemeRoleNotFoundError,
  createBootstrapCoreRegistry,
  createHexColor,
  type CoreThemeRole,
  type HexColor,
} from "../../src/theme/core-system.js";

describe("createHexColor", () => {
  test("accepts valid 6-digit hex colors", () => {
    const color = createHexColor("#FFFFFF");
    expect(color).toBe("#FFFFFF");
  });

  test("accepts valid 8-digit hex colors with alpha", () => {
    const color = createHexColor("#FFFFFF80");
    expect(color).toBe("#FFFFFF80");
  });

  test("accepts lowercase hex colors", () => {
    const color = createHexColor("#ffffff");
    expect(color).toBe("#ffffff");
  });

  test("rejects invalid hex colors", () => {
    expect(() => createHexColor("FFFFFF")).toThrow();
    expect(() => createHexColor("#GGGGGG")).toThrow();
    expect(() => createHexColor("#FFF")).toThrow();
    expect(() => createHexColor("#FFFFFFF")).toThrow();
  });
});

describe("CoreTheme instantiation", () => {
  test("creates theme with all roles defined", () => {
    const theme = new CoreTheme("test-theme", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
      dark: {
        text: createHexColor("#FFFFFF"),
        background: createHexColor("#000000"),
        accent: createHexColor("#66CCFF"),
      },
    });

    expect(theme.id()).toBe("test-theme");
    expect(theme.parent()).toBeUndefined();
  });

  test("creates theme with partial roles and parent fallback", () => {
    const parent = new CoreTheme("parent-theme", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
      dark: {
        text: createHexColor("#FFFFFF"),
        background: createHexColor("#000000"),
        accent: createHexColor("#66CCFF"),
      },
    });

    const child = new CoreTheme(
      "child-theme",
      {
        light: {
          text: createHexColor("#1A1A1A"),
        },
      },
      parent,
    );

    expect(child.id()).toBe("child-theme");
    expect(child.parent()).toBe(parent);
    expect(child.getRole("text", "light")).toBe(createHexColor("#1A1A1A"));
    expect(child.getRole("background", "light")).toBe(
      createHexColor("#FFFFFF"),
    );
  });

  test("throws when role cannot be resolved", () => {
    const theme = new CoreTheme("incomplete-theme", {
      light: {
        text: createHexColor("#000000"),
      },
    });

    expect(() => theme.getRole("background", "light")).toThrow(
      CoreThemeRoleNotFoundError,
    );
    expect(() => theme.getRole("accent", "dark")).toThrow(
      CoreThemeRoleNotFoundError,
    );
  });

  test("theme is immutable after creation", () => {
    const theme = new CoreTheme("immutable-theme", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
    });

    const lightVariant = theme.getVariant("light");
    expect(Object.isFrozen(lightVariant)).toBe(true);
    const originalValue = lightVariant.text;
    // Attempting to modify throws in strict mode
    expect(() => {
      (lightVariant as any).text = createHexColor("#FF0000");
    }).toThrow();
    expect(lightVariant.text).toBe(originalValue); // Should not change
  });
});

describe("CoreTheme role lookup", () => {
  test("retrieves individual role for light variant", () => {
    const theme = new CoreTheme("lookup-test", {
      light: {
        text: createHexColor("#111111"),
        background: createHexColor("#EEEEEE"),
        accent: createHexColor("#0066CC"),
      },
    });

    expect(theme.getRole("text", "light")).toBe(createHexColor("#111111"));
    expect(theme.getRole("background", "light")).toBe(
      createHexColor("#EEEEEE"),
    );
    expect(theme.getRole("accent", "light")).toBe(createHexColor("#0066CC"));
  });

  test("retrieves individual role for dark variant", () => {
    const theme = new CoreTheme("lookup-test-dark", {
      dark: {
        text: createHexColor("#EEEEEE"),
        background: createHexColor("#111111"),
        accent: createHexColor("#66CCFF"),
      },
    });

    expect(theme.getRole("text", "dark")).toBe(createHexColor("#EEEEEE"));
    expect(theme.getRole("background", "dark")).toBe(
      createHexColor("#111111"),
    );
    expect(theme.getRole("accent", "dark")).toBe(createHexColor("#66CCFF"));
  });

  test("role lookup cascades through inheritance chain", () => {
    const grandparent = new CoreTheme("grandparent", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
    });

    const parent = new CoreTheme(
      "parent",
      {
        light: {
          text: createHexColor("#1A1A1A"),
        },
      },
      grandparent,
    );

    const child = new CoreTheme(
      "child",
      {
        light: {
          accent: createHexColor("#FF00FF"),
        },
      },
      parent,
    );

    // Child overrides accent
    expect(child.getRole("accent", "light")).toBe(createHexColor("#FF00FF"));
    // Parent overrides text, so child gets parent's override
    expect(child.getRole("text", "light")).toBe(createHexColor("#1A1A1A"));
    // Neither child nor parent define background, so cascade to grandparent
    expect(child.getRole("background", "light")).toBe(
      createHexColor("#FFFFFF"),
    );
  });

  test("role not found throws CoreThemeRoleNotFoundError", () => {
    const theme = new CoreTheme("error-test", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
    });

    expect(() => theme.getRole("text", "dark")).toThrow(
      CoreThemeRoleNotFoundError,
    );
    expect(() => theme.getRole("background", "dark")).toThrow(
      CoreThemeRoleNotFoundError,
    );
  });
});

describe("CoreTheme variant selection", () => {
  test("returns all colors for light variant", () => {
    const theme = new CoreTheme("variant-test", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
      dark: {
        text: createHexColor("#FFFFFF"),
        background: createHexColor("#000000"),
        accent: createHexColor("#66CCFF"),
      },
    });

    const lightVariant = theme.getVariant("light");
    expect(lightVariant.text).toBe(createHexColor("#000000"));
    expect(lightVariant.background).toBe(createHexColor("#FFFFFF"));
    expect(lightVariant.accent).toBe(createHexColor("#0066CC"));
  });

  test("returns all colors for dark variant", () => {
    const theme = new CoreTheme("variant-test-dark", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
      dark: {
        text: createHexColor("#FFFFFF"),
        background: createHexColor("#000000"),
        accent: createHexColor("#66CCFF"),
      },
    });

    const darkVariant = theme.getVariant("dark");
    expect(darkVariant.text).toBe(createHexColor("#FFFFFF"));
    expect(darkVariant.background).toBe(createHexColor("#000000"));
    expect(darkVariant.accent).toBe(createHexColor("#66CCFF"));
  });

  test("returns all variants snapshot", () => {
    const theme = new CoreTheme("all-variants-test", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
      dark: {
        text: createHexColor("#FFFFFF"),
        background: createHexColor("#000000"),
        accent: createHexColor("#66CCFF"),
      },
    });

    const variants = theme.getAllVariants();
    expect(variants.light.text).toBe(createHexColor("#000000"));
    expect(variants.dark.text).toBe(createHexColor("#FFFFFF"));
    expect(Object.isFrozen(variants)).toBe(true);
  });

  test("variant selection respects inheritance", () => {
    const parent = new CoreTheme("parent-variant", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
      dark: {
        text: createHexColor("#FFFFFF"),
        background: createHexColor("#000000"),
        accent: createHexColor("#66CCFF"),
      },
    });

    const child = new CoreTheme(
      "child-variant",
      {
        light: {
          accent: createHexColor("#FF6600"),
        },
      },
      parent,
    );

    const childLight = child.getVariant("light");
    expect(childLight.accent).toBe(createHexColor("#FF6600"));
    expect(childLight.text).toBe(createHexColor("#000000"));
    expect(childLight.background).toBe(createHexColor("#FFFFFF"));
  });
});

describe("CoreTheme cycle detection", () => {
  test("validates no cycles in single theme", () => {
    const theme = new CoreTheme("single", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
    });

    // Should not throw for a single theme with no parent
    expect(() => theme.validateNoCycles()).not.toThrow();
  });

  test("validates no cycles in linear inheritance", () => {
    const grandparent = new CoreTheme("gp", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
    });

    const parent = new CoreTheme(
      "parent",
      {
        light: {
          text: createHexColor("#111111"),
          background: createHexColor("#EEEEEE"),
          accent: createHexColor("#0066CC"),
        },
      },
      grandparent,
    );

    const child = new CoreTheme(
      "child",
      {
        light: {
          text: createHexColor("#222222"),
          background: createHexColor("#DDDDDD"),
          accent: createHexColor("#0066CC"),
        },
      },
      parent,
    );

    // Should not throw
    child.validateNoCycles();
    parent.validateNoCycles();
    grandparent.validateNoCycles();
  });
});

describe("CoreThemeRegistry", () => {
  test("registers single theme", () => {
    const registry = new CoreThemeRegistry();
    const theme = new CoreTheme("registered", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
    });

    registry.register(theme);
    expect(registry.get("registered")).toBe(theme);
    expect(registry.has("registered")).toBe(true);
  });

  test("registers multiple themes", () => {
    const registry = new CoreThemeRegistry();
    const theme1 = new CoreTheme("theme1", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
    });

    const theme2 = new CoreTheme("theme2", {
      light: {
        text: createHexColor("#111111"),
        background: createHexColor("#EEEEEE"),
        accent: createHexColor("#0066CC"),
      },
    });

    registry.register(theme1, theme2);
    expect(registry.get("theme1")).toBe(theme1);
    expect(registry.get("theme2")).toBe(theme2);
    expect(registry.has("theme1")).toBe(true);
    expect(registry.has("theme2")).toBe(true);
  });

  test("rejects duplicate theme IDs", () => {
    const registry = new CoreThemeRegistry();
    const theme = new CoreTheme("dup", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
    });

    registry.register(theme);
    expect(() => registry.register(theme)).toThrow(
      'Theme "dup" is already registered',
    );
  });

  test("returns sorted theme IDs", () => {
    const registry = new CoreThemeRegistry();
    const themeZ = new CoreTheme("z-theme", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
    });

    const themeA = new CoreTheme("a-theme", {
      light: {
        text: createHexColor("#111111"),
        background: createHexColor("#EEEEEE"),
        accent: createHexColor("#0066CC"),
      },
    });

    const themeM = new CoreTheme("m-theme", {
      light: {
        text: createHexColor("#222222"),
        background: createHexColor("#DDDDDD"),
        accent: createHexColor("#0066CC"),
      },
    });

    registry.register(themeZ, themeA, themeM);
    expect(registry.ids()).toEqual(["a-theme", "m-theme", "z-theme"]);
  });

  test("prevents registration after freeze", () => {
    const registry = new CoreThemeRegistry();
    const theme = new CoreTheme("frozen", {
      light: {
        text: createHexColor("#000000"),
        background: createHexColor("#FFFFFF"),
        accent: createHexColor("#0066CC"),
      },
    });

    registry.freeze();
    expect(() => registry.register(theme)).toThrow(
      "CoreThemeRegistry is frozen",
    );
  });

  test("freeze() makes registry immutable", () => {
    const registry = new CoreThemeRegistry();
    registry.freeze();
    expect(registry.isFrozen()).toBe(true);
    expect(Object.isFrozen(registry)).toBe(true);
  });

  test("returns undefined for non-existent theme", () => {
    const registry = new CoreThemeRegistry();
    expect(registry.get("nonexistent")).toBeUndefined();
    expect(registry.has("nonexistent")).toBe(false);
  });
});

describe("Bootstrap core registry", () => {
  test("creates registry with default themes", () => {
    const registry = createBootstrapCoreRegistry();
    expect(registry.isFrozen()).toBe(true);
    expect(registry.ids()).toContain("core-base-light");
    expect(registry.ids()).toContain("core-base-dark");
    expect(registry.ids()).toContain("core-light");
    expect(registry.ids()).toContain("core-dark");
  });

  test("bootstrap themes have valid colors", () => {
    const registry = createBootstrapCoreRegistry();
    const lightTheme = registry.get("core-light");
    expect(lightTheme).toBeDefined();
    expect(lightTheme!.getRole("text", "light")).toBeDefined();
    expect(lightTheme!.getRole("background", "light")).toBeDefined();
    expect(lightTheme!.getRole("accent", "light")).toBeDefined();
  });

  test("bootstrap dark theme inherits from base", () => {
    const registry = createBootstrapCoreRegistry();
    const darkTheme = registry.get("core-dark");
    expect(darkTheme).toBeDefined();
    expect(darkTheme!.parent()).toBeDefined();

    // Should be able to get dark variant roles
    expect(darkTheme!.getRole("text", "dark")).toBeDefined();
    expect(darkTheme!.getRole("background", "dark")).toBeDefined();
    expect(darkTheme!.getRole("accent", "dark")).toBeDefined();
  });

  test("bootstrap light and dark themes use different colors", () => {
    const registry = createBootstrapCoreRegistry();
    const lightTheme = registry.get("core-light")!;
    const darkTheme = registry.get("core-dark")!;

    expect(lightTheme.getRole("text", "light")).not.toBe(
      darkTheme.getRole("text", "dark"),
    );
    expect(lightTheme.getRole("background", "light")).not.toBe(
      darkTheme.getRole("background", "dark"),
    );
  });
});
