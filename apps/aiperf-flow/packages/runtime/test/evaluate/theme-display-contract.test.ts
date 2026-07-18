// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ThemeRole, ThemeValueIr } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import type { ResolvedTheme } from "../../src/theme/index.js";
import {
  buildThemeDisplayMapping,
  displayInstructionFromThemeValue,
  validateDisplayInstruction,
  type DisplayInstruction,
  type ThemeRoleContract,
  ThemeRoleInventory,
} from "../../src/evaluate/theme-display-contract.js";

const baseTheme: ResolvedTheme = {
  id: "test-theme",
  values: {
    "surface.canvas": { kind: "color", value: "#232526" },
    "surface.panel": { kind: "color", value: "#292C2D" },
    "ink.primary": { kind: "color", value: "#F1F3F2" },
    "ink.muted": { kind: "color", value: "#AEB4B5" },
    "accent.control": { kind: "color", value: "#71D8D0" },
    "accent.execution": { kind: "color", value: "#69C8BA" },
    "font.body": { kind: "font", value: ["Inter", "sans-serif"] },
    "font.data": { kind: "font", value: ["Mono", "monospace"] },
    "weight.regular": { kind: "number", value: 400 },
    "weight.label": { kind: "number", value: 600 },
    "size.body": { kind: "number", value: 14 },
    "size.label": { kind: "number", value: 16 },
    "stroke.standard": { kind: "number", value: 2 },
    "motion.draw": { kind: "duration", valueMs: 420 },
    "motion.enter": { kind: "duration", valueMs: 240 },
    "stroke.cap": { kind: "enum", value: "round" },
    "stroke.join": { kind: "enum", value: "round" },
  } as Partial<Record<ThemeRole, ThemeValueIr>>,
};

const testContracts: ThemeRoleContract[] = [
  {
    role: "surface.canvas",
    valueKind: "color",
    category: "background",
    required: true,
    description: "Primary canvas background",
    dependencies: [],
  },
  {
    role: "surface.panel",
    valueKind: "color",
    category: "background",
    required: true,
    description: "Panel background",
    dependencies: [],
  },
  {
    role: "ink.primary",
    valueKind: "color",
    category: "foreground",
    required: true,
    description: "Primary text color",
    dependencies: [],
  },
  {
    role: "ink.muted",
    valueKind: "color",
    category: "foreground",
    required: false,
    description: "Secondary text color",
    dependencies: [],
  },
  {
    role: "accent.control",
    valueKind: "color",
    category: "accent",
    required: false,
    description: "Control accent color",
    dependencies: [],
  },
  {
    role: "accent.execution",
    valueKind: "color",
    category: "accent",
    required: false,
    description: "Execution accent color",
    dependencies: [],
  },
  {
    role: "font.body",
    valueKind: "font",
    category: "typography",
    required: true,
    description: "Body text font",
    dependencies: [],
  },
  {
    role: "font.data",
    valueKind: "font",
    category: "typography",
    required: false,
    description: "Data display font",
    dependencies: [],
  },
  {
    role: "weight.regular",
    valueKind: "number",
    category: "typography",
    required: true,
    description: "Regular font weight",
    dependencies: [],
  },
  {
    role: "weight.label",
    valueKind: "number",
    category: "typography",
    required: false,
    description: "Label font weight",
    dependencies: [],
  },
  {
    role: "size.body",
    valueKind: "number",
    category: "typography",
    required: true,
    description: "Body font size",
    dependencies: [],
  },
  {
    role: "size.label",
    valueKind: "number",
    category: "typography",
    required: false,
    description: "Label font size",
    dependencies: [],
  },
  {
    role: "stroke.standard",
    valueKind: "number",
    category: "structure",
    required: false,
    description: "Standard stroke width",
    dependencies: [],
  },
  {
    role: "stroke.cap",
    valueKind: "enum",
    category: "structure",
    required: false,
    description: "Stroke end cap style",
    dependencies: [],
  },
  {
    role: "stroke.join",
    valueKind: "enum",
    category: "structure",
    required: false,
    description: "Stroke join style",
    dependencies: [],
  },
  {
    role: "motion.draw",
    valueKind: "duration",
    category: "motion",
    required: false,
    description: "Draw animation duration",
    dependencies: [],
  },
  {
    role: "motion.enter",
    valueKind: "duration",
    category: "motion",
    required: false,
    description: "Enter animation duration",
    dependencies: [],
  },
];

describe("theme-display-contract", () => {
  describe("displayInstructionFromThemeValue", () => {
    test("extracts color values into displayInstructions", () => {
      const value: ThemeValueIr = { kind: "color", value: "#F1F3F2" };
      const instruction = displayInstructionFromThemeValue("ink.primary", value, "foreground", 1000);

      expect(instruction).toEqual({
        role: "ink.primary",
        kind: "color",
        value: "#F1F3F2",
        category: "foreground",
        appliedAtMs: 1000,
      });
      expect(Object.isFrozen(instruction)).toBe(true);
    });

    test("extracts font array values, freezing the copy", () => {
      const value: ThemeValueIr = { kind: "font", value: ["Inter", "sans-serif"] };
      const instruction = displayInstructionFromThemeValue("font.body", value, "typography", 500);

      expect(instruction).toEqual({
        role: "font.body",
        kind: "font",
        value: ["Inter", "sans-serif"],
        category: "typography",
        appliedAtMs: 500,
      });
      expect(Object.isFrozen(instruction)).toBe(true);
      expect(Object.isFrozen(instruction.value)).toBe(true);
    });

    test("extracts number values unchanged", () => {
      const value: ThemeValueIr = { kind: "number", value: 400 };
      const instruction = displayInstructionFromThemeValue("weight.regular", value, "typography", 200);

      expect(instruction.kind).toBe("number");
      expect(instruction.value).toBe(400);
    });

    test("converts duration.valueMs to instruction.value", () => {
      const value: ThemeValueIr = { kind: "duration", valueMs: 420 };
      const instruction = displayInstructionFromThemeValue("motion.draw", value, "motion", 100);

      expect(instruction).toEqual({
        role: "motion.draw",
        kind: "duration",
        value: 420,
        category: "motion",
        appliedAtMs: 100,
      });
    });

    test("extracts enum values as strings", () => {
      const value: ThemeValueIr = { kind: "enum", value: "round" };
      const instruction = displayInstructionFromThemeValue("stroke.cap", value, "structure", 50);

      expect(instruction).toEqual({
        role: "stroke.cap",
        kind: "enum",
        value: "round",
        category: "structure",
        appliedAtMs: 50,
      });
    });

    test("throws on unknown value kind", () => {
      const badValue = { kind: "unknown" } as ThemeValueIr;
      expect(() =>
        displayInstructionFromThemeValue("some.role", badValue, "foreground", 0),
      ).toThrow(/Unknown theme value kind/);
    });
  });

  describe("validateDisplayInstruction", () => {
    test("accepts valid color hex values", () => {
      const validColors = ["#000000", "#FFFFFF", "#F1F3F2", "#F1F3F2FF"];
      for (const color of validColors) {
        const instruction: DisplayInstruction = {
          role: "ink.primary",
          kind: "color",
          value: color,
          category: "foreground",
          appliedAtMs: 0,
        };
        expect(() => validateDisplayInstruction(instruction)).not.toThrow();
      }
    });

    test("rejects invalid color formats", () => {
      const invalidColors = [
        "red",
        "#FFF",
        "#FFFFFFFF00",
        "#GGG000",
        "F1F3F2",
      ];
      for (const color of invalidColors) {
        const instruction: DisplayInstruction = {
          role: "ink.primary",
          kind: "color",
          value: color,
          category: "foreground",
          appliedAtMs: 0,
        };
        expect(() => validateDisplayInstruction(instruction)).toThrow(/hex #RRGGBB/);
      }
    });

    test("accepts valid font arrays", () => {
      const instruction: DisplayInstruction = {
        role: "font.body",
        kind: "font",
        value: ["Inter", "sans-serif"],
        category: "typography",
        appliedAtMs: 0,
      };
      expect(() => validateDisplayInstruction(instruction)).not.toThrow();
    });

    test("rejects non-array font values", () => {
      const instruction: DisplayInstruction = {
        role: "font.body",
        kind: "font",
        value: "Inter",
        category: "typography",
        appliedAtMs: 0,
      };
      expect(() => validateDisplayInstruction(instruction)).toThrow(/requires array/);
    });

    test("rejects font arrays with empty strings", () => {
      const instruction: DisplayInstruction = {
        role: "font.body",
        kind: "font",
        value: ["", "sans-serif"],
        category: "typography",
        appliedAtMs: 0,
      };
      expect(() => validateDisplayInstruction(instruction)).toThrow(
        /non-empty string array/,
      );
    });

    test("accepts valid finite non-negative numbers", () => {
      const instruction: DisplayInstruction = {
        role: "size.body",
        kind: "number",
        value: 14,
        category: "typography",
        appliedAtMs: 0,
      };
      expect(() => validateDisplayInstruction(instruction)).not.toThrow();
    });

    test("rejects infinite, NaN, or negative numbers", () => {
      for (const value of [Number.POSITIVE_INFINITY, Number.NaN, -1]) {
        const instruction: DisplayInstruction = {
          role: "size.body",
          kind: "number",
          value,
          category: "typography",
          appliedAtMs: 0,
        };
        expect(() => validateDisplayInstruction(instruction)).toThrow(
          /finite non-negative number/,
        );
      }
    });

    test("accepts valid duration values (non-negative integers)", () => {
      const instruction: DisplayInstruction = {
        role: "motion.draw",
        kind: "duration",
        value: 420,
        category: "motion",
        appliedAtMs: 0,
      };
      expect(() => validateDisplayInstruction(instruction)).not.toThrow();
    });

    test("rejects non-integer or negative durations", () => {
      for (const value of [420.5, -100, Number.POSITIVE_INFINITY]) {
        const instruction: DisplayInstruction = {
          role: "motion.draw",
          kind: "duration",
          value,
          category: "motion",
          appliedAtMs: 0,
        };
        expect(() => validateDisplayInstruction(instruction)).toThrow(
          /non-negative integer ms/,
        );
      }
    });

    test("accepts valid enum string values", () => {
      const instruction: DisplayInstruction = {
        role: "stroke.cap",
        kind: "enum",
        value: "round",
        category: "structure",
        appliedAtMs: 0,
      };
      expect(() => validateDisplayInstruction(instruction)).not.toThrow();
    });

    test("rejects non-string enum values", () => {
      const instruction: DisplayInstruction = {
        role: "stroke.cap",
        kind: "enum",
        value: 123,
        category: "structure",
        appliedAtMs: 0,
      };
      expect(() => validateDisplayInstruction(instruction)).toThrow(
        /requires string value/,
      );
    });
  });

  describe("buildThemeDisplayMapping", () => {
    test("maps semantic roles to theme-derived display instructions", () => {
      const roleMapping = new Map([
        ["request", ["ink.primary", "accent.control"]],
        ["token", ["ink.muted"]],
      ] as const);

      const mapping = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 1500);

      expect(mapping.themeId).toBe("test-theme");
      expect(mapping.appliedPhase).toBe("apply");
      expect(Object.isFrozen(mapping)).toBe(true);

      const requestInstructions = mapping.semanticRoleToInstructions["request"];
      expect(requestInstructions).toHaveLength(2);
      expect(requestInstructions[0]?.role).toBe("ink.primary");
      expect(requestInstructions[1]?.role).toBe("accent.control");

      const tokenInstructions = mapping.semanticRoleToInstructions["token"];
      expect(tokenInstructions).toHaveLength(1);
      expect(tokenInstructions[0]?.role).toBe("ink.muted");
    });

    test("freezes semanticRoleToInstructions records", () => {
      const roleMapping = new Map([["request", ["ink.primary"]]]);
      const mapping = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 0);

      expect(Object.isFrozen(mapping.semanticRoleToInstructions)).toBe(true);
      const instructions = mapping.semanticRoleToInstructions["request"];
      expect(Object.isFrozen(instructions)).toBe(true);
    });

    test("infers category from role prefix convention", () => {
      const roleMapping = new Map([
        ["test-fg", ["ink.primary"]],
        ["test-bg", ["surface.canvas"]],
        ["test-accent", ["accent.control"]],
        ["test-typo", ["font.body"]],
        ["test-struct", ["stroke.standard"]],
        ["test-motion", ["motion.draw"]],
      ] as const);

      const mapping = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 0);

      expect(mapping.semanticRoleToInstructions["test-fg"][0]?.category).toBe("foreground");
      expect(mapping.semanticRoleToInstructions["test-bg"][0]?.category).toBe("background");
      expect(mapping.semanticRoleToInstructions["test-accent"][0]?.category).toBe("accent");
      expect(mapping.semanticRoleToInstructions["test-typo"][0]?.category).toBe("typography");
      expect(mapping.semanticRoleToInstructions["test-struct"][0]?.category).toBe("structure");
      expect(mapping.semanticRoleToInstructions["test-motion"][0]?.category).toBe("motion");
    });

    test("skips theme roles absent from the resolved theme", () => {
      const roleMapping = new Map([["test", ["ink.primary", "missing.role"]]]);
      const mapping = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 0);

      const instructions = mapping.semanticRoleToInstructions["test"];
      expect(instructions).toHaveLength(1);
      expect(instructions[0]?.role).toBe("ink.primary");
    });

    test("produces empty instruction list for unknown semantic roles", () => {
      const roleMapping = new Map([["request", []]]);
      const mapping = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 0);

      expect(mapping.semanticRoleToInstructions["request"]).toEqual([]);
    });
  });

  describe("ThemeRoleInventory", () => {
    let inventory: ThemeRoleInventory;

    test("constructs from a set of contracts", () => {
      inventory = new ThemeRoleInventory(testContracts);
      expect(inventory.allRoles()).toContain("ink.primary");
      expect(inventory.allRoles()).toHaveLength(testContracts.length);
    });

    test("retrieves contracts by role", () => {
      inventory = new ThemeRoleInventory(testContracts);
      const contract = inventory.contract("ink.primary");

      expect(contract?.role).toBe("ink.primary");
      expect(contract?.required).toBe(true);
      expect(contract?.category).toBe("foreground");
    });

    test("returns undefined for unregistered roles", () => {
      inventory = new ThemeRoleInventory(testContracts);
      expect(inventory.contract("unknown.role")).toBeUndefined();
    });

    test("lists roles by category", () => {
      inventory = new ThemeRoleInventory(testContracts);
      const fgRoles = inventory.rolesByCategory("foreground");

      expect(fgRoles).toContain("ink.primary");
      expect(fgRoles).toContain("ink.muted");
    });

    test("returns undefined for unknown categories", () => {
      inventory = new ThemeRoleInventory(testContracts);
      expect(inventory.rolesByCategory("unknown-category")).toBeUndefined();
    });

    test("identifies required roles", () => {
      inventory = new ThemeRoleInventory(testContracts);
      const required = inventory.requiredRoles();

      expect(required).toContain("ink.primary");
      expect(required).toContain("surface.canvas");
      expect(required).toContain("font.body");
      expect(required).toContain("weight.regular");
      expect(required).toContain("size.body");
      expect(required).not.toContain("ink.muted");
    });

    test("validates that a theme provides all required roles", () => {
      inventory = new ThemeRoleInventory(testContracts);
      expect(() => inventory.validateThemeCompleteness(baseTheme)).not.toThrow();
    });

    test("rejects themes missing required roles", () => {
      inventory = new ThemeRoleInventory(testContracts);
      const incompleteTheme: ResolvedTheme = {
        ...baseTheme,
        values: {
          "ink.primary": baseTheme.values["ink.primary"]!,
        } as Partial<Record<ThemeRole, ThemeValueIr>>,
      };

      expect(() => inventory.validateThemeCompleteness(incompleteTheme)).toThrow(
        /missing required roles/,
      );
    });

    test("rejects themes with unknown roles", () => {
      inventory = new ThemeRoleInventory(testContracts);
      const themeWithUnknownRole: ResolvedTheme = {
        ...baseTheme,
        values: {
          ...baseTheme.values,
          "unknown.role": { kind: "color", value: "#000000" },
        } as Partial<Record<ThemeRole, ThemeValueIr>>,
      };

      expect(() => inventory.validateThemeCompleteness(themeWithUnknownRole)).toThrow(
        /unknown roles/,
      );
    });
  });

  describe("theme application lifecycle", () => {
    test("display instructions are determined by phase", () => {
      const roleMapping = new Map([["request", ["ink.primary"]]]);

      const bootstrapMapping = buildThemeDisplayMapping(baseTheme, roleMapping, "bootstrap", 0);
      expect(bootstrapMapping.appliedPhase).toBe("bootstrap");

      const resolveMapping = buildThemeDisplayMapping(baseTheme, roleMapping, "resolve", 100);
      expect(resolveMapping.appliedPhase).toBe("resolve");

      const applyMapping = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 200);
      expect(applyMapping.appliedPhase).toBe("apply");

      const finalizeMapping = buildThemeDisplayMapping(baseTheme, roleMapping, "finalize", 300);
      expect(finalizeMapping.appliedPhase).toBe("finalize");
    });

    test("appliedAtMs tracks when instructions were extracted", () => {
      const roleMapping = new Map([["request", ["ink.primary"]]]);

      const mapping1 = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 1000);
      const mapping2 = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 2000);

      expect(mapping1.semanticRoleToInstructions["request"][0]?.appliedAtMs).toBe(1000);
      expect(mapping2.semanticRoleToInstructions["request"][0]?.appliedAtMs).toBe(2000);
    });
  });

  describe("role → instruction mapping contract", () => {
    test("a semantic role maps to multiple theme roles", () => {
      const roleMapping = new Map([
        ["entity", ["ink.primary", "accent.control", "font.body"]],
      ]);

      const mapping = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 0);
      const instructions = mapping.semanticRoleToInstructions["entity"];

      expect(instructions).toHaveLength(3);
      expect(instructions.map((i) => i.role)).toEqual([
        "ink.primary",
        "accent.control",
        "font.body",
      ]);
    });

    test("instruction order matches role order in mapping", () => {
      const roleMapping = new Map([
        ["entity", ["font.body", "ink.primary", "accent.control"]],
      ]);

      const mapping = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 0);
      const instructions = mapping.semanticRoleToInstructions["entity"];

      expect(instructions.map((i) => i.role)).toEqual([
        "font.body",
        "ink.primary",
        "accent.control",
      ]);
    });

    test("all instructions share the same appliedAtMs and themeId", () => {
      const roleMapping = new Map([
        ["entity", ["ink.primary", "accent.control", "font.body"]],
      ]);

      const mapping = buildThemeDisplayMapping(baseTheme, roleMapping, "apply", 5000);
      const instructions = mapping.semanticRoleToInstructions["entity"];

      expect(instructions.every((i) => i.appliedAtMs === 5000)).toBe(true);
      expect(mapping.themeId).toBe("test-theme");
    });
  });
});
