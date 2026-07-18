// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { FlowThemeIr, SceneIr } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import type { DrawCommand } from "../../src/display-list.js";
import {
  evaluateSceneWithTheme,
  resolveStyleValue,
  resolveThemeValue,
  themeValueToStyleValue,
  type ThemeContext,
} from "../../src/evaluate/with-theme.js";

const sourceMap = {
  source: "scene.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

function createBaseTheme(id: string = "base"): FlowThemeIr {
  return {
    id,
    extends: "",
    values: {
      "surface.canvas": { kind: "color", value: "#ffffff" },
      "surface.panel": { kind: "color", value: "#f5f5f5" },
      "ink.primary": { kind: "color", value: "#000000" },
      "ink.muted": { kind: "color", value: "#808080" },
      "font.body": { kind: "font", value: ["Roboto", "sans-serif"] },
      "size.body": { kind: "number", value: 14 },
      "weight.regular": { kind: "number", value: 400 },
      "motion.draw": { kind: "duration", valueMs: 300 },
      "stroke.cap": { kind: "enum", value: "round" },
    },
    sourceMap,
  };
}

function createDarkTheme(): FlowThemeIr {
  return {
    id: "dark",
    extends: "base",
    values: {
      "surface.canvas": { kind: "color", value: "#1a1a1a" },
      "surface.panel": { kind: "color", value: "#2a2a2a" },
      "ink.primary": { kind: "color", value: "#ffffff" },
    },
    sourceMap,
  };
}

function createScene(): SceneIr {
  return {
    id: "test-scene",
    title: "Test Scene",
    summary: "Scene for theme testing",
    roots: [
      {
        kind: "group",
        id: "root",
        geometry: { x: 0, y: 0, width: 640, height: 360 },
        style: {
          fill: { kind: "theme-role", role: "surface.canvas" },
        },
        accessibility: { label: "Root group" },
        fallback: "Group unavailable",
        sourceMap,
        children: [
          {
            kind: "rect",
            id: "panel",
            geometry: { x: 20, y: 30, width: 200, height: 100 },
            style: {
              fill: { kind: "theme-role", role: "surface.panel" },
              stroke: { kind: "theme-role", role: "ink.primary" },
            },
            accessibility: { label: "Panel" },
            fallback: "Panel unavailable",
            sourceMap,
          },
          {
            kind: "text",
            id: "label",
            geometry: { x: 40, y: 50, width: 100, height: 24 },
            style: {
              fill: { kind: "theme-role", role: "ink.primary" },
              fontFamily: { kind: "theme-role", role: "font.body" },
              fontSize: { kind: "theme-role", role: "size.body" },
              fontWeight: { kind: "theme-role", role: "weight.regular" },
            },
            accessibility: { label: "Text label" },
            fallback: "Label unavailable",
            sourceMap,
            text: "Hello World",
          },
        ],
      },
    ],
    camera: [],
    timeline: [],
    narration: "",
    interactions: [],
    responsive: [],
    accessibility: {
      label: "Test scene",
      readingOrder: ["panel", "label"],
    },
    fallback: "Scene unavailable",
    sourceMap,
  };
}

describe("resolveThemeValue", () => {
  test("resolves value from active theme", () => {
    const baseTheme = createBaseTheme();
    const context: ThemeContext = {
      activeTheme: baseTheme,
      allThemes: [baseTheme],
    };

    const result = resolveThemeValue("surface.canvas", context);

    expect(result).toEqual({ kind: "color", value: "#ffffff" });
  });

  test("resolves value from inherited theme", () => {
    const baseTheme = createBaseTheme();
    const darkTheme = createDarkTheme();
    const context: ThemeContext = {
      activeTheme: darkTheme,
      allThemes: [baseTheme, darkTheme],
    };

    // "size.body" is not in darkTheme, but should be found in baseTheme
    const result = resolveThemeValue("size.body", context);

    expect(result).toEqual({ kind: "number", value: 14 });
  });

  test("prefers active theme over inherited values", () => {
    const baseTheme = createBaseTheme();
    const darkTheme = createDarkTheme();
    const context: ThemeContext = {
      activeTheme: darkTheme,
      allThemes: [baseTheme, darkTheme],
    };

    // "surface.canvas" is overridden in darkTheme
    const result = resolveThemeValue("surface.canvas", context);

    expect(result).toEqual({ kind: "color", value: "#1a1a1a" });
  });

  test("returns undefined for unknown roles", () => {
    const baseTheme = createBaseTheme();
    const context: ThemeContext = {
      activeTheme: baseTheme,
      allThemes: [baseTheme],
    };

    const result = resolveThemeValue("nonexistent.role" as any, context);

    expect(result).toBeUndefined();
  });

  test("handles multiple inheritance levels", () => {
    const rootTheme: FlowThemeIr = {
      id: "root",
      extends: "",
      values: {
        "surface.canvas": { kind: "color", value: "#aaaaaa" },
      },
      sourceMap,
    };

    const midTheme: FlowThemeIr = {
      id: "mid",
      extends: "root",
      values: {
        "surface.panel": { kind: "color", value: "#bbbbbb" },
      },
      sourceMap,
    };

    const leafTheme: FlowThemeIr = {
      id: "leaf",
      extends: "mid",
      values: {
        "ink.primary": { kind: "color", value: "#cccccc" },
      },
      sourceMap,
    };

    const context: ThemeContext = {
      activeTheme: leafTheme,
      allThemes: [rootTheme, midTheme, leafTheme],
    };

    expect(resolveThemeValue("surface.canvas", context)).toEqual({
      kind: "color",
      value: "#aaaaaa",
    });
    expect(resolveThemeValue("surface.panel", context)).toEqual({
      kind: "color",
      value: "#bbbbbb",
    });
    expect(resolveThemeValue("ink.primary", context)).toEqual({
      kind: "color",
      value: "#cccccc",
    });
  });
});

describe("themeValueToStyleValue", () => {
  test("converts color theme value to string", () => {
    const value = { kind: "color" as const, value: "#ff0000" };
    expect(themeValueToStyleValue(value)).toBe("#ff0000");
  });

  test("converts font theme value to CSS font-family string", () => {
    const value = { kind: "font" as const, value: ["Arial", "sans-serif"] };
    expect(themeValueToStyleValue(value)).toBe('"Arial", "sans-serif"');
  });

  test("converts number theme value to numeric", () => {
    const value = { kind: "number" as const, value: 42 };
    expect(themeValueToStyleValue(value)).toBe(42);
  });

  test("converts duration theme value to milliseconds", () => {
    const value = { kind: "duration" as const, valueMs: 500 };
    expect(themeValueToStyleValue(value)).toBe(500);
  });

  test("converts enum theme value to string", () => {
    const value = { kind: "enum" as const, value: "round" };
    expect(themeValueToStyleValue(value)).toBe("round");
  });
});

describe("resolveStyleValue", () => {
  test("resolves theme role reference", () => {
    const baseTheme = createBaseTheme();
    const context: ThemeContext = {
      activeTheme: baseTheme,
      allThemes: [baseTheme],
    };

    const styleValue = { kind: "theme-role" as const, role: "surface.canvas" as const };
    const result = resolveStyleValue(styleValue, context);

    expect(result).toBe("#ffffff");
  });

  test("returns scalar value when not a theme reference", () => {
    const baseTheme = createBaseTheme();
    const context: ThemeContext = {
      activeTheme: baseTheme,
      allThemes: [baseTheme],
    };

    expect(resolveStyleValue("#ff0000", context)).toBe("#ff0000");
    expect(resolveStyleValue(42, context)).toBe(42);
    expect(resolveStyleValue(true, context)).toBe(true);
  });

  test("returns undefined for unresolved theme role", () => {
    const baseTheme = createBaseTheme();
    const context: ThemeContext = {
      activeTheme: baseTheme,
      allThemes: [baseTheme],
    };

    const styleValue = { kind: "theme-role" as const, role: "nonexistent.role" as any };
    const result = resolveStyleValue(styleValue, context);

    expect(result).toBeUndefined();
  });

  test("returns scalar when context is undefined", () => {
    expect(resolveStyleValue("#ff0000", undefined)).toBe("#ff0000");
    expect(resolveStyleValue(42, undefined)).toBe(42);
    expect(resolveStyleValue(true, undefined)).toBe(true);
  });

  test("returns undefined for theme reference when context is undefined", () => {
    const styleValue = { kind: "theme-role" as const, role: "surface.canvas" as const };
    expect(resolveStyleValue(styleValue, undefined)).toBeUndefined();
  });
});

describe("evaluateSceneWithTheme", () => {
  test("evaluates scene without theme context", () => {
    const scene = createScene();

    const result = evaluateSceneWithTheme(scene);

    expect(result).toBeDefined();
    expect(result.sceneId).toBe("test-scene");
    // Root group is the single command at the top level
    expect(result.displayList.commands).toHaveLength(1);
  });

  test("evaluates scene with base theme", () => {
    const scene = createScene();
    const baseTheme = createBaseTheme();
    const context: ThemeContext = {
      activeTheme: baseTheme,
      allThemes: [baseTheme],
    };

    const result = evaluateSceneWithTheme(scene, 0, { themeContext: context });

    expect(result).toBeDefined();
    expect(result.sceneId).toBe("test-scene");
    expect(result.atMs).toBe(0);

    // Verify theme values are applied to display commands
    const flatten = (commands: readonly DrawCommand[]): readonly DrawCommand[] =>
      commands.flatMap((command) =>
        command.kind === "group" ||
        command.kind === "clip" ||
        command.kind === "layer"
          ? [command, ...flatten(command.children)]
          : [command],
      );
    const allCommands = flatten(result.displayList.commands);
    const textCommand = allCommands.find((c) => c.kind === "text");
    expect(textCommand).toBeDefined();
    if (textCommand && textCommand.kind === "text") {
      expect(textCommand.fill).toBe("#000000");
      expect(textCommand.font.family).toBe('"Roboto", "sans-serif"');
      expect(textCommand.font.sizePx).toBe(14);
      expect(textCommand.font.weight).toBe(400);
    }
  });

  test("applies dark theme correctly", () => {
    const scene = createScene();
    const baseTheme = createBaseTheme();
    const darkTheme = createDarkTheme();
    const context: ThemeContext = {
      activeTheme: darkTheme,
      allThemes: [baseTheme, darkTheme],
    };

    const result = evaluateSceneWithTheme(scene, 0, { themeContext: context });

    expect(result).toBeDefined();

    // Verify dark theme values override base theme
    const flatten = (commands: readonly DrawCommand[]): readonly DrawCommand[] =>
      commands.flatMap((command) =>
        command.kind === "group" ||
        command.kind === "clip" ||
        command.kind === "layer"
          ? [command, ...flatten(command.children)]
          : [command],
      );
    const allCommands = flatten(result.displayList.commands);
    const textCommand = allCommands.find((c) => c.kind === "text");
    expect(textCommand).toBeDefined();
    if (textCommand && textCommand.kind === "text") {
      // Dark theme has white ink.primary
      expect(textCommand.fill).toBe("#ffffff");
    }
  });

  test("handles theme inheritance correctly", () => {
    const scene = createScene();
    const baseTheme = createBaseTheme();
    const darkTheme = createDarkTheme();
    const context: ThemeContext = {
      activeTheme: darkTheme,
      allThemes: [baseTheme, darkTheme],
    };

    const result = evaluateSceneWithTheme(scene, 0, { themeContext: context });

    // Verify inherited values from base theme
    const flatten = (commands: readonly DrawCommand[]): readonly DrawCommand[] =>
      commands.flatMap((command) =>
        command.kind === "group" ||
        command.kind === "clip" ||
        command.kind === "layer"
          ? [command, ...flatten(command.children)]
          : [command],
      );
    const allCommands = flatten(result.displayList.commands);
    const textCommand = allCommands.find((c) => c.kind === "text");
    expect(textCommand).toBeDefined();
    if (textCommand && textCommand.kind === "text") {
      // font.body should be inherited from base theme
      expect(textCommand.font.family).toBe('"Roboto", "sans-serif"');
      // size.body should be inherited from base theme
      expect(textCommand.font.sizePx).toBe(14);
    }
  });

  test("preserves scene structure and semantic projection", () => {
    const scene = createScene();
    const baseTheme = createBaseTheme();
    const context: ThemeContext = {
      activeTheme: baseTheme,
      allThemes: [baseTheme],
    };

    const result = evaluateSceneWithTheme(scene, 0, { themeContext: context });

    expect(result.semantic).toBeDefined();
    expect(result.semantic.sceneId).toBe("test-scene");
    expect(result.semantic.entities).toHaveLength(2);
    expect(result.semantic.readingOrder).toEqual(["panel", "label"]);
  });

  test("respects evaluation time parameter", () => {
    const scene = createScene();
    const baseTheme = createBaseTheme();
    const context: ThemeContext = {
      activeTheme: baseTheme,
      allThemes: [baseTheme],
    };

    const result = evaluateSceneWithTheme(scene, 1500, { themeContext: context });

    expect(result.atMs).toBe(1500);
  });

  test("handles scenes with scalar style values", () => {
    const sceneWithScalars: SceneIr = {
      id: "scalar-scene",
      title: "Scalar Scene",
      summary: "Scene with scalar values",
      roots: [
        {
          kind: "rect",
          id: "rect-with-scalar",
          geometry: { x: 0, y: 0, width: 100, height: 100 },
          style: {
            fill: "#ff0000", // Scalar color
            strokeWidth: 2, // Scalar number
          },
          accessibility: { label: "Scalar rect" },
          fallback: "Rect unavailable",
          sourceMap,
        },
      ],
      camera: [],
      timeline: [],
      narration: "",
      interactions: [],
      responsive: [],
      accessibility: {
        label: "Scalar scene",
        readingOrder: ["rect-with-scalar"],
      },
      fallback: "Scene unavailable",
      sourceMap,
    };

    const baseTheme = createBaseTheme();
    const context: ThemeContext = {
      activeTheme: baseTheme,
      allThemes: [baseTheme],
    };

    const result = evaluateSceneWithTheme(sceneWithScalars, 0, { themeContext: context });

    expect(result).toBeDefined();
    const rectCommand = result.displayList.commands[0];
    if (rectCommand && rectCommand.kind === "path") {
      expect(rectCommand.fill).toBe("#ff0000");
      expect(rectCommand.strokeWidth).toBe(2);
    }
  });

  test("handles mixed theme and scalar values", () => {
    const sceneMixed: SceneIr = {
      id: "mixed-scene",
      title: "Mixed Scene",
      summary: "Scene with mixed values",
      roots: [
        {
          kind: "text",
          id: "mixed-text",
          geometry: { x: 0, y: 0, width: 100, height: 24 },
          style: {
            fill: { kind: "theme-role", role: "ink.primary" }, // Theme ref
            fontFamily: "Arial", // Scalar
            fontSize: 16, // Scalar
          },
          accessibility: { label: "Mixed text" },
          fallback: "Text unavailable",
          sourceMap,
          text: "Mixed",
        },
      ],
      camera: [],
      timeline: [],
      narration: "",
      interactions: [],
      responsive: [],
      accessibility: {
        label: "Mixed scene",
        readingOrder: ["mixed-text"],
      },
      fallback: "Scene unavailable",
      sourceMap,
    };

    const baseTheme = createBaseTheme();
    const context: ThemeContext = {
      activeTheme: baseTheme,
      allThemes: [baseTheme],
    };

    const result = evaluateSceneWithTheme(sceneMixed, 0, { themeContext: context });

    expect(result).toBeDefined();
    const textCommand = result.displayList.commands[0];
    if (textCommand && textCommand.kind === "text") {
      expect(textCommand.fill).toBe("#000000"); // Resolved from theme
      expect(textCommand.font.family).toBe("Arial"); // Kept scalar
      expect(textCommand.font.sizePx).toBe(16); // Kept scalar
    }
  });
});
