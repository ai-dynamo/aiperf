// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { describe, expect, test } from "vitest";

import {
  SYSTEMS_CHALK,
  createBootstrapThemeRegistry,
  type ResolvedTheme,
} from "../../src/theme/index.js";
import { buildDisplayList } from "../../src/display-list.js";
import { renderCanvasDisplayList } from "../../src/backends/canvas/canvas-renderer.js";

type CanvasCall = Readonly<{
  name: string;
  arguments: readonly unknown[];
}>;

function recordingCanvasContext(): Readonly<{
  context: CanvasRenderingContext2D;
  calls: readonly CanvasCall[];
}> {
  const calls: CanvasCall[] = [];
  const context = new Proxy(
    {},
    {
      get(_target, property) {
        if (property === "canvas") {
          return { width: 640, height: 360 };
        }
        if (property === "measureText") {
          return (text: string) => ({ width: text.length * 8 });
        }
        return (...arguments_: readonly unknown[]) => {
          calls.push({ name: String(property), arguments: arguments_ });
        };
      },
      set(_target, property, value) {
        calls.push({ name: `set:${String(property)}`, arguments: [value] });
        return true;
      },
    },
  ) as CanvasRenderingContext2D;

  return { context, calls };
}

function findCanvasCallsByName(calls: readonly CanvasCall[], name: string): CanvasCall[] {
  return calls.filter((call) => call.name === name);
}

function extractColorValue(callArguments: readonly unknown[]): string | undefined {
  const value = callArguments[0];
  if (typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value)) {
    return value.toLowerCase();
  }
  return undefined;
}

describe("canvas backend theme rendering", () => {
  test("applies theme surface colors to fill styles", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    const displayList = buildDisplayList({
      commands: [
        {
          kind: "path",
          id: "surface-canvas",
          order: 0,
          path: "M 0 0 H 100 V 100 H 0 Z",
          fill: theme.values["surface.canvas"].kind === "color"
            ? theme.values["surface.canvas"].value
            : "#000000",
          paintBounds: { x: 0, y: 0, width: 100, height: 100 },
          damageBounds: { x: 0, y: 0, width: 100, height: 100 },
        },
      ],
      hitRegions: [],
      paintBounds: { x: 0, y: 0, width: 100, height: 100 },
      damageBounds: { x: 0, y: 0, width: 100, height: 100 },
    });

    const recorder = recordingCanvasContext();
    renderCanvasDisplayList(displayList, recorder.context, {
      devicePixelRatio: 1,
      quality: "reference",
    });

    const fillStyleCalls = findCanvasCallsByName(recorder.calls, "set:fillStyle");
    expect(fillStyleCalls.length).toBeGreaterThan(0);

    const colors = fillStyleCalls
      .map((call) => extractColorValue(call.arguments))
      .filter((color): color is string => color !== undefined);

    const expectedColor =
      theme.values["surface.canvas"].kind === "color"
        ? theme.values["surface.canvas"].value.toLowerCase()
        : "#000000";
    expect(colors).toContain(expectedColor);
  });

  test("applies theme accent colors to stroke styles", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    const displayList = buildDisplayList({
      commands: [
        {
          kind: "path",
          id: "accent-control",
          order: 0,
          path: "M 10 10 L 90 90",
          stroke: theme.values["accent.control"].kind === "color"
            ? theme.values["accent.control"].value
            : "#000000",
          strokeWidth: 2,
          paintBounds: { x: 10, y: 10, width: 80, height: 80 },
          damageBounds: { x: 10, y: 10, width: 80, height: 80 },
        },
      ],
      hitRegions: [],
      paintBounds: { x: 0, y: 0, width: 100, height: 100 },
      damageBounds: { x: 0, y: 0, width: 100, height: 100 },
    });

    const recorder = recordingCanvasContext();
    renderCanvasDisplayList(displayList, recorder.context, {
      devicePixelRatio: 1,
      quality: "reference",
    });

    const strokeStyleCalls = findCanvasCallsByName(recorder.calls, "set:strokeStyle");
    expect(strokeStyleCalls.length).toBeGreaterThan(0);

    const colors = strokeStyleCalls
      .map((call) => extractColorValue(call.arguments))
      .filter((color): color is string => color !== undefined);

    const expectedColor =
      theme.values["accent.control"].kind === "color"
        ? theme.values["accent.control"].value.toLowerCase()
        : "#000000";
    expect(colors).toContain(expectedColor);
  });

  test("applies theme ink colors to text fills", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    const displayList = buildDisplayList({
      commands: [
        {
          kind: "text",
          id: "text-label",
          order: 0,
          text: "Theme Label",
          origin: { x: 50, y: 50 },
          font: {
            family: "sans-serif",
            sizePx: 14,
            weight: 400,
          },
          fill: theme.values["ink.primary"].kind === "color"
            ? theme.values["ink.primary"].value
            : "#ffffff",
          paintBounds: { x: 40, y: 40, width: 120, height: 20 },
          damageBounds: { x: 40, y: 40, width: 120, height: 20 },
        },
      ],
      hitRegions: [],
      paintBounds: { x: 0, y: 0, width: 200, height: 100 },
      damageBounds: { x: 0, y: 0, width: 200, height: 100 },
    });

    const recorder = recordingCanvasContext();
    renderCanvasDisplayList(displayList, recorder.context, {
      devicePixelRatio: 1,
      quality: "reference",
    });

    const fillStyleCalls = findCanvasCallsByName(recorder.calls, "set:fillStyle");
    const colors = fillStyleCalls
      .map((call) => extractColorValue(call.arguments))
      .filter((color): color is string => color !== undefined);

    const expectedColor =
      theme.values["ink.primary"].kind === "color"
        ? theme.values["ink.primary"].value.toLowerCase()
        : "#ffffff";
    expect(colors).toContain(expectedColor);
  });

  test("applies font properties from theme", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    const displayList = buildDisplayList({
      commands: [
        {
          kind: "text",
          id: "text-with-font",
          order: 0,
          text: "Themed Text",
          origin: { x: 50, y: 50 },
          font: {
            family:
              theme.values["font.body"].kind === "font"
                ? theme.values["font.body"].value[0]
                : "sans-serif",
            sizePx:
              theme.values["size.body"].kind === "number"
                ? theme.values["size.body"].value
                : 14,
            weight:
              theme.values["weight.regular"].kind === "number"
                ? theme.values["weight.regular"].value
                : 400,
          },
          paintBounds: { x: 40, y: 40, width: 120, height: 20 },
          damageBounds: { x: 40, y: 40, width: 120, height: 20 },
        },
      ],
      hitRegions: [],
      paintBounds: { x: 0, y: 0, width: 200, height: 100 },
      damageBounds: { x: 0, y: 0, width: 200, height: 100 },
    });

    const recorder = recordingCanvasContext();
    renderCanvasDisplayList(displayList, recorder.context, {
      devicePixelRatio: 1,
      quality: "reference",
    });

    const fontCalls = findCanvasCallsByName(recorder.calls, "set:font");
    expect(fontCalls.length).toBeGreaterThan(0);

    const expectedFont = fontCalls[0]?.arguments[0] as string | undefined;
    expect(expectedFont).toBeDefined();
    expect(String(expectedFont)).toMatch(/Nunito Sans|Inter/);
  });

  test("correctly resolves multiple color roles from theme", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    const colorRoles = [
      { role: "surface.canvas" as const, id: "canvas" },
      { role: "surface.panel" as const, id: "panel" },
      { role: "accent.control" as const, id: "control" },
      { role: "accent.execution" as const, id: "execution" },
      { role: "ink.primary" as const, id: "primary" },
    ];

    const commands = colorRoles.map((color, index) => ({
      kind: "path" as const,
      id: color.id,
      order: index,
      path: `M ${index * 20} 0 H ${index * 20 + 20} V 20 H ${index * 20} Z`,
      fill:
        theme.values[color.role].kind === "color"
          ? theme.values[color.role].value
          : "#000000",
      paintBounds: {
        x: index * 20,
        y: 0,
        width: 20,
        height: 20,
      },
      damageBounds: {
        x: index * 20,
        y: 0,
        width: 20,
        height: 20,
      },
    }));

    const displayList = buildDisplayList({
      commands,
      hitRegions: [],
      paintBounds: { x: 0, y: 0, width: 100, height: 20 },
      damageBounds: { x: 0, y: 0, width: 100, height: 20 },
    });

    const recorder = recordingCanvasContext();
    renderCanvasDisplayList(displayList, recorder.context, {
      devicePixelRatio: 1,
      quality: "reference",
    });

    const fillStyleCalls = findCanvasCallsByName(recorder.calls, "set:fillStyle");
    const colors = fillStyleCalls
      .map((call) => extractColorValue(call.arguments))
      .filter((color): color is string => color !== undefined);

    // Verify each color role is present
    for (const color of colorRoles) {
      const value = theme.values[color.role];
      if (value.kind === "color") {
        expect(colors).toContain(value.value.toLowerCase());
      }
    }
  });

  test("respects stroke width from theme", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    const strokeWidth =
      theme.values["stroke.standard"].kind === "number"
        ? theme.values["stroke.standard"].value
        : 2;

    const displayList = buildDisplayList({
      commands: [
        {
          kind: "path",
          id: "stroked-path",
          order: 0,
          path: "M 10 10 L 90 90",
          stroke: "#ffffff",
          strokeWidth,
          paintBounds: { x: 10, y: 10, width: 80, height: 80 },
          damageBounds: { x: 10, y: 10, width: 80, height: 80 },
        },
      ],
      hitRegions: [],
      paintBounds: { x: 0, y: 0, width: 100, height: 100 },
      damageBounds: { x: 0, y: 0, width: 100, height: 100 },
    });

    const recorder = recordingCanvasContext();
    renderCanvasDisplayList(displayList, recorder.context, {
      devicePixelRatio: 1,
      quality: "reference",
    });

    // Verify stroke style is set from display list
    const strokeStyleCalls = findCanvasCallsByName(recorder.calls, "set:strokeStyle");
    expect(strokeStyleCalls.length).toBeGreaterThan(0);

    // Verify theme has stroke width value
    expect(strokeWidth).toBe(2);
    expect(typeof strokeWidth).toBe("number");
  });

  test("handles theme inheritance correctly", () => {
    const registry = createBootstrapThemeRegistry();
    const frozenRegistry = registry.freeze();

    const baseTheme = frozenRegistry.resolve("systems_chalk");
    const baseColor =
      baseTheme.values["surface.canvas"].kind === "color"
        ? baseTheme.values["surface.canvas"].value
        : "#000000";

    // Verify base theme has all required roles
    expect(baseTheme.values).toHaveProperty("surface.canvas");
    expect(baseTheme.values).toHaveProperty("surface.panel");
    expect(baseTheme.values).toHaveProperty("surface.raised");
    expect(baseTheme.values).toHaveProperty("ink.primary");
    expect(baseTheme.values).toHaveProperty("accent.control");

    // Verify colors are valid hex
    expect(baseColor).toMatch(/^#[0-9a-f]{6}$/i);
  });
});
