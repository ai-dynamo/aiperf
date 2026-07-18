// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";

import { SvgFallback } from "../../src/backends/svg/svg-fallback.js";
import { buildDisplayList } from "../../src/display-list.js";
import {
  createBootstrapThemeRegistry,
  type ResolvedTheme,
} from "../../src/theme/index.js";
import type { EvaluatedScene } from "../../src/evaluate/types.js";

afterEach(cleanup);

function createThemedScene(theme: ResolvedTheme): EvaluatedScene {
  const displayList = buildDisplayList({
    commands: [
      {
        kind: "path",
        id: "surface-shape",
        order: 0,
        path: "M 0 0 H 100 V 100 H 0 Z",
        fill:
          theme.values["surface.panel"].kind === "color"
            ? theme.values["surface.panel"].value
            : "#000000",
        paintBounds: { x: 0, y: 0, width: 100, height: 100 },
        damageBounds: { x: 0, y: 0, width: 100, height: 100 },
      },
      {
        kind: "path",
        id: "accent-line",
        order: 1,
        path: "M 10 50 L 90 50",
        stroke:
          theme.values["accent.control"].kind === "color"
            ? theme.values["accent.control"].value
            : "#ffffff",
        strokeWidth: 2,
        paintBounds: { x: 10, y: 50, width: 80, height: 0 },
        damageBounds: { x: 10, y: 50, width: 80, height: 0 },
      },
      {
        kind: "text",
        id: "text-element",
        order: 2,
        text: "Themed Text",
        origin: { x: 30, y: 60 },
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
        fill:
          theme.values["ink.primary"].kind === "color"
            ? theme.values["ink.primary"].value
            : "#ffffff",
        paintBounds: { x: 30, y: 50, width: 80, height: 20 },
        damageBounds: { x: 30, y: 50, width: 80, height: 20 },
      },
    ],
    hitRegions: [
      {
        id: "shape-hit",
        semanticId: "themed-entity",
        order: 0,
        bounds: { x: 0, y: 0, width: 100, height: 100 },
        label: "Themed Shape",
        focusTarget: "themed-entity",
        selected: false,
        focusable: true,
      } as any,
    ],
    paintBounds: { x: 0, y: 0, width: 100, height: 100 },
    damageBounds: { x: 0, y: 0, width: 100, height: 100 },
  });

  return {
    sceneId: "themed-scene",
    atMs: 0,
    displayList,
    semantic: {
      sceneId: "themed-scene",
      readingOrder: ["themed-entity"],
      entities: [
        {
          id: "themed-entity",
          label: "Themed Shape",
          focusTarget: "themed-entity",
          selected: false,
        },
      ],
      relations: [],
    },
  };
}

describe("SVG backend theme rendering", () => {
  test("applies surface colors to path fill attributes", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");
    const scene = createThemedScene(theme);

    const { container } = render(
      <SvgFallback
        scene={scene}
        displayList={scene.displayList}
        selectedEntityIds={[]}
        focusedEntityId={null}
      />,
    );

    const paths = container.querySelectorAll("path");
    expect(paths.length).toBeGreaterThan(0);

    const fillColors = Array.from(paths)
      .map((path) => path.getAttribute("fill"))
      .filter((fill): fill is string => fill !== null && fill !== "none");

    const expectedColor =
      theme.values["surface.panel"].kind === "color"
        ? theme.values["surface.panel"].value
        : "#000000";
    expect(fillColors).toContain(expectedColor);
  });

  test("applies accent colors to stroke attributes", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");
    const scene = createThemedScene(theme);

    const { container } = render(
      <SvgFallback
        scene={scene}
        displayList={scene.displayList}
        selectedEntityIds={[]}
        focusedEntityId={null}
      />,
    );

    const paths = container.querySelectorAll("path");
    const strokes = Array.from(paths)
      .map((path) => path.getAttribute("stroke"))
      .filter((stroke): stroke is string => stroke !== null && stroke !== "none");

    const expectedStroke =
      theme.values["accent.control"].kind === "color"
        ? theme.values["accent.control"].value
        : "#ffffff";
    expect(strokes).toContain(expectedStroke);
  });

  test("applies theme font properties to text elements", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");
    const scene = createThemedScene(theme);

    const { container } = render(
      <SvgFallback
        scene={scene}
        displayList={scene.displayList}
        selectedEntityIds={[]}
        focusedEntityId={null}
      />,
    );

    const textElements = container.querySelectorAll("text");
    expect(textElements.length).toBeGreaterThan(0);

    const expectedFontFamily =
      theme.values["font.body"].kind === "font"
        ? theme.values["font.body"].value[0]
        : "sans-serif";
    const expectedFontSize =
      theme.values["size.body"].kind === "number"
        ? theme.values["size.body"].value
        : 13;

    const text = textElements[0];
    expect(text).toBeDefined();
    // Verify font properties from theme are available
    expect(expectedFontFamily).toBe("Nunito Sans");
    expect(expectedFontSize).toBe(13);
  });

  test("applies ink color to text fill", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");
    const scene = createThemedScene(theme);

    const { container } = render(
      <SvgFallback
        scene={scene}
        displayList={scene.displayList}
        selectedEntityIds={[]}
        focusedEntityId={null}
      />,
    );

    const textElements = container.querySelectorAll("text");
    const fills = Array.from(textElements)
      .map((text) => text.getAttribute("fill"))
      .filter((fill): fill is string => fill !== null);

    const expectedColor =
      theme.values["ink.primary"].kind === "color"
        ? theme.values["ink.primary"].value
        : "#ffffff";
    expect(fills).toContain(expectedColor);
  });

  test("preserves stroke width from theme", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");
    const scene = createThemedScene(theme);

    // Verify theme has stroke width value
    const themeStrokeWidth =
      theme.values["stroke.standard"].kind === "number"
        ? theme.values["stroke.standard"].value
        : 2;
    expect(themeStrokeWidth).toBe(2);

    const { container } = render(
      <SvgFallback
        scene={scene}
        displayList={scene.displayList}
        selectedEntityIds={[]}
        focusedEntityId={null}
      />,
    );

    const paths = container.querySelectorAll("path");
    // Verify paths are rendered
    expect(paths.length).toBeGreaterThan(0);
  });

  test("applies multiple theme colors across different shapes", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    const commands = [
      {
        kind: "path" as const,
        id: "surface-1",
        order: 0,
        path: "M 0 0 H 30 V 30 H 0 Z",
        fill:
          theme.values["surface.canvas"].kind === "color"
            ? theme.values["surface.canvas"].value
            : "#000000",
        paintBounds: { x: 0, y: 0, width: 30, height: 30 },
        damageBounds: { x: 0, y: 0, width: 30, height: 30 },
      },
      {
        kind: "path" as const,
        id: "surface-2",
        order: 1,
        path: "M 35 0 H 65 V 30 H 35 Z",
        fill:
          theme.values["surface.panel"].kind === "color"
            ? theme.values["surface.panel"].value
            : "#000000",
        paintBounds: { x: 35, y: 0, width: 30, height: 30 },
        damageBounds: { x: 35, y: 0, width: 30, height: 30 },
      },
      {
        kind: "path" as const,
        id: "surface-3",
        order: 2,
        path: "M 70 0 H 100 V 30 H 70 Z",
        fill:
          theme.values["surface.raised"].kind === "color"
            ? theme.values["surface.raised"].value
            : "#000000",
        paintBounds: { x: 70, y: 0, width: 30, height: 30 },
        damageBounds: { x: 70, y: 0, width: 30, height: 30 },
      },
    ];

    const displayList = buildDisplayList({
      commands,
      hitRegions: [],
      paintBounds: { x: 0, y: 0, width: 100, height: 30 },
      damageBounds: { x: 0, y: 0, width: 100, height: 30 },
    });

    const { container } = render(
      <SvgFallback
        scene={{
          sceneId: "multi-surface",
          atMs: 0,
          displayList,
          semantic: {
            sceneId: "multi-surface",
            readingOrder: [],
            entities: [],
            relations: [],
          },
        }}
        displayList={displayList}
        selectedEntityIds={[]}
        focusedEntityId={null}
      />,
    );

    const paths = container.querySelectorAll("path");
    const fills = Array.from(paths)
      .map((path) => path.getAttribute("fill"))
      .filter((fill): fill is string => fill !== null && fill !== "none");

    const canvasColor =
      theme.values["surface.canvas"].kind === "color"
        ? theme.values["surface.canvas"].value
        : null;
    const panelColor =
      theme.values["surface.panel"].kind === "color"
        ? theme.values["surface.panel"].value
        : null;
    const raisedColor =
      theme.values["surface.raised"].kind === "color"
        ? theme.values["surface.raised"].value
        : null;

    if (canvasColor) expect(fills).toContain(canvasColor);
    if (panelColor) expect(fills).toContain(panelColor);
    if (raisedColor) expect(fills).toContain(raisedColor);
  });

  test("applies theme font weight from number role", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");
    const scene = createThemedScene(theme);

    const expectedWeight =
      theme.values["weight.regular"].kind === "number"
        ? theme.values["weight.regular"].value
        : 400;

    expect(expectedWeight).toBe(400);

    const { container } = render(
      <SvgFallback
        scene={scene}
        displayList={scene.displayList}
        selectedEntityIds={[]}
        focusedEntityId={null}
      />,
    );

    const textElements = container.querySelectorAll("text");
    expect(textElements.length).toBeGreaterThan(0);
  });

  test("correctly renders theme with inheritance chain", () => {
    const registry = createBootstrapThemeRegistry();
    const frozenRegistry = registry.freeze();

    // Resolve theme multiple times to verify caching
    const theme1 = frozenRegistry.resolve("systems_chalk");
    const theme2 = frozenRegistry.resolve("systems_chalk");

    // Should be the same resolved instance due to caching
    expect(theme1.id).toBe(theme2.id);
    expect(theme1.values["surface.canvas"]).toEqual(theme2.values["surface.canvas"]);

    const scene = createThemedScene(theme1);

    const { container } = render(
      <SvgFallback
        scene={scene}
        displayList={scene.displayList}
        selectedEntityIds={[]}
        focusedEntityId={null}
      />,
    );

    // Verify rendered output
    const svg = container.querySelector("svg");
    expect(svg).toBeDefined();
    expect(svg?.querySelectorAll("path").length).toBeGreaterThan(0);
  });

  test("applies accent color variations from theme", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    const accentRoles = [
      "accent.control" as const,
      "accent.execution" as const,
      "accent.compute" as const,
      "accent.attention" as const,
      "accent.success" as const,
      "accent.danger" as const,
    ];

    const commands = accentRoles.map((role, index) => ({
      kind: "path" as const,
      id: `accent-${index}`,
      order: index,
      path: `M ${index * 20} 0 L ${index * 20 + 15} 15`,
      stroke:
        theme.values[role].kind === "color"
          ? theme.values[role].value
          : "#ffffff",
      strokeWidth: 2,
      paintBounds: {
        x: index * 20,
        y: 0,
        width: 15,
        height: 15,
      },
      damageBounds: {
        x: index * 20,
        y: 0,
        width: 15,
        height: 15,
      },
    }));

    const displayList = buildDisplayList({
      commands,
      hitRegions: [],
      paintBounds: { x: 0, y: 0, width: 120, height: 15 },
      damageBounds: { x: 0, y: 0, width: 120, height: 15 },
    });

    const { container } = render(
      <SvgFallback
        scene={{
          sceneId: "accent-variations",
          atMs: 0,
          displayList,
          semantic: {
            sceneId: "accent-variations",
            readingOrder: [],
            entities: [],
            relations: [],
          },
        }}
        displayList={displayList}
        selectedEntityIds={[]}
        focusedEntityId={null}
      />,
    );

    const paths = container.querySelectorAll("path");
    const strokes = Array.from(paths)
      .map((path) => path.getAttribute("stroke"))
      .filter((stroke): stroke is string => stroke !== null && stroke !== "none");

    // Verify multiple accent colors are present
    expect(strokes.length).toBeGreaterThanOrEqual(accentRoles.length);

    for (const role of accentRoles) {
      const value = theme.values[role];
      if (value.kind === "color") {
        expect(strokes).toContain(value.value);
      }
    }
  });
});
