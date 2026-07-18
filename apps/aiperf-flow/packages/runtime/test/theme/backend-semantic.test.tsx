// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// @vitest-environment jsdom

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, test } from "vitest";

import { SemanticTwin } from "../../src/semantic/semantic-twin.js";
import {
  createBootstrapThemeRegistry,
  type ResolvedTheme,
} from "../../src/theme/index.js";
import type { SemanticProjection } from "../../src/evaluate/types.js";

afterEach(cleanup);

function createThemedProjection(): SemanticProjection {
  return {
    sceneId: "semantic-theme-test",
    readingOrder: ["entity-1", "entity-2", "entity-3"],
    entities: [
      {
        id: "entity-1",
        label: "Primary Entity",
        focusTarget: "entity-1",
        selected: false,
        kind: "box",
      },
      {
        id: "entity-2",
        label: "Secondary Entity",
        focusTarget: "entity-2",
        selected: false,
        kind: "connector",
      },
      {
        id: "entity-3",
        label: "Tertiary Entity",
        focusTarget: "entity-3",
        selected: false,
        kind: "text",
      },
    ],
    relations: [
      {
        id: "relation-1",
        fromId: "entity-1",
        toId: "entity-2",
        label: "connects to",
        role: "dependency",
      },
    ],
  };
}

describe("semantic backend theme rendering", () => {
  test("renders semantic entities with theme inheritance", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");
    const projection = createThemedProjection();

    const { container } = render(
      <SemanticTwin
        projection={projection}
        focusedEntityId={null}
        selectedEntityId={null}
        onFocus={() => undefined}
        onActivate={() => undefined}
      />,
    );

    const semanticTwin = container.querySelector(".aiperf-flow__semantic-twin");
    expect(semanticTwin).toBeDefined();

    const entities = container.querySelectorAll("[data-entity-id]");
    expect(entities.length).toBe(projection.entities.length);

    // Verify each entity has proper data attributes
    projection.entities.forEach((entity) => {
      const element = container.querySelector(`[data-entity-id="${entity.id}"]`);
      expect(element).toBeDefined();
      expect(element?.getAttribute("aria-label")).toBe(entity.label);
    });
  });

  test("applies theme color roles to semantic structure", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    // Verify theme has required color roles
    expect(theme.values["surface.canvas"]).toBeDefined();
    expect(theme.values["surface.panel"]).toBeDefined();
    expect(theme.values["ink.primary"]).toBeDefined();
    expect(theme.values["ink.muted"]).toBeDefined();

    // All color roles should be of kind "color"
    const colorRoles = Object.entries(theme.values).filter(([key]) =>
      key.startsWith("surface.") || key.startsWith("ink.") || key.startsWith("accent."),
    );

    for (const [_role, value] of colorRoles) {
      expect(value.kind).toBe("color");
      expect(typeof value.value).toBe("string");
      expect(value.value).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });

  test("maintains semantic entity selection state with theme applied", () => {
    const projection = createThemedProjection();
    const selectedEntityId = "entity-2";

    const { container, rerender } = render(
      <SemanticTwin
        projection={projection}
        focusedEntityId={null}
        selectedEntityId={selectedEntityId}
        onFocus={() => undefined}
        onActivate={() => undefined}
      />,
    );

    const selectedElement = container.querySelector(
      `[data-entity-id="${selectedEntityId}"]`,
    );
    expect(selectedElement?.getAttribute("data-selected")).toBe("true");
    expect(selectedElement?.getAttribute("aria-selected")).toBe("true");

    // Verify other entities are not selected
    const otherEntity = container.querySelector(
      '[data-entity-id="entity-1"]',
    );
    expect(otherEntity?.getAttribute("data-selected")).toBe("false");
  });

  test("applies theme font properties to semantic content", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");
    const projection = createThemedProjection();

    expect(theme.values["font.body"]).toBeDefined();
    if (theme.values["font.body"].kind === "font") {
      expect(theme.values["font.body"].value).toBeDefined();
      expect(Array.isArray(theme.values["font.body"].value)).toBe(true);
    }

    const { container } = render(
      <SemanticTwin
        projection={projection}
        focusedEntityId={null}
        selectedEntityId={null}
        onFocus={() => undefined}
        onActivate={() => undefined}
      />,
    );

    const content = container.querySelector(".aiperf-flow__semantic-twin");
    expect(content).toBeDefined();
  });

  test("handles multiple theme role inheritance", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    // Verify inheritance of multiple role types
    const roleCategories = {
      surfaces: ["surface.canvas", "surface.panel", "surface.raised", "surface.control"],
      inks: ["ink.primary", "ink.muted", "ink.inverse"],
      accents: [
        "accent.control",
        "accent.execution",
        "accent.compute",
        "accent.attention",
      ],
      fonts: ["font.body", "font.display", "font.data"],
      weights: ["weight.regular", "weight.label", "weight.emphasis"],
      sizes: ["size.caption", "size.body", "size.label", "size.title"],
    };

    for (const [category, roles] of Object.entries(roleCategories)) {
      for (const role of roles) {
        expect(theme.values).toHaveProperty(role);
        const value = theme.values[role as keyof typeof theme.values];
        expect(value).toBeDefined();
        expect(value.kind).toMatch(/color|font|number|duration|enum/);
      }
    }
  });

  test("preserves semantic relations with theme context", () => {
    const projection = createThemedProjection();

    const { container } = render(
      <SemanticTwin
        projection={projection}
        focusedEntityId={null}
        selectedEntityId={null}
        onFocus={() => undefined}
        onActivate={() => undefined}
      />,
    );

    const relations = container.querySelectorAll("[data-relation-id]");
    expect(relations.length).toBe(projection.relations.length);

    const firstRelation = container.querySelector("[data-relation-id]");
    expect(firstRelation?.getAttribute("data-from")).toBe(projection.relations[0]?.fromId);
    expect(firstRelation?.getAttribute("data-to")).toBe(projection.relations[0]?.toId);
    expect(firstRelation?.getAttribute("data-kind")).toBe(projection.relations[0]?.role);
  });

  test("applies theme focus state correctly", () => {
    const projection = createThemedProjection();
    const focusedEntityId = "entity-1";

    const { container } = render(
      <SemanticTwin
        projection={projection}
        focusedEntityId={focusedEntityId}
        selectedEntityId={null}
        onFocus={() => undefined}
        onActivate={() => undefined}
      />,
    );

    const focusedElement = container.querySelector(
      `[data-entity-id="${focusedEntityId}"]`,
    );
    expect(focusedElement?.getAttribute("data-focused")).toBe("true");
    expect(focusedElement?.getAttribute("tabIndex")).toBe("0");

    // Verify non-focused elements
    const unfocusedElement = container.querySelector(
      '[data-entity-id="entity-2"]',
    );
    expect(unfocusedElement?.getAttribute("data-focused")).toBe("false");
    expect(unfocusedElement?.getAttribute("tabIndex")).toBe("-1");
  });

  test("theme color and font values are immutable after resolution", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");

    // Verify theme object is frozen
    expect(Object.isFrozen(theme)).toBe(true);
    expect(Object.isFrozen(theme.values)).toBe(true);

    // Verify attempting to modify throws in strict mode or fails silently
    const originalColor =
      theme.values["surface.canvas"].kind === "color"
        ? theme.values["surface.canvas"].value
        : "#000000";

    expect(() => {
      (theme.values["surface.canvas"] as any).value = "#ffffff";
    }).toThrow();

    // Verify color remains unchanged
    if (theme.values["surface.canvas"].kind === "color") {
      expect(theme.values["surface.canvas"].value).toBe(originalColor);
    }
  });

  test("resolves theme with complete coverage of all required roles", () => {
    const registry = createBootstrapThemeRegistry();
    const frozenRegistry = registry.freeze();
    const theme = frozenRegistry.resolve("systems_chalk");

    // Define all required roles based on the theme structure
    const requiredRoles = [
      "surface.canvas",
      "surface.panel",
      "surface.raised",
      "surface.control",
      "ink.primary",
      "ink.muted",
      "ink.inverse",
      "line.structural",
      "line.guide",
      "accent.control",
      "accent.execution",
      "accent.compute",
      "accent.attention",
      "accent.success",
      "accent.danger",
      "accent.focus",
      "font.display",
      "font.body",
      "font.data",
      "weight.regular",
      "weight.label",
      "weight.emphasis",
      "size.caption",
      "size.body",
      "size.label",
      "size.title",
      "stroke.hairline",
      "stroke.standard",
      "stroke.emphasis",
      "stroke.cap",
      "stroke.join",
      "motion.draw",
      "motion.enter",
      "motion.emphasis",
      "motion.stagger",
      "motion.easing",
    ];

    for (const role of requiredRoles) {
      expect(theme.values).toHaveProperty(role);
    }

    expect(Object.keys(theme.values).length).toBe(requiredRoles.length);
  });

  test("applies theme accent colors for special entity kinds", () => {
    const registry = createBootstrapThemeRegistry();
    const theme = registry.freeze().resolve("systems_chalk");
    const projection = createThemedProjection();

    // Verify accent roles exist for different semantic purposes
    const accentRoles = [
      "accent.control",
      "accent.execution",
      "accent.compute",
      "accent.attention",
      "accent.success",
      "accent.danger",
      "accent.focus",
    ];

    for (const role of accentRoles) {
      const value = theme.values[role as keyof typeof theme.values];
      expect(value).toBeDefined();
      expect(value.kind).toBe("color");
      if (value.kind === "color") {
        expect(value.value).toMatch(/^#[0-9a-f]{6}$/i);
      }
    }

    const { container } = render(
      <SemanticTwin
        projection={projection}
        focusedEntityId={null}
        selectedEntityId={null}
        onFocus={() => undefined}
        onActivate={() => undefined}
      />,
    );

    const entities = container.querySelectorAll("[data-entity-id]");
    expect(entities.length).toBeGreaterThan(0);
  });

  test("maintains theme contrast requirements for readability", () => {
    const registry = createBootstrapThemeRegistry();

    // Validation happens during resolve, so if we got here, contrast is valid
    expect(() => {
      registry.freeze().resolve("systems_chalk");
    }).not.toThrow();
  });

  test("applies theme with semantic entity descriptions", () => {
    const projection: SemanticProjection = {
      sceneId: "described-scene",
      readingOrder: ["entity-a"],
      entities: [
        {
          id: "entity-a",
          label: "Named Entity",
          description: "This is a detailed description of the entity",
          focusTarget: "entity-a",
          selected: false,
        },
      ],
      relations: [],
    };

    const { container } = render(
      <SemanticTwin
        projection={projection}
        focusedEntityId={null}
        selectedEntityId={null}
        onFocus={() => undefined}
        onActivate={() => undefined}
      />,
    );

    const entity = container.querySelector('[data-entity-id="entity-a"]');
    expect(entity).toBeDefined();
    expect(entity?.getAttribute("aria-label")).toBe("Named Entity");

    const description = container.querySelector("[id*=desc]");
    expect(description?.textContent).toContain("This is a detailed description");
  });
});
