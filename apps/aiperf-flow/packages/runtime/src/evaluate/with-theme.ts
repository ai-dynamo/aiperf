// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Theme value resolution for scene evaluation.

import type {
  FlowThemeIr,
  RenderNodeIr,
  SceneIr,
  StyleValueIr,
  ThemeRole,
  ThemeValueIr,
} from "@aiperf/flow-schema";

import { evaluateScene as evaluateSceneFoundation } from "./scene-evaluator.js";
import type { EvaluateSceneOptions } from "./scene-evaluator.js";
import type { EvaluatedScene } from "./types.js";

/** Theme context passed to evaluation. */
export type ThemeContext = Readonly<{
  /** Active theme providing value bindings. */
  activeTheme: FlowThemeIr;
  /** All available themes for inheritance resolution. */
  allThemes: readonly FlowThemeIr[];
}>;

/**
 * Resolves a theme value from a role, following inheritance chain.
 * Returns undefined if the role is not found in the active theme or inherited themes.
 */
export function resolveThemeValue(
  role: ThemeRole,
  context: ThemeContext,
): ThemeValueIr | undefined {
  // First, check if the value is in the active theme
  const activeValue = context.activeTheme.values[role];
  if (activeValue !== undefined) {
    return activeValue;
  }

  // Follow the inheritance chain
  let currentThemeId = context.activeTheme.extends;
  while (currentThemeId) {
    const theme = context.allThemes.find((t) => t.id === currentThemeId);
    if (!theme) {
      break;
    }
    const inheritedValue = theme.values[role];
    if (inheritedValue !== undefined) {
      return inheritedValue;
    }
    currentThemeId = theme.extends;
  }

  return undefined;
}

/**
 * Converts a theme value to its resolved representation based on value kind.
 * - color → hex string
 * - font → CSS font-family string
 * - number → numeric value
 * - duration → milliseconds
 * - enum → enum string value
 */
export function themeValueToStyleValue(value: ThemeValueIr): string | number {
  switch (value.kind) {
    case "color":
      return value.value;
    case "font":
      return value.value.map((font) => `"${font}"`).join(", ");
    case "number":
      return value.value;
    case "duration":
      return value.valueMs;
    case "enum":
      return value.value;
  }
}

/**
 * Resolves a style value by checking if it's a theme role reference and
 * looking it up in the theme context. Otherwise returns the scalar value.
 */
export function resolveStyleValue(
  styleValue: StyleValueIr,
  context: ThemeContext | undefined,
): string | number | boolean | undefined {
  if (context === undefined) {
    // No theme context, return scalar values only
    return typeof styleValue === "object" ? undefined : styleValue;
  }

  if (typeof styleValue === "object" && styleValue.kind === "theme-role") {
    const resolved = resolveThemeValue(styleValue.role, context);
    if (resolved !== undefined) {
      return themeValueToStyleValue(resolved);
    }
    return undefined;
  }

  return typeof styleValue !== "object" ? styleValue : undefined;
}

/**
 * Resolves all theme role references in a render node's style object.
 */
function resolveNodeStyle(
  style: Record<string, StyleValueIr>,
  context: ThemeContext,
): Record<string, string | number | boolean> {
  const resolved: Record<string, string | number | boolean> = {};

  for (const [key, value] of Object.entries(style)) {
    const resolvedValue = resolveStyleValue(value, context);
    if (resolvedValue !== undefined) {
      resolved[key] = resolvedValue;
    }
  }

  return resolved;
}

/**
 * Recursively processes a render node tree, resolving theme role references
 * in all style objects.
 */
function resolveNodeTree(
  node: RenderNodeIr,
  context: ThemeContext,
): RenderNodeIr {
  const resolvedStyle = resolveNodeStyle(node.style, context);

  const base = {
    ...node,
    style: resolvedStyle,
  };

  if (node.kind === "group" || node.kind === "component") {
    return {
      ...base,
      children: node.children.map((child) => resolveNodeTree(child, context)),
    } as RenderNodeIr;
  }

  return base as RenderNodeIr;
}

/**
 * Theme-aware scene evaluator that resolves theme role references before
 * evaluation and propagates resolved values through display instructions.
 *
 * @param scene - Scene to evaluate
 * @param atMs - Evaluation time in milliseconds (default 0)
 * @param options - Evaluation options including optional theme context
 * @returns Evaluated scene with theme values resolved in display instructions
 */
export function evaluateSceneWithTheme(
  scene: SceneIr,
  atMs = 0,
  options: Readonly<
    EvaluateSceneOptions & { themeContext?: ThemeContext | undefined }
  > = {},
): EvaluatedScene {
  const themeContext = options.themeContext;

  // If no theme context, delegate to foundation evaluator
  if (themeContext === undefined) {
    return evaluateSceneFoundation(scene, atMs, options);
  }

  // Transform scene by resolving theme values before evaluation
  const resolvedScene: SceneIr = {
    ...scene,
    roots: scene.roots.map((node) => resolveNodeTree(node, themeContext)),
  };

  // Evaluate the transformed scene
  return evaluateSceneFoundation(resolvedScene, atMs, options);
}
