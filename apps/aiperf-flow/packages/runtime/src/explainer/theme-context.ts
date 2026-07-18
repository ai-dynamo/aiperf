// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Theme integration for explainer slides and scenes.

import type { ResolvedTheme } from '../theme/types.js';
import type { ThemeValueIr } from '@aiperf/flow-schema';
import type { SceneIr, RenderNodeIr } from '@aiperf/flow-schema';

export interface SlideDefinition {
  eyebrow: string;
  title: string;
  lede: string;
  narration: string;
  term?: { word: string; meaning: string };
  points: string[];
  caption: string;
  sceneId?: string;
}

export interface SlideCss {
  backgroundColor: string;
  color: string;
  accentColor: string;
  borderColor: string;
}

/**
 * Applies ResolvedTheme to explainer slides and mental model scenes.
 * Resolves @theme.* role references to actual color values.
 */
export class ExplainerThemeContext {
  /**
   * Apply theme colors to a slide definition.
   * Generates CSS variables for background, foreground, accent, and border.
   */
  applyThemeToSlide(slide: SlideDefinition, theme: ResolvedTheme): SlideCss {
    return {
      backgroundColor: this.resolveThemeValue(theme, 'surface.primary'),
      color: this.resolveThemeValue(theme, 'ink.primary'),
      accentColor: this.resolveThemeValue(theme, 'accent.execute'),
      borderColor: this.resolveThemeValue(theme, 'structure.divider'),
    };
  }

  /**
   * Apply theme to a scene IR.
   * Recursively replaces @theme.* references in node styles with actual colors.
   */
  applyThemeToScene(sceneIr: SceneIr, theme: ResolvedTheme): SceneIr {
    return this.transformSceneWithTheme(sceneIr, theme);
  }

  /**
   * Resolve a theme role to its actual color value.
   * Returns the color string if found, else a fallback.
   */
  private resolveThemeValue(theme: ResolvedTheme, role: string): string {
    const value = theme.values[role as keyof typeof theme.values];
    if (!value) {
      return '#f2eee3'; // fallback to light text
    }

    // Handle color value discriminant
    if (typeof value === 'object' && 'kind' in value && 'value' in value) {
      const themeValue = value as ThemeValueIr;
      if (themeValue.kind === 'color' && typeof themeValue.value === 'string') {
        return themeValue.value;
      }
    }

    // Fallback if value is already a string
    if (typeof value === 'string' && value.startsWith('#')) {
      return value;
    }

    return '#f2eee3';
  }

  /**
   * Transform scene IR by recursively replacing @theme.* references.
   */
  private transformSceneWithTheme(sceneIr: SceneIr, theme: ResolvedTheme): SceneIr {
    // Deep clone to avoid mutating input
    const transformed = JSON.parse(JSON.stringify(sceneIr));

    const walkScene = (node: any) => {
      if (!node || typeof node !== 'object') return;

      // Transform style properties
      if (node.style && typeof node.style === 'object') {
        Object.entries(node.style).forEach(([key, val]) => {
          if (typeof val === 'string' && val.startsWith('@theme.')) {
            const role = val.replace('@theme.', '');
            node.style[key] = this.resolveThemeValue(theme, role);
          }
        });
      }

      // Recurse into children
      if (Array.isArray(node.children)) {
        node.children.forEach((child: any) => walkScene(child));
      }

      // Recurse into roots
      if (Array.isArray(node.roots)) {
        node.roots.forEach((root: any) => walkScene(root));
      }
    };

    walkScene(transformed);
    return transformed;
  }
}
