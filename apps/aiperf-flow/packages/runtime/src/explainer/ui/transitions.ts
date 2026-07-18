// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scene transition animations for explainer slides.
//! Supports fade, slide, and morph animations with theme-aware color palettes.

import type { ResolvedTheme } from '../theme/types.js';

/**
 * Animation type: fade, slide, or morph.
 */
export type AnimationType = 'fade' | 'slide' | 'morph';

/**
 * Slide direction: left, right, up, or down.
 */
export type SlideDirection = 'left' | 'right' | 'up' | 'down';

/**
 * Configuration for a transition animation.
 */
export interface TransitionConfig {
  type: AnimationType;
  duration: number; // milliseconds
  easing: string; // CSS easing function
  direction?: SlideDirection; // for slide animations
}

/**
 * Animation keyframes with theme-aware colors.
 */
export interface AnimationKeyframe {
  offset: number; // 0.0 to 1.0
  opacity: number;
  transform: string;
  backgroundColor?: string;
  borderColor?: string;
}

/**
 * Complete animation definition.
 */
export interface Animation {
  name: string;
  duration: number;
  easing: string;
  keyframes: AnimationKeyframe[];
}

/**
 * Manages scene transition animations.
 * Generates CSS keyframe animations with theme-aware colors.
 */
export class TransitionAnimationGenerator {
  private readonly theme: ResolvedTheme;

  constructor(theme: ResolvedTheme) {
    this.theme = theme;
  }

  /**
   * Generate a fade animation.
   * Fades from transparent to opaque (or vice versa).
   */
  generateFadeAnimation(config: TransitionConfig, direction: 'in' | 'out' = 'in'): Animation {
    const startOpacity = direction === 'in' ? 0 : 1;
    const endOpacity = direction === 'in' ? 1 : 0;

    const keyframes: AnimationKeyframe[] = [
      {
        offset: 0,
        opacity: startOpacity,
        transform: 'translate(0, 0)',
        backgroundColor: this.resolveThemeColor('surface.primary'),
      },
      {
        offset: 1,
        opacity: endOpacity,
        transform: 'translate(0, 0)',
        backgroundColor: this.resolveThemeColor('surface.primary'),
      },
    ];

    return {
      name: `fade-${direction}`,
      duration: config.duration,
      easing: config.easing,
      keyframes,
    };
  }

  /**
   * Generate a slide animation.
   * Slides from off-screen to on-screen (or vice versa).
   */
  generateSlideAnimation(config: TransitionConfig, direction: 'in' | 'out' = 'in'): Animation {
    const slideDirection = config.direction || 'right';
    const distance = 100; // percentage
    const startTransform = this.getSlideStartTransform(slideDirection, distance, direction);
    const endTransform = 'translate(0, 0)';

    const keyframes: AnimationKeyframe[] = [
      {
        offset: 0,
        opacity: direction === 'in' ? 0 : 1,
        transform: startTransform,
        backgroundColor: this.resolveThemeColor('surface.primary'),
      },
      {
        offset: 1,
        opacity: direction === 'in' ? 1 : 0,
        transform: endTransform,
        backgroundColor: this.resolveThemeColor('surface.primary'),
      },
    ];

    return {
      name: `slide-${slideDirection}-${direction}`,
      duration: config.duration,
      easing: config.easing,
      keyframes,
    };
  }

  /**
   * Generate a morph animation.
   * Scales and fades to create a morphing transition effect.
   */
  generateMorphAnimation(config: TransitionConfig, direction: 'in' | 'out' = 'in'): Animation {
    const startScale = direction === 'in' ? 0.8 : 1;
    const endScale = direction === 'in' ? 1 : 0.8;
    const startOpacity = direction === 'in' ? 0 : 1;
    const endOpacity = direction === 'in' ? 1 : 0;

    const keyframes: AnimationKeyframe[] = [
      {
        offset: 0,
        opacity: startOpacity,
        transform: `scale(${startScale})`,
        backgroundColor: this.resolveThemeColor('surface.primary'),
        borderColor: this.resolveThemeColor('structure.divider'),
      },
      {
        offset: 0.5,
        opacity: Math.max(startOpacity, endOpacity) * 0.7,
        transform: `scale(${(startScale + endScale) / 2})`,
        backgroundColor: this.resolveThemeColor('surface.raised'),
        borderColor: this.resolveThemeColor('accent.execute'),
      },
      {
        offset: 1,
        opacity: endOpacity,
        transform: `scale(${endScale})`,
        backgroundColor: this.resolveThemeColor('surface.primary'),
        borderColor: this.resolveThemeColor('structure.divider'),
      },
    ];

    return {
      name: `morph-${direction}`,
      duration: config.duration,
      easing: config.easing,
      keyframes,
    };
  }

  /**
   * Generate animation for transitioning between slides.
   * Combines fade-out of previous slide with fade-in of next slide.
   */
  generateSlideTransition(config: TransitionConfig): { out: Animation; in: Animation } {
    switch (config.type) {
      case 'fade':
        return {
          out: this.generateFadeAnimation(config, 'out'),
          in: this.generateFadeAnimation(config, 'in'),
        };
      case 'slide':
        return {
          out: this.generateSlideAnimation(config, 'out'),
          in: this.generateSlideAnimation(config, 'in'),
        };
      case 'morph':
        return {
          out: this.generateMorphAnimation(config, 'out'),
          in: this.generateMorphAnimation(config, 'in'),
        };
    }
  }

  /**
   * Convert animation to CSS keyframes string.
   */
  animationToCss(animation: Animation): string {
    const keyframesCss = animation.keyframes
      .map((kf) => {
        const offset = `${Math.round(kf.offset * 100)}%`;
        const styles: string[] = [
          `opacity: ${kf.opacity};`,
          `transform: ${kf.transform};`,
        ];

        if (kf.backgroundColor) {
          styles.push(`background-color: ${kf.backgroundColor};`);
        }
        if (kf.borderColor) {
          styles.push(`border-color: ${kf.borderColor};`);
        }

        return `${offset} { ${styles.join(' ')} }`;
      })
      .join('\n  ');

    return `@keyframes ${animation.name} {
  ${keyframesCss}
}

.${animation.name} {
  animation: ${animation.name} ${animation.duration}ms ${animation.easing} forwards;
}`;
  }

  /**
   * Get the starting transform for a slide animation.
   */
  private getSlideStartTransform(direction: SlideDirection, distance: number, animDir: 'in' | 'out'): string {
    const sign = animDir === 'in' ? 1 : -1;

    switch (direction) {
      case 'left':
        return `translate(${sign * distance}%, 0)`;
      case 'right':
        return `translate(${-sign * distance}%, 0)`;
      case 'up':
        return `translate(0, ${sign * distance}%)`;
      case 'down':
        return `translate(0, ${-sign * distance}%)`;
    }
  }

  /**
   * Resolve a theme role to its actual color value.
   */
  private resolveThemeColor(role: string): string {
    const value = this.theme.values[role as keyof typeof this.theme.values];
    if (!value) {
      return '#f2eee3'; // fallback to light text
    }

    // Handle color value discriminant
    if (typeof value === 'object' && 'kind' in value && 'value' in value) {
      const themeValue = value as any;
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
}

/**
 * Preset transition configurations.
 */
export const TRANSITION_PRESETS: Record<string, TransitionConfig> = {
  // Fast transitions for quick navigation
  fastFade: {
    type: 'fade',
    duration: 150,
    easing: 'ease-in-out',
  },
  fastSlideLeft: {
    type: 'slide',
    duration: 200,
    easing: 'ease-out',
    direction: 'left',
  },
  fastSlideRight: {
    type: 'slide',
    duration: 200,
    easing: 'ease-out',
    direction: 'right',
  },

  // Standard transitions for normal navigation
  fade: {
    type: 'fade',
    duration: 300,
    easing: 'ease-in-out',
  },
  slideLeft: {
    type: 'slide',
    duration: 400,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
    direction: 'left',
  },
  slideRight: {
    type: 'slide',
    duration: 400,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
    direction: 'right',
  },
  morph: {
    type: 'morph',
    duration: 500,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
  },

  // Slow transitions for emphasis
  slowFade: {
    type: 'fade',
    duration: 400,
    easing: 'ease-in-out',
  },
  slowMorph: {
    type: 'morph',
    duration: 500,
    easing: 'cubic-bezier(0.25, 0.46, 0.45, 0.94)',
  },
};
