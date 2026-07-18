// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect, beforeEach } from 'vitest';
import {
  TransitionAnimationGenerator,
  TRANSITION_PRESETS,
  type TransitionConfig,
  type ResolvedTheme,
} from '../../../src/explainer/ui/transitions.js';

describe('TransitionAnimationGenerator', () => {
  let generator: TransitionAnimationGenerator;
  let mockTheme: ResolvedTheme;

  beforeEach(() => {
    mockTheme = {
      id: 'test-theme',
      values: {
        'surface.primary': { kind: 'color', value: '#24282b' } as any,
        'surface.raised': { kind: 'color', value: '#303334' } as any,
        'ink.primary': { kind: 'color', value: '#f1f3f2' } as any,
        'structure.divider': { kind: 'color', value: '#d7dada' } as any,
        'accent.execute': { kind: 'color', value: '#72d6a2' } as any,
      } as any,
    };

    generator = new TransitionAnimationGenerator(mockTheme);
  });

  describe('Fade Animation', () => {
    it('generates fade-in animation with correct opacity', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'in');

      expect(animation.name).toBe('fade-in');
      expect(animation.duration).toBe(300);
      expect(animation.easing).toBe('ease-in-out');
      expect(animation.keyframes).toHaveLength(2);
      expect(animation.keyframes[0].opacity).toBe(0);
      expect(animation.keyframes[1].opacity).toBe(1);
    });

    it('generates fade-out animation with correct opacity', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'out');

      expect(animation.name).toBe('fade-out');
      expect(animation.keyframes[0].opacity).toBe(1);
      expect(animation.keyframes[1].opacity).toBe(0);
    });

    it('fade animation applies theme background color', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'in');

      expect(animation.keyframes[0].backgroundColor).toBe('#24282b');
      expect(animation.keyframes[1].backgroundColor).toBe('#24282b');
    });

    it('fade animation keeps transform at identity', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'in');

      animation.keyframes.forEach((kf) => {
        expect(kf.transform).toBe('translate(0, 0)');
      });
    });
  });

  describe('Slide Animation', () => {
    it('generates slide-right animation with correct transform', () => {
      const config: TransitionConfig = {
        type: 'slide',
        duration: 400,
        easing: 'ease-out',
        direction: 'right',
      };

      const animation = generator.generateSlideAnimation(config, 'in');

      expect(animation.name).toBe('slide-right-in');
      expect(animation.duration).toBe(400);
      expect(animation.keyframes[0].transform).toContain('translate');
      expect(animation.keyframes[1].transform).toBe('translate(0, 0)');
    });

    it('generates slide-left animation with correct direction', () => {
      const config: TransitionConfig = {
        type: 'slide',
        duration: 400,
        easing: 'ease-out',
        direction: 'left',
      };

      const animation = generator.generateSlideAnimation(config, 'in');

      expect(animation.name).toContain('slide-left');
      expect(animation.keyframes[0].transform).toContain('translate');
    });

    it('generates slide-up animation', () => {
      const config: TransitionConfig = {
        type: 'slide',
        duration: 400,
        easing: 'ease-out',
        direction: 'up',
      };

      const animation = generator.generateSlideAnimation(config, 'in');

      expect(animation.name).toContain('slide-up');
    });

    it('generates slide-down animation', () => {
      const config: TransitionConfig = {
        type: 'slide',
        duration: 400,
        easing: 'ease-out',
        direction: 'down',
      };

      const animation = generator.generateSlideAnimation(config, 'in');

      expect(animation.name).toContain('slide-down');
    });

    it('slide-in animation fades from transparent to opaque', () => {
      const config: TransitionConfig = {
        type: 'slide',
        duration: 400,
        easing: 'ease-out',
        direction: 'right',
      };

      const animation = generator.generateSlideAnimation(config, 'in');

      expect(animation.keyframes[0].opacity).toBe(0);
      expect(animation.keyframes[1].opacity).toBe(1);
    });

    it('slide-out animation fades from opaque to transparent', () => {
      const config: TransitionConfig = {
        type: 'slide',
        duration: 400,
        easing: 'ease-out',
        direction: 'right',
      };

      const animation = generator.generateSlideAnimation(config, 'out');

      expect(animation.keyframes[0].opacity).toBe(1);
      expect(animation.keyframes[1].opacity).toBe(0);
    });

    it('slide animation applies theme colors', () => {
      const config: TransitionConfig = {
        type: 'slide',
        duration: 400,
        easing: 'ease-out',
        direction: 'left',
      };

      const animation = generator.generateSlideAnimation(config, 'in');

      animation.keyframes.forEach((kf) => {
        expect(kf.backgroundColor).toBe('#24282b');
      });
    });
  });

  describe('Morph Animation', () => {
    it('generates morph-in animation with scale transform', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'in');

      expect(animation.name).toBe('morph-in');
      expect(animation.duration).toBe(500);
      expect(animation.keyframes).toHaveLength(3);
      expect(animation.keyframes[0].transform).toContain('scale(0.8)');
      expect(animation.keyframes[2].transform).toContain('scale(1)');
    });

    it('generates morph-out animation with scale transform', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'out');

      expect(animation.name).toBe('morph-out');
      expect(animation.keyframes[0].transform).toContain('scale(1)');
      expect(animation.keyframes[2].transform).toContain('scale(0.8)');
    });

    it('morph-in animation fades from transparent to opaque', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'in');

      expect(animation.keyframes[0].opacity).toBe(0);
      expect(animation.keyframes[2].opacity).toBe(1);
    });

    it('morph-out animation fades from opaque to transparent', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'out');

      expect(animation.keyframes[0].opacity).toBe(1);
      expect(animation.keyframes[2].opacity).toBe(0);
    });

    it('morph animation applies theme colors including border', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'in');

      expect(animation.keyframes[0].backgroundColor).toBe('#24282b');
      expect(animation.keyframes[0].borderColor).toBe('#d7dada');
      expect(animation.keyframes[1].backgroundColor).toBe('#303334');
      expect(animation.keyframes[1].borderColor).toBe('#72d6a2');
      expect(animation.keyframes[2].backgroundColor).toBe('#24282b');
      expect(animation.keyframes[2].borderColor).toBe('#d7dada');
    });

    it('morph animation has smooth mid-point opacity', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'in');

      // Mid-point should be less opaque than start/end
      expect(animation.keyframes[1].opacity).toBeLessThan(1);
      expect(animation.keyframes[1].opacity).toBeGreaterThan(0);
    });
  });

  describe('Slide Transition (Combined Out/In)', () => {
    it('generates paired fade transitions', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const { out, in: inAnim } = generator.generateSlideTransition(config);

      expect(out.name).toBe('fade-out');
      expect(inAnim.name).toBe('fade-in');
      expect(out.duration).toBe(300);
      expect(inAnim.duration).toBe(300);
    });

    it('generates paired slide transitions', () => {
      const config: TransitionConfig = {
        type: 'slide',
        duration: 400,
        easing: 'ease-out',
        direction: 'left',
      };

      const { out, in: inAnim } = generator.generateSlideTransition(config);

      expect(out.name).toContain('slide-left-out');
      expect(inAnim.name).toContain('slide-left-in');
    });

    it('generates paired morph transitions', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const { out, in: inAnim } = generator.generateSlideTransition(config);

      expect(out.name).toBe('morph-out');
      expect(inAnim.name).toBe('morph-in');
    });

    it('out animation ends opaque and in animation starts transparent', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const { out, in: inAnim } = generator.generateSlideTransition(config);

      // Out should go from opaque to transparent
      expect(out.keyframes[0].opacity).toBeGreaterThan(0);
      expect(out.keyframes[out.keyframes.length - 1].opacity).toBeLessThan(1);

      // In should go from transparent to opaque
      expect(inAnim.keyframes[0].opacity).toBeLessThan(1);
      expect(inAnim.keyframes[inAnim.keyframes.length - 1].opacity).toBe(1);
    });
  });

  describe('CSS Generation', () => {
    it('generates valid CSS @keyframes', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'in');
      const css = generator.animationToCss(animation);

      expect(css).toContain('@keyframes fade-in');
      expect(css).toContain('0%');
      expect(css).toContain('100%');
      expect(css).toContain('opacity:');
      expect(css).toContain('transform:');
      expect(css).toContain('.fade-in');
    });

    it('CSS includes background color', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'in');
      const css = generator.animationToCss(animation);

      expect(css).toContain('background-color');
      expect(css).toContain('#24282b');
    });

    it('CSS includes border color for morph animation', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'in');
      const css = generator.animationToCss(animation);

      expect(css).toContain('border-color');
      expect(css).toContain('#d7dada');
    });

    it('CSS animation class includes duration and easing', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'in');
      const css = generator.animationToCss(animation);

      expect(css).toContain('300ms');
      expect(css).toContain('ease-in-out');
      expect(css).toContain('forwards');
    });

    it('CSS has correct percentage offsets', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'in');
      const css = generator.animationToCss(animation);

      expect(css).toContain('0% {');
      expect(css).toContain('100% {');
    });

    it('morph animation CSS includes all three keyframes', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'in');
      const css = generator.animationToCss(animation);

      expect(css).toContain('0% {');
      expect(css).toContain('50% {');
      expect(css).toContain('100% {');
    });
  });

  describe('Theme Color Resolution', () => {
    it('resolves theme colors correctly in animations', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'in');

      expect(animation.keyframes[0].backgroundColor).toBe('#24282b');
      expect(animation.keyframes[1].backgroundColor).toBe('#24282b');
    });

    it('falls back to default color for unknown theme role', () => {
      // Create a theme with limited values
      const limitedTheme: ResolvedTheme = {
        id: 'limited',
        values: {} as any,
      };

      const limitedGenerator = new TransitionAnimationGenerator(limitedTheme);
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = limitedGenerator.generateFadeAnimation(config, 'in');

      // Should use fallback color
      expect(animation.keyframes[0].backgroundColor).toBe('#f2eee3');
    });

    it('applies different theme colors to different keyframes in morph', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'in');

      // Start: surface.primary
      expect(animation.keyframes[0].backgroundColor).toBe('#24282b');
      // Middle: surface.raised
      expect(animation.keyframes[1].backgroundColor).toBe('#303334');
      // End: surface.primary
      expect(animation.keyframes[2].backgroundColor).toBe('#24282b');
    });

    it('applies accent color to morph mid-point', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'in');

      // Mid-point should have accent border
      expect(animation.keyframes[1].borderColor).toBe('#72d6a2');
    });
  });

  describe('Transition Presets', () => {
    it('has fast fade preset', () => {
      const preset = TRANSITION_PRESETS.fastFade;

      expect(preset.type).toBe('fade');
      expect(preset.duration).toBeLessThan(200);
      expect(preset.easing).toBe('ease-in-out');
    });

    it('has standard fade preset', () => {
      const preset = TRANSITION_PRESETS.fade;

      expect(preset.type).toBe('fade');
      expect(preset.duration).toBe(300);
      expect(preset.easing).toBe('ease-in-out');
    });

    it('has slide direction presets', () => {
      expect(TRANSITION_PRESETS.fastSlideLeft.direction).toBe('left');
      expect(TRANSITION_PRESETS.fastSlideRight.direction).toBe('right');
      expect(TRANSITION_PRESETS.slideLeft.direction).toBe('left');
      expect(TRANSITION_PRESETS.slideRight.direction).toBe('right');
    });

    it('has morph presets with different easing', () => {
      const standardMorph = TRANSITION_PRESETS.morph;
      const slowMorph = TRANSITION_PRESETS.slowMorph;

      expect(standardMorph.type).toBe('morph');
      expect(slowMorph.type).toBe('morph');
      expect(standardMorph.easing).toBe('cubic-bezier(0.4, 0, 0.2, 1)');
      expect(slowMorph.easing).toBe('cubic-bezier(0.25, 0.46, 0.45, 0.94)');
    });

    it('all presets have valid easing functions', () => {
      Object.values(TRANSITION_PRESETS).forEach((preset) => {
        expect(preset.easing).toBeTruthy();
        // Easing should be either a standard keyword or cubic-bezier
        const isValid =
          preset.easing.includes('ease') || preset.easing.includes('cubic-bezier');
        expect(isValid).toBe(true);
      });
    });

    it('slow presets have appropriate durations', () => {
      expect(TRANSITION_PRESETS.slowFade.duration).toBe(400);
      expect(TRANSITION_PRESETS.slowMorph.duration).toBe(500);
      expect(TRANSITION_PRESETS.slowFade.duration).toBeGreaterThanOrEqual(
        TRANSITION_PRESETS.fade.duration
      );
    });
  });

  describe('Animation Smoothness', () => {
    it('fade animation is smooth with linear opacity', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'in');

      // Fade should have continuous opacity change
      const opacityStart = animation.keyframes[0].opacity;
      const opacityEnd = animation.keyframes[1].opacity;

      expect(Math.abs(opacityEnd - opacityStart)).toBe(1);
    });

    it('morph animation has smooth scale progression', () => {
      const config: TransitionConfig = {
        type: 'morph',
        duration: 500,
        easing: 'ease-in-out',
      };

      const animation = generator.generateMorphAnimation(config, 'in');

      // Extract scale values from transforms
      const scales = animation.keyframes.map((kf) => {
        const match = kf.transform.match(/scale\(([\d.]+)\)/);
        return match ? parseFloat(match[1]) : 1;
      });

      // Should go 0.8 -> ~0.9 -> 1.0
      expect(scales[0]).toBe(0.8);
      expect(scales[1]).toBeGreaterThan(0.8);
      expect(scales[1]).toBeLessThan(1);
      expect(scales[2]).toBe(1);
    });

    it('all transitions complete within specified duration', () => {
      const configs: TransitionConfig[] = [
        { type: 'fade', duration: 300, easing: 'ease-in-out' },
        { type: 'slide', duration: 400, easing: 'ease-out', direction: 'left' },
        { type: 'morph', duration: 500, easing: 'ease-in-out' },
      ];

      configs.forEach((config) => {
        const { out, in: inAnim } = generator.generateSlideTransition(config);

        expect(out.duration).toBe(config.duration);
        expect(inAnim.duration).toBe(config.duration);
      });
    });

    it('all slide animations maintain 500ms max performance target', () => {
      const allPresets = Object.values(TRANSITION_PRESETS);

      allPresets.forEach((preset) => {
        expect(preset.duration).toBeLessThanOrEqual(500);
      });
    });
  });

  describe('Performance & Constraints', () => {
    it('respects global 500ms max transition constraint', () => {
      Object.values(TRANSITION_PRESETS).forEach((preset) => {
        expect(preset.duration).toBeLessThanOrEqual(500);
      });
    });

    it('fade transitions are fastest', () => {
      const fadePresets = [
        TRANSITION_PRESETS.fastFade,
        TRANSITION_PRESETS.fade,
        TRANSITION_PRESETS.slowFade,
      ];

      const slideDuration = TRANSITION_PRESETS.slideLeft.duration;
      const morphDuration = TRANSITION_PRESETS.morph.duration;

      expect(TRANSITION_PRESETS.fade.duration).toBeLessThan(slideDuration);
      expect(TRANSITION_PRESETS.fade.duration).toBeLessThan(morphDuration);
    });

    it('morphing transitions use smooth easing', () => {
      expect(TRANSITION_PRESETS.morph.easing).toContain('cubic-bezier');
      expect(TRANSITION_PRESETS.slowMorph.easing).toContain('cubic-bezier');
    });
  });

  describe('Accessibility & Reduced Motion', () => {
    it('fast transitions have shortest duration for accessibility', () => {
      const fastTransitions = [
        TRANSITION_PRESETS.fastFade,
        TRANSITION_PRESETS.fastSlideLeft,
        TRANSITION_PRESETS.fastSlideRight,
      ];

      fastTransitions.forEach((preset) => {
        expect(preset.duration).toBeLessThanOrEqual(200);
      });
    });

    it('CSS includes animation class for prefers-reduced-motion override', () => {
      const config: TransitionConfig = {
        type: 'fade',
        duration: 300,
        easing: 'ease-in-out',
      };

      const animation = generator.generateFadeAnimation(config, 'in');
      const css = generator.animationToCss(animation);

      // CSS should have class name for targeting
      expect(css).toContain(`.${animation.name}`);
    });
  });
});
