import { describe, it, expect, beforeEach } from 'vitest';
import { ExplainerThemeContext } from '../../src/explainer/theme-context.js';
import type { ResolvedTheme } from '../../src/theme/types.js';

describe('ExplainerThemeContext', () => {
  let context: ExplainerThemeContext;
  let theme: ResolvedTheme;

  beforeEach(() => {
    context = new ExplainerThemeContext();
    theme = {
      id: 'systems-chalk',
      values: {
        'ink.primary': { kind: 'color', value: '#f2eee3' } as any,
        'surface.primary': { kind: 'color', value: '#24282b' } as any,
        'accent.execute': { kind: 'color', value: '#72d6a2' } as any,
      } as any,
    };
  });

  it('generates CSS for slide with theme colors', () => {
    const slide = {
      eyebrow: 'Test',
      title: 'Title',
      lede: 'Lede',
      narration: 'Narration',
      points: ['P1'],
      caption: 'Caption',
    };

    const css = context.applyThemeToSlide(slide, theme);

    expect(css.backgroundColor).toBe('#24282b'); // surface.primary
    expect(css.color).toBe('#f2eee3'); // ink.primary
  });

  it('applies theme to mental model scene', () => {
    const sceneIr = {
      type: 'scene',
      roots: [
        {
          id: 'box',
          capability: 'core.rect',
          style: { fill: '@theme.surface.primary' },
        },
      ],
    };

    const themed = context.applyThemeToScene(sceneIr, theme);

    expect(themed.roots[0].style.fill).toBe('#24282b');
  });

  it('handles theme role lookups', () => {
    const css = context.applyThemeToSlide(
      {
        eyebrow: 'T',
        title: 'T',
        lede: 'L',
        narration: 'N',
        points: [],
        caption: 'C',
      },
      theme
    );

    expect(css.accentColor).toBe('#72d6a2'); // accent.execute
  });
});
