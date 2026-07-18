import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ImmersiveExplainerContext } from '../../src/explainer/immersive-integration.js';

describe('ImmersiveExplainerContext', () => {
  let context: ImmersiveExplainerContext;

  beforeEach(() => {
    context = new ImmersiveExplainerContext();
  });

  it('expands slide scene to full viewport', () => {
    const scene = context.expandSlideToViewport('scene-0');

    expect(scene.layout).toEqual({ fullViewport: true });
    expect(scene.overlayContent).toBeDefined();
  });

  it('generates immersive controls', () => {
    const controls = context.applyImmersiveControls();

    expect(controls.playButton).toBeDefined();
    expect(controls.speedControl).toBeDefined();
    expect(controls.causalTraceToggle).toBeDefined();
  });

  it('positions narration UI in immersive mode', () => {
    const scene = context.expandSlideToViewport('scene-1');

    expect(scene.overlayContent.narrationUI).toBeDefined();
    expect(scene.overlayContent.narrationUI.position).toBe('top-right');
  });
});
