import { describe, it, expect, beforeEach, vi } from 'vitest';
import { SlideshowController } from '../../src/explainer/controller.js';
import type { ExplainerDefinition } from '@aiperf/flow-compiler';
import type { NarratorBackend } from '../../src/narrative/narrator.js';

describe('SlideshowController', () => {
  let controller: SlideshowController;
  let mockNarrator: NarratorBackend;
  let deck: ExplainerDefinition;

  beforeEach(() => {
    deck = {
      id: 'test',
      route: '/test',
      topic: 'test',
      eyebrowLabel: 'Test',
      startGateTitle: 'Go?',
      slides: [
        {
          eyebrow: 'S1',
          title: 'Slide 1',
          lede: 'First',
          narration: 'First narration.',
          points: [],
          caption: '',
        },
        {
          eyebrow: 'S2',
          title: 'Slide 2',
          lede: 'Second',
          narration: 'Second narration.',
          points: [],
          caption: '',
        },
        {
          eyebrow: 'S3',
          title: 'Slide 3',
          lede: 'Third',
          narration: 'Third narration.',
          points: [],
          caption: '',
        },
      ],
      scenesById: new Map(),
    };

    mockNarrator = {
      speak: vi.fn().mockResolvedValue(undefined),
      stop: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
    } as any;

    controller = new SlideshowController(deck, mockNarrator);
  });

  it('initializes at slide 0', () => {
    expect(controller.currentSlideIndex).toBe(0);
    expect(controller.totalSlides).toBe(3);
  });

  it('advances to next slide', async () => {
    await controller.nextSlide();
    expect(controller.currentSlideIndex).toBe(1);
  });

  it('does not advance past last slide', async () => {
    controller.jumpToSlide(2);
    await controller.nextSlide();
    expect(controller.currentSlideIndex).toBe(2);
  });

  it('retreats to previous slide', async () => {
    controller.jumpToSlide(2);
    await controller.prevSlide();
    expect(controller.currentSlideIndex).toBe(1);
  });

  it('does not retreat before first slide', async () => {
    await controller.prevSlide();
    expect(controller.currentSlideIndex).toBe(0);
  });

  it('jumps to specific slide', async () => {
    controller.jumpToSlide(1);
    expect(controller.currentSlideIndex).toBe(1);
  });

  it('speaks narration for current slide', async () => {
    await controller.nextSlide();
    expect(mockNarrator.speak).toHaveBeenCalledWith(
      expect.objectContaining({ narration: 'Second narration.' })
    );
  });

  it('stops narration when advancing', async () => {
    controller.jumpToSlide(1);
    await controller.nextSlide();
    // Should stop previous narration before playing next
    expect(mockNarrator.stop).toHaveBeenCalled();
  });
});
