import { describe, it, expect, beforeEach, vi } from 'vitest';
import { NarratorBinding } from '../../src/explainer/narrator-binding.js';
import type { SlideshowController } from '../../src/explainer/controller.js';
import type { NarratorBackend } from '../../src/narrative/narrator.js';

describe('NarratorBinding', () => {
  let binding: NarratorBinding;
  let mockController: SlideshowController;
  let mockNarrator: NarratorBackend;

  beforeEach(() => {
    mockController = {
      currentSlideIndex: 0,
      totalSlides: 3,
      isPlayingNarration: true,
      nextSlide: vi.fn().mockResolvedValue(undefined),
      prevSlide: vi.fn(),
      jumpToSlide: vi.fn(),
      getCurrentSlide: () => ({
        eyebrow: 'S1',
        title: 'Test',
        lede: 'Test',
        narration: 'Test narration.',
        points: [],
        caption: '',
      }),
    } as any;

    mockNarrator = {
      speak: vi.fn().mockResolvedValue(undefined),
      stop: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
    } as any;

    binding = new NarratorBinding(mockController, mockNarrator);
  });

  it('advances slide on narration complete', async () => {
    binding.onNarrationComplete();
    expect(mockController.nextSlide).toHaveBeenCalled();
  });

  it('pauses narrator', () => {
    binding.pauseNarration();
    expect(mockNarrator.pause).toHaveBeenCalled();
  });

  it('resumes narrator', () => {
    binding.resumeNarration();
    expect(mockNarrator.resume).toHaveBeenCalled();
  });

  it('skips to next slide on skip command', async () => {
    binding.skipNarration();
    expect(mockNarrator.stop).toHaveBeenCalled();
    expect(mockController.nextSlide).toHaveBeenCalled();
  });
});
