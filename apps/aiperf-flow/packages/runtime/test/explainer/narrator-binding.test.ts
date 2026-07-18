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
      pauseNarration: vi.fn(),
      resumeNarration: vi.fn(),
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
      available: true,
      voices: () => [],
      speak: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
      cancel: vi.fn(),
    };

    binding = new NarratorBinding(mockController, mockNarrator);
  });

  it('advances slide on narration complete', async () => {
    binding.onNarrationComplete();
    expect(mockController.nextSlide).toHaveBeenCalled();
  });

  it('pauses narration through the controller', () => {
    binding.pauseNarration();
    expect(mockController.pauseNarration).toHaveBeenCalled();
  });

  it('resumes narration through the controller', () => {
    binding.resumeNarration();
    expect(mockController.resumeNarration).toHaveBeenCalled();
  });

  it('cancels the narrator and advances on skip', async () => {
    binding.skipNarration();
    expect(mockNarrator.cancel).toHaveBeenCalled();
    expect(mockController.nextSlide).toHaveBeenCalled();
  });
});
