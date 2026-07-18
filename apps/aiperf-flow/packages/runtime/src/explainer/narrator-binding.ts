import type { SlideshowController } from './controller.js';
import type { NarratorBackend } from '../narrative/narrator.js';

export class NarratorBinding {
  private readonly controller: SlideshowController;
  private readonly narrator: NarratorBackend;

  constructor(controller: SlideshowController, narrator: NarratorBackend) {
    this.controller = controller;
    this.narrator = narrator;
  }

  onNarrationComplete(): void {
    // Auto-advance to next slide
    void this.controller.nextSlide();
  }

  pauseNarration(): void {
    this.narrator.pause();
  }

  resumeNarration(): void {
    this.narrator.resume();
  }

  skipNarration(): void {
    this.narrator.stop();
    void this.controller.nextSlide();
  }
}
