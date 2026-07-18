import type { ExplainerDefinition, SlideDefinition } from '@aiperf/flow-compiler';
import type { NarratorBackend } from '../narrative/narrator.js';

export class SlideshowController {
  private currentIndex = 0;
  private isNarrating = false;
  private readonly deck: ExplainerDefinition;
  private readonly narrator: NarratorBackend;

  constructor(deck: ExplainerDefinition, narrator: NarratorBackend) {
    this.deck = deck;
    this.narrator = narrator;
  }

  get currentSlideIndex(): number {
    return this.currentIndex;
  }

  get totalSlides(): number {
    return this.deck.slides.length;
  }

  get isPlayingNarration(): boolean {
    return this.isNarrating;
  }

  getCurrentSlide(): SlideDefinition {
    return this.deck.slides[this.currentIndex]!;
  }

  async nextSlide(): Promise<void> {
    if (this.currentIndex < this.deck.slides.length - 1) {
      this.narrator.stop();
      this.isNarrating = false;
      this.currentIndex++;
      await this.playNarrationForCurrentSlide();
    }
  }

  async prevSlide(): Promise<void> {
    if (this.currentIndex > 0) {
      this.narrator.stop();
      this.isNarrating = false;
      this.currentIndex--;
      await this.playNarrationForCurrentSlide();
    }
  }

  async jumpToSlide(index: number): Promise<void> {
    if (index >= 0 && index < this.deck.slides.length) {
      this.narrator.stop();
      this.isNarrating = false;
      this.currentIndex = index;
      await this.playNarrationForCurrentSlide();
    }
  }

  private async playNarrationForCurrentSlide(): Promise<void> {
    const slide = this.getCurrentSlide();
    if (slide.narration) {
      this.isNarrating = true;
      try {
        await this.narrator.speak({
          text: slide.narration,
          narration: slide.narration, // for narrator backend
        } as any);
      } finally {
        this.isNarrating = false;
      }
    }
  }
}
