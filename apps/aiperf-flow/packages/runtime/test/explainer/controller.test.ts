import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  SlideshowController,
  type NarrationTimer,
} from '../../src/explainer/controller.js';
import type { ExplainerDefinition } from '@aiperf/flow-compiler';
import type { NarratorBackend } from '../../src/narrative/narrator.js';
import type { SubtitleState } from '../../src/narrative/subtitle-overlay.js';

/** Deterministic clock + interval scheduler for narration timing tests. */
class FakeTimer implements NarrationTimer {
  private currentMs = 0;
  private ticks = new Set<() => void>();

  now(): number {
    return this.currentMs;
  }

  setInterval(callback: () => void): () => void {
    this.ticks.add(callback);
    return () => this.ticks.delete(callback);
  }

  /** Advance virtual time and fire every registered ticker once. */
  advance(ms: number): void {
    this.currentMs += ms;
    for (const tick of [...this.ticks]) {
      tick();
    }
  }
}

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
      available: true,
      voices: () => [],
      speak: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
      cancel: vi.fn(),
    };

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
    await controller.jumpToSlide(2);
    await controller.nextSlide();
    expect(controller.currentSlideIndex).toBe(2);
  });

  it('retreats to previous slide', async () => {
    await controller.jumpToSlide(2);
    await controller.prevSlide();
    expect(controller.currentSlideIndex).toBe(1);
  });

  it('does not retreat before first slide', async () => {
    await controller.prevSlide();
    expect(controller.currentSlideIndex).toBe(0);
  });

  it('jumps to specific slide', async () => {
    await controller.jumpToSlide(1);
    expect(controller.currentSlideIndex).toBe(1);
  });

  it('speaks narration for current slide', async () => {
    await controller.nextSlide();
    expect(mockNarrator.speak).toHaveBeenCalledWith(
      expect.objectContaining({ text: 'Second narration.' }),
      expect.any(Function)
    );
  });

  it('cancels narration when advancing', async () => {
    await controller.jumpToSlide(1);
    await controller.nextSlide();
    // Should cancel previous narration before playing next.
    expect(mockNarrator.cancel).toHaveBeenCalled();
  });
});

describe('SlideshowController narration timing', () => {
  let timer: FakeTimer;
  let deck: ExplainerDefinition;

  const makeDeck = (narration: string): ExplainerDefinition => ({
    id: 'timing',
    route: '/timing',
    topic: 'timing',
    eyebrowLabel: 'Timing',
    startGateTitle: 'Go?',
    slides: [
      { eyebrow: 'S1', title: 'One', lede: '', narration, points: [], caption: '' },
      { eyebrow: 'S2', title: 'Two', lede: '', narration: 'Second slide narration here.', points: [], caption: '' },
    ],
    scenesById: new Map(),
  });

  beforeEach(() => {
    timer = new FakeTimer();
    // 10 words, 150 wpm, rate 1 -> 400ms/word -> 4000ms total.
    deck = makeDeck('one two three four five six seven eight nine ten');
  });

  it('emits word-synchronized subtitle progress that stays locked without drift', () => {
    const narrator: NarratorBackend = {
      available: true,
      voices: () => [],
      speak: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
      cancel: vi.fn(),
    };
    const controller = new SlideshowController(deck, narrator, {
      timer,
      tickMs: 100,
    });
    const states: SubtitleState[] = [];
    controller.subscribeSubtitles((s) => states.push(s));

    void controller.jumpToSlide(0);
    // First cue emitted immediately at progress 0.
    expect(states.at(-1)?.activeCue?.text).toContain('one two');
    expect(states.at(-1)?.activeCue?.progress).toBe(0);

    timer.advance(2000); // halfway through 4000ms narration
    expect(states.at(-1)?.activeCue?.progress).toBeCloseTo(0.5, 5);

    timer.advance(1000); // 3000ms -> 75%
    expect(states.at(-1)?.activeCue?.progress).toBeCloseTo(0.75, 5);

    // Progress is a pure function of elapsed virtual time: no drift.
    timer.advance(500); // 3500ms -> 87.5%
    expect(states.at(-1)?.activeCue?.progress).toBeCloseTo(0.875, 5);
  });

  it('completes via audio callback and clears captions', () => {
    let onComplete: (() => void) | undefined;
    const narrator: NarratorBackend = {
      available: true,
      voices: () => [],
      speak: vi.fn((_u, cb) => {
        onComplete = cb;
      }),
      pause: vi.fn(),
      resume: vi.fn(),
      cancel: vi.fn(),
    };
    const done = vi.fn();
    const controller = new SlideshowController(deck, narrator, {
      timer,
      onNarrationComplete: done,
    });
    controller.subscribeSubtitles(() => {});

    void controller.jumpToSlide(0);
    expect(controller.isPlayingNarration).toBe(true);
    onComplete?.();
    expect(done).toHaveBeenCalledTimes(1);
    expect(controller.isPlayingNarration).toBe(false);
    expect(controller.subtitle.activeCue).toBeNull();
  });

  it('completes on estimated timing when no audio backend is available', () => {
    const narrator: NarratorBackend = {
      available: false,
      voices: () => [],
      speak: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
      cancel: vi.fn(),
    };
    const done = vi.fn();
    const controller = new SlideshowController(deck, narrator, {
      timer,
      tickMs: 100,
      onNarrationComplete: done,
    });

    void controller.jumpToSlide(0);
    expect(narrator.speak).not.toHaveBeenCalled();
    timer.advance(4000); // full estimated duration
    expect(done).toHaveBeenCalledTimes(1);
    expect(controller.isPlayingNarration).toBe(false);
  });

  it('pauses and resumes captions at the exact frozen position', () => {
    const narrator: NarratorBackend = {
      available: true,
      voices: () => [],
      speak: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
      cancel: vi.fn(),
    };
    const controller = new SlideshowController(deck, narrator, {
      timer,
      tickMs: 100,
    });
    const states: SubtitleState[] = [];
    controller.subscribeSubtitles((s) => states.push(s));

    void controller.jumpToSlide(0);
    timer.advance(1000); // 25%
    expect(states.at(-1)?.activeCue?.progress).toBeCloseTo(0.25, 5);

    controller.pauseNarration();
    expect(narrator.pause).toHaveBeenCalled();
    // Time passes while paused; progress must not advance.
    timer.advance(2000);
    expect(controller.subtitle.activeCue?.progress).toBeCloseTo(0.25, 5);

    controller.resumeNarration();
    expect(narrator.resume).toHaveBeenCalled();
    timer.advance(1000); // another 25% of real speaking time -> 50%
    expect(controller.subtitle.activeCue?.progress).toBeCloseTo(0.5, 5);
  });

  it('hides captions when subtitles are disabled', () => {
    const narrator: NarratorBackend = {
      available: true,
      voices: () => [],
      speak: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
      cancel: vi.fn(),
    };
    const controller = new SlideshowController(deck, narrator, { timer });
    void controller.jumpToSlide(0);
    expect(controller.subtitle.activeCue).not.toBeNull();
    controller.setSubtitlesEnabled(false);
    expect(controller.subtitle.enabled).toBe(false);
    expect(controller.subtitle.activeCue).toBeNull();
  });
});
