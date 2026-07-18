// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Slideshow narration controller.
//!
//! Coordinates spoken narration with word-synchronized subtitles for a slide
//! deck. The controller owns the timing model: it drives subtitle progress from
//! an injectable clock so captions advance in lock-step with speech, stays
//! locked for the full narration duration, and pauses/resumes as one unit.
//!
//! Word-level timing is derived analytically from the narration length and the
//! configured speech rate (fallback timing). Browser speech synthesis does not
//! expose reliable per-word boundary events, so the deterministic estimate keeps
//! captions and audio aligned without drifting seconds behind the voice.

import type { ExplainerDefinition, SlideDefinition } from '@aiperf/flow-compiler';
import type { NarratorBackend, NarratorUtterance } from '../narrative/narrator.js';
import type { SubtitleCue, SubtitleState } from '../narrative/subtitle-overlay.js';

/** Average speaking pace used to estimate narration duration. */
const DEFAULT_WORDS_PER_MINUTE = 150;

/** Subtitle refresh cadence; small enough to read as continuous karaoke. */
const DEFAULT_TICK_MS = 60;

/**
 * Injected time source for narration timing.
 *
 * Keeping the clock and interval scheduler behind an interface keeps wall-clock
 * globals out of the timing logic and lets tests advance narration
 * deterministically.
 */
export interface NarrationTimer {
  now(): number;
  setInterval(callback: () => void, intervalMs: number): () => void;
}

const defaultNarrationTimer: NarrationTimer = {
  now: () => Date.now(),
  setInterval: (callback, intervalMs) => {
    const handle = setInterval(callback, intervalMs);
    return () => clearInterval(handle);
  },
};

/** Optional configuration for narration timing and lifecycle. */
export interface SlideshowControllerOptions {
  /** Fired once when the current slide's narration reaches its end. */
  onNarrationComplete?: () => void;
  /** Injected time source; defaults to wall-clock time. */
  timer?: NarrationTimer;
  /** Speech rate multiplier applied to both audio and subtitle timing. */
  rate?: number;
  /** Backend voice id, or null for the backend default. */
  voiceId?: string | null;
  /** Estimated speaking pace for fallback subtitle timing. */
  wordsPerMinute?: number;
  /** Subtitle refresh cadence in milliseconds. */
  tickMs?: number;
  /** Whether subtitles are shown; toggled at runtime via setSubtitlesEnabled. */
  subtitlesEnabled?: boolean;
}

/** Receives every subtitle state transition for the active narration. */
export type SubtitleListener = (state: SubtitleState) => void;

const HIDDEN_SUBTITLE_STATE: SubtitleState = Object.freeze({
  enabled: false,
  activeCue: null,
});

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

export class SlideshowController {
  private currentIndex = 0;
  private isNarrating = false;
  private readonly deck: ExplainerDefinition;
  private readonly narrator: NarratorBackend;
  private readonly onNarrationComplete: (() => void) | null;
  private readonly timer: NarrationTimer;
  private readonly rate: number;
  private readonly voiceId: string | null;
  private readonly wordsPerMinute: number;
  private readonly tickMs: number;

  private readonly subtitleListeners = new Set<SubtitleListener>();
  private subtitlesEnabled: boolean;
  private subtitleState: SubtitleState = HIDDEN_SUBTITLE_STATE;

  // Active narration timing state.
  private generation = 0;
  private stopTicker: (() => void) | null = null;
  private activeText: string | null = null;
  private activeDurationMs = 0;
  private startedAtMs = 0;
  private accumulatedPauseMs = 0;
  private pausedAtMs: number | null = null;

  constructor(
    deck: ExplainerDefinition,
    narrator: NarratorBackend,
    options: SlideshowControllerOptions = {},
  ) {
    this.deck = deck;
    this.narrator = narrator;
    this.onNarrationComplete = options.onNarrationComplete ?? null;
    this.timer = options.timer ?? defaultNarrationTimer;
    this.rate = options.rate ?? 1;
    this.voiceId = options.voiceId ?? null;
    this.wordsPerMinute = options.wordsPerMinute ?? DEFAULT_WORDS_PER_MINUTE;
    this.tickMs = options.tickMs ?? DEFAULT_TICK_MS;
    this.subtitlesEnabled = options.subtitlesEnabled ?? true;
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

  /** Current subtitle state, reflecting live word-level progress. */
  get subtitle(): SubtitleState {
    return this.subtitleState;
  }

  getCurrentSlide(): SlideDefinition {
    return this.deck.slides[this.currentIndex]!;
  }

  /**
   * Subscribes to subtitle state transitions.
   *
   * The listener is invoked immediately with the current state so late
   * subscribers render the active caption without waiting for the next tick.
   */
  subscribeSubtitles(listener: SubtitleListener): () => void {
    this.subtitleListeners.add(listener);
    listener(this.subtitleState);
    return () => this.subtitleListeners.delete(listener);
  }

  /** Toggles subtitle visibility without altering narration timing. */
  setSubtitlesEnabled(enabled: boolean): void {
    if (this.subtitlesEnabled === enabled) {
      return;
    }
    this.subtitlesEnabled = enabled;
    this.emitSubtitle(this.subtitleState.activeCue, true);
  }

  async nextSlide(): Promise<void> {
    if (this.currentIndex < this.deck.slides.length - 1) {
      this.currentIndex++;
      await this.playNarrationForCurrentSlide();
    }
  }

  async prevSlide(): Promise<void> {
    if (this.currentIndex > 0) {
      this.currentIndex--;
      await this.playNarrationForCurrentSlide();
    }
  }

  async jumpToSlide(index: number): Promise<void> {
    if (index >= 0 && index < this.deck.slides.length) {
      this.currentIndex = index;
      await this.playNarrationForCurrentSlide();
    }
  }

  /** Freezes narration and its subtitles at the current word. */
  pauseNarration(): void {
    if (!this.isNarrating || this.pausedAtMs !== null) {
      return;
    }
    this.pausedAtMs = this.timer.now();
    this.stopTicker?.();
    this.stopTicker = null;
    this.narrator.pause();
  }

  /** Resumes narration and subtitles at the exact frozen position. */
  resumeNarration(): void {
    if (!this.isNarrating || this.pausedAtMs === null) {
      return;
    }
    this.accumulatedPauseMs += this.timer.now() - this.pausedAtMs;
    this.pausedAtMs = null;
    this.narrator.resume();
    this.startTicker();
    this.tick();
  }

  /** Cancels the active narration and clears its subtitles. */
  stopNarration(): void {
    this.teardownNarration();
    this.narrator.cancel();
    this.isNarrating = false;
    this.emitSubtitle(null);
  }

  private async playNarrationForCurrentSlide(): Promise<void> {
    // Cancel any in-flight narration before starting the next slide so audio
    // and captions cannot overlap or drift across the slide boundary.
    this.teardownNarration();
    this.narrator.cancel();

    const slide = this.getCurrentSlide();
    const text = slide.narration?.trim();
    if (text === undefined || text === '') {
      this.isNarrating = false;
      this.emitSubtitle(null);
      return;
    }

    const generation = ++this.generation;
    this.isNarrating = true;
    this.activeText = text;

    const words = Math.max(1, countWords(text));
    const msPerWord = 60_000 / this.wordsPerMinute / this.rate;
    this.activeDurationMs = words * msPerWord;
    this.startedAtMs = this.timer.now();
    this.accumulatedPauseMs = 0;
    this.pausedAtMs = null;

    this.emitSubtitle({ id: `slide-${this.currentIndex}`, text, progress: 0 });

    const utterance: NarratorUtterance = Object.freeze({
      cueId: `slide-${this.currentIndex}`,
      text,
      rate: this.rate,
      voiceId: this.voiceId,
    });

    if (this.narrator.available) {
      this.narrator.speak(utterance, () => this.finishNarration(generation));
    }
    // Whether or not audio is available, the injected timer drives subtitle
    // progress; when no audio backend reports completion, the ticker completes
    // the narration once estimated speech time elapses.
    this.startTicker();
  }

  private startTicker(): void {
    this.stopTicker?.();
    this.stopTicker = this.timer.setInterval(() => this.tick(), this.tickMs);
  }

  private tick(): void {
    if (!this.isNarrating || this.activeText === null) {
      return;
    }
    const elapsed =
      this.timer.now() - this.startedAtMs - this.accumulatedPauseMs;
    const progress = Math.min(1, Math.max(0, elapsed / this.activeDurationMs));
    this.emitSubtitle({
      id: `slide-${this.currentIndex}`,
      text: this.activeText,
      progress,
    });
    if (progress >= 1 && !this.narrator.available) {
      this.finishNarration(this.generation);
    }
  }

  private finishNarration(generation: number): void {
    if (generation !== this.generation || !this.isNarrating) {
      return;
    }
    this.teardownNarration();
    this.isNarrating = false;
    this.emitSubtitle(null);
    this.onNarrationComplete?.();
  }

  private teardownNarration(): void {
    this.generation++;
    this.stopTicker?.();
    this.stopTicker = null;
    this.activeText = null;
    this.pausedAtMs = null;
    this.accumulatedPauseMs = 0;
  }

  private emitSubtitle(cue: SubtitleCue | null, force = false): void {
    const next: SubtitleState = Object.freeze({
      enabled: this.subtitlesEnabled,
      activeCue: this.subtitlesEnabled ? cue : null,
    });
    const previous = this.subtitleState;
    if (
      !force &&
      next.enabled === previous.enabled &&
      next.activeCue?.id === previous.activeCue?.id &&
      next.activeCue?.progress === previous.activeCue?.progress &&
      next.activeCue?.text === previous.activeCue?.text
    ) {
      return;
    }
    this.subtitleState = next;
    for (const listener of this.subtitleListeners) {
      listener(next);
    }
  }
}
