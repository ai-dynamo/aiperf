/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/** Shared wall-clock multipliers for scene timelines and narration. */
export const PLAYBACK_SPEEDS = [0.75, 1, 1.25, 1.5, 2] as const;

export type PlaybackSpeed = (typeof PLAYBACK_SPEEDS)[number];

export const DEFAULT_PLAYBACK_SPEED: PlaybackSpeed = 1;

export const NARRATION_RATE = 1.08;
const WORDS_PER_MINUTE = 150 * NARRATION_RATE;
const POST_NARRATION_PAUSE_MS = 600;

/** Web Speech Synthesis rate is typically useful in ~0.5–2. */
export function speechRateForSpeed(speed: number): number {
  const rate = NARRATION_RATE * speed;
  return Math.min(2, Math.max(0.5, rate));
}

export function isPlaybackSpeed(value: unknown): value is PlaybackSpeed {
  return (
    typeof value === "number" &&
    (PLAYBACK_SPEEDS as readonly number[]).includes(value)
  );
}

export function narrationSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    "speechSynthesis" in window &&
    typeof SpeechSynthesisUtterance !== "undefined"
  );
}

export function splitWords(text: string): readonly string[] {
  return text.trim().split(/\s+/).filter(Boolean);
}

export function estimateNarrationMs(text: string, speed = 1): number {
  const words = splitWords(text).length;
  const baseMs = Math.max(
    2500,
    Math.round((words / WORDS_PER_MINUTE) * 60_000) + POST_NARRATION_PAUSE_MS,
  );
  const safeSpeed = speed > 0 ? speed : 1;
  return Math.max(400, Math.round(baseMs / safeSpeed));
}

export function stopNarration(): void {
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
}

/**
 * Must run inside a user-gesture handler. Browsers refuse to speak until then;
 * after a successful unlock, later slides can speak without another click.
 */
export function unlockSpeech(): boolean {
  if (!narrationSupported()) return false;
  try {
    window.speechSynthesis.getVoices();
    const prime = new SpeechSynthesisUtterance(" ");
    prime.volume = 0;
    prime.rate = NARRATION_RATE;
    window.speechSynthesis.speak(prime);
    window.speechSynthesis.cancel();
    return true;
  } catch {
    return false;
  }
}

function wordIndexFromChar(text: string, charIndex: number): number {
  if (charIndex <= 0) return 0;
  const prefix = text.slice(0, Math.min(charIndex, text.length));
  const count = splitWords(prefix).length;
  return Math.max(0, count - 1);
}

export function speakNarration(
  text: string,
  options: {
    useSpeech: boolean;
    voiceURI?: string;
    /** Wall-clock speed multiplier (same as scene playbackRate). */
    speed?: number;
    onWord?: (wordIndex: number) => void;
    onComplete: () => void;
  },
): () => void {
  stopNarration();
  const speed = options.speed && options.speed > 0 ? options.speed : 1;
  const words = splitWords(text);
  const fallbackMs = estimateNarrationMs(text, speed);
  const postPauseMs = Math.max(120, Math.round(POST_NARRATION_PAUSE_MS / speed));
  const timers: number[] = [];

  const clearTimers = () => {
    for (const timer of timers) window.clearTimeout(timer);
    timers.length = 0;
  };

  const driveEstimatedWords = () => {
    if (words.length === 0) return;
    const speakMs = Math.max(400, fallbackMs - postPauseMs);
    const stepMs = speakMs / words.length;
    words.forEach((_, index) => {
      timers.push(
        window.setTimeout(() => {
          options.onWord?.(index);
        }, Math.round(index * stepMs)),
      );
    });
  };

  if (!options.useSpeech || !narrationSupported()) {
    options.onWord?.(0);
    driveEstimatedWords();
    timers.push(window.setTimeout(options.onComplete, fallbackMs));
    return () => clearTimers();
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = speechRateForSpeed(speed);
  utterance.pitch = 1;
  if (options.voiceURI) {
    const voice = window.speechSynthesis
      .getVoices()
      .find((candidate) => candidate.voiceURI === options.voiceURI);
    if (voice) utterance.voice = voice;
  }

  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    options.onWord?.(Math.max(0, words.length - 1));
    timers.push(window.setTimeout(options.onComplete, postPauseMs));
  };

  options.onWord?.(0);
  // Always run estimated word timing — many engines skip onboundary, which
  // would otherwise leave karaoke stuck on the first word.
  driveEstimatedWords();
  utterance.onboundary = (event) => {
    if (event.name && event.name !== "word") return;
    options.onWord?.(wordIndexFromChar(text, event.charIndex));
  };
  utterance.onend = finish;
  utterance.onerror = (event) => {
    // Intentional stopNarration()/cancel must not schedule fallback advancement.
    if (done || event.error === "interrupted" || event.error === "canceled") {
      return;
    }
    // Estimated word timers already started; keep a completion fallback.
    timers.push(window.setTimeout(finish, fallbackMs));
  };

  window.speechSynthesis.speak(utterance);
  timers.push(window.setTimeout(finish, fallbackMs + Math.round(4000 / speed)));

  return () => {
    done = true;
    clearTimers();
    stopNarration();
  };
}
