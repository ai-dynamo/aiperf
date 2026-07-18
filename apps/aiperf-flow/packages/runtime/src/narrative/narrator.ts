// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { normalizeSceneTimeMs } from "../player.js";

/** One authored narration cue on the integer scene timeline. */
export type NarrationCue = Readonly<{
  id: string;
  atMs: number;
  text: string;
}>;

/** A backend-independent voice exposed by a narrator backend. */
export type NarratorVoice = Readonly<{
  id: string;
  name: string;
  language: string;
  default: boolean;
}>;

/** Speech request emitted when the scene timeline crosses a narration cue. */
export type NarratorUtterance = Readonly<{
  cueId: string;
  text: string;
  rate: number;
  voiceId: string | null;
}>;

/** Injected audible output boundary for deterministic narration control. */
export interface NarratorBackend {
  readonly available: boolean;
  voices(): readonly NarratorVoice[];
  speak(utterance: NarratorUtterance): void;
  pause(): void;
  resume(): void;
  cancel(): void;
}

export type NarratorStatus = "idle" | "playing" | "paused" | "stopped";

/** Serializable deterministic narrator state. */
export type NarratorSnapshot = Readonly<{
  timeMs: number;
  activeCueId: string | null;
  status: NarratorStatus;
  muted: boolean;
  rate: number;
  voiceId: string | null;
}>;

function sceneTimeMs(value: number): number {
  return Math.min(Number.MAX_SAFE_INTEGER, normalizeSceneTimeMs(value));
}

function canonicalCues(cues: readonly NarrationCue[]): readonly NarrationCue[] {
  const ids = new Set<string>();
  return Object.freeze(
    cues
      .map((cue, sourceIndex) => {
        if (cue.id.trim() === "") {
          throw new Error("Narration cue id must not be empty");
        }
        if (ids.has(cue.id)) {
          throw new Error(`Duplicate narration cue id: ${cue.id}`);
        }
        if (cue.text.trim() === "") {
          throw new Error(`Narration cue ${cue.id} must contain text`);
        }
        ids.add(cue.id);
        return {
          cue: Object.freeze({
            id: cue.id,
            atMs: sceneTimeMs(cue.atMs),
            text: cue.text,
          }),
          sourceIndex,
        };
      })
      .sort(
        (left, right) =>
          left.cue.atMs - right.cue.atMs ||
          left.sourceIndex - right.sourceIndex,
      )
      .map(({ cue }) => cue),
  );
}

function firstCueAtOrAfter(
  cues: readonly NarrationCue[],
  timeMs: number,
): number {
  const index = cues.findIndex((cue) => cue.atMs >= timeMs);
  return index < 0 ? cues.length : index;
}

/**
 * Coordinates audible narration with externally owned integer scene time.
 *
 * `sync()` is the only progression input, so wall time and browser globals
 * never enter deterministic controller state.
 */
export class NarratorController {
  readonly #cues: readonly NarrationCue[];
  readonly #backend: NarratorBackend;
  #timeMs = 0;
  #nextCueIndex = 0;
  #activeCueId: string | null = null;
  #status: NarratorStatus = "idle";
  #muted = false;
  #rate = 1;
  #voiceId: string | null = null;

  constructor(cues: readonly NarrationCue[], backend: NarratorBackend) {
    this.#cues = canonicalCues(cues);
    this.#backend = backend;
  }

  /** Starts narration at the current or supplied scene beat. */
  play(timeMs = this.#timeMs): NarratorSnapshot {
    if (this.#status === "playing") {
      return this.sync(timeMs);
    }
    if (this.#status === "paused" && sceneTimeMs(timeMs) === this.#timeMs) {
      return this.resume(timeMs);
    }
    if (sceneTimeMs(timeMs) !== this.#timeMs) {
      this.seek(timeMs);
    }
    this.#status = "playing";
    this.#dispatchDueCues();
    return this.snapshot();
  }

  /** Advances narration to externally evaluated scene time. */
  sync(timeMs: number): NarratorSnapshot {
    if (this.#status !== "playing") {
      return this.snapshot();
    }
    const nextTimeMs = sceneTimeMs(timeMs);
    if (nextTimeMs < this.#timeMs) {
      this.seek(nextTimeMs);
      this.#status = "playing";
    } else {
      this.#timeMs = nextTimeMs;
    }
    this.#dispatchDueCues();
    return this.snapshot();
  }

  /** Freezes narration at one exact scene beat for exploration. */
  pause(timeMs = this.#timeMs): NarratorSnapshot {
    if (this.#status === "playing") {
      this.sync(timeMs);
      this.#status = "paused";
      if (!this.#muted && this.#backend.available) {
        this.#backend.pause();
      }
    }
    return this.snapshot();
  }

  /** Continues the paused backend utterance without redispatching its cue. */
  resume(timeMs = this.#timeMs): NarratorSnapshot {
    if (this.#status !== "paused") {
      return this.snapshot();
    }
    const resumeTimeMs = sceneTimeMs(timeMs);
    if (resumeTimeMs !== this.#timeMs) {
      this.seek(resumeTimeMs);
      this.#status = "paused";
    }
    this.#status = "playing";
    if (
      this.#activeCueId !== null &&
      !this.#muted &&
      this.#backend.available
    ) {
      this.#backend.resume();
    } else {
      this.#dispatchDueCues();
    }
    return this.snapshot();
  }

  /** Repositions narration, cancelling audio from the previous beat. */
  seek(timeMs: number): NarratorSnapshot {
    this.#backend.cancel();
    this.#timeMs = sceneTimeMs(timeMs);
    this.#nextCueIndex = firstCueAtOrAfter(this.#cues, this.#timeMs);
    this.#activeCueId = null;
    if (this.#status === "playing") {
      this.#dispatchDueCues();
    }
    return this.snapshot();
  }

  /** Cancels narration and resets its explicit replay position to zero. */
  stop(): NarratorSnapshot {
    this.#backend.cancel();
    this.#timeMs = 0;
    this.#nextCueIndex = 0;
    this.#activeCueId = null;
    this.#status = "stopped";
    return this.snapshot();
  }

  /** Enables or disables audible output without changing scene progression. */
  setMuted(muted: boolean): NarratorSnapshot {
    if (muted && !this.#muted) {
      this.#backend.cancel();
      this.#activeCueId = null;
    }
    this.#muted = muted;
    return this.snapshot();
  }

  /** Selects the speech rate for subsequently dispatched cues. */
  setRate(rate: number): NarratorSnapshot {
    if (!Number.isFinite(rate) || rate < 0.1 || rate > 10) {
      throw new RangeError("Narration rate must be between 0.1 and 10");
    }
    this.#rate = rate;
    return this.snapshot();
  }

  /** Selects a backend voice id, or restores backend-default selection. */
  selectVoice(voiceId: string | null): NarratorSnapshot {
    this.#voiceId = voiceId;
    return this.snapshot();
  }

  /** Returns currently available backend voices. */
  voices(): readonly NarratorVoice[] {
    return this.#backend.voices();
  }

  snapshot(): NarratorSnapshot {
    return Object.freeze({
      timeMs: this.#timeMs,
      activeCueId: this.#activeCueId,
      status: this.#status,
      muted: this.#muted,
      rate: this.#rate,
      voiceId: this.#voiceId,
    });
  }

  #dispatchDueCues(): void {
    while (this.#nextCueIndex < this.#cues.length) {
      const cue = this.#cues[this.#nextCueIndex];
      if (cue === undefined || cue.atMs > this.#timeMs) {
        return;
      }
      this.#nextCueIndex += 1;
      if (this.#muted || !this.#backend.available) {
        continue;
      }
      // Cancel before speak so browser SpeechSynthesis cannot queue ahead of
      // the integer scene clock and drift seconds behind the visual timeline.
      this.#backend.cancel();
      this.#activeCueId = cue.id;
      this.#backend.speak(
        Object.freeze({
          cueId: cue.id,
          text: cue.text,
          rate: this.#rate,
          voiceId: this.#voiceId,
        }),
      );
    }
  }
}

/** Minimal browser voice surface used by the feature-detected backend. */
export type SpeechSynthesisVoicePort = Readonly<{
  voiceURI: string;
  name: string;
  lang: string;
  default: boolean;
}>;

/** Minimal utterance surface required by the browser backend. */
export type SpeechSynthesisUtterancePort = {
  rate: number;
  lang: string;
  voice: SpeechSynthesisVoicePort | null;
};

/** Browser speech APIs accepted as an injectable platform boundary. */
export type SpeechSynthesisPlatform = Readonly<{
  synthesis: Readonly<{
    getVoices(): readonly SpeechSynthesisVoicePort[];
    speak(utterance: SpeechSynthesisUtterancePort): void;
    pause(): void;
    resume(): void;
    cancel(): void;
  }>;
  Utterance: new (text: string) => SpeechSynthesisUtterancePort;
  /**
   * Schedules speak after cancel. Chrome silently drops speak() when it runs in
   * the same turn as cancel(); the default uses setTimeout(0).
   */
  scheduleSpeak?: (run: () => void) => () => void;
}>;

const PREFERRED_BROWSER_VOICE_NAME = "Google UK English Male";
const PREFERRED_BROWSER_VOICE_LANG = "en-GB";

class BrowserSpeechSynthesisBackend implements NarratorBackend {
  readonly available = true;
  readonly #platform: SpeechSynthesisPlatform;
  readonly #scheduleSpeak: (run: () => void) => () => void;
  #pending: NarratorUtterance | null = null;
  #cancelScheduled: (() => void) | null = null;

  constructor(platform: SpeechSynthesisPlatform) {
    this.#platform = platform;
    this.#scheduleSpeak =
      platform.scheduleSpeak ??
      ((run) => {
        const handle = setTimeout(run, 0);
        return () => clearTimeout(handle);
      });
  }

  voices(): readonly NarratorVoice[] {
    return Object.freeze(
      this.#platform.synthesis.getVoices().map((voice) =>
        Object.freeze({
          id: voice.voiceURI,
          name: voice.name,
          language: voice.lang,
          default: voice.default,
        }),
      ),
    );
  }

  speak(request: NarratorUtterance): void {
    this.#pending = request;
    this.#cancelScheduled?.();
    this.#cancelScheduled = this.#scheduleSpeak(() => {
      this.#cancelScheduled = null;
      const pending = this.#pending;
      this.#pending = null;
      if (pending === null) {
        return;
      }
      this.#speakNow(pending);
    });
  }

  pause(): void {
    this.#platform.synthesis.pause();
  }

  resume(): void {
    this.#platform.synthesis.resume();
  }

  cancel(): void {
    this.#pending = null;
    this.#cancelScheduled?.();
    this.#cancelScheduled = null;
    this.#platform.synthesis.cancel();
  }

  #speakNow(request: NarratorUtterance): void {
    const voices = this.#platform.synthesis.getVoices();
    const voice = resolveBrowserSpeechVoice(voices, request.voiceId);
    const utterance = new this.#platform.Utterance(request.text);
    utterance.rate = request.rate;
    utterance.lang = voice?.lang ?? PREFERRED_BROWSER_VOICE_LANG;
    utterance.voice = voice;
    this.#platform.synthesis.speak(utterance);
  }
}

/**
 * Prefer an explicit browser voice URI, otherwise Chrome's UK English Male
 * (or the closest en-GB male / en-GB voice available on the platform).
 */
export function resolveBrowserSpeechVoice(
  voices: readonly SpeechSynthesisVoicePort[],
  voiceId: string | null,
): SpeechSynthesisVoicePort | null {
  if (voiceId !== null) {
    const selected = voices.find((voice) => voice.voiceURI === voiceId);
    if (selected !== undefined) {
      return selected;
    }
  }

  const exact = voices.find(
    (voice) =>
      voice.name === PREFERRED_BROWSER_VOICE_NAME ||
      voice.voiceURI === PREFERRED_BROWSER_VOICE_NAME,
  );
  if (exact !== undefined) {
    return exact;
  }

  const ukMale = voices.find((voice) => isUkEnglishMaleVoice(voice));
  if (ukMale !== undefined) {
    return ukMale;
  }

  const enGb = voices.find((voice) =>
    voice.lang.toLowerCase().startsWith("en-gb"),
  );
  return enGb ?? null;
}

function isUkEnglishMaleVoice(voice: SpeechSynthesisVoicePort): boolean {
  const name = voice.name.toLowerCase();
  const lang = voice.lang.toLowerCase();
  const uri = voice.voiceURI.toLowerCase();
  const haystack = `${name} ${uri}`;
  if (/\bfemale\b/.test(haystack)) {
    return false;
  }
  const looksUk =
    lang.startsWith("en-gb") ||
    name.includes("uk english") ||
    name.includes("british") ||
    uri.includes("en-gb");
  const looksMale =
    /\bmale\b/.test(haystack) ||
    name.includes("daniel") ||
    name.includes("ravi") ||
    name.includes("george");
  return looksUk && looksMale;
}

function detectBrowserSpeechSynthesis(): SpeechSynthesisPlatform | null {
  const scope = globalThis as typeof globalThis & {
    speechSynthesis?: SpeechSynthesis;
    SpeechSynthesisUtterance?: new (text: string) => SpeechSynthesisUtterance;
  };
  if (
    typeof scope.speechSynthesis !== "object" ||
    scope.speechSynthesis === null ||
    typeof scope.SpeechSynthesisUtterance !== "function"
  ) {
    return null;
  }
  // Chrome populates voices asynchronously; prime the list now and on change.
  scope.speechSynthesis.getVoices();
  const synthesis = scope.speechSynthesis;
  if (typeof synthesis.addEventListener === "function") {
    synthesis.addEventListener("voiceschanged", () => {
      synthesis.getVoices();
    });
  }
  return {
    synthesis: scope.speechSynthesis,
    Utterance: scope.SpeechSynthesisUtterance,
  } as SpeechSynthesisPlatform;
}

/**
 * Creates immediate browser speech output when SpeechSynthesis is supported.
 *
 * Passing a platform keeps tests and non-browser runtimes free of ambient
 * globals; passing `null` explicitly disables feature detection.
 */
export function createBrowserSpeechSynthesisBackend(
  platform?: SpeechSynthesisPlatform | null,
): NarratorBackend | null {
  const detected =
    platform === undefined ? detectBrowserSpeechSynthesis() : platform;
  return detected === null
    ? null
    : new BrowserSpeechSynthesisBackend(detected);
}
