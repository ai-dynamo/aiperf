// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  createKokoroNarratorBackend,
  type KokoroNarratorBackend,
  type KokoroNarratorSnapshot,
} from "../packages/runtime/src/narrative/kokoro-narrator";
import {
  createBrowserSpeechSynthesisBackend,
  type NarratorBackend,
} from "../packages/runtime/src/narrative/narrator";

const unavailableNarratorBackend: NarratorBackend = Object.freeze({
  available: false,
  voices: () => Object.freeze([]),
  speak: () => undefined,
  pause: () => undefined,
  resume: () => undefined,
  cancel: () => undefined,
});

let sharedKokoroBackend: KokoroNarratorBackend | null = null;

/** Shared Kokoro backend for the preview shell (lazy singleton). */
export function previewKokoroBackend(): KokoroNarratorBackend | null {
  if (typeof Worker !== "function") {
    return null;
  }
  if (sharedKokoroBackend === null) {
    sharedKokoroBackend = createKokoroNarratorBackend({
      fallback: createBrowserSpeechSynthesisBackend(),
      workerFactory: () =>
        new Worker(
          new URL(
            "../packages/runtime/src/narrative/kokoro-worker.ts",
            import.meta.url,
          ),
          { type: "module", name: "aiperf-kokoro-narrator" },
        ),
    });
  }
  return sharedKokoroBackend;
}

/** Preferred local neural narrator with browser speech fallback. */
export function createPreviewNarratorBackend(): NarratorBackend {
  return (
    previewKokoroBackend() ??
    createBrowserSpeechSynthesisBackend() ??
    unavailableNarratorBackend
  );
}

/** Observable Kokoro lifecycle for preview chrome. */
export function subscribePreviewKokoroState(
  listener: (state: KokoroNarratorSnapshot) => void,
): () => void {
  const backend = previewKokoroBackend();
  if (backend === null) {
    listener(
      Object.freeze({
        status: "fallback",
        engine: "web-speech",
        progress: 0,
        progressFile: null,
        error: null,
        activeCueId: null,
        needsUserActivation: false,
      }),
    );
    return () => undefined;
  }
  return backend.subscribe(listener);
}

/** Preloads Kokoro weights without audible output. */
export function prewarmPreviewNarrator(): void {
  const backend = previewKokoroBackend();
  if (backend !== null) {
    void backend.prewarm();
  }
}

/**
 * Unlocks narration from a user gesture: Kokoro Web Audio first, then
 * SpeechSynthesis priming when neural inference is unavailable.
 */
export function unlockPreviewSpeech(): boolean {
  const kokoro = previewKokoroBackend();
  if (kokoro !== null) {
    void kokoro.activate();
    void kokoro.prewarm();
  }
  if (typeof window === "undefined" || !("speechSynthesis" in window)) {
    return kokoro !== null;
  }
  try {
    window.speechSynthesis.getVoices();
    const prime = new SpeechSynthesisUtterance(" ");
    prime.volume = 0;
    window.speechSynthesis.speak(prime);
    window.speechSynthesis.cancel();
    return true;
  } catch {
    return kokoro !== null;
  }
}
