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
 * Unlocks narration from a user gesture: awaits Kokoro Web Audio resume, then
 * primes SpeechSynthesis when needed. Call from the consent/play click itself.
 */
export async function unlockPreviewSpeech(): Promise<boolean> {
  const kokoro = previewKokoroBackend();
  if (kokoro !== null) {
    void kokoro.prewarm();
    await kokoro.activate();
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

/** True when the preferred narrator can accept audible cues now. */
export function previewNarratorReady(): boolean {
  const kokoro = previewKokoroBackend();
  if (kokoro === null) {
    return typeof window !== "undefined" && "speechSynthesis" in window;
  }
  const snapshot = kokoro.snapshot();
  return (
    snapshot.status === "ready" ||
    snapshot.status === "generating" ||
    snapshot.status === "playing" ||
    snapshot.status === "paused" ||
    snapshot.engine === "web-speech"
  );
}

/**
 * Resolves when Kokoro (or its Web Speech fallback) can synthesize. Safe to
 * call after activate(); does not produce audio by itself.
 */
export async function whenPreviewNarratorReady(): Promise<void> {
  const kokoro = previewKokoroBackend();
  if (kokoro === null) {
    return;
  }
  if (previewNarratorReady()) {
    return;
  }
  await new Promise<void>((resolve) => {
    const unsubscribe = kokoro.subscribe((state) => {
      if (
        state.status === "ready" ||
        state.status === "fallback" ||
        state.engine === "web-speech" ||
        state.status === "error"
      ) {
        unsubscribe();
        resolve();
      }
    });
    void kokoro.prewarm().catch(() => {
      unsubscribe();
      resolve();
    });
  });
}
