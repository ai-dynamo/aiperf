// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Preferred local Kokoro narration with WebGPU acceleration and WASM fallback.

import type {
  NarratorBackend,
  NarratorUtterance,
  NarratorVoice,
} from "./narrator.js";
import type {
  KokoroDevice,
  KokoroDtype,
  KokoroWorkerCommand,
  KokoroWorkerMessage,
} from "./kokoro-worker.js";

export type { KokoroWorkerMessage } from "./kokoro-worker.js";

const DEFAULT_MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const DEFAULT_VOICE: NarratorVoice = Object.freeze({
  id: "af_heart",
  name: "Heart",
  language: "en-us",
  default: true,
});

export type KokoroNarratorStatus =
  | "idle"
  | "loading"
  | "ready"
  | "generating"
  | "playing"
  | "paused"
  | "needs-user-activation"
  | "fallback"
  | "error";

/** Observable local narrator lifecycle state for loading and playback UI. */
export type KokoroNarratorSnapshot = Readonly<{
  status: KokoroNarratorStatus;
  engine: KokoroDevice | "web-speech" | null;
  progress: number;
  progressFile: string | null;
  error: string | null;
  activeCueId: string | null;
  needsUserActivation: boolean;
}>;

/** Injectable module-worker boundary used by the browser backend. */
export interface KokoroWorkerPort {
  onmessage: ((event: MessageEvent<KokoroWorkerMessage>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage(message: KokoroWorkerCommand): void;
  terminate(): void;
}

export type KokoroNarratorOptions = Readonly<{
  fallback?: NarratorBackend | null;
  modelId?: string;
  workerFactory?: () => KokoroWorkerPort;
  audioContextFactory?: () => AudioContext;
  webGpuAvailable?: () => boolean;
}>;

type ActiveRequest = Readonly<{
  requestId: string;
  utterance: NarratorUtterance;
}>;

type PendingAudio = Readonly<{
  request: ActiveRequest;
  buffer: AudioBuffer;
}>;

function defaultWorkerFactory(): KokoroWorkerPort {
  return new Worker(new URL("./kokoro-worker.js", import.meta.url), {
    type: "module",
    name: "aiperf-kokoro-narrator",
  });
}

function defaultAudioContextFactory(): AudioContext {
  const scope = globalThis as typeof globalThis & {
    webkitAudioContext?: typeof AudioContext;
  };
  const AudioContextConstructor =
    scope.AudioContext ?? scope.webkitAudioContext;
  if (AudioContextConstructor === undefined) {
    throw new Error("Web Audio is unavailable");
  }
  return new AudioContextConstructor();
}

function defaultWebGpuAvailable(): boolean {
  return (
    typeof navigator === "object" &&
    navigator !== null &&
    "gpu" in navigator
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Local Kokoro backend that keeps model inference in a dedicated worker. */
export class KokoroNarratorBackend implements NarratorBackend {
  readonly #fallback: NarratorBackend | null;
  readonly #modelId: string;
  readonly #workerFactory: () => KokoroWorkerPort;
  readonly #audioContextFactory: () => AudioContext;
  readonly #webGpuAvailable: () => boolean;
  readonly #workerSupported: boolean;
  readonly #listeners = new Set<(state: KokoroNarratorSnapshot) => void>();
  readonly #queue: NarratorUtterance[] = [];
  #state: KokoroNarratorSnapshot = Object.freeze({
    status: "idle",
    engine: null,
    progress: 0,
    progressFile: null,
    error: null,
    activeCueId: null,
    needsUserActivation: false,
  });
  #worker: KokoroWorkerPort | null = null;
  #audioContext: AudioContext | null = null;
  #source: AudioBufferSourceNode | null = null;
  #pendingAudio: PendingAudio | null = null;
  #activeRequest: ActiveRequest | null = null;
  #voices: readonly NarratorVoice[] = Object.freeze([DEFAULT_VOICE]);
  #prewarmPromise: Promise<void> | null = null;
  #resolvePrewarm: (() => void) | null = null;
  #rejectPrewarm: ((error: Error) => void) | null = null;
  #modelReady = false;
  #activated = false;
  #requestSequence = 0;

  constructor(options: KokoroNarratorOptions = {}) {
    this.#fallback = options.fallback ?? null;
    this.#modelId = options.modelId ?? DEFAULT_MODEL_ID;
    this.#workerFactory = options.workerFactory ?? defaultWorkerFactory;
    this.#workerSupported =
      options.workerFactory !== undefined || typeof Worker === "function";
    this.#audioContextFactory =
      options.audioContextFactory ?? defaultAudioContextFactory;
    this.#webGpuAvailable =
      options.webGpuAvailable ?? defaultWebGpuAvailable;
  }

  get available(): boolean {
    return this.#workerSupported || this.#fallback?.available === true;
  }

  voices(): readonly NarratorVoice[] {
    return this.#state.engine === "web-speech" && this.#fallback !== null
      ? this.#fallback.voices()
      : this.#voices;
  }

  snapshot(): KokoroNarratorSnapshot {
    return this.#state;
  }

  /** Subscribes to model loading, fallback, activation, and playback state. */
  subscribe(
    listener: (state: KokoroNarratorSnapshot) => void,
  ): () => void {
    this.#listeners.add(listener);
    listener(this.#state);
    return () => this.#listeners.delete(listener);
  }

  /** Starts model download and initialization without producing audio. */
  prewarm(): Promise<void> {
    if (this.#modelReady || this.#state.engine === "web-speech") {
      return Promise.resolve();
    }
    if (this.#prewarmPromise !== null) {
      return this.#prewarmPromise;
    }

    this.#prewarmPromise = new Promise<void>((resolve, reject) => {
      this.#resolvePrewarm = resolve;
      this.#rejectPrewarm = reject;
    });
    const device = this.#webGpuAvailable() ? "webgpu" : "wasm";
    this.#startWorker(device, device === "webgpu" ? "fp32" : "q8");
    return this.#prewarmPromise;
  }

  /**
   * Unlocks Web Audio from the same user gesture that starts narration.
   *
   * Browsers may still reject activation; that policy result remains visible
   * instead of being bypassed.
   */
  async activate(): Promise<void> {
    try {
      const context = this.#getAudioContext();
      await context.resume();
      if (context.state !== "running") {
        throw new DOMException("Audio playback requires user activation", "NotAllowedError");
      }
      this.#activated = true;
      this.#update({
        needsUserActivation: false,
        status:
          this.#state.status === "needs-user-activation"
            ? "ready"
            : this.#state.status,
      });
      await this.#playPendingAudio();
    } catch (error) {
      this.#activationBlocked(error);
    }
  }

  speak(utterance: NarratorUtterance): void {
    if (this.#state.engine === "web-speech" && this.#fallback !== null) {
      this.#fallback.speak(utterance);
      return;
    }
    this.#queue.push(utterance);
    void this.prewarm()
      .then(() => this.#pump())
      .catch(() => undefined);
  }

  pause(): void {
    if (this.#state.engine === "web-speech") {
      this.#fallback?.pause();
      return;
    }
    if (this.#audioContext !== null) {
      void this.#audioContext.suspend();
    }
    this.#update({ status: "paused" });
  }

  resume(): void {
    if (this.#state.engine === "web-speech") {
      this.#fallback?.resume();
      return;
    }
    void this.activate();
  }

  cancel(): void {
    this.#queue.length = 0;
    const active = this.#activeRequest;
    if (active !== null) {
      this.#worker?.postMessage({
        type: "cancel",
        requestId: active.requestId,
        cueId: active.utterance.cueId,
      });
    }
    this.#activeRequest = null;
    this.#pendingAudio = null;
    if (this.#source !== null) {
      try {
        this.#source.stop();
      } catch {
        // An unstarted or already-ended source has no remaining audio to stop.
      }
      this.#source = null;
    }
    this.#fallback?.cancel();
    this.#update({
      status: this.#modelReady ? "ready" : "idle",
      activeCueId: null,
      needsUserActivation: false,
    });
  }

  #startWorker(device: KokoroDevice, dtype: KokoroDtype): void {
    try {
      const worker = this.#workerFactory();
      this.#worker = worker;
      worker.onmessage = (event) => {
        if (this.#worker !== worker) {
          return;
        }
        void this.#handleWorkerMessage(event.data);
      };
      worker.onerror = (event) => {
        if (this.#worker !== worker) {
          return;
        }
        this.#handleWorkerError(event.message || "Kokoro worker failed");
      };
      this.#update({
        status: "loading",
        engine: device,
        progress: 0,
        progressFile: null,
        error: null,
      });
      worker.postMessage({
        type: "initialize",
        modelId: this.#modelId,
        device,
        dtype,
      });
    } catch (error) {
      this.#handleWorkerError(errorMessage(error));
    }
  }

  async #handleWorkerMessage(message: KokoroWorkerMessage): Promise<void> {
    if (message.type === "progress") {
      this.#update({
        progress: Math.min(1, Math.max(0, message.progress / 100)),
        progressFile: message.file,
      });
      return;
    }

    if (message.type === "ready") {
      this.#modelReady = true;
      this.#voices = Object.freeze(
        message.voices.length === 0
          ? [DEFAULT_VOICE]
          : message.voices.map((voice) => Object.freeze({ ...voice })),
      );
      this.#update({
        status: "ready",
        progress: 1,
        progressFile: null,
        error: null,
      });
      this.#resolvePrewarm?.();
      this.#clearPrewarmSettlement();
      this.#pump();
      return;
    }

    if (message.type === "error") {
      if (!this.#modelReady) {
        this.#handleWorkerError(message.message);
      } else {
        this.#switchToFallback(message.message);
      }
      return;
    }

    if (
      this.#activeRequest === null ||
      message.requestId !== this.#activeRequest.requestId
    ) {
      return;
    }
    try {
      const context = this.#getAudioContext();
      const buffer = await context.decodeAudioData(message.wav);
      if (
        this.#activeRequest === null ||
        message.requestId !== this.#activeRequest.requestId
      ) {
        return;
      }
      this.#pendingAudio = Object.freeze({
        request: this.#activeRequest,
        buffer,
      });
      await this.#playPendingAudio();
    } catch (error) {
      this.#switchToFallback(errorMessage(error));
    }
  }

  #handleWorkerError(message: string): void {
    if (this.#state.engine === "webgpu") {
      this.#worker?.terminate();
      this.#worker = null;
      this.#startWorker("wasm", "q8");
      return;
    }
    this.#switchToFallback(message);
  }

  #switchToFallback(message: string): void {
    this.#worker?.terminate();
    this.#worker = null;
    this.#modelReady = false;
    const pending = [
      ...(this.#activeRequest === null
        ? []
        : [this.#activeRequest.utterance]),
      ...this.#queue,
    ];
    this.#activeRequest = null;
    this.#queue.length = 0;

    if (this.#fallback?.available === true) {
      this.#update({
        status: "fallback",
        engine: "web-speech",
        error: message,
        activeCueId: null,
      });
      this.#resolvePrewarm?.();
      this.#clearPrewarmSettlement();
      for (const utterance of pending) {
        this.#fallback.speak(utterance);
      }
      return;
    }

    const error = new Error(message);
    this.#update({
      status: "error",
      error: message,
      activeCueId: null,
    });
    this.#rejectPrewarm?.(error);
    this.#clearPrewarmSettlement();
  }

  #pump(): void {
    if (
      !this.#modelReady ||
      this.#activeRequest !== null ||
      this.#queue.length === 0
    ) {
      return;
    }
    const utterance = this.#queue.shift();
    if (utterance === undefined) {
      return;
    }
    const request = Object.freeze({
      requestId: `${utterance.cueId}:${++this.#requestSequence}`,
      utterance,
    });
    this.#activeRequest = request;
    this.#update({
      status: "generating",
      activeCueId: utterance.cueId,
      error: null,
    });
    this.#worker?.postMessage({
      type: "synthesize",
      requestId: request.requestId,
      cueId: utterance.cueId,
      text: utterance.text,
      voiceId: utterance.voiceId,
      rate: utterance.rate,
    });
  }

  async #playPendingAudio(): Promise<void> {
    const pending = this.#pendingAudio;
    if (pending === null) {
      return;
    }
    const context = this.#getAudioContext();
    if (!this.#activated || context.state !== "running") {
      try {
        await context.resume();
      } catch (error) {
        this.#activationBlocked(error);
        return;
      }
      if (context.state !== "running") {
        this.#activationBlocked(
          new DOMException(
            "Audio playback requires user activation",
            "NotAllowedError",
          ),
        );
        return;
      }
      this.#activated = true;
    }

    if (
      this.#pendingAudio !== pending ||
      this.#activeRequest?.requestId !== pending.request.requestId
    ) {
      return;
    }
    const source = context.createBufferSource();
    source.buffer = pending.buffer;
    source.connect(context.destination);
    source.onended = () => {
      if (this.#source !== source) {
        return;
      }
      this.#source = null;
      this.#pendingAudio = null;
      this.#activeRequest = null;
      this.#update({
        status: "ready",
        activeCueId: null,
      });
      this.#pump();
    };
    this.#source = source;
    this.#pendingAudio = null;
    try {
      source.start();
      this.#update({
        status: "playing",
        activeCueId: pending.request.utterance.cueId,
        needsUserActivation: false,
      });
    } catch (error) {
      this.#pendingAudio = pending;
      this.#source = null;
      this.#activationBlocked(error);
    }
  }

  #activationBlocked(error: unknown): void {
    this.#activated = false;
    this.#update({
      status: "needs-user-activation",
      needsUserActivation: true,
      error:
        error instanceof DOMException && error.name === "NotAllowedError"
          ? null
          : errorMessage(error),
    });
  }

  #getAudioContext(): AudioContext {
    this.#audioContext ??= this.#audioContextFactory();
    return this.#audioContext;
  }

  #clearPrewarmSettlement(): void {
    this.#resolvePrewarm = null;
    this.#rejectPrewarm = null;
  }

  #update(
    update: Partial<KokoroNarratorSnapshot>,
  ): void {
    this.#state = Object.freeze({ ...this.#state, ...update });
    for (const listener of this.#listeners) {
      listener(this.#state);
    }
  }
}

/** Creates the preferred local narrator backend. */
export function createKokoroNarratorBackend(
  options: KokoroNarratorOptions = {},
): KokoroNarratorBackend {
  return new KokoroNarratorBackend(options);
}
