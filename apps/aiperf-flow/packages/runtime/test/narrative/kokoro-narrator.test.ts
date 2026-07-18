// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test, vi } from "vitest";

import type {
  NarratorBackend,
  NarratorUtterance,
} from "../../src/narrative/narrator.js";
import {
  createKokoroNarratorBackend,
  type KokoroWorkerMessage,
  type KokoroWorkerPort,
} from "../../src/narrative/kokoro-narrator.js";

class FakeWorker implements KokoroWorkerPort {
  onmessage: ((event: MessageEvent<KokoroWorkerMessage>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  readonly sent: unknown[] = [];
  terminated = false;

  postMessage(message: unknown): void {
    this.sent.push(message);
  }

  terminate(): void {
    this.terminated = true;
  }

  emit(message: KokoroWorkerMessage): void {
    this.onmessage?.({ data: message } as MessageEvent<KokoroWorkerMessage>);
  }
}

class FakeSource {
  buffer: AudioBuffer | null = null;
  onended: (() => void) | null = null;
  readonly start = vi.fn();
  readonly stop = vi.fn();
  readonly connect = vi.fn();
}

class FakeAudioContext {
  state: AudioContextState = "suspended";
  readonly destination = {};
  readonly source = new FakeSource();
  readonly resume = vi.fn(async () => {
    this.state = "running";
  });
  readonly suspend = vi.fn(async () => {
    this.state = "suspended";
  });
  readonly decodeAudioData = vi.fn(async () => ({ duration: 1 }) as AudioBuffer);

  createBufferSource(): AudioBufferSourceNode {
    return this.source as unknown as AudioBufferSourceNode;
  }
}

class FakeFallback implements NarratorBackend {
  readonly available = true;
  readonly spoken: NarratorUtterance[] = [];

  voices() {
    return [
      {
        id: "browser",
        name: "Browser",
        language: "en-US",
        default: true,
      },
    ] as const;
  }

  speak(utterance: NarratorUtterance): void {
    this.spoken.push(utterance);
  }

  pause(): void {}
  resume(): void {}
  cancel(): void {}
}

const utterance = {
  cueId: "dispatch",
  text: "Dispatch work to the runtime.",
  rate: 1.25,
  voiceId: "af_bella",
} as const;

describe("Kokoro narrator backend", () => {
  test("prewarms WebGPU fp32 and exposes model progress and voices", async () => {
    const worker = new FakeWorker();
    const backend = createKokoroNarratorBackend({
      workerFactory: () => worker,
      audioContextFactory: () => new FakeAudioContext() as unknown as AudioContext,
      webGpuAvailable: () => true,
    });

    const prewarm = backend.prewarm();
    expect(worker.sent).toEqual([
      {
        type: "initialize",
        modelId: "onnx-community/Kokoro-82M-v1.0-ONNX",
        device: "webgpu",
        dtype: "fp32",
      },
    ]);

    worker.emit({ type: "progress", progress: 42, file: "model.onnx" });
    expect(backend.snapshot()).toMatchObject({
      status: "loading",
      engine: "webgpu",
      progress: 0.42,
    });

    worker.emit({
      type: "ready",
      voices: [
        {
          id: "af_heart",
          name: "Heart",
          language: "en-us",
          default: true,
        },
      ],
    });
    await prewarm;

    expect(backend.snapshot()).toMatchObject({
      status: "ready",
      engine: "webgpu",
      progress: 1,
    });
    expect(backend.voices()[0]?.id).toBe("af_heart");
  });

  test("falls back from WebGPU fp32 to q8 WASM before Web Speech", async () => {
    const webGpuWorker = new FakeWorker();
    const wasmWorker = new FakeWorker();
    const fallback = new FakeFallback();
    const workers = [webGpuWorker, wasmWorker];
    const backend = createKokoroNarratorBackend({
      fallback,
      workerFactory: () => workers.shift()!,
      audioContextFactory: () => new FakeAudioContext() as unknown as AudioContext,
      webGpuAvailable: () => true,
    });

    const prewarm = backend.prewarm();
    webGpuWorker.emit({ type: "error", message: "WebGPU adapter lost" });

    expect(webGpuWorker.terminated).toBe(true);
    expect(wasmWorker.sent).toEqual([
      {
        type: "initialize",
        modelId: "onnx-community/Kokoro-82M-v1.0-ONNX",
        device: "wasm",
        dtype: "q8",
      },
    ]);

    wasmWorker.emit({ type: "ready", voices: [] });
    await prewarm;
    expect(backend.snapshot()).toMatchObject({
      status: "ready",
      engine: "wasm",
      error: null,
    });
    expect(fallback.spoken).toEqual([]);
  });

  test("uses the injected Web Speech backend only after local loading fails", async () => {
    const worker = new FakeWorker();
    const fallback = new FakeFallback();
    const backend = createKokoroNarratorBackend({
      fallback,
      workerFactory: () => worker,
      audioContextFactory: () => new FakeAudioContext() as unknown as AudioContext,
      webGpuAvailable: () => false,
    });

    backend.speak(utterance);
    worker.emit({ type: "error", message: "WASM unavailable" });
    await vi.waitFor(() => expect(fallback.spoken).toEqual([utterance]));

    expect(backend.snapshot()).toMatchObject({
      status: "fallback",
      engine: "web-speech",
      error: "WASM unavailable",
    });
  });

  test("one activation gesture unlocks later generated cues automatically", async () => {
    const worker = new FakeWorker();
    const audioContext = new FakeAudioContext();
    const backend = createKokoroNarratorBackend({
      workerFactory: () => worker,
      audioContextFactory: () => audioContext as unknown as AudioContext,
      webGpuAvailable: () => false,
    });

    await backend.activate();
    backend.speak(utterance);
    worker.emit({ type: "ready", voices: [] });
    await vi.waitFor(() =>
      expect(worker.sent).toContainEqual({
        type: "synthesize",
        requestId: "dispatch:1",
        cueId: "dispatch",
        text: utterance.text,
        voiceId: "af_bella",
        rate: 1.25,
      }),
    );
    worker.emit({
      type: "audio",
      requestId: "dispatch:1",
      cueId: "dispatch",
      wav: new ArrayBuffer(8),
    });
    await vi.waitFor(() => expect(audioContext.source.start).toHaveBeenCalled());

    expect(backend.snapshot()).toMatchObject({
      status: "playing",
      activeCueId: "dispatch",
      needsUserActivation: false,
    });
  });

  test("preloads generated audio but reports blocked audible playback", async () => {
    const worker = new FakeWorker();
    const audioContext = new FakeAudioContext();
    audioContext.resume.mockRejectedValueOnce(
      new DOMException("Not allowed", "NotAllowedError"),
    );
    const backend = createKokoroNarratorBackend({
      workerFactory: () => worker,
      audioContextFactory: () => audioContext as unknown as AudioContext,
      webGpuAvailable: () => false,
    });

    backend.speak(utterance);
    worker.emit({ type: "ready", voices: [] });
    await vi.waitFor(() =>
      expect(worker.sent).toContainEqual(
        expect.objectContaining({ type: "synthesize", cueId: "dispatch" }),
      ),
    );
    worker.emit({
      type: "audio",
      requestId: "dispatch:1",
      cueId: "dispatch",
      wav: new ArrayBuffer(8),
    });
    await vi.waitFor(() =>
      expect(backend.snapshot().status).toBe("needs-user-activation"),
    );

    expect(audioContext.decodeAudioData).toHaveBeenCalled();
    expect(audioContext.source.start).not.toHaveBeenCalled();
    expect(backend.snapshot().needsUserActivation).toBe(true);
  });

  test("cancel invalidates inference and stops generated audio", async () => {
    const worker = new FakeWorker();
    const audioContext = new FakeAudioContext();
    const backend = createKokoroNarratorBackend({
      workerFactory: () => worker,
      audioContextFactory: () => audioContext as unknown as AudioContext,
      webGpuAvailable: () => false,
    });

    await backend.activate();
    backend.speak(utterance);
    worker.emit({ type: "ready", voices: [] });
    await vi.waitFor(() =>
      expect(worker.sent).toContainEqual(
        expect.objectContaining({ type: "synthesize", cueId: "dispatch" }),
      ),
    );
    backend.cancel();
    worker.emit({
      type: "audio",
      requestId: "dispatch:1",
      cueId: "dispatch",
      wav: new ArrayBuffer(8),
    });

    expect(worker.sent).toContainEqual({
      type: "cancel",
      requestId: "dispatch:1",
      cueId: "dispatch",
    });
    expect(audioContext.source.start).not.toHaveBeenCalled();
    expect(backend.snapshot().activeCueId).toBeNull();
  });
});
