// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test, vi } from "vitest";

import {
  configureKokoroWasmPaths,
  createKokoroWorkerRuntime,
  type KokoroModel,
} from "../../src/narrative/kokoro-worker.js";

describe("Kokoro worker runtime", () => {
  test("pins ONNX Runtime wasmPaths to local Vite-bundled assets before load", () => {
    const env: { wasmPaths?: unknown } = {};
    configureKokoroWasmPaths(env, {
      wasm: "https://preview.test/assets/ort-wasm-simd-threaded.jsep.wasm",
      mjs: "https://preview.test/assets/ort-wasm-simd-threaded.jsep.mjs",
    });
    expect(env.wasmPaths).toEqual({
      wasm: "https://preview.test/assets/ort-wasm-simd-threaded.jsep.wasm",
      mjs: "https://preview.test/assets/ort-wasm-simd-threaded.jsep.mjs",
    });
  });

  test("loads the requested model profile and transfers generated WAV audio", async () => {
    const wav = new ArrayBuffer(16);
    const model: KokoroModel = {
      voices: {
        af_heart: { name: "Heart", language: "en-us" },
      },
      generate: vi.fn(async () => ({ toWav: () => wav })),
    };
    const load = vi.fn(async (_modelId, options) => {
      options.progress_callback({
        status: "progress",
        file: "model.onnx",
        progress: 25,
      });
      return model;
    });
    const posted: Array<{ message: unknown; transfer?: Transferable[] }> = [];
    const runtime = createKokoroWorkerRuntime({
      load,
      postMessage: (message, transfer) => posted.push({ message, transfer }),
    });

    await runtime.handle({
      type: "initialize",
      modelId: "onnx-community/Kokoro-82M-v1.0-ONNX",
      device: "webgpu",
      dtype: "fp32",
    });
    await runtime.handle({
      type: "synthesize",
      requestId: "intro:1",
      cueId: "intro",
      text: "Welcome.",
      voiceId: "af_heart",
      rate: 1.1,
    });

    expect(load).toHaveBeenCalledWith(
      "onnx-community/Kokoro-82M-v1.0-ONNX",
      expect.objectContaining({
        device: "webgpu",
        dtype: "fp32",
        progress_callback: expect.any(Function),
      }),
    );
    expect(model.generate).toHaveBeenCalledWith("Welcome.", {
      voice: "af_heart",
      speed: 1.1,
    });
    expect(posted).toContainEqual({
      message: {
        type: "audio",
        requestId: "intro:1",
        cueId: "intro",
        wav,
      },
      transfer: [wav],
    });
  });

  test("discards generated audio for a cancelled deterministic cue identity", async () => {
    let finish!: (value: { toWav: () => ArrayBuffer }) => void;
    const model: KokoroModel = {
      voices: {},
      generate: () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    };
    const posted: unknown[] = [];
    const runtime = createKokoroWorkerRuntime({
      load: async () => model,
      postMessage: (message) => posted.push(message),
    });

    await runtime.handle({
      type: "initialize",
      modelId: "model",
      device: "wasm",
      dtype: "q8",
    });
    const generation = runtime.handle({
      type: "synthesize",
      requestId: "same-cue:1",
      cueId: "same-cue",
      text: "Cancelled.",
      voiceId: null,
      rate: 1,
    });
    await runtime.handle({
      type: "cancel",
      requestId: "same-cue:1",
      cueId: "same-cue",
    });
    finish({ toWav: () => new ArrayBuffer(8) });
    await generation;

    expect(posted).not.toContainEqual(
      expect.objectContaining({ type: "audio", cueId: "same-cue" }),
    );
  });
});
