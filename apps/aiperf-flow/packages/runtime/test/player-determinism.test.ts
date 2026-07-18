// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SceneIr } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  type Clock,
  normalizeSceneTimeMs,
  TimelinePlayer,
} from "../src/player.js";
import { createInitialSceneState, sceneReducer } from "../src/store.js";

class VirtualClock implements Clock {
  #nowNs: bigint;
  #nextHandle = 1;
  readonly #callbacks = new Map<number, () => void>();

  constructor(nowNs = 0n) {
    this.#nowNs = nowNs;
  }

  nowNs(): bigint {
    return this.#nowNs;
  }

  requestFrame(callback: () => void): number {
    const handle = this.#nextHandle;
    this.#nextHandle += 1;
    this.#callbacks.set(handle, callback);
    return handle;
  }

  cancelFrame(handle: number): void {
    this.#callbacks.delete(handle);
  }

  advanceNs(elapsedNs: bigint): void {
    this.#nowNs += elapsedNs;
    const callbacks = [...this.#callbacks.values()];
    this.#callbacks.clear();
    callbacks.forEach((callback) => callback());
  }
}

const timeline = [
  {
    id: "reveal-request",
    at: 100,
    duration: 400,
    action: "reveal",
    target: "request",
    sourceMap: {
      source: "determinism.flow",
      start: { offset: 0, line: 1, column: 1 },
      end: { offset: 1, line: 1, column: 2 },
    },
  },
  {
    id: "trace-response",
    at: 500,
    duration: 750,
    action: "trace",
    target: "response",
    sourceMap: {
      source: "determinism.flow",
      start: { offset: 2, line: 2, column: 1 },
      end: { offset: 3, line: 2, column: 2 },
    },
  },
] satisfies SceneIr["timeline"];

describe("integer virtual-clock determinism", () => {
  test("canonicalizes direct and elapsed scene time to integer milliseconds", () => {
    const clock = new VirtualClock();
    const player = new TimelinePlayer(timeline, clock);

    expect(normalizeSceneTimeMs(42.9)).toBe(42);
    expect(player.seek(42.9).timeMs).toBe(42);

    player.play();
    clock.advanceNs(900_001n);

    expect(player.snapshot().timeMs).toBe(42);
    expect(Number.isInteger(player.currentTimeMs())).toBe(true);
  });

  test("continuous play and direct seek produce equal snapshots", () => {
    const targetTimeMs = 937;
    const clock = new VirtualClock(8_765_432_100_000n);
    const continuous = new TimelinePlayer(timeline, clock);

    continuous.play();
    for (const stepMs of [16, 17, 33, 101, 250, 500]) {
      clock.advanceNs(BigInt(stepMs) * 1_000_000n);
    }
    clock.advanceNs(20_999_999n);

    const direct = new TimelinePlayer(timeline, new VirtualClock()).seek(
      targetTimeMs,
    );
    expect(continuous.snapshot()).toEqual(direct);
  });

  test("wall-clock origin does not enter scene state", () => {
    const zeroOrigin = new VirtualClock();
    const arbitraryOrigin = new VirtualClock(91_234_567_890_123n);
    const first = new TimelinePlayer(timeline, zeroOrigin);
    const second = new TimelinePlayer(timeline, arbitraryOrigin);

    first.play();
    second.play();
    zeroOrigin.advanceNs(625_750_000n);
    arbitraryOrigin.advanceNs(625_750_000n);

    expect(first.snapshot()).toEqual(second.snapshot());
  });

  test("store retains only canonical integer scene time", () => {
    const state = sceneReducer(createInitialSceneState("scene"), {
      type: "set-playback",
      timeMs: 125.75,
      status: "playing",
    });

    expect(state.playbackTimeMs).toBe(125);
  });
});
