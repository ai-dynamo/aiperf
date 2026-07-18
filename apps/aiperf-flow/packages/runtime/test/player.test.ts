// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SceneIr } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  type Clock,
  TimelinePlayer,
  type TimelineSnapshot,
} from "../src/player.js";

class ManualClock implements Clock {
  #nowNs = 0n;
  #nextHandle = 1;
  readonly #callbacks = new Map<number, () => void>();

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

  advanceMs(milliseconds: number): void {
    this.#nowNs += BigInt(milliseconds) * 1_000_000n;
    const callbacks = [...this.#callbacks.values()];
    this.#callbacks.clear();
    callbacks.forEach((callback) => callback());
  }
}

const timeline = [
  {
    id: "reveal-cli",
    at: 100,
    duration: 400,
    action: "reveal",
    target: "cli",
    sourceMap: {
      source: "request-flow.flow",
      start: { offset: 0, line: 1, column: 1 },
      end: { offset: 1, line: 1, column: 2 },
    },
  },
  {
    id: "trace-spawn",
    at: 800,
    duration: 1200,
    action: "trace",
    target: "spawn",
    sourceMap: {
      source: "request-flow.flow",
      start: { offset: 0, line: 1, column: 1 },
      end: { offset: 1, line: 1, column: 2 },
    },
  },
] satisfies SceneIr["timeline"];

describe("TimelinePlayer", () => {
  test("advances deterministically from virtual clock time", () => {
    const clock = new ManualClock();
    const snapshots: TimelineSnapshot[] = [];
    const player = new TimelinePlayer(timeline, clock, (state) => {
      snapshots.push(state);
    });

    player.play();
    clock.advanceMs(250);

    expect(player.currentTimeMs()).toBe(250);
    expect(snapshots.at(-1)?.targets.cli?.progress).toBe(0.375);
  });

  test("pause freezes time and play resumes from that time", () => {
    const clock = new ManualClock();
    const player = new TimelinePlayer(timeline, clock);

    player.play();
    clock.advanceMs(300);
    player.pause();
    clock.advanceMs(500);
    expect(player.currentTimeMs()).toBe(300);

    player.play();
    clock.advanceMs(100);
    expect(player.currentTimeMs()).toBe(400);
  });

  test("seek and reset produce timeline-derived state", () => {
    const clock = new ManualClock();
    const player = new TimelinePlayer(timeline, clock);

    expect(player.seek(1_400).targets.spawn?.progress).toBe(0.5);
    expect(player.reset().timeMs).toBe(0);
    expect(player.currentTimeMs()).toBe(0);
  });

  test("returns the authored final state without running frames", () => {
    const player = new TimelinePlayer(timeline, new ManualClock());

    const final = player.finalState();
    expect(final.timeMs).toBe(2_000);
    expect(final.complete).toBe(true);
    expect(final.targets.cli?.progress).toBe(1);
    expect(final.targets.spawn?.progress).toBe(1);
  });

  test("extends playback through narrative content beyond the visual timeline", () => {
    const player = new TimelinePlayer(
      timeline,
      new ManualClock(),
      () => undefined,
      2_400,
    );

    expect(player.seek(Number.POSITIVE_INFINITY).timeMs).toBe(2_400);
    expect(player.finalState().complete).toBe(true);
  });

  test("clamps invalid seek values to finite timeline bounds", () => {
    const player = new TimelinePlayer(timeline, new ManualClock());

    expect(player.seek(-1).timeMs).toBe(0);
    expect(player.seek(Number.POSITIVE_INFINITY).timeMs).toBe(2_000);
  });
});
