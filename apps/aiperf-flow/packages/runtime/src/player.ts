// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SceneIr } from "@aiperf/flow-schema";

export interface Clock {
  nowNs(): bigint;
  requestFrame(callback: () => void): number;
  cancelFrame(handle: number): void;
}

export type TimelineTargetState = Readonly<{
  action: string;
  progress: number;
}>;

export type TimelineSnapshot = Readonly<{
  timeMs: number;
  complete: boolean;
  targets: Readonly<Record<string, TimelineTargetState>>;
}>;

type Timeline = SceneIr["timeline"];
type UnknownRecord = Readonly<Record<string, unknown>>;

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

function finiteNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** Converts an external time value to canonical non-negative scene milliseconds. */
export function normalizeSceneTimeMs(timeMs: number): number {
  return Number.isFinite(timeMs) ? Math.max(0, Math.trunc(timeMs)) : 0;
}

function duration(timeline: Timeline): number {
  return Math.ceil(
    timeline.reduce((maximum, value) => {
      const cue = record(value);
      return Math.max(
        maximum,
        finiteNumber(cue.at) + finiteNumber(cue.duration),
      );
    }, 0),
  );
}

export class PerformanceClock implements Clock {
  nowNs(): bigint {
    return BigInt(Math.floor(performance.now() * 1_000_000));
  }

  requestFrame(callback: () => void): number {
    return requestAnimationFrame(callback);
  }

  cancelFrame(handle: number): void {
    cancelAnimationFrame(handle);
  }
}

export class TimelinePlayer {
  readonly #timeline: Timeline;
  readonly #clock: Clock;
  readonly #onState: (state: TimelineSnapshot) => void;
  readonly #durationMs: number;
  #timeMs = 0;
  #playing = false;
  #startedAtNs = 0n;
  #frameHandle: number | null = null;

  constructor(
    timeline: Timeline,
    clock: Clock = new PerformanceClock(),
    onState: (state: TimelineSnapshot) => void = () => undefined,
    minimumDurationMs = 0,
  ) {
    this.#timeline = timeline;
    this.#clock = clock;
    this.#onState = onState;
    this.#durationMs = Math.max(
      duration(timeline),
      normalizeSceneTimeMs(minimumDurationMs),
    );
  }

  play(): TimelineSnapshot {
    if (this.#playing || this.#durationMs === 0) {
      return this.snapshot();
    }
    if (this.#timeMs >= this.#durationMs) {
      this.#timeMs = 0;
    }
    this.#playing = true;
    this.#startedAtNs =
      this.#clock.nowNs() - BigInt(this.#timeMs) * 1_000_000n;
    this.#scheduleFrame();
    return this.#emit();
  }

  pause(): TimelineSnapshot {
    if (this.#playing) {
      this.#timeMs = this.#liveTimeMs();
      this.#playing = false;
    }
    if (this.#frameHandle !== null) {
      this.#clock.cancelFrame(this.#frameHandle);
      this.#frameHandle = null;
    }
    return this.#emit();
  }

  seek(timeMs: number): TimelineSnapshot {
    const requested =
      timeMs === Number.POSITIVE_INFINITY
        ? this.#durationMs
        : normalizeSceneTimeMs(timeMs);
    this.#timeMs = Math.min(this.#durationMs, Math.max(0, requested));
    if (this.#playing) {
      this.#startedAtNs =
        this.#clock.nowNs() - BigInt(this.#timeMs) * 1_000_000n;
    }
    return this.#emit();
  }

  reset(): TimelineSnapshot {
    this.pause();
    this.#timeMs = 0;
    return this.#emit();
  }

  currentTimeMs(): number {
    return this.#playing ? this.#liveTimeMs() : this.#timeMs;
  }

  snapshot(): TimelineSnapshot {
    return this.#compute(this.currentTimeMs());
  }

  finalState(): TimelineSnapshot {
    return this.#compute(this.#durationMs);
  }

  #liveTimeMs(): number {
    const elapsedNs = this.#clock.nowNs() - this.#startedAtNs;
    return Math.min(
      this.#durationMs,
      normalizeSceneTimeMs(Number(elapsedNs / 1_000_000n)),
    );
  }

  #scheduleFrame(): void {
    this.#frameHandle = this.#clock.requestFrame(() => {
      this.#frameHandle = null;
      this.#timeMs = this.#liveTimeMs();
      this.#emit();
      if (this.#timeMs >= this.#durationMs) {
        this.#playing = false;
      } else if (this.#playing) {
        this.#scheduleFrame();
      }
    });
  }

  #emit(): TimelineSnapshot {
    const state = this.snapshot();
    this.#onState(state);
    return state;
  }

  #compute(timeMs: number): TimelineSnapshot {
    const canonicalTimeMs = normalizeSceneTimeMs(timeMs);
    const targets: Record<string, TimelineTargetState> = {};
    for (const value of this.#timeline) {
      const cue = record(value);
      const target = typeof cue.target === "string" ? cue.target : "";
      if (target === "") {
        continue;
      }
      const startMs = finiteNumber(cue.at);
      const cueDurationMs = finiteNumber(cue.duration);
      const progress =
        cueDurationMs === 0
          ? Number(canonicalTimeMs >= startMs)
          : Math.min(
              1,
              Math.max(0, (canonicalTimeMs - startMs) / cueDurationMs),
            );
      targets[target] = Object.freeze({
        action: typeof cue.action === "string" ? cue.action : "",
        progress,
      });
    }
    return Object.freeze({
      timeMs: canonicalTimeMs,
      complete: canonicalTimeMs >= this.#durationMs,
      targets: Object.freeze(targets),
    });
  }
}
