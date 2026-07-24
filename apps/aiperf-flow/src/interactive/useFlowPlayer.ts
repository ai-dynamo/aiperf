/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Play/step/scrub hook over a typed `FlowStep[]`, built on the app's `useStepSimulator`. Generic
//! over `FlowStep` with no domain assumptions: it just walks a caption/highlight sequence, clamped
//! and auto-stopping at the end. Pairs with `RequestParticle` to render "a request moving through
//! a graph".

import { useStepSimulator } from "../state/useStepSimulator.js";
import type { FlowStep } from "./types.js";

/** Controls + derived state returned by {@link useFlowPlayer}. */
export interface FlowPlayer {
  /** The full step sequence, echoed back for convenience. */
  steps: readonly FlowStep[];
  /** The active step, or `undefined` when `steps` is empty. */
  current: FlowStep | undefined;
  /** Active step index, clamped to `[0, total − 1]`. */
  index: number;
  /** Total number of steps. */
  total: number;
  /** Whether the head is at the first step (or empty). */
  isFirst: boolean;
  /** Whether the head is at the last step (or empty). */
  isLast: boolean;
  /** Whether autoplay is currently advancing the head. */
  isPlaying: boolean;
  /** Id of the node the active step highlights (`current?.nodeId`). */
  activeNodeId: string | undefined;
  /** Caption of the active step (`current?.caption`). */
  caption: string | undefined;
  /** Start autoplay (no-op if already playing or at the last step). */
  play: () => void;
  /** Stop autoplay (no-op if already paused). */
  pause: () => void;
  /** Toggle autoplay. */
  togglePlay: () => void;
  /** Advance one step; no-op past the last. */
  next: () => void;
  /** Step back one; no-op before the first. */
  back: () => void;
  /** Return to the first step and stop playback. */
  reset: () => void;
  /** Scrub directly to a step index (clamped). */
  scrubTo: (index: number) => void;
}

/**
 * Drives playback over `steps`. `scrubTo` is implemented with the sanctioned bounded-loop pattern
 * (never `while (!isLast) next()`): `useStepSimulator`'s `next`/`back` use functional state
 * updates, so calling them a fixed number of times composes into a single clamped jump.
 */
export function useFlowPlayer(
  steps: readonly FlowStep[],
  opts?: { autoPlayMs?: number },
): FlowPlayer {
  const sim = useStepSimulator(steps, opts);

  const scrubTo = (target: number): void => {
    const clamped = Math.max(0, Math.min(target, Math.max(sim.total - 1, 0)));
    const delta = clamped - sim.index;
    const stepOnce = delta > 0 ? sim.next : sim.back;
    for (let i = 0; i < Math.abs(delta); i++) {
      stepOnce();
    }
  };

  return {
    steps,
    current: sim.current,
    index: sim.index,
    total: sim.total,
    isFirst: sim.isFirst,
    isLast: sim.isLast,
    isPlaying: sim.isPlaying,
    activeNodeId: sim.current?.nodeId,
    caption: sim.current?.caption,
    play: () => {
      if (!sim.isPlaying && !sim.isLast) {
        sim.togglePlay();
      }
    },
    pause: () => {
      if (sim.isPlaying) {
        sim.togglePlay();
      }
    },
    togglePlay: sim.togglePlay,
    next: sim.next,
    back: sim.back,
    reset: sim.reset,
    scrubTo,
  };
}
