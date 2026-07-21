/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from "react";

const DEFAULT_AUTO_PLAY_MS = 1000;

/**
 * Result returned by {@link useStepSimulator}.
 */
export interface StepSimulator<T> {
  /** The step value at `index`, or `undefined` when `steps` is empty. */
  current: T | undefined;
  /** Current position within `steps`, clamped to `[0, total - 1]`. */
  index: number;
  /** Total number of steps. */
  total: number;
  /** Whether `index` is at the first step (or `steps` is empty). */
  isFirst: boolean;
  /** Whether `index` is at the last step (or `steps` is empty). */
  isLast: boolean;
  /** Whether autoplay is currently advancing `index` on a timer. */
  isPlaying: boolean;
  /** Advances one step forward; a no-op past the last step. */
  next: () => void;
  /** Moves one step back; a no-op before the first step. */
  back: () => void;
  /** Returns to the first step and stops playback. */
  reset: () => void;
  /** Starts or stops autoplay. */
  togglePlay: () => void;
}

/**
 * Drives a "step through the diagram" walkthrough over a fixed sequence of
 * steps: Play/Pause/Step/Reset controls over a plain array, with no
 * assumptions about what a step's payload looks like (a segment id, a node
 * highlight set, a narration string, whatever the deck slide needs).
 *
 * This is the in-memory analogue of Cursor Canvas's `useCanvasState`
 * ergonomics, deliberately without its disk-backed persistence: a deck slide
 * only needs the state to live for the render's lifetime, not to survive a
 * reload. See `./DESIGN.md`.
 */
export function useStepSimulator<T>(
  steps: readonly T[],
  opts?: { autoPlayMs?: number },
): StepSimulator<T> {
  const autoPlayMs = opts?.autoPlayMs ?? DEFAULT_AUTO_PLAY_MS;
  const total = steps.length;
  const [index, setIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const clampedIndex = total === 0 ? 0 : Math.min(index, total - 1);
  const isLast = total === 0 || clampedIndex >= total - 1;
  const isFirst = clampedIndex <= 0;

  useEffect(() => {
    if (!isPlaying || isLast) {
      return;
    }

    const timer = setTimeout(() => {
      setIndex((current) => {
        const nextIndex = current + 1;
        if (nextIndex >= total - 1) {
          setIsPlaying(false);
        }
        return Math.min(nextIndex, Math.max(total - 1, 0));
      });
    }, autoPlayMs);

    return () => clearTimeout(timer);
    // `index` is an intentional dependency: each tick must reschedule a new
    // timer for the *next* step, not just fire once when `isPlaying` flips.
  }, [isPlaying, isLast, autoPlayMs, total, index]);

  const next = () => {
    setIndex((current) => Math.min(current + 1, Math.max(total - 1, 0)));
  };

  const back = () => {
    setIndex((current) => Math.max(current - 1, 0));
  };

  const reset = () => {
    setIndex(0);
    setIsPlaying(false);
  };

  const togglePlay = () => {
    setIsPlaying((playing) => !playing);
  };

  return {
    current: total === 0 ? undefined : steps[clampedIndex],
    index: clampedIndex,
    total,
    isFirst,
    isLast,
    isPlaying,
    next,
    back,
    reset,
    togglePlay,
  };
}
