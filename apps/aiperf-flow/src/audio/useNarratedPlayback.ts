/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from "react";
import { estimateNarrationMs, speakNarration, stopNarration } from "./narration.js";

/**
 * Timed voice + word-highlight driver for one step of a narrated walkthrough.
 *
 * Speaks `narrations[index]` while `playing`, reports the spoken word position
 * for karaoke subtitles, and calls `onAdvance` when the utterance finishes.
 * Replaces the fixed-interval timer in {@link useStepSimulator} with
 * "advance when the narration ends".
 */
export function useNarratedPlayback({
  index,
  playing,
  narrationEnabled,
  voiceURI,
  narrations,
  restartKey = 0,
  speed = 1,
  onAdvance,
}: {
  index: number;
  playing: boolean;
  narrationEnabled: boolean;
  voiceURI?: string;
  narrations: readonly string[];
  /**
   * Bumped on step change, revisit, and play so narration restarts in step with
   * any diagram animation keyed off the same value. Without it, re-entering the
   * same step (e.g. Back) would not restart the voice, since `index` is unchanged.
   */
  restartKey?: number;
  /** Wall-clock speed multiplier shared with diagram playback (1 = realtime). */
  speed?: number;
  onAdvance: () => void;
}): { activeWordIndex: number } {
  // Held in a ref so a caller re-creating `onAdvance` each render does not
  // cancel and restart the in-flight utterance.
  const advanceRef = useRef(onAdvance);
  advanceRef.current = onAdvance;
  const [activeWordIndex, setActiveWordIndex] = useState(0);
  // Depend on the current step's text, not the array identity: callers commonly
  // build `narrations` inline, and word highlights re-render on every boundary.
  const currentNarration = narrations[index] ?? "";

  useEffect(() => {
    if (!playing) {
      stopNarration();
      setActiveWordIndex(-1);
      return;
    }

    setActiveWordIndex(0);
    const cancel = speakNarration(currentNarration, {
      useSpeech: narrationEnabled,
      voiceURI,
      speed,
      onWord: setActiveWordIndex,
      onComplete: () => advanceRef.current(),
    });

    return () => {
      cancel();
      setActiveWordIndex(-1);
    };
  }, [index, playing, narrationEnabled, voiceURI, currentNarration, restartKey, speed]);

  return { activeWordIndex };
}

/** Estimated spoken length of every step combined, e.g. `4 min 30 sec`. */
export function formatDeckDuration(narrations: readonly string[], speed = 1): string {
  const totalMs = narrations.reduce((sum, line) => sum + estimateNarrationMs(line, speed), 0);
  const minutes = Math.floor(totalMs / 60_000);
  const seconds = Math.round((totalMs % 60_000) / 1000);
  return minutes > 0 ? `${minutes} min ${seconds} sec` : `${seconds} sec`;
}

/** Estimated spoken length of a single step, e.g. `12s`. */
export function formatStepDuration(text: string, speed = 1): string {
  return `${Math.round(estimateNarrationMs(text, speed) / 1000)}s`;
}
