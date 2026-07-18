/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from "react";
import { estimateNarrationMs, speakNarration, stopNarration } from "./narration";

/**
 * Timed voice + word highlight driver for ExplainerShell.
 * `restartKey` must bump with SceneRenderer so narration and timelines restart together.
 */
export function useTimedSlideshow({
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
   * Bumped with slide changes / revisits / play so narration and SceneRenderer stay
   * in sync: fresh narration on the same slide (e.g. Back), and a new start
   * when auto-advance bumps restartKey alongside index (voice continues).
   */
  restartKey?: number;
  /** Shared wall-clock speed with SceneRenderer (1 = realtime). */
  speed?: number;
  onAdvance: () => void;
}): { activeWordIndex: number } {
  const advanceRef = useRef(onAdvance);
  advanceRef.current = onAdvance;
  const [activeWordIndex, setActiveWordIndex] = useState(0);
  // Depend on the current slide's text, not the narrations array identity —
  // slideNarrations() allocates a fresh array every render, and word highlights
  // re-render on every speech boundary.
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

export function formatSlideshowDuration(
  narrations: readonly string[],
  speed = 1,
): string {
  const totalMs = narrations.reduce(
    (sum, line) => sum + estimateNarrationMs(line, speed),
    0,
  );
  const minutes = Math.floor(totalMs / 60_000);
  const seconds = Math.round((totalMs % 60_000) / 1000);
  return minutes > 0 ? `${minutes} min ${seconds} sec` : `${seconds} sec`;
}

export function formatSlideDuration(text: string, speed = 1): string {
  const seconds = Math.round(estimateNarrationMs(text, speed) / 1000);
  return `${seconds}s`;
}
