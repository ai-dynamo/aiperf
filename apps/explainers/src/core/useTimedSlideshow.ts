import { useEffect, useRef, useState } from "react";
import { estimateNarrationMs, speakNarration, stopNarration } from "./narration";

export function useTimedSlideshow({
  index,
  playing,
  narrationEnabled,
  voiceURI,
  narrations,
  restartKey = 0,
  onAdvance,
}: {
  index: number;
  playing: boolean;
  narrationEnabled: boolean;
  voiceURI?: string;
  narrations: readonly string[];
  /** Bumped to force a fresh narration start on the same slide (e.g. Back). */
  restartKey?: number;
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
      onWord: setActiveWordIndex,
      onComplete: () => advanceRef.current(),
    });

    return () => {
      cancel();
      setActiveWordIndex(-1);
    };
  }, [index, playing, narrationEnabled, voiceURI, currentNarration, restartKey]);

  return { activeWordIndex };
}

export function formatSlideshowDuration(narrations: readonly string[]): string {
  const totalMs = narrations.reduce((sum, line) => sum + estimateNarrationMs(line), 0);
  const minutes = Math.floor(totalMs / 60_000);
  const seconds = Math.round((totalMs % 60_000) / 1000);
  return minutes > 0 ? `${minutes} min ${seconds} sec` : `${seconds} sec`;
}

export function formatSlideDuration(text: string): string {
  const seconds = Math.round(estimateNarrationMs(text) / 1000);
  return `${seconds}s`;
}
