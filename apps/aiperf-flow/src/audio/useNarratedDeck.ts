/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState, type KeyboardEvent } from "react";
import { useLocation } from "react-router-dom";
import { usePersistentState } from "../state/usePersistentState.js";
import { useNarratedPlayback } from "./useNarratedPlayback.js";
import { useSpeechVoices } from "./useSpeechVoices.js";
import {
  DEFAULT_PLAYBACK_SPEED,
  isPlaybackSpeed,
  narrationSupported,
  stopNarration,
  unlockSpeech,
  type PlaybackSpeed,
} from "./narration.js";

/** Everything a narrated deck's chrome needs to render its controls. */
export interface NarratedDeck {
  /** Current step, clamped into range. */
  index: number;
  /** Whether narration is advancing steps right now. */
  playing: boolean;
  /** False until the viewer clears the start gate; audio cannot begin before this. */
  started: boolean;
  /** Whether the viewer chose spoken audio (false = silent auto-advance). */
  narrationEnabled: boolean;
  /** Selected voice, or `""` for the engine default. */
  voiceURI: string;
  speed: PlaybackSpeed;
  /** English voices offered by this browser; empty until the engine enumerates them. */
  voices: readonly SpeechSynthesisVoice[];
  /** Whether this browser can speak at all. When false, only silent mode works. */
  speechAvailable: boolean;
  /** Word position within the current step's narration, or -1 when idle. */
  activeWordIndex: number;
  /**
   * Bumped on every step change, revisit, and play. Pass to diagram/animation
   * components as a React `key` (or restart dependency) to keep visuals in step
   * with the voice.
   */
  restartKey: number;
  /** Clears the start gate and begins playback, with or without spoken audio. */
  begin: (withNarration: boolean) => void;
  /** Play/pause. Rewinds to the first step when invoked on the last one. */
  togglePlayback: () => void;
  /** Jumps to a step, clamped; keeps playing if already playing. */
  goTo: (next: number) => void;
  setNarrationEnabled: (enabled: boolean) => void;
  setVoiceURI: (voiceURI: string) => void;
  setSpeed: (speed: PlaybackSpeed) => void;
  /** Arrow-key nav, space to play/pause. Attach to the deck's focusable root. */
  onKeyDown: (event: KeyboardEvent) => void;
}

/**
 * Playback state machine for a narrated deck: start gate, play/pause, step
 * advance driven by narration completion, voice/speed settings, and keyboard nav.
 *
 * Narration preferences persist under `storagePrefix`; the current step and
 * whether playback is running are session-only, so a reload always reopens at
 * the start gate on step 1.
 */
export function useNarratedDeck({
  narrations,
  storagePrefix,
}: {
  /** One narration string per step. Empty strings still consume estimated time. */
  narrations: readonly string[];
  /** `localStorage` namespace for the persisted narration settings. */
  storagePrefix: string;
}): NarratedDeck {
  const total = narrations.length;
  const location = useLocation();
  // Step and playing are session-only so a refresh always opens on step 1 / the gate.
  const [stored, setStored] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [started, setStarted] = useState(false);
  const [restartKey, setRestartKey] = useState(0);
  const [narrationEnabled, setNarrationEnabled] = usePersistentState<boolean>(
    storagePrefix,
    "narration",
    true,
  );
  const [voiceURI, setVoiceURI] = usePersistentState<string>(storagePrefix, "voice", "");
  const [speedRaw, setSpeedRaw] = usePersistentState<number>(
    storagePrefix,
    "speed",
    DEFAULT_PLAYBACK_SPEED,
  );

  // A stored speed from an older build may no longer be an offered option.
  const speed: PlaybackSpeed = isPlaybackSpeed(speedRaw) ? speedRaw : DEFAULT_PLAYBACK_SPEED;
  const voices = useSpeechVoices();
  const speechAvailable = narrationSupported();
  const index = Number.isInteger(stored) && stored >= 0 && stored < total ? stored : 0;

  const bumpRestart = () => setRestartKey((key) => key + 1);

  // Navigating to another deck must not leave the previous one talking.
  //
  // The mount-side call covers arriving somewhere new. The teardown covers *leaving* for a page
  // that narrates nothing — the home listing, say — where this hook simply unmounts and no
  // arriving deck is there to silence the old one. Without it the voice reads on over a page the
  // viewer has already left.
  useEffect(() => {
    stopNarration();
    setPlaying(false);
    return () => stopNarration();
  }, [location.pathname]);

  const advance = () => {
    if (!playing) return;
    if (index >= total - 1) {
      stopNarration();
      setPlaying(false);
      return;
    }
    setStored(index + 1);
    bumpRestart();
  };

  const { activeWordIndex } = useNarratedPlayback({
    index,
    playing: started && playing,
    narrationEnabled,
    voiceURI,
    narrations,
    restartKey,
    speed,
    onAdvance: advance,
  });

  const goTo = (next: number) => {
    const nextIndex = Math.max(0, Math.min(total - 1, next));
    stopNarration();
    setStored(nextIndex);
    if (started) {
      setPlaying(true);
      bumpRestart();
    }
  };

  const begin = (withNarration: boolean) => {
    // Must happen inside this click handler: browsers only honor the unlock
    // during a user gesture.
    if (withNarration) unlockSpeech();
    setNarrationEnabled(withNarration);
    setStarted(true);
    setPlaying(true);
    bumpRestart();
  };

  const togglePlayback = () => {
    if (playing) {
      stopNarration();
      setPlaying(false);
      return;
    }
    unlockSpeech();
    if (index >= total - 1) setStored(0);
    setStarted(true);
    setPlaying(true);
    bumpRestart();
  };

  const onKeyDown = (event: KeyboardEvent) => {
    // Let controls and text fields keep their own space/arrow behavior.
    const el = event.target as HTMLElement | null;
    if (el?.matches?.("input, textarea, select, button, [role='button']")) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      goTo(index - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      goTo(index + 1);
    } else if (event.key === " ") {
      event.preventDefault();
      togglePlayback();
    }
  };

  return {
    index,
    playing,
    started,
    narrationEnabled,
    voiceURI,
    speed,
    voices,
    speechAvailable,
    activeWordIndex,
    restartKey,
    begin,
    togglePlayback,
    goTo,
    setNarrationEnabled,
    setVoiceURI,
    setSpeed: setSpeedRaw,
    onKeyDown,
  };
}
