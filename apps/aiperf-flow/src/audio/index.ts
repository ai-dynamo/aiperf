/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Browser speech-synthesis narration: autoplay decks that advance when the
//! spoken narration for a step finishes. See `./README.md`.

export {
  DEFAULT_PLAYBACK_SPEED,
  NARRATION_RATE,
  PLAYBACK_SPEEDS,
  estimateNarrationMs,
  isPlaybackSpeed,
  narrationSupported,
  speakNarration,
  speechRateForSpeed,
  splitWords,
  stopNarration,
  unlockSpeech,
  type PlaybackSpeed,
} from "./narration.js";
export {
  formatDeckDuration,
  formatStepDuration,
  useNarratedPlayback,
} from "./useNarratedPlayback.js";
export { useNarratedDeck, type NarratedDeck } from "./useNarratedDeck.js";
export { useSpeechVoices } from "./useSpeechVoices.js";
