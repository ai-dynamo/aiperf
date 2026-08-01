/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import { Button } from "../prose/Button.js";
import { VoicePicker } from "./VoicePicker.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../theme/tokens.js";

/**
 * Pre-roll overlay shown before a narrated deck starts.
 *
 * This exists for a hard browser constraint, not for decoration: speech
 * synthesis is blocked until a user gesture, so playback needs one click to
 * unlock it. The gate turns that requirement into the deck's opening choice —
 * with audio or silent.
 */
export function StartGate({
  title,
  onStartWithNarration,
  onStartSilent,
  speechAvailable,
  voices,
  selectedVoiceURI,
  onVoiceSelect,
}: {
  title: string;
  onStartWithNarration: () => void;
  onStartSilent: () => void;
  /** False when the browser has no speech synthesis; only silent start is offered. */
  speechAvailable: boolean;
  voices: readonly SpeechSynthesisVoice[];
  selectedVoiceURI: string;
  onVoiceSelect: (voiceURI: string) => void;
}): React.JSX.Element {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Start ${title}`}
      className="absolute inset-0 z-20 flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm"
    >
      <div
        className={clsx(
          "w-full max-w-lg rounded-lg border p-6 shadow-lg",
          surfaceClassName("elevated"),
          strokeClassName("primary"),
        )}
      >
        <div className="mb-1 text-[10px] font-semibold tracking-widest text-accent-primary">
          AIPERF FLOW
        </div>
        <h2 className={clsx("text-xl font-semibold", inkClassName("primary"))}>{title}</h2>
        <p className={clsx("mt-2 mb-5 text-sm leading-relaxed", inkClassName("secondary"))}>
          {speechAvailable
            ? "Browsers block spoken audio until you click once. After that, steps advance and narrate automatically."
            : "This browser has no speech synthesis available. Steps will still advance automatically, with narration shown as subtitles."}
        </p>
        {speechAvailable && (
          <div className="mb-5">
            <VoicePicker
              voices={voices}
              selectedVoiceURI={selectedVoiceURI}
              onVoiceSelect={onVoiceSelect}
              speechAvailable={speechAvailable}
            />
          </div>
        )}
        <div className="flex flex-wrap gap-2">
          <Button
            variant="primary"
            className="flex-1 basis-44"
            onClick={onStartWithNarration}
            disabled={!speechAvailable}
          >
            {speechAvailable ? "Play with audio" : "Audio unavailable"}
          </Button>
          <Button variant="secondary" className="flex-1 basis-44" onClick={onStartSilent}>
            Play without audio
          </Button>
        </div>
      </div>
    </div>
  );
}
