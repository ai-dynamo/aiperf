/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import { Pill } from "../prose/Pill.js";
import { inkClassName } from "../theme/tokens.js";

/**
 * Narration voice selection. Voice quality varies widely across operating
 * systems and browsers, so the viewer picks; `""` keeps the engine default.
 */
export function VoicePicker({
  voices,
  selectedVoiceURI,
  onVoiceSelect,
  speechAvailable,
}: {
  voices: readonly SpeechSynthesisVoice[];
  selectedVoiceURI: string;
  onVoiceSelect: (voiceURI: string) => void;
  speechAvailable: boolean;
}): React.JSX.Element | null {
  if (!speechAvailable) return null;

  return (
    <div>
      <div
        className={clsx(
          "mb-2 text-[10px] font-semibold tracking-widest",
          inkClassName("tertiary"),
        )}
      >
        VOICE
      </div>
      <div role="radiogroup" aria-label="Narration voice" className="flex flex-wrap gap-1.5">
        <Pill active={selectedVoiceURI === ""} onClick={() => onVoiceSelect("")}>
          Default
        </Pill>
        {voices.map((voice) => (
          <Pill
            key={voice.voiceURI}
            active={selectedVoiceURI === voice.voiceURI}
            onClick={() => onVoiceSelect(voice.voiceURI)}
            ariaLabel={`${voice.name} · ${voice.lang}`}
          >
            {voice.name}
          </Pill>
        ))}
      </div>
    </div>
  );
}
