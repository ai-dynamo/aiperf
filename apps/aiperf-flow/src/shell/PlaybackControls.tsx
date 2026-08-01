/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import { Button } from "../prose/Button.js";
import { Pill } from "../prose/Pill.js";
import { PLAYBACK_SPEEDS, type PlaybackSpeed } from "../audio/narration.js";
import { inkClassName } from "../theme/tokens.js";

/**
 * Shared play/mute/speed control row for narrated decks, replacing the
 * per-deck Play/Pause button rows each deck previously built inline.
 */
export function PlaybackControls({
  playing,
  isLast,
  narrationEnabled,
  speechAvailable,
  speed,
  onTogglePlayback,
  onToggleNarration,
  onSpeedChange,
}: {
  playing: boolean;
  /** On the last step, play reads as "Replay" and rewinds to the start. */
  isLast: boolean;
  narrationEnabled: boolean;
  speechAvailable: boolean;
  speed: PlaybackSpeed;
  onTogglePlayback: () => void;
  onToggleNarration: () => void;
  onSpeedChange: (speed: PlaybackSpeed) => void;
}): React.JSX.Element {
  const playLabel = playing ? "Pause" : isLast ? "Replay" : "Play";

  return (
    <div className="flex items-center gap-2">
      <Button variant="primary" onClick={onTogglePlayback}>
        {playLabel}
      </Button>
      <Button
        variant="ghost"
        onClick={onToggleNarration}
        disabled={!speechAvailable}
        aria-pressed={narrationEnabled}
      >
        {!speechAvailable ? "No audio" : narrationEnabled ? "Mute" : "Unmute"}
      </Button>
      <div
        role="radiogroup"
        aria-label="Playback speed"
        className="flex items-center gap-1"
      >
        <span className={clsx("mr-1 text-[10px] tracking-widest", inkClassName("tertiary"))}>
          SPEED
        </span>
        {PLAYBACK_SPEEDS.map((option) => (
          <Pill
            key={option}
            active={option === speed}
            onClick={() => onSpeedChange(option)}
            ariaLabel={`${option}× speed`}
          >
            {option}×
          </Pill>
        ))}
      </div>
    </div>
  );
}
