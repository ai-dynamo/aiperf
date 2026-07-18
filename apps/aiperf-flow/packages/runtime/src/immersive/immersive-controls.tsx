// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type FocusEvent,
  type ReactNode,
  useState,
} from "react";

import type {
  FullscreenState,
  HudVisibility,
} from "../immersive-state.js";

/** Controlled edge-HUD actions for immersive Causal Field playback. */
export type ImmersiveControlsProps = Readonly<{
  playing: boolean;
  exploring: boolean;
  playbackDisabled?: boolean;
  hud: HudVisibility;
  fullscreen: FullscreenState;
  onPlayPause(): void;
  onExploreResume(): void;
  onOpenCommands(): void;
  onToggleTwin(): void;
  onToggleFullscreen(): void;
}>;

function fullscreenLabel(fullscreen: FullscreenState): string {
  return fullscreen === "windowed" ? "Enter fullscreen" : "Exit fullscreen";
}

/**
 * Compact immersive playback HUD.
 *
 * Quiet and hidden visibility only affect decorative chrome. Interactive
 * controls stay in the accessibility tree and remain keyboard-reachable; focus
 * within the HUD restores chrome so focused controls are never concealed.
 */
export function ImmersiveControls({
  playing,
  exploring,
  playbackDisabled = false,
  hud,
  fullscreen,
  onPlayPause,
  onExploreResume,
  onOpenCommands,
  onToggleTwin,
  onToggleFullscreen,
}: ImmersiveControlsProps): ReactNode {
  const [focusedWithin, setFocusedWithin] = useState(false);

  function onFocusIn(): void {
    setFocusedWithin(true);
  }

  function onFocusOut(event: FocusEvent<HTMLElement>): void {
    const next = event.relatedTarget;
    if (next instanceof Node && event.currentTarget.contains(next)) {
      return;
    }
    setFocusedWithin(false);
  }

  // Policy may request quiet/hidden chrome, but focused controls stay present.
  const chromeHud: HudVisibility =
    focusedWithin && hud !== "present" ? "present" : hud;

  return (
    <section
      aria-label="Immersive controls"
      className="aiperf-flow__immersive-controls aiperf-flow__chrome"
      data-focused-within={focusedWithin ? "true" : "false"}
      data-fullscreen={fullscreen}
      data-hud={chromeHud}
      onBlur={onFocusOut}
      onFocus={onFocusIn}
    >
      <button
        disabled={playbackDisabled}
        onClick={onPlayPause}
        type="button"
      >
        {playing ? "Pause" : "Play"}
      </button>
      <button onClick={onExploreResume} type="button">
        {exploring ? "Resume lesson" : "Explore"}
      </button>
      <button onClick={onToggleTwin} type="button">
        Semantic twin
      </button>
      <button onClick={onOpenCommands} type="button">
        Open commands
      </button>
      <button
        aria-pressed={fullscreen !== "windowed" ? true : undefined}
        onClick={onToggleFullscreen}
        type="button"
      >
        {fullscreenLabel(fullscreen)}
      </button>
    </section>
  );
}
