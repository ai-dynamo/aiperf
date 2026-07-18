// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure visibility policy for immersive playback HUD chrome.

/** Visibility level for decorative HUD chrome. */
export type HudVisibility = "present" | "quiet" | "hidden";

/** Inputs that determine HUD visibility without consulting global state. */
export type HudVisibilityInput = Readonly<{
  playing: boolean;
  exploring: boolean;
  commandOpen: boolean;
  focusedWithinHud: boolean;
  inactive: boolean;
}>;

/**
 * Resolves HUD visibility from explicit playback, interaction, and activity state.
 */
export function hudVisibilityFor(
  input: HudVisibilityInput,
): HudVisibility {
  if (
    !input.playing ||
    input.exploring ||
    input.commandOpen ||
    input.focusedWithinHud
  ) {
    return "present";
  }

  return input.inactive ? "hidden" : "quiet";
}
