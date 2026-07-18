// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ReactNode } from "react";

import "./audio-consent-modal.css";

/** User choice from the first-load audio consent dialog. */
export type AudioConsentChoice = "with-audio" | "without-audio";

export type AudioConsentModalProps = Readonly<{
  open: boolean;
  onChoose(choice: AudioConsentChoice): void;
  title?: string;
}>;

/**
 * Full-page-load gate that unlocks Web Audio from a real user gesture when the
 * viewer chooses audible playback. Choice lives in component state only, so a
 * reload asks again while in-SPA scene changes keep playing.
 */
export function AudioConsentModal({
  open,
  onChoose,
  title = "Audio preference",
}: AudioConsentModalProps): ReactNode {
  if (!open) {
    return null;
  }

  return (
    <div className="aiperf-flow__audio-consent-backdrop">
      <div
        aria-labelledby="aiperf-flow-audio-consent-title"
        aria-modal="true"
        className="aiperf-flow__audio-consent"
        role="dialog"
      >
        <h2 id="aiperf-flow-audio-consent-title">{title}</h2>
        <p>
          How would you like to watch? Choosing audio unlocks narration for
          this flow. You can mute later.
        </p>
        <div className="aiperf-flow__audio-consent-actions">
          <button
            aria-label="Play with audio"
            className="aiperf-flow__audio-consent-primary"
            onClick={() => onChoose("with-audio")}
            type="button"
          >
            Play with audio
          </button>
          <button
            aria-label="Play without audio"
            onClick={() => onChoose("without-audio")}
            type="button"
          >
            Play without audio
          </button>
        </div>
      </div>
    </div>
  );
}
