// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type ReactNode,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";

import "./subtitle-overlay.css";

/** One immutable subtitle cue on the narrative timeline. */
export type SubtitleCue = Readonly<{
  id: string;
  text: string;
  speaker?: string;
}>;

/** Controlled subtitle visibility and currently active timeline cue. */
export type SubtitleState = Readonly<{
  enabled: boolean;
  activeCue: SubtitleCue | null;
}>;

/** Presentation and control contract for the subtitle overlay. */
export type SubtitleOverlayProps = Readonly<{
  state: SubtitleState;
  onEnabledChange(enabled: boolean): void;
  contrast?: "standard" | "high";
  reducedMotion?: boolean;
}>;

function announcementFor(cue: SubtitleCue | null): string {
  if (cue === null) {
    return "";
  }
  const speaker = cue.speaker?.trim();
  return speaker === undefined || speaker === ""
    ? cue.text
    : `${speaker}: ${cue.text}`;
}

/**
 * HTML subtitle layer shared by every visual backend.
 *
 * Cue IDs identify immutable narration events. The live region updates only
 * when that identity changes, so unrelated scene renders cannot repeatedly
 * announce the active caption.
 */
export function SubtitleOverlay({
  state,
  onEnabledChange,
  contrast = "standard",
  reducedMotion = false,
}: SubtitleOverlayProps): ReactNode {
  const cueContainerId = useId();
  const initialCue = state.enabled ? state.activeCue : null;
  const announcedCueId = useRef<string | null>(initialCue?.id ?? null);
  const [announcement, setAnnouncement] = useState(() =>
    announcementFor(initialCue),
  );

  useEffect(() => {
    const cue = state.enabled ? state.activeCue : null;
    if (cue === null) {
      announcedCueId.current = null;
      setAnnouncement("");
      return;
    }
    if (announcedCueId.current === cue.id) {
      return;
    }
    announcedCueId.current = cue.id;
    setAnnouncement(announcementFor(cue));
  }, [state.activeCue, state.enabled]);

  const cue = state.enabled ? state.activeCue : null;

  return (
    <section
      aria-label="Subtitles"
      className="aiperf-flow__subtitle-overlay"
      data-contrast={contrast}
      data-cue-id={cue?.id}
      data-reduced-motion={reducedMotion ? "true" : "false"}
    >
      <button
        aria-controls={cueContainerId}
        aria-label={`Turn subtitles ${state.enabled ? "off" : "on"}`}
        aria-pressed={state.enabled}
        className="aiperf-flow__subtitle-control"
        onClick={() => onEnabledChange(!state.enabled)}
        type="button"
      >
        <span aria-hidden="true">CC</span>
      </button>

      <div className="aiperf-flow__subtitle-safe-area" id={cueContainerId}>
        {cue === null ? null : (
          <p
            aria-hidden="true"
            className="aiperf-flow__subtitle-cue"
            key={cue.id}
          >
            {cue.speaker === undefined ? null : (
              <span className="aiperf-flow__subtitle-speaker">
                {cue.speaker}
              </span>
            )}
            <span>{cue.text}</span>
          </p>
        )}
      </div>

      <div
        aria-atomic="true"
        aria-live="polite"
        className="aiperf-flow__subtitle-live"
        role="status"
      >
        {announcement}
      </div>
    </section>
  );
}
