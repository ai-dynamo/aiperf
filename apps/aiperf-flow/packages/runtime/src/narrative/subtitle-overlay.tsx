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
  /** Spoken progress through this cue, 0..1, for word-level highlighting. */
  progress?: number;
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

/**
 * Renders a cue as karaoke word spans: words already spoken read bright, the
 * word at the current spoken position is highlighted, and upcoming words stay
 * dim. A trailing space after each word preserves the plain-text reading.
 */
function karaokeWords(text: string, progress?: number): ReactNode {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const clamped = Math.min(1, Math.max(0, progress ?? 0));
  // No progress signal (progress absent) leaves every word in its resting
  // state; an active index of -1 selects "spoken" for none.
  const activeIndex =
    progress === undefined
      ? Number.POSITIVE_INFINITY
      : Math.min(words.length - 1, Math.floor(clamped * words.length));
  return words.map((word, index) => {
    const state =
      progress === undefined
        ? "spoken"
        : index < activeIndex
          ? "spoken"
          : index === activeIndex
            ? "active"
            : "upcoming";
    return (
      <span
        className="aiperf-flow__subtitle-word"
        data-state={state}
        key={`${index}-${word}`}
      >
        {word}
        {index < words.length - 1 ? " " : ""}
      </span>
    );
  });
}

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
        {state.enabled ? (
          <p
            aria-hidden="true"
            className="aiperf-flow__subtitle-cue"
            data-idle={cue === null ? "true" : "false"}
          >
            <span className="aiperf-flow__subtitle-label">
              {cue?.speaker ?? "Subtitles"}
            </span>
            <span className="aiperf-flow__subtitle-words">
              {cue === null ? null : karaokeWords(cue.text, cue.progress)}
            </span>
          </p>
        ) : null}
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
