// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure narrative cue evaluation at integer virtual time.

/** One authored spoken cue and its synchronized subtitle. */
export type NarrativeCue = Readonly<{
  id: string;
  atMs: number;
  durationMs: number;
  spokenText: string;
  subtitleText?: string;
}>;

/** The current location in the ordered narrative transcript. */
export type TranscriptPosition = Readonly<{
  cueIndex: number;
  cueCount: number;
}>;

/** Narrative state derived entirely from one virtual-time beat. */
export type NarrativeTimelineSnapshot = Readonly<{
  atMs: number;
  activeCue: NarrativeCue | null;
  subtitleText: string | null;
  progress: number;
  transcriptPosition: TranscriptPosition;
  complete: boolean;
  nextBoundaryMs: number | null;
}>;

/** Narrative preferences that cannot alter spoken or subtitle content. */
export type NarrativeTimelineOptions = Readonly<{
  reducedMotion?: boolean;
}>;

function orderedCues(cues: readonly NarrativeCue[]): readonly NarrativeCue[] {
  return [...cues].sort(
    (left, right) =>
      left.atMs - right.atMs ||
      left.id.localeCompare(right.id, "en", { sensitivity: "variant" }),
  );
}

/**
 * Evaluates synchronized narration and subtitles without wall-clock state.
 *
 * Reduced motion intentionally produces the same narrative state because
 * skipping or shortening spoken cues would drop authored content.
 */
export function evaluateNarrativeTimeline(
  cues: readonly NarrativeCue[],
  atMs: number,
  _options: NarrativeTimelineOptions = {},
): NarrativeTimelineSnapshot {
  if (!Number.isSafeInteger(atMs) || atMs < 0) {
    throw new RangeError(
      "Narrative timeline time must be a non-negative safe integer.",
    );
  }

  const ordered = orderedCues(cues);
  const activeIndex = ordered.findIndex(
    (cue) => atMs >= cue.atMs && atMs < cue.atMs + cue.durationMs,
  );
  const activeCue = activeIndex < 0 ? null : (ordered[activeIndex] ?? null);
  const cueIndex =
    activeIndex >= 0
      ? activeIndex
      : ordered.filter((cue) => atMs >= cue.atMs + cue.durationMs).length;
  const durationMs = ordered.reduce(
    (maximum, cue) => Math.max(maximum, cue.atMs + cue.durationMs),
    0,
  );
  const nextBoundaryMs =
    ordered
      .flatMap((cue) => [cue.atMs, cue.atMs + cue.durationMs])
      .filter((boundaryMs) => boundaryMs > atMs)
      .sort((left, right) => left - right)[0] ?? null;
  const progress =
    activeCue === null ? 0 : (atMs - activeCue.atMs) / activeCue.durationMs;

  return Object.freeze({
    atMs,
    activeCue,
    subtitleText:
      activeCue === null
        ? null
        : (activeCue.subtitleText ?? activeCue.spokenText),
    progress,
    transcriptPosition: Object.freeze({
      cueIndex,
      cueCount: ordered.length,
    }),
    complete: atMs >= durationMs,
    nextBoundaryMs,
  });
}
