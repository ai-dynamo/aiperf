// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SceneIr } from "@aiperf/flow-schema";

import type { NarrativeCue } from "./timeline.js";

type UnknownRecord = Readonly<Record<string, unknown>>;

function record(value: unknown): UnknownRecord {
  return typeof value === "object" && value !== null
    ? (value as UnknownRecord)
    : {};
}

function finiteNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** Returns the authored scene duration from timeline and narrative bounds. */
export function sceneDurationMs(scene: SceneIr): number {
  const timelineEnd = scene.timeline.reduce(
    (maximum, cue) =>
      Math.max(
        maximum,
        finiteNumber(record(cue).at) + finiteNumber(record(cue).duration),
      ),
    0,
  );
  const narrativeEnd =
    scene.narrativeTrack?.cues.reduce(
      (maximum, cue) => Math.max(maximum, cue.endMs),
      0,
    ) ?? 0;
  return Math.max(1, Math.ceil(Math.max(timelineEnd, narrativeEnd)));
}

/**
 * Resolves narrative cues for one scene, preferring `narrativeTrack` and
 * synthesizing one legacy cue from `narration` when no track is authored.
 */
export function sceneNarrativeCues(scene: SceneIr): readonly NarrativeCue[] {
  const track = scene.narrativeTrack;
  if (track !== undefined && track.cues.length > 0) {
    return Object.freeze(
      track.cues.map((cue) =>
        Object.freeze({
          id: cue.id,
          atMs: cue.startMs,
          durationMs: Math.max(1, cue.endMs - cue.startMs),
          spokenText: cue.spokenText,
          subtitleText: cue.subtitleText,
        }),
      ),
    );
  }

  const narration = typeof scene.narration === "string" ? scene.narration.trim() : "";
  if (narration === "") {
    return [];
  }

  return Object.freeze([
    Object.freeze({
      id: `${scene.id}:narration`,
      atMs: 0,
      durationMs: sceneDurationMs(scene),
      spokenText: narration,
      subtitleText: narration,
    }),
  ]);
}
