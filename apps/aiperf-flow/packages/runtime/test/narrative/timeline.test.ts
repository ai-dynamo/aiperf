// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  evaluateNarrativeTimeline,
  type NarrativeCue,
} from "../../src/narrative/timeline.js";

const cues = [
  {
    id: "dispatch",
    atMs: 100,
    durationMs: 300,
    spokenText: "The runtime dispatches work.",
    subtitleText: "Runtime dispatches work.",
  },
  {
    id: "observe",
    atMs: 500,
    durationMs: 200,
    spokenText: "The observer records every request.",
  },
] satisfies readonly NarrativeCue[];

describe("narrative timeline", () => {
  test("derives cue, subtitle, progress, transcript position, and boundary", () => {
    expect(evaluateNarrativeTimeline(cues, 250)).toEqual({
      atMs: 250,
      activeCue: cues[0],
      subtitleText: "Runtime dispatches work.",
      progress: 0.5,
      transcriptPosition: { cueIndex: 0, cueCount: 2 },
      complete: false,
      nextBoundaryMs: 400,
    });
  });

  test("uses half-open cue intervals and spoken text as subtitle fallback", () => {
    expect(evaluateNarrativeTimeline(cues, 400)).toEqual({
      atMs: 400,
      activeCue: null,
      subtitleText: null,
      progress: 0,
      transcriptPosition: { cueIndex: 1, cueCount: 2 },
      complete: false,
      nextBoundaryMs: 500,
    });

    expect(evaluateNarrativeTimeline(cues, 500)).toMatchObject({
      activeCue: cues[1],
      subtitleText: "The observer records every request.",
      progress: 0,
      transcriptPosition: { cueIndex: 1, cueCount: 2 },
      nextBoundaryMs: 700,
    });
  });

  test("marks the final boundary complete", () => {
    expect(evaluateNarrativeTimeline(cues, 700)).toEqual({
      atMs: 700,
      activeCue: null,
      subtitleText: null,
      progress: 0,
      transcriptPosition: { cueIndex: 2, cueCount: 2 },
      complete: true,
      nextBoundaryMs: null,
    });
  });

  test("direct seek equals evaluation reached through every playback beat", () => {
    let continuous = evaluateNarrativeTimeline(cues, 0);
    for (let atMs = 1; atMs <= 625; atMs += 1) {
      continuous = evaluateNarrativeTimeline(cues, atMs);
    }

    expect(continuous).toEqual(evaluateNarrativeTimeline(cues, 625));
  });

  test("reduced motion preserves every narrative cue and subtitle", () => {
    for (const atMs of [0, 100, 250, 400, 500, 625, 700]) {
      expect(
        evaluateNarrativeTimeline(cues, atMs, { reducedMotion: true }),
      ).toEqual(evaluateNarrativeTimeline(cues, atMs));
    }
  });

  test("re-evaluates the exact paused beat on resume", () => {
    const paused = evaluateNarrativeTimeline(cues, 275);

    expect(evaluateNarrativeTimeline(cues, paused.atMs)).toEqual(paused);
  });

  test("sorts authored cues deterministically without mutating input", () => {
    const reversed = [cues[1], cues[0]];

    expect(evaluateNarrativeTimeline(reversed, 250)).toEqual(
      evaluateNarrativeTimeline(cues, 250),
    );
    expect(reversed).toEqual([cues[1], cues[0]]);
  });

  test.each([-1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    "rejects non-canonical time %s",
    (atMs) => {
      expect(() => evaluateNarrativeTimeline(cues, atMs)).toThrow(RangeError);
    },
  );
});
