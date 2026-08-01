/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { beatFraction, type BeatAnchor } from "./useBeatClock.js";

/** Ten words each, so a word index maps cleanly onto a fraction. */
const TEN = "one two three four five six seven eight nine ten";
const BEATS: BeatAnchor[] = [
  { endAt: 0.25, narration: TEN },
  { endAt: 0.75, narration: TEN },
  { endAt: 1, narration: TEN },
];

describe("beatFraction", () => {
  it("starts a beat at the previous beat's end", () => {
    expect(beatFraction(BEATS, 1, 0, true)).toBeCloseTo(0.25, 6);
  });

  it("reaches the beat's own end on its final word", () => {
    expect(beatFraction(BEATS, 1, 9, true)).toBeCloseTo(0.75, 6);
  });

  it("interpolates linearly across the beat", () => {
    expect(beatFraction(BEATS, 1, 4, true)).toBeCloseTo(0.25 + 0.5 * (4 / 9), 6);
  });

  it("holds at the beat start while the word index is still the previous narration's", () => {
    // The bug this exists to prevent: on the render where the beat advances, activeWordIndex
    // still belongs to the old beat. Read against the new range it lands most of the way through,
    // and the caller's monotonic clamp then latches it — a lurch, then a long freeze.
    expect(beatFraction(BEATS, 1, 9, false)).toBeCloseTo(0.25, 6);
    expect(beatFraction(BEATS, 2, 9, false)).toBeCloseTo(0.75, 6);
  });

  it("treats the idle sentinel as the beat start", () => {
    expect(beatFraction(BEATS, 1, -1, true)).toBeCloseTo(0.25, 6);
  });

  it("clamps a word index past the end of its narration", () => {
    expect(beatFraction(BEATS, 0, 999, true)).toBeCloseTo(0.25, 6);
  });

  it("returns zero for a beat that does not exist", () => {
    expect(beatFraction(BEATS, 9, 3, true)).toBe(0);
  });

  it("never runs backwards across a full scripted pass", () => {
    let prev = -1;
    for (let i = 0; i < BEATS.length; i++) {
      for (let w = 0; w < 10; w++) {
        const f = beatFraction(BEATS, i, w, true);
        expect(f).toBeGreaterThanOrEqual(prev - 1e-9);
        prev = f;
      }
    }
    expect(prev).toBeCloseTo(1, 6);
  });
});
