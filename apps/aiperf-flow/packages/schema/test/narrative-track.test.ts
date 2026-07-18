/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, test } from "vitest";

import {
  parseNarrativeTrackIr,
  parseSceneIr,
  type NarrativeTrackIr,
} from "../src/ir.js";

const sourceMap = {
  source: "narrative.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 10, line: 1, column: 11 },
} as const;

const narrativeTrack = {
  language: "en-US",
  voice: "nvidia-narrator",
  cues: [
    {
      id: "introduce-runtime",
      startMs: 0,
      endMs: 1_500,
      spokenText: "AIPerf begins at the native command-line interface.",
      subtitleText: "AIPerf begins at the native CLI.",
      audioAsset: "audio/introduce-runtime.opus",
    },
    {
      id: "dispatch-request",
      startMs: 1_500,
      endMs: 3_000,
      spokenText: "The runtime dispatches the request.",
      subtitleText: "The runtime dispatches the request.",
    },
  ],
} satisfies NarrativeTrackIr;

const legacyScene = {
  id: "request-lifecycle",
  title: "Request lifecycle",
  summary: "A request moving through AIPerf.",
  roots: [],
  camera: [],
  timeline: [],
  narration: "AIPerf begins at the native command-line interface.",
  interactions: [],
  responsive: [],
  accessibility: {
    label: "Request lifecycle",
    readingOrder: [],
  },
  fallback: "A request lifecycle description.",
  sourceMap,
};

describe("narrative track IR", () => {
  test("parses timed narration, subtitles, audio, and locale metadata", () => {
    expect(parseNarrativeTrackIr(narrativeTrack)).toEqual(narrativeTrack);
    expect(
      parseSceneIr({ ...legacyScene, narrativeTrack }).narrativeTrack,
    ).toEqual(narrativeTrack);
  });

  test("preserves legacy scenes with narration only", () => {
    const parsed = parseSceneIr(legacyScene);

    expect(parsed.narration).toBe(legacyScene.narration);
    expect(parsed.narrativeTrack).toBeUndefined();
  });

  test("rejects unknown track and cue fields", () => {
    expect(() =>
      parseNarrativeTrackIr({ ...narrativeTrack, playbackRate: 1 }),
    ).toThrow();
    expect(() =>
      parseNarrativeTrackIr({
        ...narrativeTrack,
        cues: [{ ...narrativeTrack.cues[0], speaker: "host" }],
      }),
    ).toThrow();
  });

  test.each([
    ["fractional start", { startMs: 0.5 }],
    ["negative start", { startMs: -1 }],
    ["non-finite end", { endMs: Number.POSITIVE_INFINITY }],
    ["unsafe end", { endMs: Number.MAX_SAFE_INTEGER + 1 }],
    ["empty range", { startMs: 10, endMs: 10 }],
    ["reversed range", { startMs: 11, endMs: 10 }],
  ])("rejects %s", (_name, range) => {
    expect(() =>
      parseNarrativeTrackIr({
        ...narrativeTrack,
        cues: [{ ...narrativeTrack.cues[0], ...range }],
      }),
    ).toThrow();
  });

  test("rejects overlapping cues while allowing adjacent cues", () => {
    expect(() =>
      parseNarrativeTrackIr({
        ...narrativeTrack,
        cues: [
          narrativeTrack.cues[0],
          { ...narrativeTrack.cues[1], startMs: 1_499 },
        ],
      }),
    ).toThrow(/overlap/i);
    expect(parseNarrativeTrackIr(narrativeTrack)).toEqual(narrativeTrack);
  });

  test("rejects duplicate stable cue ids", () => {
    expect(() =>
      parseNarrativeTrackIr({
        ...narrativeTrack,
        cues: [
          narrativeTrack.cues[0],
          {
            ...narrativeTrack.cues[1],
            id: narrativeTrack.cues[0].id,
          },
        ],
      }),
    ).toThrow(/unique/i);
  });
});
