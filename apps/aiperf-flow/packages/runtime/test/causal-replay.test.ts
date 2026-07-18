// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { SceneIr } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  activeCausalBeat,
  adjacentCausalBeat,
  projectCausalBeats,
} from "../src/causal-replay.js";

const sourceMap = {
  source: "causal.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
} as const;

function scene(overrides: Partial<SceneIr> = {}): SceneIr {
  return {
    id: "scene",
    title: "Scene",
    summary: "Summary",
    roots: [],
    camera: [],
    timeline: [
      { id: "first", at: 0, duration: 100, target: "a", action: "Reveal", sourceMap },
      { id: "second", at: 100, duration: 100, target: "b", action: "Trace", sourceMap },
    ],
    narration: "Narration",
    interactions: [],
    responsive: [],
    accessibility: { label: "Scene", readingOrder: [] },
    fallback: "Fallback",
    sourceMap,
    ...overrides,
  };
}

describe("projectCausalBeats", () => {
  test("projects deterministic active state and bounded traversal", () => {
    const beats = projectCausalBeats(scene());

    expect(beats.map(({ id }) => id)).toEqual(["first", "second"]);
    expect(activeCausalBeat(beats, 100)?.id).toBe("second");
    expect(adjacentCausalBeat(beats, "first", "previous")).toBeNull();
    expect(adjacentCausalBeat(beats, "first", "next")?.id).toBe("second");
  });

  test("rejects duplicate authored timeline ids", () => {
    const duplicate = scene({
      timeline: [
        { id: "same", at: 0, duration: 10, target: "a", action: "Reveal", sourceMap },
        { id: "same", at: 20, duration: 10, target: "b", action: "Trace", sourceMap },
      ],
    });

    expect(() => projectCausalBeats(duplicate)).toThrow(/duplicate.*same/iu);
  });
});
