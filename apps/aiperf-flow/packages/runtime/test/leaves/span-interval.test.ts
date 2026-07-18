// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  buildSpanIntervalIndex,
  projectCoverage,
  type SpanEdge,
  type SpanInterval,
} from "../../src/leaves/span-interval.js";

const tokenSpanMorphEdges: readonly SpanEdge[] = [
  {
    id: "e0",
    sourceSpanIds: ["g0", "g1", "g2"],
    targetSpanIds: ["t0"],
    kind: "map",
  },
  {
    id: "e1",
    sourceSpanIds: ["g3"],
    targetSpanIds: ["t1"],
    kind: "map",
  },
  {
    id: "e2",
    sourceSpanIds: ["g5"],
    targetSpanIds: ["t2", "t3"],
    kind: "map",
  },
  {
    id: "e3",
    sourceSpanIds: [],
    targetSpanIds: ["t4"],
    kind: "special-insert",
  },
];

function graphemeSpans(ids: readonly string[]): readonly SpanInterval[] {
  return ids.map((id, index) => ({
    id,
    start: index,
    end: index + 1,
  }));
}

describe("leaf.span-interval", () => {
  test("detects half-open interval overlaps", () => {
    const index = buildSpanIntervalIndex([
      { id: "a", start: 0, end: 5 },
      { id: "b", start: 3, end: 8 },
    ]);

    expect(index.overlaps).toEqual([{ left: "a", right: "b" }]);
    expect(index.uncovered).toEqual([]);
    expect(index.covered).toBe(true);
  });

  test("reports TokenSpanMorph source coverage gaps for café 🚀 graphemes", () => {
    const allGraphemes = graphemeSpans(["g0", "g1", "g2", "g3", "g4", "g5"]);
    const uncovered = projectCoverage(allGraphemes, tokenSpanMorphEdges, "source");

    expect(uncovered.covered).toBe(false);
    expect(uncovered.uncovered).toEqual(["g4"]);
    expect(uncovered.overlaps).toEqual([]);
  });

  test("accepts TokenSpanMorph source coverage when g4 is omitted", () => {
    const coveredGraphemes = graphemeSpans(["g0", "g1", "g2", "g3", "g5"]);
    const covered = projectCoverage(coveredGraphemes, tokenSpanMorphEdges, "source");

    expect(covered.covered).toBe(true);
    expect(covered.uncovered).toEqual([]);
    expect(covered.overlaps).toEqual([]);
  });
});
