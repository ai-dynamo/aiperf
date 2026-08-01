/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { cumulative, flattenRagged, raggedNodeSize } from "./raggedLayout.js";

describe("flattenRagged", () => {
  it("packs values back to back and records each value's owner", () => {
    const { values, recordIndices, offsets } = flattenRagged([[5, 7], [3], [9, 1, 4]]);

    expect(values).toEqual([5, 7, 3, 9, 1, 4]);
    expect(recordIndices).toEqual([0, 0, 1, 2, 2, 2]);
    expect(offsets).toEqual([0, 2, 3]);
  });

  it("gives an empty record offset -1, not the next record's start", () => {
    const { offsets, values } = flattenRagged([[5, 7], [], [9]]);

    // Record 1 contributed nothing; offset 2 would be indistinguishable from record 2's start.
    expect(offsets).toEqual([0, -1, 2]);
    expect(values).toEqual([5, 7, 9]);
  });

  it("handles no records at all", () => {
    expect(flattenRagged([])).toEqual({ values: [], recordIndices: [], offsets: [] });
  });
});

describe("cumulative", () => {
  it("turns per-chunk gaps into absolute end times", () => {
    expect(cumulative([5, 7, 3])).toEqual([5, 12, 15]);
  });
});

describe("raggedNodeSize", () => {
  it("widens to the flat row, which is at least as long as the longest record", () => {
    const lists = [[1, 2], [3], [4, 5, 6]];
    const withFlat = raggedNodeSize({ lists, hasTitle: true, showFlat: true });
    const withoutFlat = raggedNodeSize({ lists, hasTitle: true, showFlat: false });

    expect(withFlat.width).toBeGreaterThan(withoutFlat.width);
    expect(withFlat.height).toBeGreaterThan(withoutFlat.height);
  });
});
