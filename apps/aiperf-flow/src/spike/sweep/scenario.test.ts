/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { buildScenario, TARGET_CONCURRENCY } from "./scenario.js";

describe("scenario", () => {
  const s = buildScenario(1);

  it("is deterministic, so every panel draws the same run", () => {
    expect(buildScenario(1).concurrency.values).toEqual(s.concurrency.values);
  });

  it("ramps toward the target, holds, and drains back to exactly zero", () => {
    expect(Math.max(...s.concurrency.values)).toBeGreaterThanOrEqual(TARGET_CONCURRENCY - 2);
    expect(s.concurrency.values[s.concurrency.values.length - 1]).toBe(0);
  });

  it("contains a genuine event collision, so the tie-break has something to show", () => {
    expect(s.collisions.length).toBeGreaterThan(0);
    const i = s.collisions[0]!;
    expect(s.sortedEvents[i]!.timestampNs).toBe(s.sortedEvents[i - 1]!.timestampNs);
    // The end must have sorted first.
    expect(s.sortedEvents[i - 1]!.delta).toBeLessThanOrEqual(s.sortedEvents[i]!.delta);
  });

  it("detects a threshold window that excludes both transients", () => {
    const w = s.thresholdWindow!;
    expect(w).not.toBeNull();
    expect(w.threshold).toBe(Math.ceil(0.8 * TARGET_CONCURRENCY));
    expect(w.startNs).toBeGreaterThan(s.runStartNs);
    expect(w.endNs).toBeLessThanOrEqual(s.runEndNs);
  });

  it("keeps every column and the ragged series index-aligned", () => {
    expect(s.store.rows).toBe(s.records.length);
    for (const c of s.store.columns) expect(c.values).toHaveLength(s.records.length);
    expect(s.store.icl.offsets).toHaveLength(s.records.length);
  });

  it("gives the ICL-aware token curve a different shape from the coarse one", () => {
    // Same totals, different shape. That difference is the whole point of ICL awareness.
    expect(s.iclTokens.values).not.toEqual(s.coarseTokens.values);
  });
});
