/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import { buildTrace, buildWarpMap, rawTimeFor } from "./warpTrace.js";
import {
  buildCuts,
  derive,
  idleGaps,
  mapWarp,
} from "../decks/weka-timing-transforms-interactive/logic.js";

const CAP = 1.5;

describe("buildTrace", () => {
  it("freezes the same trace for the same seed", () => {
    expect(buildTrace(3, 12_000).reqs).toEqual(buildTrace(3, 12_000).reqs);
  });

  it("keeps only closed intervals, since the warp is defined over completed requests", () => {
    const { reqs } = buildTrace(3, 12_000);
    expect(reqs.length).toBeGreaterThan(0);
    for (const r of reqs) expect(r.api).toBeGreaterThan(0);
  });

  it("lists every lane that actually has a request", () => {
    const { reqs, lanes } = buildTrace(5, 16_000);
    for (const r of reqs) expect(lanes).toContain(r.agent);
  });

  it("keeps the whole session, not just the live sim's scrolling window", () => {
    // The live sim discards turns older than its 14s window. Reading only the final state
    // returned a few seconds of history no matter how long the recording ran.
    const short = buildTrace(3, 20_000);
    const long = buildTrace(3, 75_000);
    expect(long.reqs.length).toBeGreaterThan(short.reqs.length * 2);
    expect(long.rawSpan).toBeGreaterThan(50);
    // The early requests must survive to the end of a long recording.
    expect(Math.min(...long.reqs.map((r) => r.t))).toBeLessThan(5);
  });
});

describe("rawTimeFor", () => {
  it("inverts mapWarp for every recorded request start", () => {
    // The playhead is driven in warped time and the raw reading recovered from it, so this
    // inverse is the only thing standing between the two heads and disagreeing.
    const { reqs } = buildTrace(3, 20_000);
    const cuts = buildCuts(reqs.map((r) => [r.t, r.t + r.api]), CAP);
    const map = buildWarpMap(idleGaps(reqs, CAP), CAP);
    for (const r of reqs) {
      const warped = mapWarp(cuts, r.t);
      expect(rawTimeFor(warped, map)).toBeCloseTo(r.t, 5);
    }
  });

  it("is the identity when no gap exceeds the cap", () => {
    const map = buildWarpMap([], 5);
    for (const t of [0, 0.5, 2.75, 3]) expect(rawTimeFor(t, map)).toBe(t);
  });

  it("spreads a collapsed gap across the surviving cap rather than stepping over it", () => {
    // One 10s idle stretch (raw 1..11) against a 1s cap. The warped second 1..2 must cover the
    // whole raw stretch, so the raw head accelerates through it instead of teleporting.
    const gaps = idleGaps(
      [{ id: "a", agent: "m", t: 0, api: 1 }, { id: "b", agent: "m", t: 11, api: 1 }],
      1,
    );
    const map = buildWarpMap(gaps, 1);
    expect(rawTimeFor(0.5, map)).toBeCloseTo(0.5, 6);
    expect(rawTimeFor(1.5, map)).toBeCloseTo(6, 6);
    expect(rawTimeFor(2, map)).toBeCloseTo(11, 6);
  });
});

describe("the warp's invariant", () => {
  it("never changes a request's duration, only where it sits", () => {
    // warped_end - warped_start == api, always. This is the claim the spike draws by using one
    // pixels-per-second for both tracks: if it broke, bar widths would visibly disagree.
    const { reqs } = buildTrace(7, 20_000);
    const nodes = derive(reqs, CAP);
    for (const n of nodes) {
      const raw = n.rawEnd - n.rawStart;
      expect(n.warpEnd - n.warpStart).toBeCloseTo(raw, 9);
    }
  });

  it("compresses the session overall while preserving request order", () => {
    const { reqs, rawSpan } = buildTrace(7, 20_000);
    const nodes = derive(reqs, CAP);
    const warpSpan = nodes.reduce((m, n) => Math.max(m, n.warpEnd), 0);
    expect(warpSpan).toBeLessThan(rawSpan);

    const byRaw = [...nodes].sort((a, b) => a.rawStart - b.rawStart).map((n) => n.id);
    const byWarp = [...nodes].sort((a, b) => a.warpStart - b.warpStart).map((n) => n.id);
    expect(byWarp).toEqual(byRaw);
  });
});
