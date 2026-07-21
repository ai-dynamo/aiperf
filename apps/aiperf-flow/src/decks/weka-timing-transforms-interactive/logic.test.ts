/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import {
  SCENARIOS,
  MINI_TRACES,
  buildCuts,
  computeEdges,
  derive,
  fmt,
  idleGaps,
  laneColorIndex,
  lanesOf,
  mapWarp,
  warmupIds,
} from "./logic.js";

describe("lanesOf", () => {
  it("returns agents in first-seen order, deduplicated", () => {
    expect(lanesOf(SCENARIOS.agent!.reqs)).toEqual(["main", "researcher", "coder"]);
  });
});

describe("buildCuts / mapWarp — the agent scenario at cap=60", () => {
  const reqs = SCENARIOS.agent!.reqs;
  const intervals = reqs.map((r) => [r.t, r.t + r.api] as [number, number]);
  const cuts = buildCuts(intervals, 60);

  it("cuts exactly the one true idle gap longer than the cap", () => {
    // m1 ends at 9, m2 (the next active interval in the union) starts at 95 -> idle 86s over the
    // cap of 60s -> excess 26s, recorded at the interval's own start.
    expect(cuts).toEqual([[95, 26]]);
  });

  it("warps timestamps before the cut unchanged and after by the excess", () => {
    expect(mapWarp(cuts, 0)).toBe(0);
    expect(mapWarp(cuts, 6)).toBe(6);
    expect(mapWarp(cuts, 95)).toBe(69);
    expect(mapWarp(cuts, 130)).toBe(104);
  });
});

describe("derive", () => {
  it("with cap=null passes raw timestamps straight through as warped", () => {
    const nodes = derive(SCENARIOS.dense!.reqs, null);
    for (const n of nodes) {
      expect(n.warpStart).toBe(n.rawStart);
      expect(n.warpEnd).toBe(n.rawEnd);
    }
  });

  it("never cuts inside a request: warpEnd - warpStart always equals api_time", () => {
    const reqs = SCENARIOS.agent!.reqs;
    const nodes = derive(reqs, 60);
    nodes.forEach((n, i) => {
      expect(n.warpEnd - n.warpStart).toBeCloseTo(reqs[i]!.api);
    });
  });
});

describe("idleGaps", () => {
  it("reports every gap between consecutive active intervals, flagging only the one over cap", () => {
    const gaps = idleGaps(SCENARIOS.agent!.reqs, 60);
    expect(gaps.filter((g) => g.capped)).toEqual([{ start: 9, end: 95, idle: 86, capped: true }]);
    expect(gaps).toHaveLength(4);
  });

  it("finds only small sub-cap gaps in the dense scenario, none capped", () => {
    const gaps = idleGaps(SCENARIOS.dense!.reqs, 60);
    expect(gaps.every((g) => !g.capped)).toBe(true);
  });
});

describe("computeEdges", () => {
  it("roots the very first node at START", () => {
    const nodes = derive(SCENARIOS.agent!.reqs, 60);
    const edges = computeEdges(nodes);
    expect(edges[0]).toMatchObject({ id: "m0", firesAfter: "START", rootsAtStart: true });
  });

  it("binds a turn to the latest-ending completed cause with a nonnegative delay", () => {
    const nodes = derive(SCENARIOS.agent!.reqs, 60);
    const edges = computeEdges(nodes);
    const m1 = edges.find((e) => e.id === "m1")!;
    // m0 runs [0,4); m1 starts at raw 6 -> m0 completed before it started.
    expect(m1.firesAfter).toBe("m0");
    expect(m1.rootsAtStart).toBe(false);
    expect(m1.delayMs).toBeGreaterThanOrEqual(0);
  });

  it("collects AND-fan-in waits from every other completed cause", () => {
    // subagents scenario: a1 starts at raw 102, after m0(0-3), m1(90-93), and a0(94-102, ends
    // exactly at 102) have all completed. a0 is the latest-ending -> binding cause; m0 and m1
    // are AND-fan-in waits with delay 0.
    const nodes = derive(SCENARIOS.subagents!.reqs, 60);
    const edges = computeEdges(nodes);
    const a1 = edges.find((e) => e.id === "a1")!;
    expect(a1.rootsAtStart).toBe(false);
    expect(a1.firesAfter).toBe("a0");
    expect([...a1.andInputs].sort()).toEqual(["m0", "m1"]);
  });
});

describe("warmupIds", () => {
  it("picks the lane-local turn closest to tStar from below", () => {
    const nodes = derive(MINI_TRACES[1]!.reqs, 60);
    const lanes = lanesOf(MINI_TRACES[1]!.reqs);
    // one-sub trace: main[0-2), main[3-5), sub[4-7), sub[8-10), main[11-13)
    const warm = warmupIds(nodes, lanes, 9);
    expect(warm.has("m1")).toBe(true); // last main turn before t*=9 (starts at 3)
    expect(warm.has("s1")).toBe(true); // last sub turn before t*=9 (starts at 8)
    expect(warm.has("s0")).toBe(false); // s0 starts at 4, closer-below turn s1 wins the lane
  });

  it("returns an empty set when tStar is before every node", () => {
    const nodes = derive(MINI_TRACES[0]!.reqs, 60);
    const lanes = lanesOf(MINI_TRACES[0]!.reqs);
    expect(warmupIds(nodes, lanes, 0).size).toBe(0);
  });
});

describe("laneColorIndex", () => {
  it("assigns sequential indices per first-seen lane and wraps at 8", () => {
    const lanes = ["main", "researcher", "coder"];
    expect(laneColorIndex("main", lanes)).toBe(0);
    expect(laneColorIndex("researcher", lanes)).toBe(1);
    expect(laneColorIndex("coder", lanes)).toBe(2);
    expect(laneColorIndex("unknown", lanes)).toBe(0);
  });
});

describe("fmt", () => {
  it("renders integers bare and fractions to one decimal", () => {
    expect(fmt(5)).toBe("5");
    expect(fmt(5.5)).toBe("5.5");
    expect(fmt(0)).toBe("0");
  });
});
