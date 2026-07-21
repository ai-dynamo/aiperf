/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure logic ported verbatim (semantics-preserving) from
//! `docs/canvases/weka-timing-transforms-interactive.canvas.tsx`, which itself mirrors
//! `src/aiperf/dataset/loader/graph/adapters/_weka_trie_build.py`'s `_ActiveIdleWarp` and
//! interval-order edge construction. Kept dependency-free (no React) so it's directly unit
//! testable and shared by every visualization in this deck.

/** A recorded weka request: which flattened agent/subagent lane it came from, its raw start
 * offset `t` (seconds from conversation start), and its server-processing duration `api`
 * (seconds). These are the only fields the trie IR's timing derives from. */
export type Req = { id: string; agent: string; t: number; api: number };

export type Scenario = { label: string; reqs: Req[] };

export const SCENARIOS: Record<string, Scenario> = {
  agent: {
    label: "Agent session (subagents past a long idle)",
    reqs: [
      { id: "m0", agent: "main", t: 0, api: 4 },
      { id: "m1", agent: "main", t: 6, api: 3 },
      { id: "m2", agent: "main", t: 95, api: 4 },
      { id: "r0", agent: "researcher", t: 100, api: 6 },
      { id: "c0", agent: "coder", t: 102, api: 10 },
      { id: "r1", agent: "researcher", t: 110, api: 5 },
      { id: "m3", agent: "main", t: 130, api: 3 },
    ],
  },
  subagents: {
    label: "Overlapping subagents past a long idle",
    reqs: [
      { id: "m0", agent: "main", t: 0, api: 3 },
      { id: "m1", agent: "main", t: 90, api: 3 },
      { id: "a0", agent: "sub-A", t: 94, api: 8 },
      { id: "b0", agent: "sub-B", t: 96, api: 10 },
      { id: "a1", agent: "sub-A", t: 102, api: 5 },
      { id: "b1", agent: "sub-B", t: 106, api: 4 },
      { id: "m2", agent: "main", t: 120, api: 2 },
    ],
  },
  dense: {
    label: "Dense turns (no idle to cut)",
    reqs: [
      { id: "m0", agent: "main", t: 0, api: 1.5 },
      { id: "s0", agent: "tool", t: 0.5, api: 2 },
      { id: "m1", agent: "main", t: 2.5, api: 1.5 },
      { id: "s1", agent: "tool", t: 3, api: 1.5 },
      { id: "m2", agent: "main", t: 5, api: 1.5 },
    ],
  },
  bursty: {
    label: "Async tool past a dead-air gap",
    reqs: [
      { id: "m0", agent: "main", t: 0, api: 2 },
      { id: "m1", agent: "main", t: 4, api: 2 },
      { id: "t0", agent: "tool", t: 95, api: 4 },
      { id: "m2", agent: "main", t: 100, api: 3 },
      { id: "m3", agent: "main", t: 105, api: 2 },
    ],
  },
};

/** Three small, self-contained traces for the independent-t* comparison section. */
export const MINI_TRACES: Array<{ key: string; label: string; reqs: Req[] }> = [
  {
    key: "linear",
    label: "Linear chat (single lane)",
    reqs: [
      { id: "m0", agent: "main", t: 0, api: 2 },
      { id: "m1", agent: "main", t: 3, api: 2 },
      { id: "m2", agent: "main", t: 6, api: 2 },
      { id: "m3", agent: "main", t: 9, api: 2 },
      { id: "m4", agent: "main", t: 12, api: 2 },
    ],
  },
  {
    key: "one-sub",
    label: "One subagent",
    reqs: [
      { id: "m0", agent: "main", t: 0, api: 2 },
      { id: "m1", agent: "main", t: 3, api: 2 },
      { id: "s0", agent: "sub", t: 4, api: 3 },
      { id: "s1", agent: "sub", t: 8, api: 2 },
      { id: "m2", agent: "main", t: 11, api: 2 },
    ],
  },
  {
    key: "two-subs",
    label: "Two overlapping subagents",
    reqs: [
      { id: "m0", agent: "main", t: 0, api: 2 },
      { id: "a0", agent: "alpha", t: 2, api: 4 },
      { id: "b0", agent: "beta", t: 3, api: 5 },
      { id: "a1", agent: "alpha", t: 7, api: 2 },
      { id: "m1", agent: "main", t: 10, api: 2 },
    ],
  },
];

export function lanesOf(reqs: readonly Req[]): string[] {
  const seen: string[] = [];
  for (const r of reqs) if (!seen.includes(r.agent)) seen.push(r.agent);
  return seen;
}

/** Ascending (next_start, cumulative_excess): every timestamp >= next_start shifts left by
 * cumulative. Built by a sweep over active intervals sorted by start, collapsing any TRUE idle
 * gap (running_max_end -> next start) greater than `cap`. */
export function buildCuts(
  intervals: ReadonlyArray<readonly [number, number]>,
  cap: number,
): Array<[number, number]> {
  const cuts: Array<[number, number]> = [];
  if (intervals.length === 0) return cuts;
  const ordered = [...intervals].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let runningEnd = ordered[0]![1];
  let cumulative = 0;
  for (let i = 1; i < ordered.length; i++) {
    const [start, end] = ordered[i]!;
    if (start > runningEnd) {
      const idle = start - runningEnd;
      if (idle > cap) {
        cumulative += idle - cap;
        cuts.push([start, cumulative]);
      }
    }
    if (end > runningEnd) runningEnd = end;
  }
  return cuts;
}

export function mapWarp(cuts: ReadonlyArray<readonly [number, number]>, t: number): number {
  let shift = 0;
  for (const [nextStart, cumulative] of cuts) {
    if (t < nextStart) break;
    shift = cumulative;
  }
  return t - shift;
}

export type DNode = {
  id: string;
  agent: string;
  rawStart: number;
  rawEnd: number;
  warpStart: number;
  warpEnd: number;
};

export function derive(reqs: readonly Req[], cap: number | null): DNode[] {
  const intervals = reqs.map((r) => [r.t, r.t + r.api] as [number, number]);
  const cuts = cap === null ? [] : buildCuts(intervals, cap);
  return reqs.map((r) => {
    const ws = cap === null ? r.t : mapWarp(cuts, r.t);
    return {
      id: r.id,
      agent: r.agent,
      rawStart: r.t,
      rawEnd: r.t + r.api,
      warpStart: ws,
      warpEnd: ws + r.api,
    };
  });
}

/** A true idle gap (running_max_end -> next start), with cap-classification. */
export type Gap = { start: number; end: number; idle: number; capped: boolean };

export function idleGaps(reqs: readonly Req[], cap: number): Gap[] {
  const intervals = reqs.map((r) => [r.t, r.t + r.api] as [number, number]);
  if (intervals.length === 0) return [];
  const ordered = [...intervals].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let runningEnd = ordered[0]![1];
  const gaps: Gap[] = [];
  for (let i = 1; i < ordered.length; i++) {
    const [start, end] = ordered[i]!;
    if (start > runningEnd) {
      const idle = start - runningEnd;
      gaps.push({ start: runningEnd, end: start, idle, capped: idle > cap });
    }
    if (end > runningEnd) runningEnd = end;
  }
  return gaps;
}

/** A cause is real only if it COMPLETED before the turn started (raw end <= raw start). The
 * binding cause is the latest-ending completed cause; the firing delay is the warped
 * end-to-start gap. If nothing had finished, the turn roots at START at its own warped arrival
 * offset. */
export type EdgeRow = {
  id: string;
  firesAfter: string;
  delayMs: number;
  andInputs: string[];
  rootsAtStart: boolean;
};

export function computeEdges(nodes: readonly DNode[]): EdgeRow[] {
  return nodes.map((n) => {
    const completed = nodes.filter((c) => c.id !== n.id && c.rawEnd <= n.rawStart);
    if (completed.length === 0) {
      return {
        id: n.id,
        firesAfter: "START",
        delayMs: n.warpStart * 1000,
        andInputs: [],
        rootsAtStart: true,
      };
    }
    const binding = completed.reduce((a, b) => (b.rawEnd > a.rawEnd ? b : a));
    const delayMs = Math.max(0, n.warpStart - binding.warpEnd) * 1000;
    return {
      id: n.id,
      firesAfter: binding.id,
      delayMs,
      andInputs: completed.filter((c) => c.id !== binding.id).map((c) => c.id),
      rootsAtStart: false,
    };
  });
}

/** Per lane, the warmed turn (warpStart < tStar) closest to tStar from below. This is the last
 * turn that ran before the snapshot in that lane, whose KV the resuming survivor still names in
 * its prompt path. */
export function warmupIds(nodes: readonly DNode[], lanes: readonly string[], tStar: number): Set<string> {
  const s = new Set<string>();
  for (const lane of lanes) {
    const warmed = nodes.filter((n) => n.agent === lane && n.warpStart < tStar);
    if (warmed.length) {
      const best = warmed.reduce((a, b) => (b.warpStart > a.warpStart ? b : a));
      s.add(best.id);
    }
  }
  return s;
}

export function fmt(n: number): string {
  return Number.isInteger(n) ? `${n}` : n.toFixed(1);
}

/** Distinct lane hue per subagent, cycled from the shared 8-color category palette. The source
 * canvas's palette includes a "pink" hue this app's `CategoryRole` doesn't have; `red` (unused
 * by the other seven) stands in for it, same substitution `PayloadsPage.tsx` uses for Media. */
export const LANE_KEYS = [
  "blue",
  "green",
  "purple",
  "orange",
  "red",
  "cyan",
  "yellow",
  "gray",
] as const;

export function laneColorIndex(agent: string, lanes: readonly string[]): number {
  const i = Math.max(0, lanes.indexOf(agent));
  return i % LANE_KEYS.length;
}
