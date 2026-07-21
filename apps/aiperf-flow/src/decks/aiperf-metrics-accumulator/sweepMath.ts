/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure sweep-line math ported from `docs/canvases/aiperf-metrics-accumulator.canvas.tsx`: turn a
//! set of request intervals into signed +/- weight events, sort them by time, and take a running
//! cumulative sum to reconstruct the exact step function (concurrency / tokens-in-flight /
//! decode-throughput curves) in `O(E log E)` without scanning the timeline point by point.

export type SweepRequest = {
  id: string;
  /** Request start (arrival). */
  start: number;
  /** generation_start = start + TTFT. */
  gen: number;
  /** Request end. */
  end: number;
  /** Output tokens. */
  tokens: number;
};

export type SweepCurveId = "concurrency" | "tokens" | "throughput";

export type SweepInterval = { start: number; end: number; weight: number };
export type SweepStepPoint = { t: number; v: number };
export type SweepEvent = { t: number; d: number; kind: "start" | "end"; id: string };

/** Maps each request to the weighted interval the given curve sweeps over. */
export function intervalFor(r: SweepRequest, curve: SweepCurveId): SweepInterval {
  if (curve === "tokens") {
    return { start: r.start, end: r.end, weight: r.tokens };
  }
  if (curve === "throughput") {
    const dur = r.end - r.gen;
    return { start: r.gen, end: r.end, weight: dur > 0 ? r.tokens / dur : 0 };
  }
  return { start: r.start, end: r.end, weight: 1 };
}

/** Turns each request's interval into a `+weight` event at its start and a `-weight` at its end. */
export function buildEvents(reqs: readonly SweepRequest[], curve: SweepCurveId): SweepEvent[] {
  const evts: SweepEvent[] = [];
  for (const r of reqs) {
    const iv = intervalFor(r, curve);
    if (iv.end <= iv.start) continue;
    evts.push({ t: iv.start, d: iv.weight, kind: "start", id: r.id });
    evts.push({ t: iv.end, d: -iv.weight, kind: "end", id: r.id });
  }
  // Ends sort before starts on ties so touching intervals don't double-count.
  evts.sort((a, b) => a.t - b.t || a.d - b.d);
  return evts;
}

/** Cumulative-sums the sorted event stream into one point per distinct event time. */
export function stepPoints(evts: readonly SweepEvent[]): SweepStepPoint[] {
  const pts: SweepStepPoint[] = [];
  let acc = 0;
  let i = 0;
  while (i < evts.length) {
    const t = evts[i].t;
    while (i < evts.length && evts[i].t === t) {
      acc += evts[i].d;
      i++;
    }
    pts.push({ t, v: acc });
  }
  return pts;
}

/** Builds an SVG path `d` string for the step function, extended flat to `tMin`/`tMax`. */
export function stepPathD(
  pts: readonly SweepStepPoint[],
  x: (t: number) => number,
  y: (v: number) => number,
  tMin: number,
  tMax: number,
): string {
  if (pts.length === 0) return `M ${x(tMin)} ${y(0)} L ${x(tMax)} ${y(0)}`;
  let d = `M ${x(tMin)} ${y(0)}`;
  let prev = 0;
  for (const p of pts) {
    d += ` L ${x(p.t)} ${y(prev)} L ${x(p.t)} ${y(p.v)}`;
    prev = p.v;
  }
  return d + ` L ${x(tMax)} ${y(prev)}`;
}

/** Rounds `v` up to a "nice" axis maximum (1/2/5 * 10^n). */
export function niceMax(v: number): number {
  if (v <= 1) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / pow;
  const step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return step * pow;
}
