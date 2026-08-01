/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — freeze a recorded trace out of the live agent session, so the warp has real input.
//!
//! The agent spike generates a session; this runs that same generator headlessly to completion
//! and hands the result to the *actual* weka warp math in
//! `decks/weka-timing-transforms-interactive/logic.ts`. Nothing about the compression is
//! reimplemented here — the point of the spike is watching the real transform run, not a mock
//! that resembles it.

import {
  createAgentSim,
  stepAgents,
  type AgentSimConfig,
} from "./agentSim.js";
import type { Gap, Req } from "../decks/weka-timing-transforms-interactive/logic.js";

export type FrozenTrace = {
  /** Recorded requests, in the shape the warp math consumes. Seconds, not ms. */
  reqs: Req[];
  /** Lane labels in depth-first order, so subagents sit under their parent. */
  lanes: string[];
  /** Nesting depth per lane label, for colouring. */
  depthOf: Map<string, number>;
  /** Wall seconds the recorded session spanned. */
  rawSpan: number;
};

/**
 * Run the agent session headlessly and freeze what it produced.
 *
 * Only completed turns are kept: a turn still streaming when the recording stopped has no
 * duration yet, and the warp is defined over closed intervals.
 */
/**
 * A sparser session than the live agent spike runs.
 *
 * The default config keeps several lanes busy almost continuously, which leaves nothing for the
 * warp to remove — measured at 3-8% idle. Long think time and shallow fan-out produce the dead
 * air a recorded agent session actually contains.
 */
export const TRACE_CONFIG: AgentSimConfig = {
  // A real agent session is mostly waiting: a human reads, a tool runs, nothing is in flight.
  // Short think time keeps a lane busy and leaves the warp nothing to remove.
  thinkMs: 6500,
  spawnChance: 0.3,
  maxDepth: 1,
  maxActive: 2,
  serviceScale: 1,
};

export function buildTrace(
  seed: number,
  durationMs: number,
  config: AgentSimConfig = TRACE_CONFIG,
): FrozenTrace {
  let sim = createAgentSim(seed);

  // Harvest as we go. The live sim is built for a scrolling view and discards agents and turns
  // older than its window, so reading only the final state would silently return just the last
  // few seconds of a long session — which is exactly what made an early version of this look
  // like a 4%-compressible trace.
  const reqs: Req[] = [];
  const seenTurns = new Set<number>();
  const labelOf = new Map<number, string>();
  const depthOf = new Map<string, number>();
  const bornOrder: string[] = [];

  // Fixed quanta, so a frozen trace is a pure function of (seed, duration, config).
  for (let elapsed = 0; elapsed < durationMs; elapsed += 20) {
    sim = stepAgents(sim, 20, config);

    for (const a of sim.agents) {
      if (labelOf.has(a.id)) continue;
      labelOf.set(a.id, a.label);
      depthOf.set(a.label, a.depth);
      bornOrder.push(a.label);
    }
    for (const t of sim.turns) {
      if (t.endAt === null || seenTurns.has(t.id)) continue;
      seenTurns.add(t.id);
      reqs.push({
        id: `t${t.id}`,
        agent: labelOf.get(t.agentId) ?? `a${t.agentId}`,
        t: t.startAt / 1000,
        api: (t.endAt - t.startAt) / 1000,
      });
    }
  }

  reqs.sort((a, b) => a.t - b.t);
  const lanes = bornOrder.filter((l) => reqs.some((r) => r.agent === l));
  const rawSpan = reqs.reduce((m, r) => Math.max(m, r.t + r.api), 0);

  return { reqs, lanes, depthOf, rawSpan };
}

/** One piece of the warped→raw mapping: warped `[w0,w1]` covers raw `[r0,r1]`. */
export type WarpSegment = { w0: number; w1: number; r0: number; r1: number };

/**
 * Piecewise-linear map from warped time back to raw time.
 *
 * `mapWarp` cannot be inverted directly: it is a *step*, jumping from raw 10.9 → warped 10.9 to
 * raw 11 → warped 2 across a collapsed gap. That is harmless where it is used, because it is only
 * ever applied to request start times and no request starts inside an idle gap. A playhead does
 * traverse that region, so it needs the gap spread across the surviving `cap` seconds instead —
 * which is also the right picture: the raw head visibly accelerates through dead air while the
 * warped head advances steadily.
 */
export function buildWarpMap(gaps: readonly Gap[], cap: number): WarpSegment[] {
  const segments: WarpSegment[] = [];
  let raw = 0;
  let warped = 0;
  for (const gap of gaps) {
    if (!gap.capped) continue;
    // Active stretch before this gap passes through untouched.
    const active = gap.start - raw;
    if (active > 0) {
      segments.push({ w0: warped, w1: warped + active, r0: raw, r1: gap.start });
      warped += active;
    }
    // The gap itself: `cap` warped seconds stand in for the whole raw stretch.
    segments.push({ w0: warped, w1: warped + cap, r0: gap.start, r1: gap.end });
    warped += cap;
    raw = gap.end;
  }
  // Everything after the last collapsed gap is 1:1 and unbounded.
  segments.push({ w0: warped, w1: Infinity, r0: raw, r1: Infinity });
  return segments;
}

/** Raw time corresponding to `warpedNow`, via the piecewise map. */
export function rawTimeFor(warpedNow: number, segments: readonly WarpSegment[]): number {
  for (const s of segments) {
    if (warpedNow > s.w1) continue;
    if (!Number.isFinite(s.w1)) return s.r0 + (warpedNow - s.w0);
    const span = s.w1 - s.w0;
    if (span <= 0) return s.r1;
    return s.r0 + ((warpedNow - s.w0) / span) * (s.r1 - s.r0);
  }
  return warpedNow;
}
