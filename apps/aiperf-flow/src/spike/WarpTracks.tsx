/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the two-clock chart, as a pure function of `(trace, cap, rawNow, warpedNow)`.
//!
//! Extracted so the free-running spike and the narrated one draw the same picture, and because a
//! chart that is a pure function of its playhead is the shape a deck node would need anyway.

import { useMemo } from "react";
import { derive, idleGaps } from "../decks/weka-timing-transforms-interactive/logic.js";
import type { FrozenTrace } from "./warpTrace.js";

const W = 1280;
const LEFT = 132;
const TOP = 92;
const LANE_H = 22;
const BAR_H = 13;
const BLOCK_GAP = 58;

const DEPTH_COLOR = [
  "var(--color-category-blue)",
  "var(--color-category-green)",
  "var(--color-category-purple)",
  "var(--color-category-orange)",
] as const;

export type WarpTracksProps = {
  trace: FrozenTrace;
  cap: number;
  rawNow: number;
  warpedNow: number;
  /** Dim everything the warped head has not reached, for a narrated build-up. */
  revealWithHead?: boolean;
};

export function WarpTracks({
  trace,
  cap,
  rawNow,
  warpedNow,
  revealWithHead = false,
}: WarpTracksProps): React.JSX.Element {
  const { reqs, lanes, depthOf, rawSpan } = trace;
  const nodes = useMemo(() => derive(reqs, cap), [reqs, cap]);
  const gaps = useMemo(() => idleGaps(reqs, cap), [reqs, cap]);
  const warpSpan = nodes.reduce((m, n) => Math.max(m, n.warpEnd), 0);
  const saved = Math.max(0, rawSpan - warpSpan);

  // One scale for both tracks: that is what makes the warped track being shorter mean something,
  // and keeps every bar's width identical across the two clocks.
  const px = Math.max(2, (W - LEFT - 30) / Math.max(rawSpan, 1));
  const x = (t: number) => LEFT + t * px;

  const rawTop = TOP;
  const warpTop = rawTop + lanes.length * LANE_H + BLOCK_GAP;
  const H = warpTop + lanes.length * LANE_H + 46;
  const laneY = (top: number, lane: string) => top + Math.max(0, lanes.indexOf(lane)) * LANE_H;
  const colorOf = (lane: string) =>
    DEPTH_COLOR[Math.min(depthOf.get(lane) ?? 0, DEPTH_COLOR.length - 1)]!;

  const block = (top: number, key: "raw" | "warp", head: number) => (
    <>
      {lanes.map((lane) => (
        <line key={`rule-${key}-${lane}`} x1={LEFT} y1={laneY(top, lane) + BAR_H / 2}
          x2={W - 26} y2={laneY(top, lane) + BAR_H / 2}
          stroke="var(--color-stroke-tertiary)" strokeWidth={1} opacity={0.25} />
      ))}
      {nodes.map((n) => {
        const s = key === "raw" ? n.rawStart : n.warpStart;
        const e = key === "raw" ? n.rawEnd : n.warpEnd;
        const c = colorOf(n.agent);
        const inside = head >= s && head < e;
        const reached = head >= s;
        const ahead = revealWithHead && !reached;
        return (
          <rect key={`${key}-${n.id}`} x={x(s)} y={laneY(top, n.agent)}
            width={Math.max(1.5, (e - s) * px)} height={BAR_H} rx={1.5}
            fill={c} fillOpacity={ahead ? 0.05 : head >= e ? 0.4 : inside ? 0.72 : 0.12}
            stroke={c} strokeWidth={inside ? 1.5 : 0.7}
            strokeOpacity={ahead ? 0.2 : reached ? 1 : 0.4} />
        );
      })}
      <line x1={x(head)} y1={top - 12} x2={x(head)} y2={top + lanes.length * LANE_H + 2}
        stroke="var(--color-category-red)" strokeWidth={1.6} />
    </>
  );

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block" role="img"
      aria-label="raw and warped clocks over one recorded session">
      <text x={LEFT} y={rawTop - 24} fontSize={11} fontWeight={700}
        fill="var(--color-ink-secondary)" letterSpacing={1.1}>RAW CLOCK — AS RECORDED</text>

      {gaps.filter((g) => g.capped).map((g, i) => (
        <g key={`gap-${i}`}>
          <rect x={x(g.start)} y={rawTop - 8} width={Math.max(2, (g.end - g.start) * px)}
            height={lanes.length * LANE_H + 12} fill="var(--color-category-orange)" opacity={0.11} />
          <text x={(x(g.start) + x(g.end)) / 2} y={rawTop - 12} textAnchor="middle" fontSize={9}
            fill="var(--color-category-orange)">−{(g.idle - cap).toFixed(1)}s</text>
        </g>
      ))}

      {lanes.map((lane) => (
        <text key={`l-raw-${lane}`} x={LEFT - 10} y={laneY(rawTop, lane) + BAR_H - 2}
          textAnchor="end" fontSize={9.5} fill={colorOf(lane)}>{lane}</text>
      ))}
      {block(rawTop, "raw", rawNow)}

      <text x={LEFT} y={warpTop - 24} fontSize={11} fontWeight={700}
        fill="var(--color-category-green)" letterSpacing={1.1}>
        WARPED CLOCK — WHAT THE RUNTIME ISSUES
      </text>
      <rect x={x(warpSpan)} y={warpTop - 8} width={Math.max(0, (rawSpan - warpSpan) * px)}
        height={lanes.length * LANE_H + 12} fill="var(--color-category-green)" opacity={0.07} />
      <text x={x(warpSpan) + 8} y={warpTop + lanes.length * LANE_H + 18} fontSize={10}
        fill="var(--color-category-green)">{saved.toFixed(1)}s never replayed</text>

      {lanes.map((lane) => (
        <text key={`l-warp-${lane}`} x={LEFT - 10} y={laneY(warpTop, lane) + BAR_H - 2}
          textAnchor="end" fontSize={9.5} fill={colorOf(lane)}>{lane}</text>
      ))}
      {block(warpTop, "warp", warpedNow)}
    </svg>
  );
}

/** Warped span and saving for a trace at a given cap, for readouts beside the chart. */
export function warpSummary(trace: FrozenTrace, cap: number) {
  const nodes = derive(trace.reqs, cap);
  const warpSpan = nodes.reduce((m, n) => Math.max(m, n.warpEnd), 0);
  return { warpSpan, saved: Math.max(0, trace.rawSpan - warpSpan) };
}
