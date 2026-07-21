/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Combined swimlane stack, ported from `CombinedTimeline` in
//! `docs/canvases/weka-timing-transforms-interactive.canvas.tsx`. Flattens the three
//! `MINI_TRACES` into one stack of (trace, lane) rows, each shifted so its own t* lands on a
//! single shared vertical line — warmup turns fall left, profiled turns fall right, regardless
//! of each trace's absolute timing.

import { LANE_KEYS, MINI_TRACES, derive, laneColorIndex, lanesOf, warmupIds } from "./logic.js";
import { categoryClassName, inkClassName, strokeClassName } from "../../theme/tokens.js";

const CATEGORY_TEXT: Record<(typeof LANE_KEYS)[number], string> = {
  blue: "text-category-blue",
  green: "text-category-green",
  purple: "text-category-purple",
  orange: "text-category-orange",
  red: "text-category-red",
  cyan: "text-category-cyan",
  yellow: "text-category-yellow",
  gray: "text-category-gray",
};

function laneTextClassName(agent: string, lanes: readonly string[]): string {
  return CATEGORY_TEXT[LANE_KEYS[laneColorIndex(agent, lanes)]!];
}

const LANE_H = 22;
const LANE_GAP = 5;
const TOP = 30;
const LEFT = 150;

export type CombinedTimelineProps = {
  /** Each trace's current t*, in `MINI_TRACES` order (linear, one-sub, two-subs). */
  tStars: readonly [number, number, number];
};

/** All three mini traces aligned so every trace's own t* lands on one shared vertical line. */
export function CombinedTimeline({ tStars }: CombinedTimelineProps): React.JSX.Element {
  const traces = MINI_TRACES.map((tr, i) => {
    const lanes = lanesOf(tr.reqs);
    const nodes = derive(tr.reqs, 60);
    const warpMax = Math.max(...nodes.map((n) => n.warpEnd));
    const tStar = Math.min(tStars[i]!, Math.ceil(warpMax));
    return { ...tr, lanes, nodes, tStar, warmup: warmupIds(nodes, lanes, tStar) };
  });

  const rows = traces.flatMap((tr) => tr.lanes.map((agent) => ({ tr, agent })));

  let leftExtent = 0;
  let rightExtent = 1;
  for (const tr of traces) {
    for (const n of tr.nodes) {
      leftExtent = Math.max(leftExtent, tr.tStar - n.warpStart);
      rightExtent = Math.max(rightExtent, n.warpEnd - tr.tStar);
    }
  }
  const span = leftExtent + rightExtent;
  const px = Math.max(8, Math.min(22, Math.floor((790 - LEFT) / Math.max(span, 1))));
  const alignX = LEFT + leftExtent * px;
  const xRel = (t: number, tStar: number) => alignX + (t - tStar) * px;
  const width = alignX + rightExtent * px + 24;

  const rowY = (i: number) => TOP + i * (LANE_H + LANE_GAP);
  const blockBottom = TOP + rows.length * (LANE_H + LANE_GAP) - LANE_GAP;
  const axisY = blockBottom + 12;
  const svgH = axisY + 24;

  const tickStep = span > 18 ? 5 : 2;
  const startTick = -Math.ceil(leftExtent / tickStep) * tickStep;
  const endTick = Math.ceil(rightExtent / tickStep) * tickStep;
  const ticks: number[] = [];
  for (let v = startTick; v <= endTick; v += tickStep) ticks.push(v);

  return (
    <div className="overflow-x-auto">
      <svg width={width} height={svgH} role="img" aria-label="combined timeline aligned at t-star">
        <line x1={alignX} y1={TOP - 8} x2={alignX} y2={blockBottom + 2} className={categoryClassName("blue")} stroke="currentColor" strokeWidth={1.5} strokeDasharray="5 3" />
        <text x={alignX} y={TOP - 12} textAnchor="middle" fontSize={11} fontWeight={700} className={categoryClassName("blue")} fill="currentColor">
          t*
        </text>

        {rows.map((r, i) => (
          <text
            key={`lbl-${r.tr.key}-${r.agent}`}
            x={8}
            y={rowY(i) + LANE_H / 2 + 4}
            fontSize={10}
            fontWeight={600}
            className={laneTextClassName(r.agent, r.tr.lanes)}
            fill="currentColor"
          >
            {r.tr.key}·{r.agent}
          </text>
        ))}

        {rows.map((r, i) =>
          r.tr.nodes
            .filter((n) => n.agent === r.agent)
            .map((n) => {
              const profiled = n.warpStart >= r.tr.tStar;
              const warm = r.tr.warmup.has(n.id);
              const w = Math.max((n.warpEnd - n.warpStart) * px, 10);
              const bx = xRel(n.warpStart, r.tr.tStar);
              const strokeCls = warm ? categoryClassName("orange") : profiled ? laneTextClassName(r.agent, r.tr.lanes) : strokeClassName("tertiary");
              const textCls = warm ? categoryClassName("orange") : profiled ? inkClassName("primary") : inkClassName("tertiary");
              return (
                <g key={`cn-${r.tr.key}-${n.id}`}>
                  <rect
                    x={bx}
                    y={rowY(i)}
                    width={w}
                    height={LANE_H}
                    fill={profiled && !warm ? "currentColor" : "none"}
                    fillOpacity={profiled && !warm ? 0.12 : undefined}
                    className={profiled && !warm ? laneTextClassName(r.agent, r.tr.lanes) : strokeCls}
                    stroke="currentColor"
                    strokeWidth={warm ? 2 : 1.5}
                    strokeDasharray={!profiled && !warm ? "4 4" : undefined}
                  />
                  <rect x={bx} y={rowY(i)} width={w} height={LANE_H} fill="none" className={strokeCls} stroke="currentColor" strokeWidth={warm ? 2 : 1.5} strokeDasharray={!profiled && !warm ? "4 4" : undefined} />
                  <text x={bx + w / 2} y={rowY(i) + LANE_H / 2 + 4} textAnchor="middle" fontSize={10} fontWeight={600} className={textCls} fill="currentColor">
                    {n.id}
                  </text>
                </g>
              );
            }),
        )}

        <line x1={LEFT - 4} y1={axisY} x2={width - 12} y2={axisY} className={strokeClassName("secondary")} stroke="currentColor" strokeWidth={1} />
        {ticks.map((v) => (
          <g key={`ct-${v}`}>
            <line x1={xRel(v, 0)} y1={axisY - 3} x2={xRel(v, 0)} y2={axisY + 3} className={strokeClassName("secondary")} stroke="currentColor" strokeWidth={1} />
            <text x={xRel(v, 0)} y={axisY + 15} textAnchor="middle" fontSize={10} className={inkClassName("tertiary")} fill="currentColor">
              {v > 0 ? `+${v}` : v}s
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
