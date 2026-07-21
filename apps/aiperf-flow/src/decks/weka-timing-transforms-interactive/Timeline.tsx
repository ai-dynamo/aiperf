/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Raw-vs-warped swimlane Gantt, ported from `Timeline` in
//! `docs/canvases/weka-timing-transforms-interactive.canvas.tsx`. Not a box-and-arrow diagram
//! (React Flow's node/edge vocabulary doesn't fit a time-scaled Gantt), so this is a minimal
//! one-off SVG scoped to this deck, following the hand-computed-layout pattern already
//! established by `src/prose/Chart.tsx`.

import { LANE_KEYS, fmt, laneColorIndex, type DNode, type Gap } from "./logic.js";
import { categoryBgClassName, categoryClassName, inkClassName, strokeClassName } from "../../theme/tokens.js";

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

function laneFillClassName(agent: string, lanes: readonly string[]): string {
  return categoryBgClassName(LANE_KEYS[laneColorIndex(agent, lanes)]!);
}

const LEFT = 92;
const LANE_H = 24;
const LANE_GAP = 6;

export type TimelineProps = {
  nodes: DNode[];
  gaps: Gap[];
  lanes: string[];
  warpOn: boolean;
};

/** Two stacked swimlane blocks (raw clock, warped clock) sharing one time axis, with dashed
 * idle-gap bands spanning every lane in the raw block. */
export function Timeline({ nodes, gaps, lanes, warpOn }: TimelineProps): React.JSX.Element {
  const maxEnd = Math.max(...nodes.map((n) => n.rawEnd), 1);
  const px = Math.max(8, Math.min(24, Math.floor((740 - LEFT) / maxEnd)));
  const x = (t: number) => LEFT + t * px;
  const width = x(maxEnd) + 24;

  const blockH = lanes.length * (LANE_H + LANE_GAP) - LANE_GAP;
  const laneIndex = (agent: string) => Math.max(0, lanes.indexOf(agent));
  const laneY = (top: number, agent: string) => top + laneIndex(agent) * (LANE_H + LANE_GAP);

  const rawTitleY = 16;
  const rawTop = 26;
  const rawBottom = rawTop + blockH;
  const warpTitleY = rawBottom + 26;
  const warpTop = warpTitleY + 10;
  const warpBottom = warpTop + blockH;
  const axisY = warpBottom + 14;
  const svgH = axisY + 22;

  const tickStep = maxEnd > 45 ? 15 : maxEnd > 20 ? 10 : 5;
  const ticks: number[] = [];
  for (let t = 0; t <= maxEnd; t += tickStep) ticks.push(t);

  const laneLabels = (top: number) =>
    lanes.map((agent) => (
      <text
        key={`ll-${top}-${agent}`}
        x={8}
        y={laneY(top, agent) + LANE_H / 2 + 4}
        fontSize={10.5}
        fontWeight={600}
        className={laneTextClassName(agent, lanes)}
        fill="currentColor"
      >
        {agent}
      </text>
    ));

  const bar = (n: DNode, top: number, s: number, e: number) => {
    const w = Math.max((e - s) * px, 10);
    const y = laneY(top, n.agent);
    return (
      <g key={`${n.id}-${top}`}>
        <rect
          x={x(s)}
          y={y}
          width={w}
          height={LANE_H}
          className={laneFillClassName(n.agent, lanes)}
          fillOpacity={0.18}
          stroke="currentColor"
          strokeWidth={1.5}
        />
        <text
          x={x(s) + w / 2}
          y={y + LANE_H / 2 + 4}
          textAnchor="middle"
          fontSize={10.5}
          fontWeight={600}
          className={inkClassName("primary")}
          fill="currentColor"
        >
          {n.id}
        </text>
      </g>
    );
  };

  return (
    <div className="overflow-x-auto">
      <svg width={width} height={svgH} role="img" aria-label="raw vs warped timeline">
        {gaps.map((gp, i) => (
          <g key={`gap-${i}`}>
            <rect
              x={x(gp.start)}
              y={rawTop - 4}
              width={Math.max(x(gp.end) - x(gp.start), 2)}
              height={blockH + 8}
              fill="none"
              className={gp.capped ? categoryClassName("orange") : strokeClassName("tertiary")}
              stroke="currentColor"
              strokeWidth={1}
              strokeDasharray="4 4"
            />
            <text
              x={(x(gp.start) + x(gp.end)) / 2}
              y={rawTop - 8}
              textAnchor="middle"
              fontSize={9.5}
              className={gp.capped ? categoryClassName("orange") : inkClassName("tertiary")}
              fill="currentColor"
            >
              idle {fmt(gp.idle)}s{gp.capped ? " > cap" : ""}
            </text>
          </g>
        ))}

        <text x={8} y={rawTitleY} fontSize={11} fontWeight={700} className={inkClassName("secondary")} fill="currentColor">
          raw clock
        </text>
        {laneLabels(rawTop)}
        {nodes.map((n) => bar(n, rawTop, n.rawStart, n.rawEnd))}

        <line x1={8} y1={rawBottom + 12} x2={width - 12} y2={rawBottom + 12} className={strokeClassName("tertiary")} stroke="currentColor" strokeWidth={1} />
        <text x={8} y={warpTitleY} fontSize={11} fontWeight={700} className={inkClassName("secondary")} fill="currentColor">
          {warpOn ? "warped clock" : "warped clock (no cap)"}
        </text>
        {laneLabels(warpTop)}
        {nodes.map((n) => bar(n, warpTop, n.warpStart, n.warpEnd))}

        <line x1={LEFT} y1={axisY} x2={width - 12} y2={axisY} className={strokeClassName("secondary")} stroke="currentColor" strokeWidth={1} />
        {ticks.map((t) => (
          <g key={`tick-${t}`}>
            <line x1={x(t)} y1={axisY - 3} x2={x(t)} y2={axisY + 3} className={strokeClassName("secondary")} stroke="currentColor" strokeWidth={1} />
            <text x={x(t)} y={axisY + 15} textAnchor="middle" fontSize={10} className={inkClassName("tertiary")} fill="currentColor">
              {t}s
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
