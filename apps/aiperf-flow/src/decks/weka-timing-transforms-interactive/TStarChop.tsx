/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! t* snapshot-chop diagram, ported from `TStarChop`/`laneNodeBox` in
//! `docs/canvases/weka-timing-transforms-interactive.canvas.tsx`. Time-scaled swimlanes with a
//! dependency arrow from the re-rooted `S*` start node, so — like `Timeline.tsx` — this stays a
//! hand-computed SVG rather than a React Flow graph.

import { LANE_KEYS, fmt, laneColorIndex, warmupIds, type DNode, type EdgeRow } from "./logic.js";
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

const LANE_H = 24;
const LANE_GAP = 6;
const BOX_W = 40;

function laneBox(
  key: string,
  cx: number,
  y: number,
  label: string,
  fillClassName: string | undefined,
  textClassName: string,
  dropped: boolean,
  warmup: boolean,
  w = BOX_W,
) {
  const strokeCls = warmup ? categoryClassName("orange") : dropped ? strokeClassName("tertiary") : textClassName;
  return (
    <g key={key}>
      <rect
        x={cx}
        y={y}
        width={w}
        height={LANE_H}
        fill={warmup || dropped ? "none" : "currentColor"}
        fillOpacity={warmup || dropped ? undefined : 0.12}
        className={warmup || dropped ? undefined : fillClassName}
        stroke="currentColor"
        strokeWidth={warmup ? 2 : 1.5}
        strokeDasharray={dropped && !warmup ? "4 4" : undefined}
      />
      <rect x={cx} y={y} width={w} height={LANE_H} fill="none" className={strokeCls} stroke="currentColor" strokeWidth={warmup ? 2 : 1.5} strokeDasharray={dropped && !warmup ? "4 4" : undefined} />
      <text
        x={cx + w / 2}
        y={y + LANE_H / 2 + 4}
        textAnchor="middle"
        fontSize={10.5}
        fontWeight={600}
        className={warmup ? categoryClassName("orange") : dropped ? inkClassName("tertiary") : inkClassName("primary")}
        fill="currentColor"
      >
        {label}
      </text>
    </g>
  );
}

export type TStarChopProps = {
  nodes: DNode[];
  edges: EdgeRow[];
  lanes: string[];
  tStar: number;
  /** Renders only the "before" swimlane block (used by the per-trace mini charts). */
  beforeOnly?: boolean;
};

/** Before/after view of dropping every node that arrived before t* (warmed, dashed) and
 * re-rooting surviving nodes whose binding cause was dropped to a synthetic `S*` start box. */
export function TStarChop({ nodes, edges, lanes, tStar, beforeOnly = false }: TStarChopProps): React.JSX.Element {
  const maxEnd = Math.max(...nodes.map((n) => n.warpEnd), 1);
  const LEFT = 96;
  const px = Math.max(8, Math.min(26, Math.floor((720 - LEFT) / maxEnd)));
  const x = (t: number) => LEFT + t * px;
  const width = x(maxEnd) + 24;

  const blockH = lanes.length * (LANE_H + LANE_GAP) - LANE_GAP;
  const laneIndex = (agent: string) => Math.max(0, lanes.indexOf(agent));
  const laneY = (top: number, agent: string) => top + laneIndex(agent) * (LANE_H + LANE_GAP);

  const beforeTitleY = 16;
  const beforeTop = 26;
  const beforeBottom = beforeTop + blockH;
  const afterTitleY = beforeBottom + 24;
  const afterTop = afterTitleY + 10;
  const afterBottom = afterTop + blockH;
  const svgH = (beforeOnly ? beforeBottom : afterBottom) + 20;

  const tx = x(tStar);
  const edgeOf = (id: string) => edges.find((e) => e.id === id)!;
  const droppedIds = new Set(nodes.filter((n) => n.warpStart < tStar).map((n) => n.id));
  const warmup = warmupIds(nodes, lanes, tStar);
  const survivors = nodes.filter((n) => n.warpStart >= tStar);
  const starX = 52;
  const starW = 30;
  const starMidY = afterTop + blockH / 2 - LANE_H / 2;

  const laneLabels = (top: number) =>
    lanes.map((agent) => (
      <text
        key={`cl-${top}-${agent}`}
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

  return (
    <div className="overflow-x-auto">
      <svg width={width} height={svgH} role="img" aria-label="t-star snapshot chop">
        <defs>
          <marker id="tt-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" className={inkClassName("quaternary")} fill="currentColor" />
          </marker>
        </defs>

        <line x1={tx} y1={beforeTop - 6} x2={tx} y2={beforeOnly ? beforeBottom : afterBottom} className={categoryClassName("blue")} stroke="currentColor" strokeWidth={1.25} strokeDasharray="5 3" />
        <text x={tx + 4} y={beforeTitleY} fontSize={11} fontWeight={600} className={categoryClassName("blue")} fill="currentColor">
          t* = {fmt(tStar)}s
        </text>

        {!beforeOnly && (
          <text x={8} y={beforeTitleY} fontSize={11} fontWeight={700} className={inkClassName("secondary")} fill="currentColor">
            before
          </text>
        )}
        {laneLabels(beforeTop)}
        {nodes.map((n) =>
          laneBox(
            `b-${n.id}`,
            x(n.warpStart),
            laneY(beforeTop, n.agent),
            n.id,
            laneTextClassName(n.agent, lanes),
            laneTextClassName(n.agent, lanes),
            droppedIds.has(n.id),
            warmup.has(n.id),
          ),
        )}

        {!beforeOnly && (
          <g>
            <line x1={8} y1={beforeBottom + 12} x2={width - 12} y2={beforeBottom + 12} className={strokeClassName("tertiary")} stroke="currentColor" strokeWidth={1} />
            <text x={8} y={afterTitleY} fontSize={11} fontWeight={700} className={inkClassName("secondary")} fill="currentColor">
              after
            </text>
            {laneLabels(afterTop)}
            {laneBox("start", starX, starMidY, "S*", undefined, categoryClassName("blue"), false, false, starW)}
            {survivors.map((n) => {
              const e = edgeOf(n.id);
              const bindingDropped = e.rootsAtStart || droppedIds.has(e.firesAfter);
              const y = laneY(afterTop, n.agent);
              const relDelay = n.warpStart - tStar;
              return (
                <g key={`a-${n.id}`}>
                  {bindingDropped && (
                    <line
                      x1={starX + starW}
                      y1={starMidY + LANE_H / 2}
                      x2={x(n.warpStart)}
                      y2={y + LANE_H / 2}
                      className={inkClassName("quaternary")}
                      stroke="currentColor"
                      strokeWidth={1.25}
                      strokeDasharray="4 4"
                      markerEnd="url(#tt-arrow)"
                    />
                  )}
                  {laneBox(
                    `an-${n.id}`,
                    x(n.warpStart),
                    y,
                    n.id,
                    laneTextClassName(n.agent, lanes),
                    laneTextClassName(n.agent, lanes),
                    false,
                    false,
                  )}
                  {bindingDropped && (
                    <text x={x(n.warpStart) + BOX_W / 2} y={y - 3} textAnchor="middle" fontSize={9} className={inkClassName("tertiary")} fill="currentColor">
                      +{fmt(relDelay)}s
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        )}
      </svg>
    </div>
  );
}
