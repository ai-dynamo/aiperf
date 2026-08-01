/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import clsx from "clsx";
import { surfaceClassName, inkClassName, categoryStrokeClassName } from "../theme/tokens.js";
import { NodeAnchorHandles } from "./anchors.js";
import { layoutIntervals, resolveRanks } from "./intervalsLayout.js";
import type { IntervalsNodeData } from "./types.js";

export type IntervalsNodeType = Node<IntervalsNodeData, "intervals">;

/**
 * Intervals on one clock, one row each, with the global rank badged on every end.
 *
 * This is the picture interval-order edge derivation reads: an edge exists from A to B only if
 * A ended before B started *and* A outranks B, so the badge and the bar's right edge are the two
 * quantities the rule compares. A dashed outline marks an async-launched interval, which never
 * serializes a successor outside its own subtree.
 *
 * Presentational only — rank is derived from `sort(start, end, id)` unless a row overrides it.
 */
export function IntervalsNode({ data }: NodeProps<IntervalsNodeType>): React.JSX.Element {
  const { rows, title } = data;
  const layout = layoutIntervals({ rows, hasTitle: title !== undefined, width: data.width });
  const ranks = resolveRanks(rows);
  const { x, rowHeight } = layout;

  return (
    <div
      className={clsx(
        "rounded-[13px] border border-white/10 px-4 py-3.5",
        "shadow-[0_12px_28px_rgba(0,0,0,0.28)]",
        surfaceClassName(data.surfaceRole ?? "elevated"),
        data.className,
      )}
    >
      {title !== undefined && (
        <div className={`mb-1.5 text-sm font-semibold tracking-tight ${inkClassName("primary")}`}>
          {title}
        </div>
      )}
      {/* `block` drops the inline baseline strut: an inline <svg> reserves descender space
          below it, making the rendered box ~3px taller than the declared *NodeSize() height. */}
      <svg
        className="block"
        width={layout.svgWidth}
        height={layout.svgHeight}
        role="img"
        aria-label={data.ariaLabel ?? "intervals on the warped clock"}
      >
        {layout.gridTicks.map((t) => (
          <g key={`grid-${t}`}>
            <line
              x1={x(t)}
              y1={layout.gridTop}
              x2={x(t)}
              y2={layout.gridBottom}
              className={inkClassName("tertiary")}
              stroke="currentColor"
              strokeWidth={1}
              strokeOpacity={0.4}
            />
            <text
              x={x(t)}
              y={layout.gridBottom + 14}
              textAnchor="middle"
              fontSize={10}
              className={inkClassName("tertiary")}
              fill="currentColor"
            >
              {t}s
            </text>
          </g>
        ))}

        {rows.map((row, i) => {
          const y = layout.rowY(i);
          // Floors at 12px so a near-instant interval still shows a bar and a reachable badge.
          const w = Math.max((row.end - row.start) * layout.px, 12);
          const stroke = categoryStrokeClassName(row.role);
          return (
            <g key={row.id}>
              <text
                x={8}
                y={y + rowHeight / 2 + 4}
                fontSize={11}
                fontWeight={600}
                className={inkClassName("secondary")}
                fill="currentColor"
              >
                {row.id}
              </text>
              <rect
                x={x(row.start)}
                y={y}
                width={w}
                height={rowHeight}
                fill="var(--color-surface-panel)"
                className={stroke}
                strokeWidth={row.dashed === true ? 2 : 1.5}
                strokeDasharray={row.dashed === true ? "5 3" : undefined}
              />
              <text
                x={x(row.start) + 8}
                y={y + rowHeight / 2 + 4}
                fontSize={10.5}
                fontWeight={600}
                className={inkClassName("primary")}
                fill="currentColor"
              >
                {row.label}
              </text>
              <circle
                cx={x(row.end)}
                cy={y + rowHeight / 2}
                r={8}
                fill="var(--color-surface-page)"
                className={stroke}
                strokeWidth={1.5}
              />
              <text
                x={x(row.end)}
                y={y + rowHeight / 2 + 3.5}
                textAnchor="middle"
                fontSize={9}
                fontWeight={700}
                className={inkClassName("secondary")}
                fill="currentColor"
              >
                {ranks.get(row.id)}
              </text>
            </g>
          );
        })}
      </svg>
      <NodeAnchorHandles />
    </div>
  );
}
