/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { EdgeProps, Edge } from "@xyflow/react";
import { BaseEdge, getBezierPath } from "@xyflow/react";

/** Animation speed presets, mapped to a CSS `animation-duration`. */
export type FlowEdgeSpeed = "slow" | "normal" | "fast";

export interface FlowEdgeData extends Record<string, unknown> {
  /** Stroke color, as any value valid in an SVG `stroke` attribute (hex, `var(--...)`, etc). */
  color?: string;
  /** Dash animation speed. Defaults to `"normal"`. */
  speed?: FlowEdgeSpeed;
}

export type FlowEdgeType = Edge<FlowEdgeData, "flow">;

const DEFAULT_COLOR = "var(--color-accent-primary)";

/**
 * Dash geometry. The animation must travel exactly one dash period per iteration —
 * animating any other distance leaves the pattern out of phase at the loop point, and
 * the snap back to offset 0 reads as a stutter rather than continuous travel.
 */
const DASH_LENGTH = 7;
const DASH_GAP = 7;
const DASH_PERIOD = DASH_LENGTH + DASH_GAP;

const SPEED_DURATIONS: Record<FlowEdgeSpeed, string> = {
  slow: "2.4s",
  normal: "1.2s",
  fast: "0.6s",
};

/**
 * Animated dashed connector signaling data/request flow along an edge.
 *
 * Renders a continuous bezier curve (research: continuous curved paths are followed more easily
 * than polylines, and fewer/shallower bends improve path-tracing) with a dashed `strokeDasharray`
 * and a CSS keyframe animation that continuously decrements `stroke-dashoffset`, giving the
 * appearance of dashes traveling from source to target. Respects `prefers-reduced-motion`.
 */
export function FlowEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  data,
}: EdgeProps<FlowEdgeType>): React.JSX.Element {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const color = data?.color ?? DEFAULT_COLOR;
  const duration = SPEED_DURATIONS[data?.speed ?? "normal"];
  const markerId = `flow-edge-arrow-${id}`;

  return (
    <>
      <style>
        {`
          @keyframes flow-edge-dash {
            to {
              stroke-dashoffset: -${DASH_PERIOD};
            }
          }
          .flow-edge__path {
            animation: flow-edge-dash var(--flow-edge-duration, 1.2s) linear infinite;
          }
          @media (prefers-reduced-motion: reduce) {
            .flow-edge__path {
              animation: none;
            }
          }
        `}
      </style>
      <defs>
        {/*
          `markerUnits="userSpaceOnUse"` pins the arrowhead to these dimensions. The SVG
          default is `strokeWidth`, which multiplies every number here by the 2px stroke
          and renders an arrowhead twice the intended size.
        */}
        <marker
          id={markerId}
          markerUnits="userSpaceOnUse"
          markerWidth={9}
          markerHeight={9}
          refX={8}
          refY={4.5}
          orient="auto-start-reverse"
        >
          <path d="M0,0.75 L9,4.5 L0,8.25 Z" fill={color} />
        </marker>
      </defs>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd ?? `url(#${markerId})`}
        className="flow-edge__path"
        // Stroke has to be inline style, not SVG presentation attributes: React Flow's
        // stylesheet sets `.react-flow__edge-path { stroke; stroke-width }`, and any CSS
        // rule outranks a presentation attribute. Passing them as attributes leaves the
        // edge rendering in React Flow's default gray at its default width, while the
        // attribute still reads back as the requested color.
        style={
          {
            "--flow-edge-duration": duration,
            stroke: color,
            strokeWidth: 2,
            strokeDasharray: `${DASH_LENGTH} ${DASH_GAP}`,
          } as React.CSSProperties
        }
      />
    </>
  );
}
