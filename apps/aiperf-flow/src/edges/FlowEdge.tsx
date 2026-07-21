/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { EdgeProps, Edge } from "@xyflow/react";
import { BaseEdge, getSmoothStepPath } from "@xyflow/react";

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

const SPEED_DURATIONS: Record<FlowEdgeSpeed, string> = {
  slow: "2.4s",
  normal: "1.2s",
  fast: "0.6s",
};

/**
 * Animated dashed connector signaling data/request flow along an edge.
 *
 * Renders a bezier path with a dashed `strokeDasharray` and a CSS keyframe
 * animation that continuously decrements `stroke-dashoffset`, giving the
 * appearance of dashes traveling from source to target. Respects
 * `prefers-reduced-motion` via a CSS media query that disables the animation.
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
  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 8,
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
              stroke-dashoffset: -16;
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
        <marker
          id={markerId}
          markerWidth={12}
          markerHeight={12}
          refX={9}
          refY={6}
          orient="auto-start-reverse"
        >
          <path d="M0,0 L12,6 L0,12 Z" fill={color} />
        </marker>
      </defs>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd ?? `url(#${markerId})`}
        className="flow-edge__path"
        stroke={color}
        strokeWidth={2.5}
        strokeDasharray="8 8"
        style={{ "--flow-edge-duration": duration } as React.CSSProperties}
      />
    </>
  );
}
