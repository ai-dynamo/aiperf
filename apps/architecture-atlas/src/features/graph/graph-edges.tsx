// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  BaseEdge,
  EdgeLabelRenderer,
  MarkerType,
  getSmoothStepPath,
  type Edge,
  type EdgeProps,
} from "@xyflow/react";

import type { GraphEdge } from "../../domain/architecture";
import type { GraphPathState } from "./types";

export interface RuntimeGraphEdgeData extends Record<string, unknown> {
  edge: GraphEdge;
  onSelect(edgeId: string): void;
  pathState: GraphPathState;
}

export function RuntimeGraphEdge({
  data,
  id,
  markerEnd,
  sourcePosition,
  sourceX,
  sourceY,
  targetPosition,
  targetX,
  targetY,
}: EdgeProps<Edge<RuntimeGraphEdgeData>>) {
  if (!data) {
    return null;
  }

  const [path, labelX, labelY] = getSmoothStepPath({
    sourcePosition,
    sourceX,
    sourceY,
    targetPosition,
    targetX,
    targetY,
  });

  return (
    <>
      <BaseEdge id={id} markerEnd={markerEnd} path={path} />
      <EdgeLabelRenderer>
        <button
          data-graph-entity-id={id}
          data-graph-entity-trigger="true"
          data-path-state={data.pathState}
          data-testid={`graph-edge-${id}`}
          onClick={() => data.onSelect(id)}
          style={{
            position: "absolute",
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
          }}
          type="button"
        >
          <span>{id}</span>
          <span>{data.edge.protocol}</span>
          <span>{data.edge.channel}</span>
        </button>
      </EdgeLabelRenderer>
    </>
  );
}

export function edgeMarker() {
  return {
    color: "#76B900",
    height: 16,
    type: MarkerType.ArrowClosed,
    width: 16,
  } as const;
}
