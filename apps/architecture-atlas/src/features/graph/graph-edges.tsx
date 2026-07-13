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
import type { CSSProperties } from "react";

import type { GraphEdge } from "../../domain/architecture";
import type { GraphFlavorClass, GraphPathState } from "./types";

const flavorColors: Record<GraphFlavorClass, string> = {
  "compare-only": "#A78BFA",
  "primary-only": "#45C7F4",
  shared: "#76B900",
};

export interface RuntimeGraphEdgeData extends Record<string, unknown> {
  edge: GraphEdge;
  flavorClass: GraphFlavorClass;
  onSelect(edgeId: string): void;
  pathState: GraphPathState;
}

function edgePathPresentation(data: RuntimeGraphEdgeData): {
  className: string;
  style: CSSProperties;
} {
  const planned = data.edge.status.state === "planned";
  return {
    className: [
      "graph-edge-path",
      `graph-edge-path-${data.pathState}`,
      `graph-edge-flavor-${data.flavorClass}`,
      planned ? "graph-edge-planned" : "graph-edge-built",
    ].join(" "),
    style: {
      opacity: data.pathState === "default" ? 0.72 : 1,
      stroke: planned ? "#FF7A7A" : flavorColors[data.flavorClass],
      strokeDasharray: planned ? "8 6" : undefined,
      strokeWidth:
        data.pathState === "focused"
          ? 4
          : data.pathState === "upstream" || data.pathState === "downstream"
            ? 3
            : 2,
    },
  };
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
  const presentation = edgePathPresentation(data);

  return (
    <>
      <BaseEdge
        className={presentation.className}
        id={id}
        markerEnd={markerEnd}
        path={path}
        style={presentation.style}
      />
      <EdgeLabelRenderer>
        <button
          aria-label={`${data.edge.protocol} ${data.edge.channel} ${data.flavorClass} ${data.edge.status.state} ${data.pathState}`}
          data-flavor-class={data.flavorClass}
          data-graph-entity-id={id}
          data-graph-entity-trigger="true"
          data-implementation-state={data.edge.status.state}
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
          <span>{data.flavorClass}</span>
          <span>{data.edge.status.state}</span>
        </button>
      </EdgeLabelRenderer>
    </>
  );
}

export function edgeMarker(
  flavorClass: GraphFlavorClass,
  planned: boolean,
) {
  return {
    color: planned ? "#FF7A7A" : flavorColors[flavorClass],
    height: 16,
    type: MarkerType.ArrowClosed,
    width: 16,
  } as const;
}
