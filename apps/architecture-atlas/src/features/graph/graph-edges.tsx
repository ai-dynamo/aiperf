// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  BaseEdge,
  EdgeLabelRenderer,
  MarkerType,
  getSmoothStepPath,
  useReactFlow,
  type Edge,
  type EdgeProps,
} from "@xyflow/react";
import type { CSSProperties } from "react";

import type { GraphEdge } from "../../domain/architecture";
import {
  EdgeWaypointControls,
  createWaypointPath,
  waypointLabelPosition,
  type EdgeWaypoint,
  type EdgeWaypointUpdate,
} from "./edge-waypoints";
import type {
  GraphFlavorClass,
  GraphPathState,
  GraphPulseChannelState,
  GraphPulseState,
} from "./types";

const flavorColors: Record<GraphFlavorClass, string> = {
  "compare-only": "#b691d4",
  "primary-only": "#7aa9d6",
  shared: "#94d340",
};

export interface RuntimeGraphEdgeData extends Record<string, unknown> {
  edge: GraphEdge;
  flavorClass: GraphFlavorClass;
  onSelect(edgeId: string): void;
  onWaypointsChange?(update: EdgeWaypointUpdate): void;
  onWaypointsReset?(edgeId: string): void;
  pathState: GraphPathState;
  pulseEdgeState?: {
    channelState: GraphPulseChannelState;
    phase: GraphPulseState;
    reducedMotion: boolean;
  };
  waypoints?: readonly EdgeWaypoint[];
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
    ]
      .filter((value): value is string => Boolean(value))
      .join(" "),
    style: {
      opacity: data.pathState === "default" ? 0.72 : 1,
      stroke: planned ? "#ef5350" : flavorColors[data.flavorClass],
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

function pulseFillColor(phase: GraphPulseState): string {
  if (phase === "active") {
    return "#26c6da";
  }
  if (phase === "completed") {
    return "#94d340";
  }
  return "transparent";
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
  const reactFlow = useReactFlow();
  if (!data) {
    return null;
  }

  const waypoints = data.waypoints ?? [];
  const hasWaypointOverrides = waypoints.length > 0;
  const [smoothPath, smoothLabelX, smoothLabelY] = getSmoothStepPath({
    sourcePosition,
    sourceX,
    sourceY,
    targetPosition,
    targetX,
    targetY,
  });
  const waypointLabel = waypointLabelPosition({
    points: waypoints,
    source: { x: sourceX, y: sourceY },
    target: { x: targetX, y: targetY },
  });
  const path = hasWaypointOverrides
    ? createWaypointPath({
        points: waypoints,
        source: { x: sourceX, y: sourceY },
        target: { x: targetX, y: targetY },
      })
    : smoothPath;
  const labelX = hasWaypointOverrides ? waypointLabel.x : smoothLabelX;
  const labelY = hasWaypointOverrides ? waypointLabel.y : smoothLabelY;
  const presentation = edgePathPresentation(data);
  const controlsVisible = data.pathState === "focused";
  const pulseEdgeState = data.pulseEdgeState ?? {
    channelState: "idle",
    phase: "idle",
    reducedMotion: false,
  };
  const pulseVisible = pulseEdgeState.phase !== "idle";
  const pulseTestId = `graph-edge-pulse-${id}`;
  const pulseRadius = pulseEdgeState.phase === "active" ? 4 : 3;
  const pulseAnimated =
    pulseEdgeState.phase === "active" && !pulseEdgeState.reducedMotion;
  const pulseMotion = pulseAnimated
    ? "animated"
    : pulseEdgeState.reducedMotion
      ? "reduced"
      : "static";

  return (
    <>
      <BaseEdge
        className={presentation.className}
        id={id}
        markerEnd={markerEnd}
        path={path}
        style={presentation.style}
      />
      {pulseVisible ? (
        <g
          data-channel-state={pulseEdgeState.channelState}
          data-motion={pulseMotion}
          data-pulse-phase={pulseEdgeState.phase}
          data-testid={pulseTestId}
        >
          {!pulseAnimated ? (
            <circle
              cx={labelX}
              cy={labelY}
              fill={pulseFillColor(pulseEdgeState.phase)}
              opacity={pulseEdgeState.phase === "active" ? 1 : 0.78}
              r={pulseRadius}
              stroke="#0f1215"
              strokeWidth={1}
            />
          ) : (
            <circle
              fill={pulseFillColor(pulseEdgeState.phase)}
              opacity={1}
              r={pulseRadius}
              stroke="#0f1215"
              strokeWidth={1}
            >
              <animateMotion
                dur="1.4s"
                path={path}
                repeatCount="indefinite"
                rotate="auto"
              />
            </circle>
          )}
        </g>
      ) : null}
      <EdgeLabelRenderer>
        <EdgeWaypointControls
          edgeId={id}
          onChange={data.onWaypointsChange ?? (() => undefined)}
          onReset={data.onWaypointsReset ?? (() => undefined)}
          points={waypoints}
          source={{ x: sourceX, y: sourceY }}
          target={{ x: targetX, y: targetY }}
          toFlowPosition={({ x, y }) => reactFlow.screenToFlowPosition({ x, y })}
          visible={controlsVisible}
        />
        <button
          aria-label={`${data.edge.protocol} ${data.edge.channel} ${data.flavorClass} ${data.edge.status.state} ${data.pathState} pulse:${pulseEdgeState.phase} channel:${pulseEdgeState.channelState}`}
          className="graph-edge-label"
          data-flavor-class={data.flavorClass}
          data-graph-entity-id={id}
          data-graph-entity-trigger="true"
          data-implementation-state={data.edge.status.state}
          data-path-state={data.pathState}
          data-pulse-channel-state={pulseEdgeState.channelState}
          data-pulse-phase={pulseEdgeState.phase}
          data-reduced-motion={String(pulseEdgeState.reducedMotion)}
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
    color: planned ? "#ef5350" : flavorColors[flavorClass],
    height: 16,
    type: MarkerType.ArrowClosed,
    width: 16,
  } as const;
}

export type { EdgeWaypoint } from "./edge-waypoints";
