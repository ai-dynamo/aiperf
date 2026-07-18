// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useRef, type PointerEvent as ReactPointerEvent } from "react";

export interface EdgeWaypoint {
  x: number;
  y: number;
}

export interface EdgeWaypointUpdate {
  edgeId: string;
  points: EdgeWaypoint[];
}

export interface EdgeWaypointAnchor {
  x: number;
  y: number;
}

export interface AppendWaypointInput {
  edgeId: string;
  points: readonly EdgeWaypoint[];
  source: EdgeWaypointAnchor;
  target: EdgeWaypointAnchor;
}

export interface EdgeWaypointControlsProps {
  edgeId: string;
  points: readonly EdgeWaypoint[];
  source: EdgeWaypointAnchor;
  target: EdgeWaypointAnchor;
  visible: boolean;
  toFlowPosition?(point: { x: number; y: number }): EdgeWaypoint;
  onChange(update: EdgeWaypointUpdate): void;
  onReset(edgeId: string): void;
}

const KEYBOARD_STEP = 12;

function midpoint(source: EdgeWaypointAnchor, target: EdgeWaypointAnchor): EdgeWaypoint {
  return {
    x: Math.round((source.x + target.x) / 2),
    y: Math.round((source.y + target.y) / 2),
  };
}

export function createWaypointPath(input: {
  source: EdgeWaypointAnchor;
  points: readonly EdgeWaypoint[];
  target: EdgeWaypointAnchor;
}): string {
  const nodes = [input.source, ...input.points, input.target];
  const [first, ...rest] = nodes;
  const segments = rest.map((point) => `L ${point.x} ${point.y}`);
  return [`M ${first.x} ${first.y}`, ...segments].join(" ");
}

export function waypointLabelPosition(input: {
  source: EdgeWaypointAnchor;
  points: readonly EdgeWaypoint[];
  target: EdgeWaypointAnchor;
}): EdgeWaypointAnchor {
  const nodes = [input.source, ...input.points, input.target];
  const midpointIndex = Math.floor((nodes.length - 1) / 2);
  const start = nodes[midpointIndex];
  const end = nodes[Math.min(midpointIndex + 1, nodes.length - 1)];
  return {
    x: Math.round((start.x + end.x) / 2),
    y: Math.round((start.y + end.y) / 2),
  };
}

export function appendWaypoint(input: AppendWaypointInput): EdgeWaypointUpdate {
  return {
    edgeId: input.edgeId,
    points: [...input.points, midpoint(input.source, input.target)],
  };
}

export function removeWaypointByIndex(
  points: readonly EdgeWaypoint[],
  index: number,
): EdgeWaypoint[] {
  return points.filter((_, pointIndex) => pointIndex !== index);
}

function moveWaypointByIndex(
  points: readonly EdgeWaypoint[],
  index: number,
  deltaX: number,
  deltaY: number,
): EdgeWaypoint[] {
  return points.map((point, pointIndex) =>
    pointIndex === index ? { x: point.x + deltaX, y: point.y + deltaY } : point
  );
}

export function EdgeWaypointControls({
  edgeId,
  points,
  source,
  target,
  visible,
  toFlowPosition,
  onChange,
  onReset,
}: EdgeWaypointControlsProps) {
  const dragState = useRef<{
    index: number;
    pointerId: number;
    x: number;
    y: number;
  } | null>(null);

  if (!visible) {
    return null;
  }

  const emitPoints = (nextPoints: EdgeWaypoint[]) => {
    onChange({
      edgeId,
      points: nextPoints,
    });
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const activeDrag = dragState.current;
    if (!activeDrag || activeDrag.pointerId !== event.pointerId) {
      return;
    }
    const flowPoint = toFlowPosition
      ? toFlowPosition({ x: event.clientX, y: event.clientY })
      : { x: Math.round(event.clientX), y: Math.round(event.clientY) };
    const previousPoint = points[activeDrag.index];
    if (
      !previousPoint ||
      (flowPoint.x === previousPoint.x && flowPoint.y === previousPoint.y)
    ) {
      return;
    }
    activeDrag.x = event.clientX;
    activeDrag.y = event.clientY;
    emitPoints(
      points.map((point, pointIndex) =>
        pointIndex === activeDrag.index ? flowPoint : point
      ),
    );
  };

  const finishPointerDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (dragState.current?.pointerId !== event.pointerId) {
      return;
    }
    dragState.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  return (
    <div aria-label="Edge waypoint controls" style={{ pointerEvents: "all" }}>
      <button
        aria-label="Add waypoint"
        data-testid={`graph-edge-waypoint-add-${edgeId}`}
        onClick={() => emitPoints(appendWaypoint({ edgeId, points, source, target }).points)}
        style={{
          position: "absolute",
          zIndex: 1000,
          transform: `translate(-50%, -50%) translate(${(source.x + target.x) / 2}px, ${Math.min(
            source.y,
            target.y,
          ) - 28}px)`,
        }}
        type="button"
      >
        +
      </button>
      <button
        aria-label="Reset waypoints"
        data-testid={`graph-edge-waypoint-reset-${edgeId}`}
        onClick={() => onReset(edgeId)}
        style={{
          position: "absolute",
          zIndex: 1000,
          transform: `translate(-50%, -50%) translate(${(source.x + target.x) / 2 + 36}px, ${Math.min(
            source.y,
            target.y,
          ) - 28}px)`,
        }}
        type="button"
      >
        reset
      </button>
      {points.map((point, index) => (
        <button
          aria-label={`Move waypoint ${index + 1}`}
          data-testid={`graph-edge-waypoint-handle-${edgeId}-${index}`}
          key={`${edgeId}-waypoint-${index}`}
          onKeyDown={(event) => {
            if (event.key === "Delete" || event.key === "Backspace") {
              emitPoints(removeWaypointByIndex(points, index));
              return;
            }
            if (!event.key.startsWith("Arrow")) {
              return;
            }
            event.preventDefault();
            const step = event.shiftKey ? 1 : KEYBOARD_STEP;
            const deltaX = event.key === "ArrowRight" ? step : event.key === "ArrowLeft" ? -step : 0;
            const deltaY = event.key === "ArrowDown" ? step : event.key === "ArrowUp" ? -step : 0;
            emitPoints(moveWaypointByIndex(points, index, deltaX, deltaY));
          }}
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            dragState.current = {
              index,
              pointerId: event.pointerId,
              x: event.clientX,
              y: event.clientY,
            };
          }}
          onPointerCancel={finishPointerDrag}
          onPointerMove={handlePointerMove}
          onPointerUp={finishPointerDrag}
          style={{
            position: "absolute",
            zIndex: 1000,
            transform: `translate(-50%, -50%) translate(${point.x}px, ${point.y}px)`,
          }}
          type="button"
        >
          {index + 1}
        </button>
      ))}
      {points.map((point, index) => (
        <button
          aria-label={`Remove waypoint ${index + 1}`}
          data-testid={`graph-edge-waypoint-remove-${edgeId}-${index}`}
          key={`${edgeId}-waypoint-remove-${index}`}
          onClick={() => emitPoints(removeWaypointByIndex(points, index))}
          style={{
            position: "absolute",
            zIndex: 1000,
            transform: `translate(-50%, -50%) translate(${point.x + 18}px, ${point.y - 16}px)`,
          }}
          type="button"
        >
          -
        </button>
      ))}
    </div>
  );
}
