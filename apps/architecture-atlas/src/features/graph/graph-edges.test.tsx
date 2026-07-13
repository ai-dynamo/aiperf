// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render } from "@testing-library/react";
import {
  Position,
  ReactFlowProvider,
  type Edge,
  type EdgeProps,
} from "@xyflow/react";
import { describe, expect, it, vi } from "vitest";

import type { GraphEdge } from "../../domain/architecture";
import { RuntimeGraphEdge, type EdgeWaypoint, type RuntimeGraphEdgeData } from "./graph-edges";

const edge: GraphEdge = {
  channel: "request_data",
  evidence: [{ path: "AGENTS.md" }],
  flavors: ["native_http"],
  footnotes: [],
  id: "edge.runner.transport",
  protocol: "http",
  source: { nodeId: "node.runner", portId: "port.runner.out" },
  status: { delivery: "unconditional", state: "planned" },
  target: { nodeId: "node.transport", portId: "port.transport.in" },
};

describe("runtime graph edge", () => {
  it("styles the BaseEdge path with path, overlay, and planned classifications", () => {
    const props = {
      data: {
        edge,
        flavorClass: "compare-only",
        onSelect: vi.fn(),
        pathState: "downstream",
      },
      id: edge.id,
      markerEnd: "marker",
      sourcePosition: Position.Right,
      sourceX: 0,
      sourceY: 0,
      targetPosition: Position.Left,
      targetX: 200,
      targetY: 0,
    } as unknown as EdgeProps<Edge<RuntimeGraphEdgeData>>;

    const { container } = render(
      <ReactFlowProvider>
        <svg>
          <RuntimeGraphEdge {...props} />
        </svg>
      </ReactFlowProvider>,
    );
    const path = container.querySelector("path");

    expect(path).toHaveClass(
      "graph-edge-path",
      "graph-edge-path-downstream",
      "graph-edge-flavor-compare-only",
      "graph-edge-planned",
    );
    expect(path).toHaveStyle({
      strokeDasharray: "8 6",
      strokeWidth: "3",
    });
  });

  it("renders focused edge path through waypoint overrides", () => {
    const waypoints: EdgeWaypoint[] = [{ x: 80, y: 24 }];
    const props = {
      data: {
        edge,
        flavorClass: "shared",
        onSelect: vi.fn(),
        onWaypointsChange: vi.fn(),
        onWaypointsReset: vi.fn(),
        pathState: "focused",
        waypoints,
      },
      id: edge.id,
      markerEnd: "marker",
      sourcePosition: Position.Right,
      sourceX: 0,
      sourceY: 0,
      targetPosition: Position.Left,
      targetX: 200,
      targetY: 0,
    } as unknown as EdgeProps<Edge<RuntimeGraphEdgeData>>;

    const { container } = render(
      <ReactFlowProvider>
        <svg>
          <RuntimeGraphEdge {...props} />
        </svg>
      </ReactFlowProvider>,
    );
    const path = container.querySelector("path");
    expect(path).toHaveAttribute("d", "M 0 0 L 80 24 L 200 0");
    expect(path).toHaveClass("graph-edge-path-focused", "graph-edge-planned");
  });

  it("animates a pulse particle on the resolved edge path", () => {
    const props = {
      data: {
        edge,
        flavorClass: "shared",
        onSelect: vi.fn(),
        pathState: "focused",
        pulseEdgeState: {
          channelState: "active",
          phase: "active",
          reducedMotion: false,
        },
        waypoints: [{ x: 80, y: 24 }],
      },
      id: edge.id,
      markerEnd: "marker",
      sourcePosition: Position.Right,
      sourceX: 0,
      sourceY: 0,
      targetPosition: Position.Left,
      targetX: 200,
      targetY: 0,
    } as unknown as EdgeProps<Edge<RuntimeGraphEdgeData>>;

    const { container } = render(
      <ReactFlowProvider>
        <svg>
          <RuntimeGraphEdge {...props} />
        </svg>
      </ReactFlowProvider>,
    );
    const pulseParticle = container.querySelector(
      `[data-testid="graph-edge-pulse-${edge.id}"]`,
    );
    const edgePath = container.querySelector("path");
    const animateMotion = container.querySelector("animateMotion");

    expect(edgePath).toBeInTheDocument();
    expect(pulseParticle).toBeInTheDocument();
    expect(pulseParticle).toHaveAttribute("data-pulse-phase", "active");
    expect(pulseParticle).toHaveAttribute("data-channel-state", "active");
    expect(pulseParticle).toHaveAttribute("data-motion", "animated");
    expect(edgePath).toHaveAttribute("d", "M 0 0 L 80 24 L 200 0");
    expect(animateMotion).toHaveAttribute("path", edgePath?.getAttribute("d") ?? "");
  });

  it("does not animate a channel peer when an exact edge reference is active", () => {
    const props = {
      data: {
        edge,
        flavorClass: "shared",
        onSelect: vi.fn(),
        pathState: "default",
        pulseEdgeState: {
          channelState: "active",
          phase: "idle",
          reducedMotion: false,
        },
      },
      id: edge.id,
      markerEnd: "marker",
      sourcePosition: Position.Right,
      sourceX: 0,
      sourceY: 0,
      targetPosition: Position.Left,
      targetX: 200,
      targetY: 0,
    } as unknown as EdgeProps<Edge<RuntimeGraphEdgeData>>;

    const { container } = render(
      <ReactFlowProvider>
        <svg>
          <RuntimeGraphEdge {...props} />
        </svg>
      </ReactFlowProvider>,
    );

    expect(
      container.querySelector(`[data-testid="graph-edge-pulse-${edge.id}"]`),
    ).not.toBeInTheDocument();
    expect(container.querySelector("animateMotion")).not.toBeInTheDocument();
  });

  it("renders completed normal-motion state as a static marker", () => {
    const props = {
      data: {
        edge,
        flavorClass: "shared",
        onSelect: vi.fn(),
        pathState: "default",
        pulseEdgeState: {
          channelState: "completed",
          phase: "completed",
          reducedMotion: false,
        },
      },
      id: edge.id,
      markerEnd: "marker",
      sourcePosition: Position.Right,
      sourceX: 0,
      sourceY: 0,
      targetPosition: Position.Left,
      targetX: 200,
      targetY: 0,
    } as unknown as EdgeProps<Edge<RuntimeGraphEdgeData>>;

    const { container } = render(
      <ReactFlowProvider>
        <svg>
          <RuntimeGraphEdge {...props} />
        </svg>
      </ReactFlowProvider>,
    );
    const pulseMarker = container.querySelector(
      `[data-testid="graph-edge-pulse-${edge.id}"]`,
    );
    const circle = pulseMarker?.querySelector("circle");

    expect(pulseMarker).toHaveAttribute("data-pulse-phase", "completed");
    expect(pulseMarker).toHaveAttribute("data-channel-state", "completed");
    expect(pulseMarker).toHaveAttribute("data-motion", "static");
    expect(circle).toHaveAttribute("cx", "100");
    expect(circle).toHaveAttribute("cy", "0");
    expect(circle).toHaveAttribute("fill", "#94d340");
    expect(pulseMarker?.querySelector("animateMotion")).not.toBeInTheDocument();
  });

  it("keeps pulse semantics while disabling movement for reduced motion", () => {
    const builtReplayEdge: GraphEdge = {
      ...edge,
      id: "edge.dynamo.online.replay",
      status: { delivery: "runtime_conditional", state: "built" },
    };
    const props = {
      data: {
        edge: builtReplayEdge,
        flavorClass: "compare-only",
        onSelect: vi.fn(),
        pathState: "default",
        pulseEdgeState: {
          channelState: "completed",
          phase: "completed",
          reducedMotion: true,
        },
      },
      id: builtReplayEdge.id,
      markerEnd: "marker",
      sourcePosition: Position.Right,
      sourceX: 0,
      sourceY: 0,
      targetPosition: Position.Left,
      targetX: 200,
      targetY: 0,
    } as unknown as EdgeProps<Edge<RuntimeGraphEdgeData>>;

    const { container } = render(
      <ReactFlowProvider>
        <svg>
          <RuntimeGraphEdge {...props} />
        </svg>
      </ReactFlowProvider>,
    );
    const path = container.querySelector("path");
    const pulseParticle = container.querySelector(
      `[data-testid="graph-edge-pulse-${builtReplayEdge.id}"]`,
    );
    const animateMotion = container.querySelector("animateMotion");

    expect(path).toHaveClass("graph-edge-built");
    expect(path).not.toHaveClass("graph-edge-planned", "graph-edge-dynamo-online");
    expect(path).toHaveStyle({ strokeDasharray: "" });
    expect(pulseParticle).toHaveAttribute("data-pulse-phase", "completed");
    expect(pulseParticle).toHaveAttribute("data-channel-state", "completed");
    expect(pulseParticle).toHaveAttribute("data-motion", "reduced");
    expect(animateMotion).not.toBeInTheDocument();
  });
});
