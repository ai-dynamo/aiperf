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
import {
  RuntimeGraphEdge,
  type RuntimeGraphEdgeData,
} from "./graph-edges";

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
});
