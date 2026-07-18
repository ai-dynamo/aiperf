// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { GraphEdge, GraphNode } from "../../domain/architecture";
import type {
  DirectedNeighborhood,
  FlavorOverlay,
} from "../../domain/graph-derivation";
import type { LayoutRequest, LayoutResult } from "../atlas/layout";
import {
  completeNodeDrag,
  fitGraphView,
  GraphCanvas,
  resolveCanvasPulseEdgeState,
} from "./graph-canvas";
import type { GraphCanvasLayoutService } from "./types";

function createNode(input: {
  id: string;
  owner: GraphNode["owner"];
  parentId?: string | null;
  ports: Array<{ id: string; name: string; channel: GraphNode["seamPorts"][number]["channel"] }>;
  tier: GraphNode["tier"];
  title: string;
  state?: GraphNode["status"]["state"];
  childIds?: string[];
}): GraphNode {
  return {
    audience: {
      autoExpandDepth: { developer: 2, executive: 1, maintainer: 3 },
      visibility: ["executive", "developer", "maintainer"],
    },
    childIds: input.childIds ?? [],
    evidence: [{ path: "AGENTS.md" }],
    flavors: ["native_http"],
    footnotes: [],
    id: input.id,
    owner: input.owner,
    parentId: input.parentId ?? null,
    seamPorts: input.ports,
    status: { delivery: "unconditional", state: input.state ?? "built" },
    summary: {
      developer: `${input.title} summary`,
      executive: `${input.title} summary`,
      maintainer: `${input.title} summary`,
    },
    tier: input.tier,
    title: {
      developer: input.title,
      executive: input.title,
      maintainer: input.title,
    },
  };
}

function createEdge(input: {
  channel: GraphEdge["channel"];
  id: string;
  protocol: string;
  sourceNodeId: string;
  sourcePortId: string;
  targetNodeId: string;
  targetPortId: string;
}): GraphEdge {
  return {
    channel: input.channel,
    evidence: [{ path: "AGENTS.md" }],
    flavors: ["native_http"],
    footnotes: [],
    id: input.id,
    protocol: input.protocol,
    source: { nodeId: input.sourceNodeId, portId: input.sourcePortId },
    status: { delivery: "unconditional", state: "built" },
    target: { nodeId: input.targetNodeId, portId: input.targetPortId },
  };
}

const visibleNodes: GraphNode[] = [
  createNode({
    id: "node.python",
    owner: "python",
    ports: [{ channel: "control", id: "port.python.out", name: "Config out" }],
    tier: 0,
    title: "Python control",
  }),
  createNode({
    childIds: ["node.transport"],
    id: "node.runner",
    owner: "rust",
    ports: [
      { channel: "control", id: "port.runner.in", name: "Runner in" },
      { channel: "request_data", id: "port.runner.out", name: "Dispatch out" },
    ],
    tier: 1,
    title: "Runner core",
  }),
  createNode({
    id: "node.transport",
    owner: "rust",
    ports: [{ channel: "request_data", id: "port.transport.in", name: "Transport in" }],
    tier: 2,
    title: "Transport sink",
  }),
];

const visibleEdges: GraphEdge[] = [
  createEdge({
    channel: "control",
    id: "edge.python.runner",
    protocol: "jsonl",
    sourceNodeId: "node.python",
    sourcePortId: "port.python.out",
    targetNodeId: "node.runner",
    targetPortId: "port.runner.in",
  }),
  createEdge({
    channel: "request_data",
    id: "edge.runner.transport",
    protocol: "http",
    sourceNodeId: "node.runner",
    sourcePortId: "port.runner.out",
    targetNodeId: "node.transport",
    targetPortId: "port.transport.in",
  }),
];

const neighborhood: DirectedNeighborhood = {
  downstreamNodeIds: ["node.transport"],
  upstreamNodeIds: ["node.python"],
};

const overlay: FlavorOverlay = {
  compareOnlyEdgeIds: ["edge.runner.transport"],
  compareOnlyNodeIds: ["node.transport"],
  primaryOnlyEdgeIds: [],
  primaryOnlyNodeIds: [],
  sharedEdgeIds: ["edge.python.runner"],
  sharedNodeIds: ["node.python", "node.runner"],
};

const layoutResult: LayoutResult = {
  bands: [],
  degraded: false,
  positions: [
    { bandId: "flow", id: "node.python", x: 20, y: 80 },
    { bandId: "flow", id: "node.runner", x: 320, y: 80 },
    { bandId: "flow", id: "node.transport", x: 620, y: 80 },
  ],
};

function createLayoutRequest(): LayoutRequest {
  return {
    bands: [],
    edges: [],
    key: "graph.canvas.test.layout",
    nodes: [],
    perspective: "ownership",
    version: 1,
  };
}

function createLayoutService(result: LayoutResult): GraphCanvasLayoutService {
  return {
    layout: vi.fn(async () => result),
  };
}

describe("graph canvas", () => {
  it("renders task-2 derived nodes and edges with graph controls", async () => {
    const onFocusEntity = vi.fn();
    render(
      <GraphCanvas
        audience="developer"
        focusedEntityId={null}
        layoutRequest={createLayoutRequest()}
        layoutService={createLayoutService(layoutResult)}
        neighborhood={neighborhood}
        onFocusEntity={onFocusEntity}
        overlay={overlay}
        visibleEdges={visibleEdges}
        visibleNodes={visibleNodes}
      />,
    );

    await screen.findByTestId("graph-node-node.python");
    expect(within(screen.getByTestId("graph-node-node.python")).getByText("Python control")).toBeInTheDocument();
    expect(within(screen.getByTestId("graph-node-node.runner")).getByText("Runner core")).toBeInTheDocument();
    expect(within(screen.getByTestId("graph-node-node.transport")).getByText("Transport sink")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Graph viewport controls" })).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Graph canvas minimap" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("status", { name: "Graph layout status" }),
    ).toHaveClass("canvas-status-chip");
    const runnerPorts = within(screen.getByTestId("graph-node-node.runner")).getAllByRole(
      "listitem",
      { hidden: true },
    );
    expect(runnerPorts.map((item) => item.textContent)).toEqual([
      "Runner in - control - target",
      "Dispatch out - request_data - source",
    ]);
    expect(onFocusEntity).not.toHaveBeenCalled();
  });

  it("supports selection and directed path highlighting", async () => {
    const onFocusEntity = vi.fn();
    render(
      <GraphCanvas
        audience="developer"
        focusedEntityId="node.runner"
        layoutRequest={createLayoutRequest()}
        layoutService={createLayoutService(layoutResult)}
        neighborhood={neighborhood}
        onFocusEntity={onFocusEntity}
        overlay={overlay}
        visibleEdges={visibleEdges}
        visibleNodes={visibleNodes}
      />,
    );

    await screen.findByTestId("graph-node-node.runner");
    expect(screen.getByTestId("graph-node-node.runner")).toHaveAttribute(
      "data-path-state",
      "focused",
    );
    expect(screen.getByTestId("graph-node-node.python")).toHaveAttribute(
      "data-path-state",
      "upstream",
    );
    expect(screen.getByTestId("graph-node-node.transport")).toHaveAttribute(
      "data-path-state",
      "downstream",
    );
    fireEvent.click(
      within(screen.getByTestId("graph-node-node.runner")).getByText("Runner core"),
    );
    expect(onFocusEntity).toHaveBeenCalledWith("node.runner");
  });

  it("threads flavor overlay and planned classifications into graph entities", async () => {
    const plannedNodes = visibleNodes.map((node) =>
      node.id === "node.transport"
        ? { ...node, status: { ...node.status, state: "planned" as const } }
        : node,
    );

    render(
      <GraphCanvas
        audience="developer"
        focusedEntityId={null}
        layoutRequest={createLayoutRequest()}
        layoutService={createLayoutService(layoutResult)}
        neighborhood={{ downstreamNodeIds: [], upstreamNodeIds: [] }}
        onFocusEntity={vi.fn()}
        overlay={overlay}
        visibleEdges={visibleEdges}
        visibleNodes={plannedNodes}
      />,
    );

    await screen.findByTestId("graph-node-node.runner");
    expect(screen.getByTestId("graph-node-node.runner")).toHaveAttribute(
      "data-flavor-class",
      "shared",
    );
    expect(screen.getByTestId("graph-node-node.transport")).toHaveAttribute(
      "data-flavor-class",
      "compare-only",
    );
    expect(screen.getByTestId("graph-node-node.transport")).toHaveAttribute(
      "data-implementation-state",
      "planned",
    );
    expect(
      within(screen.getByTestId("graph-node-node.transport")).getByText("planned"),
    ).toBeInTheDocument();
  });

  it("executes a typed fit-view command through the React Flow API", async () => {
    const fitView = vi.fn(async () => true);

    await fitGraphView(
      { fitView },
      ["node.python", "node.runner", "node.transport"],
    );

    expect(fitView).toHaveBeenCalledWith({
      nodes: [
        { id: "node.python" },
        { id: "node.runner" },
        { id: "node.transport" },
      ],
      padding: 0.14,
    });
  });

  it("shows loading then a degraded fallback layout notice", async () => {
    let resolveLayout: ((result: LayoutResult) => void) | undefined;
    const layoutPromise = new Promise<LayoutResult>((resolve) => {
      resolveLayout = resolve;
    });
    const service: GraphCanvasLayoutService = {
      layout: vi.fn(async () => layoutPromise),
    };

    render(
      <GraphCanvas
        audience="developer"
        focusedEntityId={null}
        layoutRequest={createLayoutRequest()}
        layoutService={service}
        neighborhood={{ downstreamNodeIds: [], upstreamNodeIds: [] }}
        onFocusEntity={vi.fn()}
        overlay={overlay}
        visibleEdges={visibleEdges}
        visibleNodes={visibleNodes}
      />,
    );

    expect(
      screen.getByRole("status", { name: "Graph layout status" }),
    ).toHaveTextContent("Positioning graph layout");

    resolveLayout?.({
      ...layoutResult,
      degraded: true,
      reason: "worker unavailable",
    });

    await waitFor(() =>
      expect(
        screen.getByRole("status", { name: "Graph layout status" }),
      ).toHaveTextContent("degraded"),
    );
    expect(screen.getByRole("status", { name: "Graph layout status" })).toHaveTextContent(
      "worker unavailable",
    );
  });

  it("falls back deterministically when layout service rejects", async () => {
    const service: GraphCanvasLayoutService = {
      layout: vi.fn(async () => {
        throw new Error("worker protocol mismatch");
      }),
    };
    render(
      <GraphCanvas
        audience="developer"
        focusedEntityId={null}
        layoutRequest={createLayoutRequest()}
        layoutService={service}
        neighborhood={{ downstreamNodeIds: [], upstreamNodeIds: [] }}
        onFocusEntity={vi.fn()}
        overlay={overlay}
        visibleEdges={visibleEdges}
        visibleNodes={visibleNodes}
      />,
    );

    await waitFor(() =>
      expect(screen.getByRole("status", { name: "Graph layout status" })).toHaveTextContent(
        "Graph layout degraded",
      ),
    );
    expect(screen.getByRole("status", { name: "Graph layout status" })).toHaveTextContent(
      "worker protocol mismatch",
    );
    expect(await screen.findByTestId("graph-node-node.runner")).toBeInTheDocument();
  });

  it("enables dragging and emits a typed manual position on completion", async () => {
    const onNodeDragComplete = vi.fn();
    render(
      <GraphCanvas
        audience="developer"
        focusedEntityId={null}
        layoutRequest={createLayoutRequest()}
        layoutService={createLayoutService(layoutResult)}
        neighborhood={{ downstreamNodeIds: [], upstreamNodeIds: [] }}
        onFocusEntity={vi.fn()}
        onNodeDragComplete={onNodeDragComplete}
        overlay={overlay}
        visibleEdges={visibleEdges}
        visibleNodes={visibleNodes}
      />,
    );

    await screen.findByTestId("graph-node-node.runner");
    expect(screen.getByTestId("rf__node-node.runner")).toHaveClass("draggable");

    completeNodeDrag(
      { id: "node.runner", position: { x: 412, y: 144 } },
      onNodeDragComplete,
    );
    expect(onNodeDragComplete).toHaveBeenCalledWith({
      nodeId: "node.runner",
      x: 412,
      y: 144,
    });
  });

  it("exposes keyboard-operable expansion and trace controls inside nodes", async () => {
    const user = userEvent.setup();
    const onCollapseNode = vi.fn();
    const onTraceModeChange = vi.fn();
    render(
      <GraphCanvas
        audience="developer"
        expandedNodeIds={["node.runner"]}
        focusedEntityId="node.runner"
        layoutRequest={createLayoutRequest()}
        layoutService={createLayoutService(layoutResult)}
        neighborhood={neighborhood}
        onCollapseNode={onCollapseNode}
        onFocusEntity={vi.fn()}
        onTraceModeChange={onTraceModeChange}
        overlay={overlay}
        traceMode="upstream"
        visibleEdges={visibleEdges}
        visibleNodes={visibleNodes}
      />,
    );

    await screen.findByTestId("graph-node-node.runner");
    const runner = within(screen.getByTestId("graph-node-node.runner"));
    const collapse = runner.getByLabelText("Collapse Runner core");
    const downstream = runner.getByLabelText(
      "Trace downstream from Runner core",
    );

    collapse.focus();
    await user.keyboard("{Enter}");
    downstream.focus();
    await user.keyboard("{Enter}");

    expect(onCollapseNode).toHaveBeenCalledWith("node.runner");
    expect(onTraceModeChange).toHaveBeenCalledWith(
      "node.runner",
      "downstream",
    );
  });

  it("renders breadcrumb focus context with typed focus callbacks", async () => {
    const onFocusBreadcrumb = vi.fn();
    render(
      <GraphCanvas
        audience="developer"
        breadcrumbNodeIds={["node.python", "node.runner"]}
        focusedEntityId="node.runner"
        layoutRequest={createLayoutRequest()}
        layoutService={createLayoutService(layoutResult)}
        neighborhood={neighborhood}
        onFocusBreadcrumb={onFocusBreadcrumb}
        onFocusEntity={vi.fn()}
        overlay={overlay}
        visibleEdges={visibleEdges}
        visibleNodes={visibleNodes}
      />,
    );

    const breadcrumbs = await screen.findByRole("navigation", {
      name: "Graph focus context",
    });
    const python = within(breadcrumbs).getByRole("button", {
      name: "Python control",
    });
    expect(
      within(breadcrumbs).getByRole("button", { name: "Runner core" }),
    ).toHaveAttribute("aria-current", "location");

    fireEvent.click(python);
    expect(onFocusBreadcrumb).toHaveBeenCalledWith("node.python");
  });

  it("preserves unaffected positions and exposes relayout and pulse states", async () => {
    const partialLayout: LayoutResult = {
      ...layoutResult,
      partialRelayout: {
        preservedManualNodeIds: ["node.python"],
        relaidOutNodeIds: ["node.runner"],
      },
      positions: layoutResult.positions.map((position) =>
        position.id === "node.python"
          ? { ...position, x: 777, y: 555 }
          : position,
      ),
    };
    render(
      <GraphCanvas
        activePulseNodeIds={["node.runner"]}
        audience="developer"
        completedPulseNodeIds={["node.python"]}
        focusedEntityId="node.runner"
        layoutRequest={createLayoutRequest()}
        layoutService={createLayoutService(partialLayout)}
        neighborhood={neighborhood}
        onFocusEntity={vi.fn()}
        overlay={overlay}
        visibleEdges={visibleEdges}
        visibleNodes={visibleNodes}
      />,
    );

    await screen.findByTestId("graph-node-node.runner");
    expect(screen.getByTestId("graph-node-node.python")).toHaveAttribute(
      "data-relayout-state",
      "preserved",
    );
    expect(screen.getByTestId("graph-node-node.python")).toHaveAttribute(
      "data-pulse-state",
      "completed",
    );
    expect(screen.getByTestId("graph-node-node.runner")).toHaveAttribute(
      "data-relayout-state",
      "relaid-out",
    );
    expect(screen.getByTestId("graph-node-node.runner")).toHaveAttribute(
      "data-pulse-state",
      "active",
    );
    expect(screen.getByTestId("graph-node-node.runner")).toHaveClass(
      "graph-node",
      "graph-node-tier-1",
      "graph-node-flavor-shared",
      "graph-node-path-focused",
      "graph-node-built",
      "graph-node-pulse-active",
      "graph-node-relayout-relaid-out",
    );
    expect(screen.getByTestId("rf__node-node.python")).toHaveStyle({
      transform: "translate(777px,555px)",
    });
  });

  it("threads typed pulse edge ids and channels into runtime edge data", () => {
    const pulseEdges = {
      activeChannels: ["request_data"] as const,
      activeEdgeIds: ["edge.runner.transport"],
      completedChannels: ["control"] as const,
      completedEdgeIds: ["edge.python.runner"],
      reducedMotion: true,
    };

    expect(resolveCanvasPulseEdgeState(visibleEdges[1], pulseEdges)).toEqual({
      channelState: "active",
      phase: "active",
      reducedMotion: true,
    });
    expect(resolveCanvasPulseEdgeState(visibleEdges[0], pulseEdges)).toEqual({
      channelState: "completed",
      phase: "completed",
      reducedMotion: true,
    });
  });
});
