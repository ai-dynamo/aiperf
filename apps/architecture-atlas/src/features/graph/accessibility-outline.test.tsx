// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { GraphEdge, GraphNode } from "../../domain/architecture";
import { AccessibilityOutline } from "./accessibility-outline";

function buildNode(overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id: "node.runtime-composition",
    tier: 1,
    parentId: null,
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: ["native_http"],
    title: {
      executive: "Runtime composition",
      developer: "Runtime composition",
      maintainer: "Runtime composition",
    },
    summary: {
      executive: "Executive runtime summary.",
      developer: "Developer runtime summary.",
      maintainer: "Maintainer runtime summary.",
    },
    evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 12 }, role: "source" }],
    seamPorts: [{ id: "port.runtime.out", name: "dispatch", channel: "request_data" }],
    audience: {
      visibility: ["executive", "developer", "maintainer"],
      autoExpandDepth: { executive: 1, developer: 2, maintainer: 3 },
    },
    footnotes: [],
    ...overrides,
  };
}

function buildEdge(overrides: Partial<GraphEdge> = {}): GraphEdge {
  return {
    id: "edge.runtime.dispatch.metrics",
    source: { nodeId: "node.runtime-composition", portId: "port.runtime.out" },
    target: { nodeId: "node.metrics-telemetry", portId: "port.metrics.in" },
    channel: "telemetry",
    status: { state: "built", delivery: "unconditional" },
    flavors: ["native_http"],
    protocol: "RequestObserver callbacks",
    evidence: [{ path: "AGENTS.md", lines: { start: 1, end: 12 }, role: "source" }],
    footnotes: [],
    ...overrides,
  };
}

describe("AccessibilityOutline", () => {
  it("is collapsed by default and expands on demand", async () => {
    const user = userEvent.setup();
    render(
      <AccessibilityOutline
        audience="developer"
        expandedNodeIds={[]}
        onCollapseNode={() => {}}
        onExpandNode={() => {}}
        onInspectEntity={() => {}}
        onIsolateEntity={() => {}}
        onSelectEntity={() => {}}
        visibleEdges={[]}
        visibleNodes={[buildNode()]}
      />,
    );

    expect(screen.queryByRole("tree", { name: "Visible graph outline" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Show graph accessibility outline" }));

    expect(screen.getByRole("tree", { name: "Visible graph outline" })).toBeInTheDocument();
  });

  it("mirrors visible directed nodes and edges", async () => {
    const user = userEvent.setup();
    const sourceNode = buildNode();
    const targetNode = buildNode({
      id: "node.metrics-telemetry",
      title: {
        executive: "Metrics and telemetry",
        developer: "Metrics and telemetry",
        maintainer: "Metrics and telemetry",
      },
      seamPorts: [{ id: "port.metrics.in", name: "observer-events", channel: "telemetry" }],
    });
    render(
      <AccessibilityOutline
        audience="developer"
        expandedNodeIds={[]}
        onCollapseNode={() => {}}
        onExpandNode={() => {}}
        onInspectEntity={() => {}}
        onIsolateEntity={() => {}}
        onSelectEntity={() => {}}
        visibleEdges={[buildEdge()]}
        visibleNodes={[sourceNode, targetNode]}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Show graph accessibility outline" }));

    expect(screen.getByRole("button", { name: "Select node Runtime composition" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Select node Metrics and telemetry" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: "Select edge Runtime composition -> Metrics and telemetry via RequestObserver callbacks",
      }),
    ).toBeInTheDocument();
  });

  it("exposes synchronized expansion state on expandable tree items", async () => {
    const user = userEvent.setup();
    const node = buildNode({ childIds: ["node.clock-seam"] });
    const child = buildNode({
      id: "node.clock-seam",
      parentId: node.id,
      title: {
        executive: "Clock seam",
        developer: "Clock seam",
        maintainer: "Clock seam",
      },
    });
    const { rerender } = render(
      <AccessibilityOutline
        audience="developer"
        expandedNodeIds={[]}
        onCollapseNode={() => {}}
        onExpandNode={() => {}}
        onInspectEntity={() => {}}
        onIsolateEntity={() => {}}
        onSelectEntity={() => {}}
        visibleEdges={[]}
        visibleNodes={[node, child]}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Show graph accessibility outline" }));
    const runtimeTreeItem = screen.getByRole("treeitem", {
      name: "Node Runtime composition",
    });
    expect(runtimeTreeItem).toHaveAttribute("aria-expanded", "false");
    expect(runtimeTreeItem).toHaveAttribute("aria-level", "1");
    expect(screen.getByRole("treeitem", { name: "Node Clock seam" })).toHaveAttribute(
      "aria-level",
      "2",
    );

    rerender(
      <AccessibilityOutline
        audience="developer"
        expandedNodeIds={[node.id]}
        onCollapseNode={() => {}}
        onExpandNode={() => {}}
        onInspectEntity={() => {}}
        onIsolateEntity={() => {}}
        onSelectEntity={() => {}}
        visibleEdges={[]}
        visibleNodes={[node, child]}
      />,
    );
    expect(
      screen.getByRole("treeitem", { name: "Node Runtime composition" }),
    ).toHaveAttribute("aria-expanded", "true");
  });

  it("supports keyboard callbacks for select, expand/collapse, isolate, and inspect", async () => {
    const user = userEvent.setup();
    const onSelectEntity = vi.fn();
    const onExpandNode = vi.fn();
    const onCollapseNode = vi.fn();
    const onIsolateEntity = vi.fn();
    const onInspectEntity = vi.fn();

    render(
      <AccessibilityOutline
        audience="developer"
        expandedNodeIds={[]}
        onCollapseNode={onCollapseNode}
        onExpandNode={onExpandNode}
        onInspectEntity={onInspectEntity}
        onIsolateEntity={onIsolateEntity}
        onSelectEntity={onSelectEntity}
        visibleEdges={[]}
        visibleNodes={[buildNode()]}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Show graph accessibility outline" }));

    const runtimeNodeRow = screen.getByRole("treeitem", {
      name: "Node Runtime composition",
    });
    runtimeNodeRow.focus();

    fireEvent.keyDown(runtimeNodeRow, { key: "Enter" });
    fireEvent.keyDown(runtimeNodeRow, { key: "ArrowRight" });
    fireEvent.keyDown(runtimeNodeRow, { key: "ArrowLeft" });
    fireEvent.keyDown(runtimeNodeRow, { key: "i" });
    fireEvent.keyDown(runtimeNodeRow, { key: "x" });

    expect(onSelectEntity).toHaveBeenCalledWith("node.runtime-composition");
    expect(onExpandNode).toHaveBeenCalledWith("node.runtime-composition");
    expect(onCollapseNode).toHaveBeenCalledWith("node.runtime-composition");
    expect(onIsolateEntity).toHaveBeenCalledWith("node.runtime-composition");
    expect(onInspectEntity).toHaveBeenCalledWith("node.runtime-composition");
  });
});
