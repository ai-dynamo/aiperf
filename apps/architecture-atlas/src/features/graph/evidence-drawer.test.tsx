// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";

import type { GraphEdge, GraphNode } from "../../domain/architecture";
import { EvidenceDrawer } from "./evidence-drawer";

function buildNode(overrides: Partial<GraphNode> = {}): GraphNode {
  return {
    id: "node.runtime-composition",
    tier: 1,
    parentId: null,
    childIds: [],
    owner: "rust",
    status: { state: "built", delivery: "unconditional" },
    flavors: ["native_http", "native_grpc"],
    title: {
      executive: "Runtime composition",
      developer: "Runtime composition",
      maintainer: "Runtime composition",
    },
    summary: {
      executive: "Executive summary for runtime composition.",
      developer: "Developer summary for runtime composition.",
      maintainer: "Maintainer summary for runtime composition.",
    },
    evidence: [
      {
        path: "crates/runner/src/application.rs",
        lines: { start: 34, end: 65 },
        role: "source",
      },
      {
        path: "docs/superpowers/specs/2026-07-12-architecture-atlas-graph-first-redesign.md",
        role: "design",
      },
    ],
    seamPorts: [
      { id: "port.runtime.in", name: "entry", channel: "control" },
      { id: "port.runtime.out", name: "dispatch", channel: "request_data" },
    ],
    audience: {
      visibility: ["executive", "developer", "maintainer"],
      autoExpandDepth: { executive: 1, developer: 2, maintainer: 3 },
    },
    footnotes: [
      {
        executive: "Executive footnote for parity context.",
        developer: "Developer legacy migration footnote.",
        maintainer: "Maintainer parity and legacy detail.",
      },
    ],
    ...overrides,
  };
}

function buildEdge(overrides: Partial<GraphEdge> = {}): GraphEdge {
  return {
    id: "edge.runtime.dispatch.metrics",
    source: {
      nodeId: "node.runtime-composition",
      portId: "port.runtime.out",
    },
    target: {
      nodeId: "node.metrics-telemetry",
      portId: "port.metrics.in",
    },
    channel: "telemetry",
    status: { state: "built", delivery: "unconditional" },
    flavors: ["native_http"],
    protocol: "RequestObserver callbacks",
    evidence: [
      {
        path: "crates/loadgen-core/src/sink.rs",
        lines: { start: 85, end: 133 },
        role: "source",
      },
    ],
    footnotes: [],
    ...overrides,
  };
}

describe("EvidenceDrawer", () => {
  it("renders node evidence, contracts, flavors, and footnotes", () => {
    render(
      <EvidenceDrawer
        audience="developer"
        entity={{ kind: "node", node: buildNode(), relatedEdges: [buildEdge()] }}
        fallbackFocusRef={createRef<HTMLInputElement>()}
        onClose={() => {}}
        sourceBaseUrl="https://github.com/ai-dynamo/aiperf/blob/main"
      />,
    );

    expect(
      screen.getByRole("heading", { name: "Runtime composition" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Developer summary for runtime composition."),
    ).toBeInTheDocument();
    expect(screen.getByText("entry (control)")).toBeInTheDocument();
    expect(screen.getByText("dispatch (request_data)")).toBeInTheDocument();
    expect(screen.getByText("RequestObserver callbacks")).toBeInTheDocument();
    expect(screen.getByText("native_http, native_grpc")).toBeInTheDocument();
    expect(
      screen.getByText("built / unconditional"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Developer legacy migration footnote."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", {
        name: "crates/runner/src/application.rs:34-65",
      }),
    ).toHaveAttribute(
      "href",
      "https://github.com/ai-dynamo/aiperf/blob/main/crates/runner/src/application.rs#L34-L65",
    );
    expect(
      screen.getByRole("link", {
        name: "docs/superpowers/specs/2026-07-12-architecture-atlas-graph-first-redesign.md",
      }),
    ).toBeInTheDocument();
  });

  it("renders edge evidence and directed endpoint metadata", () => {
    const sourceNode = buildNode();
    const targetNode = buildNode({
      id: "node.metrics-telemetry",
      title: {
        executive: "Metrics and telemetry",
        developer: "Metrics and telemetry",
        maintainer: "Metrics and telemetry",
      },
      summary: {
        executive: "Executive summary for metrics and telemetry.",
        developer: "Developer summary for metrics and telemetry.",
        maintainer: "Maintainer summary for metrics and telemetry.",
      },
      seamPorts: [{ id: "port.metrics.in", name: "observer-events", channel: "telemetry" }],
    });

    render(
      <EvidenceDrawer
        audience="maintainer"
        entity={{
          kind: "edge",
          edge: buildEdge(),
          sourceNode,
          targetNode,
        }}
        fallbackFocusRef={createRef<HTMLInputElement>()}
        onClose={() => {}}
      />,
    );

    expect(
      screen.getByRole("heading", { name: "edge.runtime.dispatch.metrics" }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText("Runtime composition -> Metrics and telemetry"),
    ).toHaveLength(2);
    expect(screen.getByText("dispatch -> observer-events")).toBeInTheDocument();
    expect(screen.getByText("RequestObserver callbacks")).toBeInTheDocument();
    expect(screen.getByText("telemetry")).toBeInTheDocument();
  });

  it("closes on Escape and restores focus to the trigger", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const trigger = document.createElement("button");
    trigger.textContent = "Node trigger";
    document.body.append(trigger);
    const fallback = createRef<HTMLInputElement>();

    const raf = vi
      .spyOn(window, "requestAnimationFrame")
      .mockImplementation((callback: FrameRequestCallback) => {
        callback(0);
        return 1;
      });

    render(
      <EvidenceDrawer
        audience="developer"
        entity={{ kind: "node", node: buildNode(), relatedEdges: [] }}
        fallbackFocusRef={fallback}
        getTriggerElement={() => trigger}
        onClose={onClose}
      />,
    );

    await user.keyboard("{Escape}");

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(trigger).toHaveFocus();

    raf.mockRestore();
    trigger.remove();
  });

  it("falls back to search focus when trigger is not visible", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const fallback = createRef<HTMLInputElement>();
    const hiddenDetails = document.createElement("details");
    const hiddenTrigger = document.createElement("button");
    hiddenTrigger.textContent = "Hidden trigger";
    hiddenDetails.append(hiddenTrigger);
    document.body.append(hiddenDetails);

    const raf = vi
      .spyOn(window, "requestAnimationFrame")
      .mockImplementation((callback: FrameRequestCallback) => {
        callback(0);
        return 1;
      });

    render(
      <>
        <input aria-label="Graph search" ref={fallback} />
        <EvidenceDrawer
          audience="developer"
          entity={{ kind: "node", node: buildNode(), relatedEdges: [] }}
          fallbackFocusRef={fallback}
          getTriggerElement={() => hiddenTrigger}
          onClose={onClose}
        />
      </>,
    );

    await user.click(screen.getByRole("button", { name: "Close evidence panel" }));

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("textbox", { name: "Graph search" })).toHaveFocus();

    raf.mockRestore();
    hiddenDetails.remove();
  });
});
