// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouterProvider, createMemoryHistory } from "@tanstack/react-router";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { decompressFromEncodedURIComponent } from "lz-string";
import { describe, expect, it, vi } from "vitest";

import {
  canonicalGraphState,
  encodeGraphStateForUrl,
} from "../domain/graph-state";
import { createAppRouter } from "./router";

function renderAtlas(path: string) {
  const router = createAppRouter({
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  render(<RouterProvider router={router} />);
  return router;
}

describe("graph scene routes", () => {
  it("renders the graph-first runtime scene at / without guided atlas content", async () => {
    renderAtlas("/?audience=developer");

    expect(
      await screen.findByRole("heading", { name: "Runtime composition" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Graph canvas" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("searchbox", { name: "Search atlas" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("list", { name: "Graph edge controls" }),
    ).not.toBeInTheDocument();
  });

  it.each([
    ["/", "Runtime composition"],
    ["/scenes/runner-protocol-registries", "Runner protocol and registries"],
    ["/scenes/scheduling-phase-lifecycle", "Scheduling and phase lifecycle"],
    ["/scenes/dataset-segment-pipeline", "Dataset and segment pipeline"],
    ["/scenes/endpoint-bindings-transports", "Endpoint bindings and HTTP/gRPC transports"],
    ["/scenes/graph-ir-execution", "Graph-IR execution"],
    ["/scenes/metrics-telemetry", "Metrics and telemetry"],
    ["/scenes/accuracy-evaluator-hosting", "Accuracy and evaluator hosting"],
    ["/scenes/crate-dependency-topology", "Crate dependency topology"],
  ])("renders canonical graph scene %s", async (path, title) => {
    renderAtlas(`${path}?audience=developer`);
    expect(await screen.findByRole("heading", { name: title })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Graph canvas" })).toBeInTheDocument();
  });

  it("applies flavor and search state to derived topology visibility", async () => {
    const user = userEvent.setup();
    renderAtlas("/scenes/runner-protocol-registries?audience=executive&primary=native_http");

    expect(
      await screen.findByTestId("graph-node-node.runner-protocol-registries"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("graph-node-node.dynamo-online-runner-pair"),
    ).not.toBeInTheDocument();

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Compare flavor" }),
      "dynamo_online",
    );

    await user.type(
      screen.getByRole("searchbox", { name: "Graph search" }),
      "Planned runner backend and pair",
    );

    expect(
      await screen.findByTestId("graph-node-node.dynamo-online-runner-pair"),
    ).toBeInTheDocument();
  });

  it("supports selection path highlighting and evidence focus restoration", async () => {
    const user = userEvent.setup();
    renderAtlas("/scenes/metrics-telemetry?audience=developer");

    await user.click(
      await screen.findByRole("button", { name: "Show graph accessibility outline" }),
    );
    const outline = screen.getByRole("tree", { name: "Visible graph outline" });
    const metricsNode = within(outline).getByRole("button", {
      name: "Select node Metrics accumulator and telemetry producers",
    });
    await user.click(metricsNode);

    expect(
      await screen.findByRole("heading", {
        name: "Metrics accumulator and telemetry producers",
      }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("graph-node-node.metrics-telemetry")).toHaveAttribute(
      "data-path-state",
      "focused",
    );
    expect(screen.getByTestId("graph-node-node.runtime-composition")).toHaveAttribute(
      "data-path-state",
      "upstream",
    );

    await user.click(screen.getByRole("button", { name: "Close evidence panel" }));
    await waitFor(() => {
      expect(document.activeElement).toHaveAttribute(
        "data-graph-entity-id",
        "node.metrics-telemetry",
      );
    });
  });

  it("expands and collapses topology using accessibility outline callbacks", async () => {
    const user = userEvent.setup();
    renderAtlas("/?audience=executive");

    await screen.findByTestId("graph-node-node.runtime-composition");
    expect(screen.queryByTestId("graph-node-node.clock-seam")).not.toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "Show graph accessibility outline" }),
    );
    const outline = screen.getByRole("tree", { name: "Visible graph outline" });
    const runtimeNodeItem = within(outline).getByRole("treeitem", {
      name: "Node Runtime composition",
    });

    await user.click(
      within(runtimeNodeItem).getByRole("button", { name: "Expand" }),
    );
    expect(await screen.findByTestId("graph-node-node.clock-seam")).toBeInTheDocument();

    await user.click(
      within(runtimeNodeItem).getByRole("button", { name: "Collapse" }),
    );
    await waitFor(() => {
      expect(screen.queryByTestId("graph-node-node.clock-seam")).not.toBeInTheDocument();
    });
  });

  it("cleans hidden descendant overrides while preserving the collapsed node position", async () => {
    const user = userEvent.setup();
    const initialState = encodeGraphStateForUrl(
      canonicalGraphState({
        audience: "executive",
        edgeWaypoints: [
          {
            edgeId: "edge.request-sink.token.metrics",
            points: [{ x: 120, y: 80 }],
          },
          {
            edgeId: "edge.runtime.dispatch.metrics",
            points: [{ x: 220, y: 180 }],
          },
        ],
        expandedNodeIds: ["node.runtime-composition"],
        focusedEntityId: "node.request-sink-seam",
        nodePositions: [
          { nodeId: "node.request-sink-seam", x: 440, y: 260 },
          { nodeId: "node.runtime-composition", x: 240, y: 160 },
        ],
        primaryFlavor: "native_http",
        sceneId: "scene.runtime-composition",
        traceMode: "upstream",
      }),
    );
    const router = renderAtlas(`/?audience=executive&s=${initialState}`);

    await user.click(
      await screen.findByRole("button", { name: "Show graph accessibility outline" }),
    );
    const runtimeNodeItem = within(
      screen.getByRole("tree", { name: "Visible graph outline" }),
    ).getByRole("treeitem", { name: "Node Runtime composition" });
    await user.click(
      within(runtimeNodeItem).getByRole("button", { name: "Collapse" }),
    );

    await waitFor(() => {
      const encodedState = String(router.state.location.search.s);
      const decodedState = JSON.parse(
        decompressFromEncodedURIComponent(encodedState) ?? "{}",
      );
      expect(decodedState.nodePositions).toEqual([
        { nodeId: "node.runtime-composition", x: 240, y: 160 },
      ]);
      expect(decodedState.edgeWaypoints).toEqual([
        {
          edgeId: "edge.runtime.dispatch.metrics",
          points: [{ x: 220, y: 180 }],
        },
      ]);
      expect(decodedState.focusedEntityId).toBe("node.runtime-composition");
      expect(decodedState.traceMode).toBe("none");
    });
  });

  it("applies trace mode controls to visible path states", async () => {
    const user = userEvent.setup();
    renderAtlas("/scenes/metrics-telemetry?audience=developer");

    const metricsNode = await screen.findByTestId("graph-node-node.metrics-telemetry");
    await user.click(
      within(metricsNode).getByText("upstream"),
    );

    expect(screen.getByTestId("graph-node-node.runtime-composition")).toHaveAttribute(
      "data-path-state",
      "upstream",
    );

    await user.click(
      screen.getByRole("button", { name: "Show graph accessibility outline" }),
    );
    const outline = screen.getByRole("tree", { name: "Visible graph outline" });
    const metricsItem = within(outline).getByRole("treeitem", {
      name: "Node Metrics accumulator and telemetry producers",
    });
    await user.click(within(metricsItem).getByRole("button", { name: "Isolate" }));

    expect(screen.getByTestId("graph-node-node.metrics-telemetry")).toHaveAttribute(
      "data-path-state",
      "focused",
    );
    expect(screen.getByTestId("graph-node-node.runtime-composition")).toHaveAttribute(
      "data-path-state",
      "default",
    );
  });

  it("wires the exact active timeline edge into the live graph canvas", async () => {
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: vi.fn().mockReturnValue({
        matches: false,
        media: "(prefers-reduced-motion: reduce)",
        onchange: null,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }),
    });
    const state = encodeGraphStateForUrl(
      canonicalGraphState({
        audience: "developer",
        edgeWaypoints: [
          {
            edgeId: "edge.runtime.dispatch.metrics",
            points: [{ x: 180, y: 120 }],
          },
        ],
        primaryFlavor: "native_http",
        sceneId: "scene.metrics-telemetry",
        timelinePosition: 5 / 6,
      }),
    );
    renderAtlas(
      `/scenes/metrics-telemetry?audience=developer&primary=native_http&s=${state}`,
    );

    const canvas = await screen.findByRole("region", { name: "Graph canvas" });
    expect(canvas).toHaveAttribute(
      "data-active-pulse-edge-ids",
      "edge.runtime.dispatch.metrics",
    );
    expect(canvas).toHaveAttribute(
      "data-active-pulse-channels",
      "telemetry",
    );
    expect(canvas).toHaveAttribute("data-reduced-motion", "false");
  });

  it("renders pulse controls and reduced-motion overlay semantics", async () => {
    const matchMedia = vi.fn().mockReturnValue({
      matches: true,
      media: "(prefers-reduced-motion: reduce)",
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    });
    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      value: matchMedia,
    });
    renderAtlas("/scenes/metrics-telemetry?audience=developer");

    expect(
      await screen.findByRole("region", { name: "Pulse edge overlay" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Pulse channels legend" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("pulse-active-particle")).toHaveAttribute(
      "data-motion",
      "reduced",
    );
  });
});
