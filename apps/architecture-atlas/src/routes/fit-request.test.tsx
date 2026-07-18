// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { RouterProvider, createMemoryHistory } from "@tanstack/react-router";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { decompressFromEncodedURIComponent } from "lz-string";

import { architectureCatalog } from "../content";
import {
  canonicalGraphState,
  decodeGraphStateFromUrl,
  encodeGraphStateForUrl,
} from "../domain/graph-state";
import type { FlavorOverlay } from "../domain/graph-derivation";
import { executionFlavorSchema } from "../domain/architecture";
import { canonicalSceneIds } from "../domain/routes";
import type { GraphFitViewCommand } from "../features/graph/types";

interface FitAwareCanvasProps {
  fitViewCommand?: GraphFitViewCommand;
  onFitViewComplete?(requestId: number): void;
  overlay?: FlavorOverlay;
  onNodeDragComplete?(position: { nodeId: string; x: number; y: number }): void;
  onWaypointsChange?(update: { edgeId: string; points: { x: number; y: number }[] }): void;
  onWaypointsReset?(edgeId: string): void;
}

vi.mock("../features/graph/graph-canvas", () => ({
  GraphCanvas: ({
    fitViewCommand,
    onFitViewComplete,
    overlay,
    onNodeDragComplete,
    onWaypointsChange,
    onWaypointsReset,
  }: FitAwareCanvasProps) => (
    <>
      <output aria-label="Observed graph fit request">
        {fitViewCommand?.requestId ?? "none"}
      </output>
      {fitViewCommand ? (
        <button
          onClick={() => onFitViewComplete?.(fitViewCommand.requestId)}
          type="button"
        >
          Acknowledge fit request
        </button>
      ) : null}
      <output aria-label="Observed graph flavor overlay">
        {JSON.stringify(overlay ?? null)}
      </output>
      <button
        onClick={() =>
          onNodeDragComplete?.({ nodeId: "node.runtime-composition", x: 222, y: 111 })
        }
        type="button"
      >
        Persist drag override
      </button>
      <button
        onClick={() =>
          onWaypointsChange?.({
            edgeId: "edge.runtime.dispatch.metrics",
            points: [{ x: 40, y: 32 }],
          })
        }
        type="button"
      >
        Persist waypoint override
      </button>
      <button onClick={() => onWaypointsReset?.("edge.runtime.dispatch.metrics")} type="button">
        Clear waypoint override
      </button>
    </>
  ),
}));

import { createAppRouter } from "./router";

function decodeSearchState(encoded: string) {
  return JSON.parse(decompressFromEncodedURIComponent(encoded) ?? "{}");
}

function buildCanonicalDomain() {
  return {
    defaultState: canonicalGraphState({
      audience: "developer",
      primaryFlavor: "native_http",
      sceneId: "scene.runtime-composition",
    }),
    edgeIds: new Set(architectureCatalog.graphEdges.map(({ id }) => id)),
    nodeIds: new Set(architectureCatalog.graphNodes.map(({ id }) => id)),
    sceneIds: new Set(canonicalSceneIds),
    supportedFlavors: new Set(executionFlavorSchema.options),
  };
}

describe("graph fit command integration", () => {
  it("clears an acknowledged request across scene remounts and advances the sequence", async () => {
    const user = userEvent.setup();
    const router = createAppRouter({
      history: createMemoryHistory({
        initialEntries: ["/?audience=developer"],
      }),
    });
    render(<RouterProvider router={router} />);

    expect(
      await screen.findByRole("status", {
        name: "Observed graph fit request",
      }),
    ).toHaveTextContent("none");

    await user.click(screen.getByRole("button", { name: "Fit graph" }));
    expect(
      screen.getByRole("status", { name: "Observed graph fit request" }),
    ).toHaveTextContent("1");

    await user.click(
      screen.getByRole("button", { name: "Acknowledge fit request" }),
    );
    expect(
      screen.getByRole("status", { name: "Observed graph fit request" }),
    ).toHaveTextContent("none");
    await user.click(screen.getByRole("button", { name: "Fit graph" }));
    expect(
      screen.getByRole("status", { name: "Observed graph fit request" }),
    ).toHaveTextContent("2");

    await user.click(
      screen.getByRole("button", { name: "Acknowledge fit request" }),
    );
    await user.click(
      screen.getByRole("link", { name: "Metrics and telemetry" }),
    );
    expect(
      await screen.findByRole("status", {
        name: "Observed graph fit request",
      }),
    ).toHaveTextContent("none");
  });

  it("passes the derived comparison overlay through GraphScene", async () => {
    const router = createAppRouter({
      history: createMemoryHistory({
        initialEntries: [
          "/?audience=developer&primary=native_http&compare=dynamo_offline",
        ],
      }),
    });
    render(<RouterProvider router={router} />);

    expect(
      await screen.findByRole("status", {
        name: "Observed graph flavor overlay",
      }),
    ).toHaveTextContent('"sharedNodeIds"');
    expect(
      screen.getByRole("status", {
        name: "Observed graph flavor overlay",
      }),
    ).toHaveTextContent("node.runtime-composition");
  });

  it("persists drag and waypoint overrides into URL state", async () => {
    const user = userEvent.setup();
    const router = createAppRouter({
      history: createMemoryHistory({
        initialEntries: ["/?audience=developer"],
      }),
    });
    render(<RouterProvider router={router} />);

    await user.click(await screen.findByRole("button", { name: "Persist drag override" }));
    await user.click(screen.getByRole("button", { name: "Persist waypoint override" }));

    const encoded = String(router.state.location.search.s);
    const decoded = decodeSearchState(encoded);
    expect(decoded.nodePositions).toContainEqual({
      nodeId: "node.runtime-composition",
      x: 222,
      y: 111,
    });
    expect(decoded.edgeWaypoints).toContainEqual({
      edgeId: "edge.runtime.dispatch.metrics",
      points: [{ x: 40, y: 32 }],
    });
    expect(decoded.edgeWaypoints[0]).not.toHaveProperty("source");
    expect(decoded.edgeWaypoints[0]).not.toHaveProperty("target");
  });

  it("resets only manual layout overrides", async () => {
    const user = userEvent.setup();
    const seeded = encodeGraphStateForUrl(
      canonicalGraphState({
        audience: "developer",
        edgeWaypoints: [
          {
            edgeId: "edge.runtime.dispatch.metrics",
            points: [{ x: 9, y: 10 }],
          },
        ],
        expandedNodeIds: ["node.runtime-composition"],
        nodePositions: [{ nodeId: "node.runtime-composition", x: 5, y: 7 }],
        primaryFlavor: "native_http",
        sceneId: "scene.runtime-composition",
      }),
    );
    const router = createAppRouter({
      history: createMemoryHistory({
        initialEntries: [`/?audience=developer&s=${seeded}`],
      }),
    });
    render(<RouterProvider router={router} />);

    await user.click(await screen.findByRole("button", { name: "Reset graph" }));
    const encoded = String(router.state.location.search.s);
    const decoded = decodeSearchState(encoded);
    expect(decoded.nodePositions).toEqual([]);
    expect(decoded.edgeWaypoints).toEqual([]);
    expect(decoded.expandedNodeIds).toEqual(["node.runtime-composition"]);
  });

  it("copies compressed share URL containing current graph state", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn<(text: string) => Promise<void>>(async () => undefined);
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    const router = createAppRouter({
      history: createMemoryHistory({
        initialEntries: ["/?audience=developer"],
      }),
    });
    render(<RouterProvider router={router} />);

    await user.click(await screen.findByRole("button", { name: "Persist drag override" }));
    await user.click(screen.getByRole("button", { name: "Share graph state" }));

    expect(writeText).toHaveBeenCalledTimes(1);
    const firstCall = writeText.mock.calls.at(0);
    if (!firstCall) {
      throw new Error("expected clipboard call");
    }
    const sharedUrl = new URL(String(firstCall[0]));
    const encoded = sharedUrl.searchParams.get("s");
    expect(encoded).toBeTruthy();
    const decoded = decodeGraphStateFromUrl(String(encoded), buildCanonicalDomain());
    expect(decoded.state.nodePositions).toContainEqual({
      nodeId: "node.runtime-composition",
      x: 222,
      y: 111,
    });
  });

  it("shares reset state with scene and flavor context preserved", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn<(text: string) => Promise<void>>(async () => undefined);
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    const router = createAppRouter({
      history: createMemoryHistory({
        initialEntries: [
          "/scenes/metrics-telemetry?audience=maintainer&primary=native_grpc&compare=dynamo_online",
        ],
      }),
    });
    render(<RouterProvider router={router} />);

    await user.click(await screen.findByRole("button", { name: "Persist drag override" }));
    await user.click(screen.getByRole("button", { name: "Persist waypoint override" }));
    await user.click(screen.getByRole("button", { name: "Reset graph" }));
    await user.click(screen.getByRole("button", { name: "Share graph state" }));

    expect(writeText).toHaveBeenCalledTimes(1);
    const firstCall = writeText.mock.calls.at(0);
    if (!firstCall) {
      throw new Error("expected clipboard call");
    }
    const sharedUrl = new URL(String(firstCall[0]));
    const encoded = sharedUrl.searchParams.get("s");
    const decoded = decodeGraphStateFromUrl(String(encoded), buildCanonicalDomain());

    expect(decoded.state.sceneId).toBe("scene.metrics-telemetry");
    expect(decoded.state.primaryFlavor).toBe("native_grpc");
    expect(decoded.state.compareFlavor).toBe("dynamo_online");
    expect(decoded.state.audience).toBe("maintainer");
    expect(decoded.state.nodePositions).toEqual([]);
    expect(decoded.state.edgeWaypoints).toEqual([]);
  });
});
