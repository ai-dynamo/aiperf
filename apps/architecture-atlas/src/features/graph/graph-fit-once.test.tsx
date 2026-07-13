// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { FlavorOverlay } from "../../domain/graph-derivation";
import type { LayoutRequest, LayoutResult } from "../atlas/layout";

const { fitView } = vi.hoisted(() => ({
  fitView: vi.fn(async () => true),
}));

vi.mock("@xyflow/react", async () => {
  const { useEffect } = await import("react");
  return {
    Background: () => null,
    BaseEdge: () => null,
    Controls: () => null,
    EdgeLabelRenderer: ({ children }: { children: unknown }) => children,
    Handle: () => null,
    MarkerType: { ArrowClosed: "arrowclosed" },
    MiniMap: () => null,
    Panel: ({ children }: { children: unknown }) => children,
    Position: { Left: "left", Right: "right" },
    ReactFlow: ({
      children,
      onInit,
    }: {
      children: unknown;
      onInit?(instance: { fitView: typeof fitView }): void;
    }) => {
      useEffect(() => {
        onInit?.({ fitView });
      }, [onInit]);
      return children;
    },
    getSmoothStepPath: () => ["", 0, 0],
  };
});

import { GraphCanvas } from "./graph-canvas";

const overlay: FlavorOverlay = {
  compareOnlyEdgeIds: [],
  compareOnlyNodeIds: [],
  primaryOnlyEdgeIds: [],
  primaryOnlyNodeIds: [],
  sharedEdgeIds: [],
  sharedNodeIds: [],
};

const layoutResult: LayoutResult = {
  bands: [],
  degraded: false,
  positions: [{ bandId: "tier.0", id: "node.runtime", x: 0, y: 0 }],
};

function layoutRequest(key: string): LayoutRequest {
  return {
    bands: [],
    edges: [],
    key,
    nodes: [],
    perspective: "ownership",
    version: 1,
  };
}

describe("graph fit command consumption", () => {
  it("executes each request id once across later layout changes", async () => {
    const layoutService = {
      layout: vi.fn(async () => layoutResult),
    };
    const commonProps = {
      audience: "developer" as const,
      focusedEntityId: null,
      layoutService,
      neighborhood: { downstreamNodeIds: [], upstreamNodeIds: [] },
      onFocusEntity: vi.fn(),
      overlay,
      visibleEdges: [],
      visibleNodes: [],
    };
    const { rerender } = render(
      <GraphCanvas
        {...commonProps}
        fitViewCommand={{ requestId: 1 }}
        layoutRequest={layoutRequest("layout.1")}
      />,
    );

    await waitFor(() => {
      expect(fitView).toHaveBeenCalledTimes(1);
    });

    rerender(
      <GraphCanvas
        {...commonProps}
        fitViewCommand={{ requestId: 1 }}
        layoutRequest={layoutRequest("layout.2")}
      />,
    );
    await waitFor(() => {
      expect(layoutService.layout).toHaveBeenCalledTimes(2);
    });
    expect(fitView).toHaveBeenCalledTimes(1);

    rerender(
      <GraphCanvas
        {...commonProps}
        fitViewCommand={{ requestId: 2 }}
        layoutRequest={layoutRequest("layout.2")}
      />,
    );
    await waitFor(() => {
      expect(fitView).toHaveBeenCalledTimes(2);
    });
  });
});
