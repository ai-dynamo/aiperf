/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock only React Flow's context hooks so the hook runs outside a live <ReactFlow>: report the
// nodes as initialized and hand back measured sizes, exercising the real layoutGraph path.
const fitView = vi.fn();
let initialized = true;
let measured: Node[] = [];

vi.mock("@xyflow/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@xyflow/react")>();
  return {
    ...actual,
    useNodesInitialized: () => initialized,
    useReactFlow: () => ({ getNodes: () => measured, fitView }),
  };
});

// Imported after the mock is registered.
const { useElkLayout } = await import("./useElkLayout.js");

const NODES: Node[] = [
  { id: "a", position: { x: 0, y: 0 }, data: {} },
  { id: "b", position: { x: 0, y: 0 }, data: {} },
];
const EDGES: Edge[] = [{ id: "e-a-b", source: "a", target: "b" }];

beforeEach(() => {
  initialized = true;
  fitView.mockClear();
  measured = [
    { id: "a", position: { x: 0, y: 0 }, data: {}, measured: { width: 200, height: 80 } },
    { id: "b", position: { x: 0, y: 0 }, data: {}, measured: { width: 200, height: 80 } },
  ];
});

describe("useElkLayout", () => {
  it("lays out nodes and flips laidOut once initialized", async () => {
    const { result } = renderHook(() => useElkLayout(NODES, EDGES, { direction: "RIGHT" }));
    await waitFor(() => expect(result.current.laidOut).toBe(true));
    const map = Object.fromEntries(result.current.nodes.map((n) => [n.id, n]));
    expect(map.a!.position.x).toBeLessThan(map.b!.position.x);
    expect(result.current.nodes).toHaveLength(2);
  });

  it("does not lay out until nodes are initialized", async () => {
    initialized = false;
    const { result } = renderHook(() => useElkLayout(NODES, EDGES, {}));
    // Give any pending microtasks a chance; laidOut must stay false.
    await new Promise((r) => setTimeout(r, 20));
    expect(result.current.laidOut).toBe(false);
  });

  it("fits the view after a layout pass", async () => {
    renderHook(() => useElkLayout(NODES, EDGES, {}));
    await waitFor(() => expect(fitView).toHaveBeenCalled());
  });
});
