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
let measured: Node[] = [];

vi.mock("@xyflow/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@xyflow/react")>();
  return {
    ...actual,
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

  it("lays out even when React Flow has measured no nodes yet", async () => {
    // Nested/animated canvases may never report measurements; the layout must still run (with
    // fallback/estimated sizes) so the diagram is never left blank.
    measured = [];
    const { result } = renderHook(() => useElkLayout(NODES, EDGES, {}));
    await waitFor(() => expect(result.current.laidOut).toBe(true));
    const map = Object.fromEntries(result.current.nodes.map((n) => [n.id, n]));
    expect(map.a!.position.x).toBeLessThan(map.b!.position.x);
  });

  it("fits the view after a layout pass", async () => {
    renderHook(() => useElkLayout(NODES, EDGES, {}));
    await waitFor(() => expect(fitView).toHaveBeenCalled(), { timeout: 3000 });
  });

  it("reflects live node data changes (same ids) without relayout", async () => {
    const initial: Node[] = [{ id: "a", position: { x: 0, y: 0 }, data: { title: "before" } }];
    const { result, rerender } = renderHook(({ nodes }) => useElkLayout(nodes, [], {}), {
      initialProps: { nodes: initial },
    });
    await waitFor(() => expect(result.current.laidOut).toBe(true));
    const laidPosition = result.current.nodes[0]!.position;

    // New data, SAME id: data must update, position must be preserved (no relayout needed).
    const updated: Node[] = [{ id: "a", position: { x: 0, y: 0 }, data: { title: "after" } }];
    rerender({ nodes: updated });
    expect(result.current.nodes[0]!.data).toEqual({ title: "after" });
    expect(result.current.nodes[0]!.position).toEqual(laidPosition);
  });
});
