/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { layoutGraph, fallbackLayout, DEFAULT_NODE_WIDTH } from "./elkEngine.js";

/** A→B→C chain with no authored positions. */
const CHAIN: Node[] = [
  { id: "a", position: { x: 0, y: 0 }, data: {} },
  { id: "b", position: { x: 0, y: 0 }, data: {} },
  { id: "c", position: { x: 0, y: 0 }, data: {} },
];
const CHAIN_EDGES: Edge[] = [
  { id: "e-a-b", source: "a", target: "b" },
  { id: "e-b-c", source: "b", target: "c" },
];

function byId(nodes: Node[]): Record<string, Node> {
  return Object.fromEntries(nodes.map((n) => [n.id, n]));
}

/** Two boxes overlap iff their measured footprints intersect on both axes. */
function overlaps(a: Node, b: Node): boolean {
  const aw = a.measured?.width ?? DEFAULT_NODE_WIDTH;
  const ah = a.measured?.height ?? 90;
  const bw = b.measured?.width ?? DEFAULT_NODE_WIDTH;
  const bh = b.measured?.height ?? 90;
  const ax = a.position.x;
  const ay = a.position.y;
  const bx = b.position.x;
  const by = b.position.y;
  return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
}

describe("layoutGraph", () => {
  it("places a chain left→right with strictly increasing x", async () => {
    const laid = byId(await layoutGraph(CHAIN, CHAIN_EDGES, { direction: "RIGHT" }));
    expect(laid.a.position.x).toBeLessThan(laid.b.position.x);
    expect(laid.b.position.x).toBeLessThan(laid.c.position.x);
  });

  it("assigns a position to every node and preserves data + identity", async () => {
    const nodes: Node[] = [{ id: "solo", position: { x: 7, y: 9 }, data: { title: "keep me" } }];
    const laid = await layoutGraph(nodes, [], {});
    expect(laid).toHaveLength(1);
    expect(laid[0]!.id).toBe("solo");
    expect(laid[0]!.data).toEqual({ title: "keep me" });
    expect(laid[0]!.position).toBeDefined();
  });

  it("returns an empty array unchanged", async () => {
    expect(await layoutGraph([], [], {})).toEqual([]);
  });

  it("stacks a top→down chain with strictly increasing y", async () => {
    const laid = byId(await layoutGraph(CHAIN, CHAIN_EDGES, { direction: "DOWN" }));
    expect(laid.a.position.y).toBeLessThan(laid.b.position.y);
    expect(laid.b.position.y).toBeLessThan(laid.c.position.y);
  });
});

describe("fallbackLayout", () => {
  it("produces non-overlapping boxes for a chain", () => {
    const laid = fallbackLayout(CHAIN, CHAIN_EDGES, { direction: "RIGHT" });
    const map = byId(laid);
    expect(overlaps(map.a!, map.b!)).toBe(false);
    expect(overlaps(map.b!, map.c!)).toBe(false);
    expect(overlaps(map.a!, map.c!)).toBe(false);
  });

  it("layers by longest incoming path (fan-in lands past both parents)", () => {
    // a→c, b→c: c must sit in a later layer (greater x) than both a and b.
    const nodes: Node[] = [
      { id: "a", position: { x: 0, y: 0 }, data: {} },
      { id: "b", position: { x: 0, y: 0 }, data: {} },
      { id: "c", position: { x: 0, y: 0 }, data: {} },
    ];
    const edges: Edge[] = [
      { id: "e-a-c", source: "a", target: "c" },
      { id: "e-b-c", source: "b", target: "c" },
    ];
    const map = byId(fallbackLayout(nodes, edges, { direction: "RIGHT" }));
    expect(map.c!.position.x).toBeGreaterThan(map.a!.position.x);
    expect(map.c!.position.x).toBeGreaterThan(map.b!.position.x);
  });

  it("does not loop forever on a cycle", () => {
    const nodes: Node[] = [
      { id: "x", position: { x: 0, y: 0 }, data: {} },
      { id: "y", position: { x: 0, y: 0 }, data: {} },
    ];
    const edges: Edge[] = [
      { id: "e-x-y", source: "x", target: "y" },
      { id: "e-y-x", source: "y", target: "x" },
    ];
    const laid = fallbackLayout(nodes, edges, {});
    expect(laid).toHaveLength(2);
  });
});
