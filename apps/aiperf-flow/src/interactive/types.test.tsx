/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import type { FlowStep, ZoomTree, ZoomTreeNode } from "./types.js";

describe("interactive/types", () => {
  it("models a ZoomTree as an id -> { label, nodes, edges, children } map", () => {
    const nodes: Node[] = [{ id: "a", position: { x: 0, y: 0 }, data: {} }];
    const edges: Edge[] = [];
    const tree: ZoomTree = {
      root: { label: "Root", nodes, edges, children: ["a"] },
      a: { label: "A", nodes, edges },
    };

    expect(Object.keys(tree)).toEqual(["root", "a"]);
    expect(tree.root.children).toEqual(["a"]);
    expect(tree.root.label).toBe("Root");
    expect(tree.a.children).toBeUndefined();
  });

  it("threads a payload type through ZoomTreeNode<T>", () => {
    const node: ZoomTreeNode<{ anchor: string }> = {
      label: "Clock",
      nodes: [],
      edges: [],
      data: { anchor: "runtime/src/clock/clock.rs:12" },
    };
    expect(node.data?.anchor).toBe("runtime/src/clock/clock.rs:12");
  });

  it("models a FlowStep as node highlight + caption plus optional timing/variant", () => {
    const step: FlowStep = {
      nodeId: "transport",
      caption: "Dispatcher hands the request to the chosen WorkerSink",
      timingMs: 900,
      variant: "http",
    };
    expect(step.nodeId).toBe("transport");
    expect(step.caption).toContain("WorkerSink");
    expect(step.timingMs).toBe(900);
    expect(step.variant).toBe("http");
  });
});
