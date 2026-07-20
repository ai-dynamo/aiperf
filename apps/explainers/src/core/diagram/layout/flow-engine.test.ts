// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from "vitest";
import { layoutFlow, type FlowNode } from "./flow-engine.js";

function leaf(id: string, width: number, height: number): FlowNode {
  return { id, measure: () => ({ width, height }) };
}

describe("layoutFlow", () => {
  it("lays out a simple row with gap", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      gap: 10,
      children: [leaf("a", 20, 30), leaf("b", 40, 30), leaf("c", 20, 30)],
    };
    const boxes = layoutFlow(root, { maxWidth: 200 });
    expect(boxes.get("a")).toEqual({ x: 0, y: 0, width: 20, height: 30 });
    expect(boxes.get("b")).toEqual({ x: 30, y: 0, width: 40, height: 30 });
    expect(boxes.get("c")).toEqual({ x: 80, y: 0, width: 20, height: 30 });
    expect(boxes.get("root")!.width).toBe(100); // 20+10+40+10+20
    expect(boxes.get("root")!.height).toBe(30);
  });

  it("distributes free space with justify: space-between in a column", () => {
    const root: FlowNode = {
      id: "root",
      direction: "column",
      justify: "space-between",
      fixedHeight: 100,
      children: [leaf("a", 50, 10), leaf("b", 50, 10)],
    };
    const boxes = layoutFlow(root, { maxWidth: 50 });
    expect(boxes.get("a")!.y).toBe(0);
    expect(boxes.get("b")!.y).toBe(90); // pushed to the bottom, 100 - 10
  });

  it("aligns cross-axis with align: center", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      align: "center",
      fixedHeight: 100,
      children: [leaf("a", 20, 20)],
    };
    const boxes = layoutFlow(root, { maxWidth: 50 });
    expect(boxes.get("a")!.y).toBe(40); // (100 - 20) / 2
  });

  it("nests a row of columns without throwing", () => {
    const col = (id: string): FlowNode => ({
      id,
      direction: "column",
      gap: 4,
      children: [leaf(`${id}-1`, 30, 10), leaf(`${id}-2`, 30, 10)],
    });
    const root: FlowNode = {
      id: "root",
      direction: "row",
      gap: 8,
      children: [col("c1"), col("c2")],
    };
    const boxes = layoutFlow(root, { maxWidth: 200 });
    expect(boxes.get("c1-2")!.y).toBe(14); // 10 + 4
    expect(boxes.get("c2")!.x).toBe(38); // 30 + 8
  });

  it("grows the container when a leaf's measured size exceeds the constraint (auto-grow)", () => {
    const root: FlowNode = {
      id: "root",
      direction: "column",
      children: [leaf("tall", 50, 500)],
    };
    const boxes = layoutFlow(root, { maxWidth: 50, maxHeight: 100 });
    // must not throw/clip — container reports the leaf's real height
    expect(boxes.get("root")!.height).toBe(500);
    expect(boxes.get("tall")!.height).toBe(500);
  });

  it("passes the constrained width down to a leaf's measure function", () => {
    let receivedWidth = -1;
    const root: FlowNode = {
      id: "root",
      direction: "column",
      fixedWidth: 120,
      children: [
        {
          id: "leaf",
          measure: (constraint) => {
            receivedWidth = constraint.maxWidth;
            return { width: constraint.maxWidth, height: 20 };
          },
        },
      ],
    };
    layoutFlow(root, { maxWidth: 999 });
    expect(receivedWidth).toBe(120);
  });
});
