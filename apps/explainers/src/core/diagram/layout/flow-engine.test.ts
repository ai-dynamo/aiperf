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

  it("wraps children onto a new line when a row exceeds the width budget", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      wrap: "wrap",
      gap: 10,
      children: [leaf("a", 40, 20), leaf("b", 40, 20), leaf("c", 40, 20)],
    };
    // budget fits exactly 2 per line: 40+10+40 = 90
    const boxes = layoutFlow(root, { maxWidth: 90 });
    expect(boxes.get("a")).toMatchObject({ x: 0, y: 0 });
    expect(boxes.get("b")).toMatchObject({ x: 50, y: 0 });
    // c wraps to a second line, back at x=0, below the first line's height + gap
    expect(boxes.get("c")).toMatchObject({ x: 0, y: 30 }); // 20 + 10 (rowGap==gap)
    expect(boxes.get("root")!.height).toBe(50); // 20 + 10 + 20
  });

  it("never drops or splits a single child wider than the wrap budget", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      wrap: "wrap",
      children: [leaf("wide", 500, 20)],
    };
    const boxes = layoutFlow(root, { maxWidth: 90 });
    expect(boxes.get("wide")).toMatchObject({ x: 0, y: 0, width: 500 });
  });

  it("distributes multiple lines with alignContent: space-between", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      wrap: "wrap",
      alignContent: "space-between",
      fixedHeight: 100,
      children: [leaf("a", 40, 20), leaf("b", 40, 20), leaf("c", 40, 20)],
    };
    const boxes = layoutFlow(root, { maxWidth: 90 });
    // two lines (a+b on line 1, c on line 2), 20px tall each, spread across 100px
    expect(boxes.get("a")!.y).toBe(0);
    expect(boxes.get("c")!.y).toBe(80); // 100 - 20
  });

  it("distributes extra space proportionally via grow", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      fixedWidth: 200,
      children: [
        { ...leaf("a", 50, 10), grow: 1 },
        { ...leaf("b", 50, 10), grow: 3 },
      ],
    };
    const boxes = layoutFlow(root, { maxWidth: 200 });
    // 100px free (200 - 100 base), split 1:3 => a gets +25, b gets +75
    expect(boxes.get("a")!.width).toBe(75);
    expect(boxes.get("b")!.width).toBe(125);
  });

  it("shrinks children proportionally to their base size when content overflows", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      fixedWidth: 100,
      children: [leaf("a", 80, 10), leaf("b", 80, 10)],
    };
    const boxes = layoutFlow(root, { maxWidth: 100 });
    // 160 base vs 100 available => -60 overflow, split by (shrink*basis) weight,
    // both equal here (default shrink=1, equal basis) => -30 each
    expect(boxes.get("a")!.width).toBe(50);
    expect(boxes.get("b")!.width).toBe(50);
  });

  it("applies per-child margin as extra spacing on all four sides", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      children: [{ ...leaf("a", 20, 20), margin: 5 }, leaf("b", 20, 20)],
    };
    const boxes = layoutFlow(root, { maxWidth: 200 });
    expect(boxes.get("a")).toMatchObject({ x: 5, y: 5, width: 20, height: 20 });
    // b starts after a's box (20) + a's right margin (5) + a's left margin already consumed at start
    expect(boxes.get("b")!.x).toBe(30); // 5 (a's left) + 20 (a) + 5 (a's right)
    expect(boxes.get("root")!.width).toBe(50); // 5+20+5 + 20
  });

  it("clamps a leaf's measured size to minWidth/maxWidth/minHeight/maxHeight", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row",
      children: [
        { ...leaf("tiny", 5, 5), minWidth: 30, minHeight: 30 },
        { ...leaf("huge", 500, 500), maxWidth: 60, maxHeight: 60 },
      ],
    };
    const boxes = layoutFlow(root, { maxWidth: 1000 });
    expect(boxes.get("tiny")).toMatchObject({ width: 30, height: 30 });
    expect(boxes.get("huge")).toMatchObject({ width: 60, height: 60 });
  });

  it("supports row-reverse, mirroring child order along the main axis", () => {
    const root: FlowNode = {
      id: "root",
      direction: "row-reverse",
      gap: 10,
      children: [leaf("a", 20, 10), leaf("b", 20, 10)],
    };
    const boxes = layoutFlow(root, { maxWidth: 200 });
    // reversed: b comes first visually (at x=0), a follows
    expect(boxes.get("b")!.x).toBe(0);
    expect(boxes.get("a")!.x).toBe(30);
  });

  it("measures every leaf exactly once, even across grow/shrink/wrap distribution", () => {
    let calls = 0;
    const countingLeaf: FlowNode = {
      id: "counted",
      measure: () => {
        calls += 1;
        return { width: 40, height: 20 };
      },
    };
    const root: FlowNode = {
      id: "root",
      direction: "row",
      wrap: "wrap",
      fixedWidth: 200,
      children: [{ ...countingLeaf, grow: 1 }],
    };
    layoutFlow(root, { maxWidth: 200 });
    expect(calls).toBe(1);
  });
});
