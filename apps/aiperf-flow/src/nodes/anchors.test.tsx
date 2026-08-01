/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from "@testing-library/react";
import { ReactFlowProvider, type Edge, type Node } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { CardNode } from "./Card.js";
import { NODE_ANCHORS, autoRouteEdges, chooseAnchors } from "./anchors.js";

/** Square nodes, so the 2:1 midpoint-vs-corner threshold is easy to reason about. */
function node(id: string, x: number, y: number): Node {
  return { id, position: { x, y }, width: 100, height: 100, data: {} };
}

const edge = (source: string, target: string): Edge => ({ id: `${source}-${target}`, source, target });

describe("chooseAnchors", () => {
  it("uses facing midpoints for runs along one axis", () => {
    expect(chooseAnchors({ x: 0, y: 0 }, { x: 300, y: 0 })).toEqual({ source: "e", target: "w" });
    expect(chooseAnchors({ x: 300, y: 0 }, { x: 0, y: 0 })).toEqual({ source: "w", target: "e" });
    expect(chooseAnchors({ x: 0, y: 0 }, { x: 0, y: 300 })).toEqual({ source: "s", target: "n" });
    expect(chooseAnchors({ x: 0, y: 300 }, { x: 0, y: 0 })).toEqual({ source: "n", target: "s" });
  });

  it("uses facing corners for a genuinely diagonal run", () => {
    expect(chooseAnchors({ x: 0, y: 0 }, { x: 200, y: 200 })).toEqual({
      source: "se",
      target: "nw",
    });
    expect(chooseAnchors({ x: 0, y: 200 }, { x: 200, y: 0 })).toEqual({
      source: "ne",
      target: "sw",
    });
  });

  it("treats a shallow diagonal as horizontal at the 2:1 threshold", () => {
    // dx is exactly twice dy — still a mostly-horizontal run, so midpoints win.
    expect(chooseAnchors({ x: 0, y: 0 }, { x: 200, y: 100 })).toEqual({ source: "e", target: "w" });
    // Past the threshold it becomes a corner hop.
    expect(chooseAnchors({ x: 0, y: 0 }, { x: 200, y: 101 })).toEqual({
      source: "se",
      target: "nw",
    });
  });
});

describe("autoRouteEdges", () => {
  it("anchors each edge from its endpoints' geometry", () => {
    const nodes = [node("a", 0, 0), node("b", 400, 0), node("c", 0, 400)];
    const routed = autoRouteEdges(nodes, [edge("a", "b"), edge("a", "c")]);

    expect(routed[0]).toMatchObject({ sourceHandle: "e", targetHandle: "w" });
    expect(routed[1]).toMatchObject({ sourceHandle: "s", targetHandle: "n" });
  });

  it("leaves an explicitly anchored edge alone", () => {
    const nodes = [node("a", 0, 0), node("b", 400, 0)];
    const authored = { ...edge("a", "b"), sourceHandle: "nw", targetHandle: "se" };

    expect(autoRouteEdges(nodes, [authored])[0]).toBe(authored);
  });

  it("passes through an edge whose endpoints are missing", () => {
    const dangling = edge("a", "ghost");
    expect(autoRouteEdges([node("a", 0, 0)], [dangling])[0]).toBe(dangling);
  });
});

describe("node anchor handles", () => {
  it("exposes every anchor as both a source and a target", () => {
    const { container } = render(
      <ReactFlowProvider>
        <CardNode
          id="n"
          type="card"
          data={{ title: "t" }}
          selected={false}
          zIndex={0}
          isConnectable={false}
          positionAbsoluteX={0}
          positionAbsoluteY={0}
          dragging={false}
          draggable={false}
          selectable={false}
          deletable={false}
        />
      </ReactFlowProvider>,
    );

    for (const anchor of NODE_ANCHORS) {
      const handles = container.querySelectorAll(`[data-handleid="${anchor}"]`);
      // One source and one target per anchor: an edge may attach to either end.
      expect(handles.length, `anchor ${anchor}`).toBe(2);
    }
  });
});
