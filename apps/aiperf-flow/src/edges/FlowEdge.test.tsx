/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from "@testing-library/react";
import { Position, ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { FlowEdge } from "./FlowEdge.js";

const baseProps = {
  id: "e1",
  source: "a",
  target: "b",
  sourceX: 0,
  sourceY: 0,
  targetX: 200,
  targetY: 100,
  sourcePosition: Position.Right,
  targetPosition: Position.Left,
} as const;

function renderEdge(extra?: Partial<React.ComponentProps<typeof FlowEdge>>) {
  return render(
    <ReactFlowProvider>
      <svg>
        <FlowEdge {...baseProps} {...extra} />
      </svg>
    </ReactFlowProvider>,
  );
}

describe("FlowEdge", () => {
  it("renders an SVG path with a non-empty d attribute", () => {
    const { container } = renderEdge();
    const path = container.querySelector("path.flow-edge__path");
    expect(path).not.toBeNull();
    expect(path?.getAttribute("d")).toBeTruthy();
  });

  it("sets a dashed strokeDasharray on the path", () => {
    const { container } = renderEdge();
    const path = container.querySelector("path.flow-edge__path") as HTMLElement;
    expect(path.style.strokeDasharray).toBeTruthy();
  });

  it("carries a distinctive animation marker class that CSS animates via stroke-dashoffset", () => {
    const { container } = renderEdge();
    const path = container.querySelector("path.flow-edge__path");
    expect(path?.classList.contains("flow-edge__path")).toBe(true);
    expect(container.querySelector("style")).not.toBeNull();
  });

  // These assert inline style, not the SVG `stroke` attribute. React Flow's stylesheet
  // styles `.react-flow__edge-path`, and a CSS rule outranks a presentation attribute —
  // so an attribute-based assertion passes while the edge renders in default gray.
  it("defaults stroke color to the accent-primary CSS variable", () => {
    const { container } = renderEdge();
    const path = container.querySelector("path.flow-edge__path") as HTMLElement;
    expect(path.style.stroke).toBe("var(--color-accent-primary)");
  });

  it("uses a caller-supplied color from data", () => {
    const { container } = renderEdge({ data: { color: "var(--color-category-blue)" } });
    const path = container.querySelector("path.flow-edge__path") as HTMLElement;
    expect(path.style.stroke).toBe("var(--color-category-blue)");
  });

  it("carries stroke width as style so the stylesheet cannot override it", () => {
    const { container } = renderEdge();
    const path = container.querySelector("path.flow-edge__path") as HTMLElement;
    expect(path.style.strokeWidth).toBe("2");
  });

  it("maps speed to a distinct animation-duration custom property", () => {
    const { container: slow } = renderEdge({ data: { speed: "slow" } });
    const { container: fast } = renderEdge({ data: { speed: "fast" } });
    const slowPath = slow.querySelector("path.flow-edge__path") as HTMLElement;
    const fastPath = fast.querySelector("path.flow-edge__path") as HTMLElement;
    expect(slowPath.style.getPropertyValue("--flow-edge-duration")).not.toBe(
      fastPath.style.getPropertyValue("--flow-edge-duration"),
    );
  });
});
