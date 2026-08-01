/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { SweepNode } from "./Sweep.js";
import type { SweepNodeData } from "./types.js";

const DATA: SweepNodeData = {
  title: "Sweep line",
  requests: [
    { id: "r0", start: 0, gen: 6, end: 20, tokens: 120 },
    { id: "r1", start: 3, gen: 10, end: 30, tokens: 200 },
  ],
  valueLabel: "concurrent requests",
  axisLabel: "time (relative ns)",
};

function renderNode(data: SweepNodeData) {
  return render(
    <ReactFlowProvider>
      <SweepNode
        id="sw"
        type="sweep"
        data={data}
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
}

describe("SweepNode", () => {
  it("labels each request row and both axes", () => {
    renderNode(DATA);
    expect(screen.getByText("r0")).toBeInTheDocument();
    expect(screen.getByText("r1")).toBeInTheDocument();
    expect(screen.getByText("concurrent requests")).toBeInTheDocument();
    expect(screen.getByText("time (relative ns)")).toBeInTheDocument();
  });

  it("draws one event tick per interval endpoint, which is what moves the curve", () => {
    const { container } = renderNode(DATA);
    // Two requests contribute a +delta and a -delta each.
    const ticks = [...container.querySelectorAll("line")].filter(
      (l) => Math.abs(Number(l.getAttribute("y2")) - Number(l.getAttribute("y1"))) === 8,
    );
    expect(ticks).toHaveLength(4);
  });

  it("splits each bar into a prefill and a decode span", () => {
    const { container } = renderNode(DATA);
    expect(container.querySelectorAll("rect")).toHaveLength(DATA.requests.length * 2);
  });
});
