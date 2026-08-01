/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { IntervalsNode } from "./Intervals.js";
import type { IntervalsNodeData } from "./types.js";

const DATA: IntervalsNodeData = {
  title: "Intervals on the warped clock",
  rows: [
    { id: "P0", label: "parent", start: 0, end: 1, role: "blue" },
    { id: "B0", label: "Explore #2", start: 1.3, end: 5, role: "green", dashed: true },
    { id: "A0", label: "Explore #1", start: 1.2, end: 4, role: "green" },
  ],
};

function renderNode(data: IntervalsNodeData) {
  return render(
    <ReactFlowProvider>
      <IntervalsNode
        id="iv"
        type="intervals"
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

describe("IntervalsNode", () => {
  it("draws an id and label per row", () => {
    renderNode(DATA);
    expect(screen.getByText("P0")).toBeInTheDocument();
    expect(screen.getByText("Explore #1")).toBeInTheDocument();
    expect(screen.getByText("Explore #2")).toBeInTheDocument();
  });

  it("badges the derived global rank, not the authored row position", () => {
    const { container } = renderNode(DATA);
    const badges = [...container.querySelectorAll("text")]
      .map((t) => t.textContent)
      .filter((t) => t === "0" || t === "1" || t === "2");
    // A0 is authored last but starts before B0, so it ranks 1 and B0 ranks 2.
    expect(badges).toEqual(["0", "2", "1"]);
  });

  it("dashes an async-launched interval so it reads as non-serializing", () => {
    const { container } = renderNode(DATA);
    const dashed = [...container.querySelectorAll("rect[stroke-dasharray]")];
    expect(dashed).toHaveLength(1);
  });
});
