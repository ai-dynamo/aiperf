/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { SlicesNode } from "./Slices.js";
import type { SlicesNodeData } from "./types.js";

const DATA: SlicesNodeData = {
  title: "slice_duration = 15 ns",
  requests: [
    { id: "r0", start: 0, end: 20 },
    { id: "r1", start: 3, end: 30 },
    { id: "r2", start: 28, end: 50 },
  ],
  duration: 15,
};

function renderNode(data: SlicesNodeData) {
  return render(
    <ReactFlowProvider>
      <SlicesNode
        id="sl"
        type="slices"
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

describe("SlicesNode", () => {
  it("stars the incomplete trailing slice so a diluted rate is visible", () => {
    renderNode(DATA);
    expect(screen.getByText("slice 0")).toBeInTheDocument();
    expect(screen.getByText("slice 3 *")).toBeInTheDocument();
  });

  it("leaves every slice unstarred when the grid divides the span exactly", () => {
    renderNode({ ...DATA, duration: 25 });
    expect(screen.getByText("slice 0")).toBeInTheDocument();
    expect(screen.getByText("slice 1")).toBeInTheDocument();
    expect(screen.queryByText(/\*/)).not.toBeInTheDocument();
  });

  it("marks each request's start, the key it bins on", () => {
    const { container } = renderNode(DATA);
    expect(container.querySelectorAll("circle")).toHaveLength(DATA.requests.length);
  });
});
