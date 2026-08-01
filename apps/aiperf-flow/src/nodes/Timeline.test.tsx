/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { TimelineNode } from "./Timeline.js";
import type { TimelineNodeData } from "./types.js";

const DATA: TimelineNodeData = {
  title: "Idle warp",
  lanes: ["main", "sub"],
  bars: [
    { id: "m0", lane: "main", rawStart: 0, rawEnd: 4, warpStart: 0, warpEnd: 4 },
    { id: "s0", lane: "sub", rawStart: 20, rawEnd: 26, warpStart: 6, warpEnd: 12 },
  ],
  gaps: [{ start: 4, end: 20, idle: 16, capped: true }],
};

function renderNode(data: TimelineNodeData) {
  return render(
    <ReactFlowProvider>
      <TimelineNode
        id="tl"
        type="timeline"
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

describe("TimelineNode", () => {
  it("draws both clocks, with every bar appearing once per block", () => {
    renderNode(DATA);

    expect(screen.getByText("raw clock")).toBeInTheDocument();
    expect(screen.getByText("warped clock")).toBeInTheDocument();
    expect(screen.getAllByText("m0")).toHaveLength(2);
    expect(screen.getAllByText("s0")).toHaveLength(2);
    // One label per lane per block.
    expect(screen.getAllByText("main")).toHaveLength(2);
  });

  it("drops the warped block and its bars when showWarp is false", () => {
    renderNode({ ...DATA, showWarp: false });

    expect(screen.getByText("raw clock")).toBeInTheDocument();
    expect(screen.queryByText("warped clock")).not.toBeInTheDocument();
    expect(screen.getAllByText("m0")).toHaveLength(1);
    expect(screen.getAllByText("main")).toHaveLength(1);
  });

  it("marks a gap that exceeds the cap, since that is the dead air the warp collapses", () => {
    renderNode(DATA);
    expect(screen.getByText(/idle 16s > cap/)).toBeInTheDocument();

    renderNode({ ...DATA, gaps: [{ start: 4, end: 8, idle: 4, capped: false }] });
    expect(screen.getByText(/idle 4s$/)).toBeInTheDocument();
  });
});
