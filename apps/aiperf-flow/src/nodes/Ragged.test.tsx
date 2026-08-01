/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { RaggedNode } from "./Ragged.js";
import type { RaggedNodeData } from "./types.js";

const DATA: RaggedNodeData = {
  title: "inter_chunk_latency",
  lists: [[5, 7], [], [9, 1, 4]],
  highlight: 2,
};

function renderNode(data: RaggedNodeData) {
  return render(
    <ReactFlowProvider>
      <RaggedNode
        id="rg"
        type="ragged"
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

describe("RaggedNode", () => {
  it("shows the flat arrays alongside the ragged rows", () => {
    renderNode(DATA);
    expect(screen.getByText("values")).toBeInTheDocument();
    expect(screen.getByText("record_indices")).toBeInTheDocument();
    expect(screen.getByText("offsets")).toBeInTheDocument();
  });

  it("renders an empty record as absent, with offset -1", () => {
    renderNode(DATA);
    expect(screen.getByText("empty")).toBeInTheDocument();
    expect(screen.getByText("−1")).toBeInTheDocument();
  });

  it("drops the flat section when showFlat is false", () => {
    renderNode({ ...DATA, showFlat: false });
    expect(screen.getByText("r0")).toBeInTheDocument();
    expect(screen.queryByText("record_indices")).not.toBeInTheDocument();
  });
});
