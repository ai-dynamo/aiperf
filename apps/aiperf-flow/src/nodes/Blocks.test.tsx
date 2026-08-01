/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { BlocksNode } from "./Blocks.js";
import { blocksNodeSize } from "./blocksLayout.js";
import type { BlocksNodeData } from "./types.js";

const DATA: BlocksNodeData = {
  title: "advance_turn relabels block 20",
  strips: [
    { label: "parent chain", cells: [...Array(3).fill("blue"), "purple"] },
    { label: "forking subagent", cells: [...Array(4).fill("blue")] },
  ],
  highlight: 3,
};

function renderNode(data: BlocksNodeData) {
  return render(
    <ReactFlowProvider>
      <BlocksNode
        id="bl"
        type="blocks"
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

describe("BlocksNode", () => {
  it("labels every strip", () => {
    renderNode(DATA);
    expect(screen.getByText("parent chain")).toBeInTheDocument();
    expect(screen.getByText("forking subagent")).toBeInTheDocument();
  });

  it("outlines the highlighted column on every strip, so the eye compares like with like", () => {
    const { container } = renderNode(DATA);
    const outlined = [...container.querySelectorAll("span[style]")].filter((el) =>
      (el as HTMLElement).style.outline.includes("solid"),
    );
    expect(outlined).toHaveLength(DATA.strips.length);
  });

  it("dims cells away from the highlight", () => {
    const { container } = renderNode(DATA);
    const cells = [...container.querySelectorAll("span[aria-hidden]")] as HTMLElement[];
    expect(cells).toHaveLength(8);
    expect(cells[0]!.style.opacity).toBe("0.55");
    expect(cells[3]!.style.opacity).toBe("1");
  });
});

describe("blocksNodeSize", () => {
  it("widens with the longest strip and heightens with strip count", () => {
    const one = blocksNodeSize({ strips: [DATA.strips[0]!], hasTitle: true, hasDetail: false });
    const two = blocksNodeSize({ strips: DATA.strips, hasTitle: true, hasDetail: false });
    const wide = blocksNodeSize({
      strips: [{ label: "long", cells: Array(23).fill("blue") }],
      hasTitle: true,
      hasDetail: false,
    });

    expect(two.height).toBeGreaterThan(one.height);
    expect(wide.width).toBeGreaterThan(one.width);
  });
});
