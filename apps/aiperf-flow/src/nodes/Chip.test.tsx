/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { ChipNode } from "./Chip.js";

// Chip now carries (invisible) React Flow handles so it can be an edge endpoint, so its tests must
// render inside a provider like the other handled node types.
const baseProps = {
  type: "chip" as const,
  selected: false,
  zIndex: 0,
  isConnectable: false,
  positionAbsoluteX: 0,
  positionAbsoluteY: 0,
  dragging: false,
  draggable: false,
  selectable: false,
  deletable: false,
};

describe("ChipNode", () => {
  it("renders its label", () => {
    render(
      <ReactFlowProvider>
        <ChipNode id="chip0a" data={{ label: "one identity" }} {...baseProps} />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("one identity")).toBeInTheDocument();
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(
      <ReactFlowProvider>
        <ChipNode id="c" data={{ label: "L", className: "extra-chip-class" }} {...baseProps} />
      </ReactFlowProvider>,
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-chip-class");
    expect(root.className).toMatch(/rounded-(md|lg|xl)/);
  });
});
