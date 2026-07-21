/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { PanelNode } from "./Panel.js";

describe("PanelNode", () => {
  it("renders title and detail", () => {
    render(
      <ReactFlowProvider>
        <PanelNode
          id="turn-body"
          type="panel"
          data={{ title: "Turn.body", detail: "SmallVec handles" }}
          selected={false}
          zIndex={0}
          isConnectable={false}
          positionAbsoluteX={0}
          positionAbsoluteY={0}
          dragging={false}
        />
      </ReactFlowProvider>,
    );
    expect(screen.getByText("Turn.body")).toBeInTheDocument();
    expect(screen.getByText("SmallVec handles")).toBeInTheDocument();
  });

  it("omits the detail line when absent", () => {
    render(
      <ReactFlowProvider>
        <PanelNode
          id="p"
          type="panel"
          data={{ title: "Only title" }}
          selected={false}
          zIndex={0}
          isConnectable={false}
          positionAbsoluteX={0}
          positionAbsoluteY={0}
          dragging={false}
        />
      </ReactFlowProvider>,
    );
    expect(screen.queryByText("SmallVec handles")).not.toBeInTheDocument();
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(
      <ReactFlowProvider>
        <PanelNode
          id="p"
          type="panel"
          data={{ title: "T", className: "extra-panel-class" }}
          selected={false}
          zIndex={0}
          isConnectable={false}
          positionAbsoluteX={0}
          positionAbsoluteY={0}
          dragging={false}
        />
      </ReactFlowProvider>,
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-panel-class");
    expect(root.className).toContain("rounded-none");
  });
});
