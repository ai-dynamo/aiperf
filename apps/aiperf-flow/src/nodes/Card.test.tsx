/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { CardNode } from "./Card.js";

describe("CardNode", () => {
  it("renders title, subtitle, and detail", () => {
    render(
      <ReactFlowProvider>
        <CardNode
          id="authored"
          type="card"
          data={{ title: "Authored run", subtitle: "Config v2 request", detail: "one identity" }}
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
    expect(screen.getByText("Authored run")).toBeInTheDocument();
    expect(screen.getByText("Config v2 request")).toBeInTheDocument();
    expect(screen.getByText("one identity")).toBeInTheDocument();
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(
      <ReactFlowProvider>
        <CardNode
          id="c"
          type="card"
          data={{ title: "T", className: "extra-card-class" }}
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
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-card-class");
    expect(root.className).toContain("rounded-none");
  });
});
