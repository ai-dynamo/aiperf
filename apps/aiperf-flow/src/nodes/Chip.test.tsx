/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ChipNode } from "./Chip.js";

describe("ChipNode", () => {
  it("renders its label", () => {
    render(
      <ChipNode
        id="chip0a"
        type="chip"
        data={{ label: "one identity" }}
        selected={false}
        zIndex={0}
        isConnectable={false}
        positionAbsoluteX={0}
        positionAbsoluteY={0}
        dragging={false}
        draggable={false}
        selectable={false}
        deletable={false}
      />,
    );
    expect(screen.getByText("one identity")).toBeInTheDocument();
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(
      <ChipNode
        id="c"
        type="chip"
        data={{ label: "L", className: "extra-chip-class" }}
        selected={false}
        zIndex={0}
        isConnectable={false}
        positionAbsoluteX={0}
        positionAbsoluteY={0}
        dragging={false}
        draggable={false}
        selectable={false}
        deletable={false}
      />,
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-chip-class");
    expect(root.className).toContain("rounded-none");
  });
});
