/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HeaderNode } from "./Header.js";

describe("HeaderNode", () => {
  it("renders title and caption", () => {
    render(
      <HeaderNode
        id="header"
        type="header"
        data={{ title: "ROWS IN → WIRE BYTES OUT", caption: "BUILD · FREEZE · DISPATCH" }}
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
    expect(screen.getByText("ROWS IN → WIRE BYTES OUT")).toBeInTheDocument();
    expect(screen.getByText("BUILD · FREEZE · DISPATCH")).toBeInTheDocument();
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(
      <HeaderNode
        id="header"
        type="header"
        data={{ title: "T", className: "extra-header-class" }}
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
    expect(root.className).toContain("extra-header-class");
    expect(root.className).toMatch(/rounded-(md|lg|xl)/);
  });
});
