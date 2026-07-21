/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Stack } from "./Stack.js";

describe("Stack", () => {
  it("renders children in a vertical flex column", () => {
    const { container } = render(
      <Stack>
        <span>one</span>
        <span>two</span>
      </Stack>,
    );
    expect(screen.getByText("one")).toBeInTheDocument();
    expect(screen.getByText("two")).toBeInTheDocument();
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("flex");
    expect(root.className).toContain("flex-col");
  });

  it("applies a default gap via inline style", () => {
    const { container } = render(<Stack>content</Stack>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.style.gap).toBe("16px");
  });

  it("applies a caller-supplied gap in pixels", () => {
    const { container } = render(<Stack gap={8}>content</Stack>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.style.gap).toBe("8px");
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(<Stack className="extra-stack-class">content</Stack>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-stack-class");
    expect(root.className).toContain("flex-col");
  });
});
