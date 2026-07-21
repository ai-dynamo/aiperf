/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Pill } from "./Pill.js";

describe("Pill", () => {
  it("renders as a span by default", () => {
    render(<Pill>WEKA_UNIFIED_STORE</Pill>);
    const el = screen.getByText("WEKA_UNIFIED_STORE");
    expect(el.tagName).toBe("SPAN");
  });

  it("applies the inactive palette by default", () => {
    render(<Pill>tag</Pill>);
    expect(screen.getByText("tag").className).toContain("text-ink-secondary");
  });

  it("applies the active accent palette when active", () => {
    render(<Pill active>tag</Pill>);
    expect(screen.getByText("tag").className).toContain("text-accent-primary");
  });

  it("colors by category when a tone is given, overriding active styling", () => {
    render(
      <Pill tone="green" active>
        ready
      </Pill>,
    );
    const el = screen.getByText("ready");
    expect(el.className).toContain("text-category-green");
    expect(el.className).toContain("bg-category-green/10");
    expect(el.className).not.toContain("text-accent-primary");
  });

  it("renders as a clickable button with aria-pressed when onClick is given", () => {
    const onClick = vi.fn();
    render(
      <Pill active onClick={onClick}>
        cell-0
      </Pill>,
    );
    const button = screen.getByRole("button", { name: "cell-0" });
    expect(button).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(button);
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("merges a caller-supplied className", () => {
    render(<Pill className="extra-pill-class">tag</Pill>);
    expect(screen.getByText("tag").className).toContain("extra-pill-class");
  });

  it("applies an accessible label when the visible text alone doesn't convey meaning", () => {
    render(<Pill ariaLabel="Implementation status: Rejected">Rejected</Pill>);
    expect(screen.getByLabelText("Implementation status: Rejected")).toBeInTheDocument();
  });
});
