/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Toggle } from "./Toggle.js";

describe("Toggle", () => {
  it("renders a button with role switch", () => {
    render(<Toggle checked={false} onChange={vi.fn()} />);
    const toggle = screen.getByRole("switch");
    expect(toggle.tagName).toBe("BUTTON");
  });

  it("reflects the checked state via aria-checked", () => {
    const { rerender } = render(<Toggle checked={false} onChange={vi.fn()} />);
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "false");
    rerender(<Toggle checked={true} onChange={vi.fn()} />);
    expect(screen.getByRole("switch")).toHaveAttribute("aria-checked", "true");
  });

  it("calls onChange with the inverted value when clicked", () => {
    const onChange = vi.fn();
    render(<Toggle checked={false} onChange={onChange} />);
    fireEvent.click(screen.getByRole("switch"));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("applies the accent-primary background when checked", () => {
    render(<Toggle checked={true} onChange={vi.fn()} />);
    expect(screen.getByRole("switch").className).toContain("bg-accent-primary");
  });

  it("does not apply the accent-primary background when unchecked", () => {
    render(<Toggle checked={false} onChange={vi.fn()} />);
    expect(screen.getByRole("switch").className).not.toContain("bg-accent-primary");
  });

  it("renders adjacent label text when label is given", () => {
    render(<Toggle checked={false} onChange={vi.fn()} label="Enable streaming" />);
    expect(screen.getByText("Enable streaming")).toBeDefined();
  });

  it("merges a caller-supplied className", () => {
    render(<Toggle checked={false} onChange={vi.fn()} className="extra-class" />);
    expect(screen.getByRole("switch").className).toContain("extra-class");
  });
});
