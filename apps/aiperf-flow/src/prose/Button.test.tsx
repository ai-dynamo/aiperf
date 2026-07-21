/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Button } from "./Button.js";

describe("Button", () => {
  it("renders its label and calls onClick", () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Run all</Button>);
    fireEvent.click(screen.getByRole("button", { name: "Run all" }));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it("applies primary variant classes by default differently from secondary", () => {
    const { rerender } = render(<Button variant="primary">A</Button>);
    expect(screen.getByRole("button").className).toContain("bg-accent-primary");
    rerender(<Button variant="secondary">A</Button>);
    expect(screen.getByRole("button").className).not.toContain("bg-accent-primary");
  });

  it("merges a caller-supplied className", () => {
    render(<Button className="extra-button-class">A</Button>);
    expect(screen.getByRole("button").className).toContain("extra-button-class");
  });

  it("respects the disabled prop", () => {
    render(<Button disabled>A</Button>);
    expect(screen.getByRole("button")).toBeDisabled();
  });
});
