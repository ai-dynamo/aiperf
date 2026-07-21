/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Eyebrow } from "./Eyebrow.js";

describe("Eyebrow", () => {
  it("renders its children uppercase-styled", () => {
    render(<Eyebrow>Product landscape</Eyebrow>);
    expect(screen.getByText("Product landscape").className).toContain("uppercase");
  });

  it("defaults to tertiary ink", () => {
    render(<Eyebrow>Section</Eyebrow>);
    expect(screen.getByText("Section").className).toContain("text-ink-tertiary");
  });

  it("colors by category when a tone is given", () => {
    render(<Eyebrow tone="green">Built</Eyebrow>);
    const el = screen.getByText("Built");
    expect(el.className).toContain("text-category-green");
    expect(el.className).not.toContain("text-ink-tertiary");
  });

  it("merges a caller-supplied className", () => {
    render(<Eyebrow className="extra-eyebrow-class">Section</Eyebrow>);
    expect(screen.getByText("Section").className).toContain("extra-eyebrow-class");
  });
});
