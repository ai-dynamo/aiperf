/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Swatch } from "./Swatch.js";

describe("Swatch", () => {
  it("renders a colored square using the category background class", () => {
    const { container } = render(<Swatch color="green" />);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("bg-category-green");
  });

  it("maps each category role to its background class", () => {
    const roles: Array<"green" | "yellow" | "purple" | "blue" | "red" | "orange" | "cyan" | "gray"> = [
      "green",
      "yellow",
      "purple",
      "blue",
      "red",
      "orange",
      "cyan",
      "gray",
    ];
    for (const role of roles) {
      const { container } = render(<Swatch color={role} />);
      const root = container.firstElementChild as HTMLElement;
      expect(root.className).toContain(`bg-category-${role}`);
    }
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(<Swatch color="blue" className="extra-swatch-class" />);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-swatch-class");
    expect(root.className).toContain("bg-category-blue");
  });
});
