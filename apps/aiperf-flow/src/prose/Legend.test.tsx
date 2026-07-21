/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Legend } from "./Legend.js";

describe("Legend", () => {
  it("renders a label for each entry", () => {
    render(
      <Legend
        entries={[
          { color: "green", label: "Healthy" },
          { color: "red", label: "Failed" },
        ]}
      />,
    );
    expect(screen.getByText("Healthy")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("renders one swatch per entry with the matching category color", () => {
    const { container } = render(
      <Legend
        entries={[
          { color: "green", label: "Healthy" },
          { color: "red", label: "Failed" },
        ]}
      />,
    );
    expect(container.querySelectorAll(".bg-category-green")).toHaveLength(1);
    expect(container.querySelectorAll(".bg-category-red")).toHaveLength(1);
  });

  it("lays out entries in a wrapping flex row", () => {
    const { container } = render(<Legend entries={[{ color: "blue", label: "Active" }]} />);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("flex");
    expect(root.className).toContain("flex-wrap");
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(
      <Legend entries={[{ color: "blue", label: "Active" }]} className="extra-legend-class" />,
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-legend-class");
  });
});
