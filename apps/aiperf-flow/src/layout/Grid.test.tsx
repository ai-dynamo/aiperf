/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Grid } from "./Grid.js";

describe("Grid", () => {
  it("renders children in a grid", () => {
    const { container } = render(
      <Grid columns={2}>
        <span>one</span>
        <span>two</span>
      </Grid>,
    );
    expect(screen.getByText("one")).toBeInTheDocument();
    expect(screen.getByText("two")).toBeInTheDocument();
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("grid");
  });

  it("maps a numeric columns prop to a static grid-cols-N class", () => {
    const { container } = render(<Grid columns={4}>content</Grid>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("grid-cols-4");
    expect(root.style.gridTemplateColumns).toBe("");
  });

  it("maps columns=1 and columns=12 (lookup table bounds)", () => {
    const { container: c1 } = render(<Grid columns={1}>content</Grid>);
    expect((c1.firstElementChild as HTMLElement).className).toContain("grid-cols-1");
    const { container: c12 } = render(<Grid columns={12}>content</Grid>);
    expect((c12.firstElementChild as HTMLElement).className).toContain("grid-cols-12");
  });

  it("applies a string columns prop as an inline gridTemplateColumns style", () => {
    const { container } = render(<Grid columns="1fr 2fr">content</Grid>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.style.gridTemplateColumns).toBe("1fr 2fr");
    expect(root.className).not.toMatch(/grid-cols-\d/);
  });

  it("applies a default gap via inline style", () => {
    const { container } = render(<Grid columns={2}>content</Grid>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.style.gap).toBe("16px");
  });

  it("applies a caller-supplied gap in pixels", () => {
    const { container } = render(
      <Grid columns={2} gap={24}>
        content
      </Grid>,
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.style.gap).toBe("24px");
  });

  it("maps align to the matching items-* class", () => {
    const { container } = render(
      <Grid columns={2} align="end">
        content
      </Grid>,
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("items-end");
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(
      <Grid columns={2} className="extra-grid-class">
        content
      </Grid>,
    );
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-grid-class");
    expect(root.className).toContain("grid");
  });
});
