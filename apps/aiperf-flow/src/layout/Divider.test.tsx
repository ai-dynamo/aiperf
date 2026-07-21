/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Divider } from "./Divider.js";

describe("Divider", () => {
  it("renders a horizontal rule", () => {
    const { container } = render(<Divider />);
    const hr = container.querySelector("hr");
    expect(hr).toBeInTheDocument();
  });

  it("applies secondary stroke className", () => {
    const { container } = render(<Divider />);
    const hr = container.querySelector("hr") as HTMLElement;
    expect(hr.className).toContain("border-t");
    expect(hr.className).toContain("border-stroke-secondary");
  });

  it("merges a caller-supplied className", () => {
    const { container } = render(<Divider className="my-custom-class" />);
    const hr = container.querySelector("hr") as HTMLElement;
    expect(hr.className).toContain("my-custom-class");
    expect(hr.className).toContain("border-stroke-secondary");
  });
});
