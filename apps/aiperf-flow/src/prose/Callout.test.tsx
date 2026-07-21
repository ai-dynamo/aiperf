/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Callout } from "./Callout.js";

describe("Callout", () => {
  it("renders body content", () => {
    render(<Callout>Rolling deploy is in progress.</Callout>);
    expect(screen.getByText("Rolling deploy is in progress.")).toBeInTheDocument();
  });

  it("renders an optional title line", () => {
    render(<Callout title="Heads up">Body text</Callout>);
    expect(screen.getByText("Heads up")).toBeInTheDocument();
  });

  it("omits the title line when absent", () => {
    render(<Callout>Body text</Callout>);
    expect(screen.queryByText("Heads up")).not.toBeInTheDocument();
  });

  it("defaults to the info tone (blue category)", () => {
    const { container } = render(<Callout>Body</Callout>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("border-l-category-blue");
  });

  it("maps each tone to its category color", () => {
    const cases: Array<["info" | "warning" | "danger" | "success" | "neutral", string]> = [
      ["info", "border-l-category-blue"],
      ["warning", "border-l-category-yellow"],
      ["danger", "border-l-category-red"],
      ["success", "border-l-category-green"],
      ["neutral", "border-l-category-gray"],
    ];
    for (const [tone, expected] of cases) {
      const { container } = render(<Callout tone={tone}>Body</Callout>);
      const root = container.firstElementChild as HTMLElement;
      expect(root.className).toContain(expected);
    }
  });

  it("merges a caller-supplied className onto its own root classes", () => {
    const { container } = render(<Callout className="extra-callout-class">Body</Callout>);
    const root = container.firstElementChild as HTMLElement;
    expect(root.className).toContain("extra-callout-class");
    expect(root.className).toMatch(/rounded-(sm|md|lg|xl|full)/);
  });
});
