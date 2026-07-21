/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PageTabs } from "./PageTabs.js";

type ExamplePageId = "overview" | "detail" | "trace";

const pages: ReadonlyArray<{ id: ExamplePageId; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "detail", label: "Detail" },
  { id: "trace", label: "Trace" },
];

describe("PageTabs", () => {
  it("renders a tab button for every page", () => {
    render(<PageTabs pages={pages} current="overview" onChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Overview" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Detail" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Trace" })).toBeInTheDocument();
  });

  it("marks the current page's tab as pressed and visually distinguished", () => {
    render(<PageTabs pages={pages} current="detail" onChange={vi.fn()} />);
    const current = screen.getByRole("button", { name: "Detail" });
    const other = screen.getByRole("button", { name: "Overview" });
    expect(current).toHaveAttribute("aria-pressed", "true");
    expect(other).toHaveAttribute("aria-pressed", "false");
    expect(current.className).toContain("bg-accent-primary");
    expect(other.className).not.toContain("bg-accent-primary");
  });

  it("calls onChange with the clicked page id when a non-current tab is clicked", () => {
    const onChange = vi.fn();
    render(<PageTabs pages={pages} current="overview" onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Trace" }));
    expect(onChange).toHaveBeenCalledWith("trace");
  });

  it("still calls onChange when the current tab is clicked", () => {
    const onChange = vi.fn();
    render(<PageTabs pages={pages} current="overview" onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Overview" }));
    expect(onChange).toHaveBeenCalledWith("overview");
  });

  it("merges an optional className onto the root", () => {
    const { container } = render(
      <PageTabs pages={pages} current="overview" onChange={vi.fn()} className="my-extra" />,
    );
    expect(container.firstElementChild?.className).toContain("my-extra");
  });
});
