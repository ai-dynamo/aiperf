/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { StoryPage } from "./StoryPage.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <StoryPage />
    </ReactFlowProvider>,
  );
}

describe("StoryPage", () => {
  it("opens on page 1 with the launch thesis and invariant", () => {
    renderPage();
    expect(screen.getByText("One run. Many cells. One report.")).toBeInTheDocument();
    expect(screen.getByText(/Cellular story · Launch · Page 1 of 20/)).toBeInTheDocument();
    expect(
      screen.getByText("Scaling changes placement, not the measurement contract."),
    ).toBeInTheDocument();
    // Page-1 introduced nodes are rendered in the graph.
    expect(screen.getAllByText("Authored run").length).toBeGreaterThan(0);
    expect(screen.getAllByText("One report").length).toBeGreaterThan(0);
  });

  it("advances to the next story page on Next", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("Author the Config v2 run")).toBeInTheDocument();
    expect(screen.getByText(/Page 2 of 20/)).toBeInTheDocument();
  });

  it("jumps to the full atlas on page 20", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Jump to full atlas" }));
    expect(screen.getByText("Full cellular system atlas")).toBeInTheDocument();
    expect(screen.getByText(/Page 20 of 20/)).toBeInTheDocument();
  });

  it("shows the reduction simulation strip on the Reduce pages", () => {
    renderPage();
    // Rail jump to page 16 (retain reduction).
    fireEvent.click(screen.getByRole("button", { name: "Page 16: Retain rows for exact artifacts" }));
    expect(screen.getByText("Retain rows for exact artifacts")).toBeInTheDocument();
    expect(screen.getByText("WIRE OUTPUT")).toBeInTheDocument();
    expect(screen.getByText("O(records) · exact percentiles")).toBeInTheDocument();
  });

  it("shows the fidelity ladder on page 19", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Page 19: Merge, publish, and understand the boundary" }));
    expect(screen.getByText("Cellular fidelity ladder")).toBeInTheDocument();
    expect(screen.getByText("T2 Hierarchical")).toBeInTheDocument();
  });
});
