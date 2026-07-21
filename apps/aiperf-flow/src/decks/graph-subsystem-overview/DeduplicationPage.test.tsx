/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { DeduplicationPage } from "./DeduplicationPage.js";

function renderPage(audience: "manager" | "developer" = "manager") {
  return render(
    <ReactFlowProvider>
      <DeduplicationPage audience={audience} />
    </ReactFlowProvider>,
  );
}

describe("DeduplicationPage", () => {
  it("shows the dedup explanation and toggles to naive replay", () => {
    renderPage();
    expect(screen.getByText("Segment-trie dedup")).toBeInTheDocument();
    expect(screen.getByText(/those blocks hash to the/)).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("switch")[0]!);
    expect(screen.getByText("Naive replay")).toBeInTheDocument();
    expect(screen.getByText(/each request is an independent blob/)).toBeInTheDocument();
  });

  it("renders the content-parent trie nodes", () => {
    renderPage();
    expect(screen.getByText("shared root")).toBeInTheDocument();
    expect(screen.getByText("branches at req 1")).toBeInTheDocument();
  });

  it("renders the prefix-cache reuse composition table", () => {
    renderPage();
    expect(screen.getByText("Reused (prefix-cache) tokens")).toBeInTheDocument();
  });

  it("hides the on-disk store shapes in manager view, shows them in developer view", () => {
    const { unmount } = renderPage("manager");
    expect(screen.queryByText("GraphSegmentUnifiedBackingStore")).not.toBeInTheDocument();
    unmount();
    renderPage("developer");
    expect(screen.getByText("GraphSegmentUnifiedBackingStore")).toBeInTheDocument();
    expect(screen.getByText("A2 interned layout")).toBeInTheDocument();
  });
});
