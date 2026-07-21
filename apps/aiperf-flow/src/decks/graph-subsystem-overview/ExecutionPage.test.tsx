/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { ExecutionPage } from "./ExecutionPage.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <ExecutionPage />
    </ReactFlowProvider>,
  );
}

describe("ExecutionPage", () => {
  it("renders the executor firing demo and advances the tick", () => {
    renderPage();
    expect(screen.getByText("Tick 0 / 5")).toBeInTheDocument();
    expect(screen.getByText(/START is the entry sentinel/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Advance" }));
    expect(screen.getByText("Tick 1 / 5")).toBeInTheDocument();
    expect(screen.getByText(/plan fires first/)).toBeInTheDocument();
  });

  it("selects a materialization branch from the store + cache-bust toggles", () => {
    renderPage();
    // unified + no cache-bust → bytes path selected.
    expect(screen.getByText("materialize_graph_request_unified_bytes")).toBeInTheDocument();
    expect(screen.getByText("Unified A2 · bytes")).toBeInTheDocument();
  });

  it("switches the barrier policy and shows its closure reason", () => {
    renderPage();
    expect(screen.getByText(/closure reason = "quorum"/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "any" }));
    expect(screen.getByText(/closure reason = "any"/)).toBeInTheDocument();
  });

  it("changes the loop aggregator output", () => {
    renderPage();
    // Default concat → "ABC".
    expect(screen.getByText('"ABC"')).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "list" }));
    expect(screen.getByText('["A", "B", "C"]')).toBeInTheDocument();
  });

  it("renders the branch resolution ladder", () => {
    renderPage();
    expect(screen.getByText("trace.selected_branches[source]")).toBeInTheDocument();
    expect(screen.getByText("edge.branch_weights")).toBeInTheDocument();
  });
});
