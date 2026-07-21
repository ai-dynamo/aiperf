/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { TreePage } from "./TreePage.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <TreePage />
    </ReactFlowProvider>,
  );
}

describe("TreePage", () => {
  it("defaults to the folded tree with two aggregators and the tree caption", () => {
    renderPage();
    expect(screen.getByText("aggregator 0")).toBeInTheDocument();
    expect(screen.getByText("cells 0–3")).toBeInTheDocument();
    expect(screen.getByText("aggregator 1")).toBeInTheDocument();
    expect(screen.getByText("8 cell stores → 2 subtree stores → 1 report")).toBeInTheDocument();
  });

  it("switches to flat records with the global-order caption", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Flat records" }));
    expect(screen.getByText("8 raw partitions → controller global-order merge")).toBeInTheDocument();
    expect(screen.queryByText("aggregator 0")).not.toBeInTheDocument();
  });

  it("updates payload volume from the slider", () => {
    renderPage();
    fireEvent.change(screen.getByLabelText("Payload volume"), { target: { value: "40" } });
    expect(screen.getByText("40u")).toBeInTheDocument();
  });
});
