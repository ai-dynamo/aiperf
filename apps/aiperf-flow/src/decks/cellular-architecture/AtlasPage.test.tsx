/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { AtlasPage } from "./AtlasPage.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <AtlasPage />
    </ReactFlowProvider>,
  );
}

describe("AtlasPage", () => {
  it("renders the heading, plane legend, and fidelity readouts", () => {
    renderPage();
    expect(
      screen.getByText("One benchmark. Many autonomous cells. One measurement contract."),
    ).toBeInTheDocument();
    expect(screen.getByText("Control plane")).toBeInTheDocument();
    expect(screen.getByText("Execution plane")).toBeInTheDocument();
    // T1 (default) sketch route.
    expect(screen.getByText("Approximate · t-digest")).toBeInTheDocument();
    expect(screen.getByText("Cells → controller")).toBeInTheDocument();
  });

  it("re-derives the route topology when a recipe is selected", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "T2 Hierarchical" }));
    expect(screen.getByText("Cells → aggregators → controller")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "T3 External sink" }));
    expect(screen.getByText("Cells → external ingest (planned)")).toBeInTheDocument();
    expect(
      screen.getByText(/T3 no-central-merge external streaming remains planned/),
    ).toBeInTheDocument();
  });

  it("renders atlas nodes and the cell cross-section ownership rows", () => {
    renderPage();
    expect(screen.getAllByText("Controller").length).toBeGreaterThan(0);
    expect(screen.getByText("Inside one cell")).toBeInTheDocument();
    expect(screen.getByText("Cell 0 owns")).toBeInTheDocument();
    expect(screen.getByText("Cell 2 owns")).toBeInTheDocument();
  });

  it("shows engineer-inspector evidence for the default controller selection", () => {
    renderPage();
    expect(screen.getByText("Engineer inspector")).toBeInTheDocument();
    expect(screen.getByText("run_cellular")).toBeInTheDocument();
    expect(screen.getByText("rust/runtime/src/engine/cellular_controller.rs")).toBeInTheDocument();
  });
});
