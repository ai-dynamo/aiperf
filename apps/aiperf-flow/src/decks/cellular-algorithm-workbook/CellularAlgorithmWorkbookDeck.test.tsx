/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { CellularAlgorithmWorkbookDeck } from "./CellularAlgorithmWorkbookDeck.js";

function renderDeck() {
  return render(
    <ReactFlowProvider>
      <CellularAlgorithmWorkbookDeck />
    </ReactFlowProvider>,
  );
}

describe("CellularAlgorithmWorkbookDeck", () => {
  it("renders the deck title and defaults to the Workbook page", () => {
    renderDeck();
    expect(screen.getByText("Reason from gate to artifact")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Step" })).toBeInTheDocument();
  });

  it("switches to Compose and Decisions pages via the tabs", () => {
    renderDeck();
    fireEvent.click(screen.getByRole("button", { name: "Compose" }));
    expect(screen.getByText("Compose an execution route")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Decisions" }));
    expect(screen.getByText("Decision laboratory")).toBeInTheDocument();
  });
});
