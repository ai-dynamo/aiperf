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
  it("renders the flat controller merge and hierarchy-refusal caption", () => {
    renderPage();
    expect(screen.getByText(/8 cell partitions → controller merge → one report/i)).toBeInTheDocument();
    expect(screen.queryByText("aggregator 0")).not.toBeInTheDocument();
  });

  it("updates payload volume from the slider", () => {
    renderPage();
    fireEvent.change(screen.getByLabelText("Payload volume"), { target: { value: "40" } });
    expect(screen.getByText("40u")).toBeInTheDocument();
  });
});
