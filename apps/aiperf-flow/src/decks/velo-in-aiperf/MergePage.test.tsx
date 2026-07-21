/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import { MergePage } from "./MergePage.js";

function renderPage() {
  return render(
    <ReactFlowProvider>
      <MergePage />
    </ReactFlowProvider>,
  );
}

describe("MergePage", () => {
  it("starts in records mode with the four ordinal feeds", () => {
    renderPage();
    expect(screen.getByText("Select radial feeds to complete the reduction.")).toBeInTheDocument();
    expect(screen.getByText("#8")).toBeInTheDocument();
    expect(screen.getByText("#11")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Switch to folded stores" })).toBeInTheDocument();
  });

  it("switches to folded-store mode showing the summation feeds", () => {
    renderPage();
    fireEvent.click(screen.getByRole("button", { name: "Switch to folded stores" }));
    expect(screen.getByText("Σ c0")).toBeInTheDocument();
    expect(screen.getByText("Σ c3")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Switch to exact records" })).toBeInTheDocument();
  });
});
